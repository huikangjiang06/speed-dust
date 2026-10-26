import os,json,time,hashlib,random,argparse
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from datasets import load_dataset
from posttraining.hf_dust import HFDUST,autocast
from posttraining.configuration import TrainingSettings,DustSettings,NoiseSettings
ROOT=Path(__file__).parent
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
def save(p,x):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def prepare():
 from huggingface_hub import snapshot_download,HfApi
 rev='c1899de289a04d12100db370d81485cdf75e47ca'  # checkpoint used in the pilot
 path=snapshot_download('Qwen/Qwen3-0.6B',revision=rev)
 ds=load_dataset('glue','sst2'); ids=np.random.default_rng(20261026).permutation(len(ds['train']))
 def rows(ix,split):return [dict(id=int(i),text=ds[split][int(i)]['sentence'],label=ds[split][int(i)]['label']) for i in ix]
 data=dict(train=rows(ids[:1024],'train'),dev=rows(ids[1024:1152],'train'),test=rows(range(len(ds['validation'])),'validation'))
 save(ROOT/'data.json',data);save(ROOT/'model.json',dict(path=path,revision=rev,dataset_fingerprint=ds['train']._fingerprint))
def run(method,steps):
 lr=float(os.environ.get('OPDT_LR','1e-6')); draws=8
 out=ROOT/'sentiment-audit';out.mkdir(exist_ok=True)
 if (out/'metrics.jsonl').exists():raise RuntimeError('Output already contains metrics; archive it before rerunning')
 torch.manual_seed(42);np.random.seed(42);random.seed(42)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 meta=json.load(open(ROOT/'model.json'));data=json.load(open(ROOT/'data.json'))
 tok=AutoTokenizer.from_pretrained(meta['path']);tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(meta['path'],torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval().requires_grad_(False)
 answers=[tok.encode(' terrible',add_special_tokens=False),tok.encode(' great',add_special_tokens=False)]
 assert all(len(x)==1 for x in answers),answers
 answer_ids=[x[0] for x in answers]
 import posttraining.hf_dust as hd
 def classification_losses(logits,labels):
  targets=torch.full_like(labels,-100)
  for k,token in enumerate(answer_ids):targets[labels==token]=k
  return torch.nn.functional.cross_entropy(logits[...,answer_ids].float().reshape(-1,2),targets.reshape(-1),ignore_index=-100,reduction='none').view_as(labels)
 hd.token_losses=classification_losses
 class ClassificationDUST(HFDUST):
  def head_gradient(self):
   n,b,t=self.settings.nrep,self.batch,self.length
   raw=self.outputs['lm_head'][...,answer_ids].float()
   targets=torch.full_like(self.y,-100)
   for k,token in enumerate(answer_ids):targets[self.y==token]=k
   draws=self.settings.head_draws_per_slab
   sigma=self.sigma('lm_head');error=torch.zeros_like(raw)
   for _ in range(draws//n):
    a=self.noise(n,b,t,2)
    changed=raw[None]+sigma*a
    loss=torch.nn.functional.cross_entropy(changed.reshape(-1,2),targets.repeat(n,1).reshape(-1),ignore_index=-100,reduction='none').view(n,b,t)
    reward=(self.clean_loss[None]-loss)*self.valid[None]
    error-=torch.einsum('nbt,nbtd->btd',reward/sigma,a)
   grad=torch.zeros_like(self.layout.head.weight)
   grad[answer_ids]=torch.einsum('btd,bti->di',error/draws,self.inputs['lm_head'])
   self.probe_counts['head_draws']=draws
   return grad
 # Two-answer CE equals CE on the answers' full-vocabulary log probabilities.
 z=torch.randn(7,11);y=torch.arange(7)%2
 assert torch.allclose(torch.nn.functional.cross_entropy(z[:,[2,8]],y),torch.nn.functional.cross_entropy(z.log_softmax(-1)[:,[2,8]],y),atol=1e-6)
 z2=z.clone();z2[:,0]+=100
 assert torch.equal(z[:,[2,8]],z2[:,[2,8]])
 def batch(rows):
  prompts=[r['text'].strip()+' It was' for r in rows]
  # Preserve the answer cue when truncating long reviews.
  seq=[tok.encode(p,add_special_tokens=False)[-64:] for p in prompts];n=max(map(len,seq))
  x=torch.full((len(rows),n),tok.pad_token_id,device='cuda',dtype=torch.long);mask=torch.zeros_like(x);y=torch.full_like(x,-100)
  for i,(r,ids) in enumerate(zip(rows,seq)):
   x[i,:len(ids)]=torch.tensor(ids,device='cuda');mask[i,:len(ids)]=1;y[i,len(ids)-1]=answer_ids[r['label']]
  return dict(input_ids=x,attention_mask=mask,labels=y)
 @torch.no_grad()
 def evaluate(rows):
  loss=full_loss=correct=count=0;conf=torch.zeros(2,2,dtype=torch.long)
  probs=torch.zeros(2,dtype=torch.float64)
  for j in range(0,len(rows),4):
   rr=rows[j:j+4];b=batch(rr)
   with autocast(torch.device('cuda'),'bf16'):logits=model(input_ids=b['input_ids'],attention_mask=b['attention_mask'],use_cache=False).logits
   ix=b['attention_mask'].sum(1)-1;z=logits[torch.arange(len(rr),device='cuda'),ix].float();labels=torch.tensor([r['label'] for r in rr],device='cuda')
   pair=z[:,answer_ids];pred=pair.argmax(1)
   loss+=torch.nn.functional.cross_entropy(pair,labels,reduction='sum').item()
   full_loss+=torch.nn.functional.cross_entropy(z,torch.tensor(answer_ids,device='cuda')[labels],reduction='sum').item()
   correct+=(pred==labels).sum().item();count+=len(rr);probs+=pair.softmax(-1).sum(0).cpu().double()
   for y,p in zip(labels.cpu(),pred.cpu()):conf[y,p]+=1
  perclass=[conf[k,k].item()/max(1,conf[k].sum().item()) for k in range(2)]
  return dict(loss=loss/count,full_vocabulary_loss=full_loss/count,accuracy=correct/count,n=count,label_order=['negative','positive'],confusion_true_rows_pred_columns=conf.tolist(),prediction_counts=conf.sum(0).tolist(),true_counts=conf.sum(1).tolist(),class_accuracy=perclass,balanced_accuracy=sum(perclass)/2,mean_candidate_probability=(probs/count).tolist())
 train=TrainingSettings(precision='bf16',layers='all',train_embedding=True,train_head=True)
 L=model.config.num_hidden_layers
 settings=DustSettings(direct_population=draws*(4*L+1),nrep=2,allocation={**{k:[draws]*L for k in ['writers','gate','up','attention']},'embedding':draws},local_q=draws,local_k=draws,local_v=draws,head_slab_size=1024,head_draws_per_slab=draws,head_centering='none',noise=NoiseSettings())
 est=ClassificationDUST(model,train,settings,42)
 # Architecture checks before training: native/cached forward and Qwen3 q/k normalization.
 b=batch(data['train'][:2]);est.capture(b);baseline=est.clean_loss.clone();site=f'model.layers.{L//2}.mlp.down_proj';est.jitter={site:torch.zeros_like(est.outputs[site])}
 cached=est.losses(b['input_ids'],b['labels']);full=est.losses(b['input_ids'],b['labels'],force_full=True);est.jitter={}
 assert torch.allclose(cached,full,atol=.02,rtol=.01),(cached-full).abs().max().item()
 attention_errors=[]
 for l in [0,L//2,L-1]:
  q,k,v,qp,kp,process,attend=est.attention_state(l)
  diff=(attend(qp,kp,v).flatten(-2).float()-est.outputs[f'o@{l}'].float()).abs().max().item();attention_errors.append(diff);assert diff<.04,diff
 save(out/'checks.json',dict(cached_max_error=(cached-full).abs().max().item(),attention_max_errors=attention_errors))
 # Match fixed activation-relative scales using training-only examples in BOTH arms.
 for s in est.outputs:
  est.site_sigmas[s]=max(1e-5,.03*est.outputs[s].float().square().mean().sqrt().item())
 est.site_sigmas['lm_head']=.05
 extraction_seconds=0.;direction_ids=[];directions={}
 if method=='semantic':
  start=time.time();sums={};counts=[0,0];picked=[];vectors={}
  for r in data['train']:
   if counts[r['label']]<32:picked.append(r);counts[r['label']]+=1
   if counts==[32,32]:break
  for r in picked:
   est.capture(batch([r]));direction_ids.append(r['id'])
   for s,x in est.outputs.items():
    if s=='lm_head' or s.endswith(('.q_proj','.k_proj','.v_proj')):continue
    if s not in sums:sums[s]=[torch.zeros(x.shape[-1],device='cuda'),torch.zeros(x.shape[-1],device='cuda')]
    sums[s][r['label']]+=x[0,-1].float()/32
    if s in ['model.embed_tokens',f'model.layers.{L//2}.mlp.down_proj',f'model.layers.{L-1}.mlp.down_proj']:
     vectors.setdefault(s,[]).append(x[0,-1].float().cpu().numpy())
  for s,(neg,pos) in sums.items():
   v=pos-neg
   if v.norm()>1e-8:directions[s]=v/v.norm()
  extraction_seconds=time.time()-start
  class Sampler:
   def sample(self,site,shape,clean,sigma):
    a=est.noise(*shape);u=directions.get(site)
    if u is None:return a
    parallel=(a*u).sum(-1,keepdim=True)*u;d=u.numel()
    return .5**.5*(a-parallel)+(.5+.5*d)**.5*parallel
   def score(self,site,a):
    u=directions.get(site)
    if u is None:return a
    p=(a*u).sum(-1,keepdim=True)*u
    return (a-p)/.5+p/(.5+.5*u.numel())
   def observe_injection(self,*a):pass
   def observe_reward(self,*a):pass
  est.direct_sampler=Sampler();torch.save({k:v.cpu() for k,v in directions.items()},out/'directions.pt')
  # C^-1 correction sanity check on an exactly linear objective.
  u=torch.tensor([1.,0.,0.],device='cuda');a=torch.randn(100000,3,device='cuda');a[:,0]*=2**.5;a[:,1:]*=.5**.5;score=a/torch.tensor([2.,.5,.5],device='cuda');g=torch.tensor([.2,-.4,.7],device='cuda');ghat=((a@g)[:,None]*score).mean(0);assert torch.allclose(ghat,g,atol=.015)
 # Save original vectors and an independent, balanced training-only probe set.
 probe=[];counts=[0,0]
 for r in data['train']:
  if r['id'] not in direction_ids and counts[r['label']]<32:probe.append(r);counts[r['label']]+=1
  if counts==[32,32]:break
 other={s:[] for s in vectors}
 for r in probe:
  est.capture(batch([r]))
  for s in other:other[s].append(est.outputs[s][0,-1].float().cpu().numpy())
 arrays={}
 for i,s in enumerate(vectors):
  arrays['fit_'+str(i)]=np.stack(vectors[s]);arrays['probe_'+str(i)]=np.stack(other[s])
 arrays['fit_labels']=np.array([r['label'] for r in picked]);arrays['probe_labels']=np.array([r['label'] for r in probe])
 np.savez_compressed(out/'activations.npz',**arrays)
 save(out/'samples.json',dict(sites=list(vectors),fit=picked,probe=probe,description='Fit: original 32+32 direction examples. Probe: different 32+32 training examples, never used for direction extraction. Pretrained weights, no updates.'))
 est.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--steps',type=int,default=100);a=p.parse_args()
 if a.mode=='prepare':prepare()
 else:run(a.mode,a.steps)
