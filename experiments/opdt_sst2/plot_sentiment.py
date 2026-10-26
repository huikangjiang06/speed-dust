import numpy as np,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path(__file__).parent/'sentiment-audit');args=p.parse_args()
R=args.input;a=np.load(R/'activations.npz');meta=json.loads((R/'samples.json').read_text());y=a['fit_labels'];v=a['probe_labels']
def axis(x,y):
 n=x[y==0].mean(0);p=x[y==1].mean(0);u=p-n;norm=np.linalg.norm(u)
 return u/max(norm,1e-30),(p+n)/2,norm
def score(x,u,c):
 z=x-c;return np.sum(z*u,axis=1)/np.maximum(np.linalg.norm(z,axis=1),1e-30)
def auc(s,y):
 d=s[y==1,None]-s[y==0][None,:];return float(((d>0)+.5*(d==0)).mean())
fig,axs=plt.subplots(3,3,figsize=(14,9),sharex=True)
results=[];rng=np.random.default_rng(813)
for i,site in enumerate(meta['sites']):
 x=a['fit_'+str(i)].astype(float);q=a['probe_'+str(i)].astype(float);u,c,n=axis(x,y)
 fit=score(x,u,c);probe=score(q,u,c);loo=[]
 for k in range(len(y)):
  keep=np.arange(len(y))!=k;ul,cl,_=axis(x[keep],y[keep]);loo.append(score(x[k:k+1],ul,cl)[0])
 loo=np.array(loo);au=auc(probe,v)
 null=[auc(probe,rng.permutation(v)) for _ in range(2000)]
 p=(1+sum(z>=au for z in null))/2001
 results.append(dict(site=site,direction_norm=float(n),fit_auc=auc(fit,y),leave_one_out_auc=auc(loo,y),independent_auc=au,independent_accuracy_at_zero=float(((probe>0)==v).mean()),permutation_p_one_sided=p))
 for j,(scores,labels) in enumerate([(fit,y),(loo,y),(probe,v)]):
  ax=axs[i,j]
  for lab,col,name in [(0,'#d86525','Negative'),(1,'#2778c4','Positive')]:
   vals=scores[labels==lab];ax.scatter(vals,lab+rng.uniform(-.14,.14,len(vals)),color=col,alpha=.8,s=28,label=name,edgecolors='white',linewidths=.3)
  ax.axvline(0,color='grey',ls='--',lw=.8);ax.set_yticks([0,1]);ax.set_yticklabels(['Negative','Positive']);ax.set_ylim(-.5,1.5);ax.grid(axis='x',alpha=.2)
  name=['Embedding (same final cue token)','Middle layer 14: MLP output','Final layer 27: MLP output'][i]
  ax.set_title(name+'\nAUC '+format(auc(scores,labels),'.3f'),fontsize=10)
  if i==2:ax.set_xlabel('Cosine alignment with sentiment direction')
  ax.set_xlim(-1,1)
for j,title in enumerate(['Original 64: direction fitted on these\n(apparent separation can be inflated)','Original 64: leave-one-out direction\n(each review excluded from its direction)','Different 64: frozen direction\n(independent check, training split only)']):axs[0,j].text(.5,1.42,title,transform=axs[0,j].transAxes,ha='center',va='bottom',fontsize=11,weight='bold')
fig.suptitle('Does positive minus negative define a sentiment axis?',fontsize=18,y=.995)
fig.text(.5,.01,'Each dot is one review. Activations are centered at the class-mean midpoint. AUC 0.5 = no separation; 1.0 = perfect ordering.\nPretrained Qwen3-0.6B, SST-2, 32 positive + 32 negative per set. Three diagnostic sites, not all layers.',ha='center',fontsize=10)
fig.subplots_adjust(top=.79,bottom=.11,hspace=.65,wspace=.27)
fig.savefig(R/'sentiment-alignment.png',dpi=180);fig.savefig(R/'sentiment-alignment.svg')
(R/'statistics.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
