import os,sys,subprocess,time,json,html
from pathlib import Path
R=Path(__file__).parent
os.chdir(R.parents[1])
started=time.time();jobs=[]
gpu_ids=os.environ.get('OPDT_GPUS','0,1,2,3').split(',')
if len(gpu_ids)!=4:raise ValueError('OPDT_GPUS must contain four GPU IDs')
for gpu,(method,lr) in enumerate([('ordinary','1e-6'),('semantic','1e-6'),('ordinary','1e-7'),('semantic','1e-7')]):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu_ids[gpu],OPDT_LR=lr)
 log=open(R/(method+'-'+lr+'.log'),'w')
 p=subprocess.Popen([sys.executable,'-u','-m','experiments.opdt_sst2.pilot',method,'--steps','100'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 jobs.append((p,method,lr))
while any(p.poll() is None for p,_,_ in jobs) and time.time()-started<27*60:time.sleep(5)
for p,_,_ in jobs:
 if p.poll() is None:
  import signal
  os.killpg(p.pid,signal.SIGTERM)
for p,_,_ in jobs:
 try:p.wait(timeout=15)
 except subprocess.TimeoutExpired:os.killpg(p.pid,9);p.wait()
summary=[]
for p,m,lr in jobs:
 folder=R/'results'/(m+'-lr'+str(float(lr)))
 item=dict(method=m,lr=float(lr),returncode=p.returncode)
 for name in ['final','status','gradient-diagnostic','config']:
  f=folder/(name+'.json')
  if f.exists():item[name]=json.loads(f.read_text())
 summary.append(item)
(R/'summary.json').write_text(json.dumps(summary,indent=2))
parts=['<html><meta charset="utf-8"><title>OPDT 30-minute pilot</title><body><h1>Ordinary vs semantic-direction DUST</h1><p>Qwen3-0.6B, SST-2. Two learning rates, seed 42, 100 updates, 8 draws per site, batch 2. Identical training data. Training-derived sentiment directions. Development-only tuning; official SST-2 validation not evaluated. No agent. Single-seed feasibility study, not evidence of benchmark superiority.</p>']
parts.append('<table border="1" cellpadding="8"><tr><th>Method</th><th>LR</th><th>Exit</th><th>Dev accuracy</th><th>Dev loss</th><th>Predictions [negative,positive]</th><th>Gradient cosine</th></tr>')
for s in summary:
 d=s.get('final',{}).get('dev',s.get('status',{}).get('dev',{}));g=s.get('gradient-diagnostic',{})
 parts.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [s['method'],s['lr'],s['returncode'],d.get('accuracy'),d.get('loss'),d.get('prediction_counts'),g.get('cosine')])+'</tr>')
parts.append('</table>')
try:
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(15,4))
 for s in summary:
  f=R/'results'/(s['method']+'-lr'+str(s['lr']))/'metrics.jsonl'
  if not f.exists():continue
  rows=[json.loads(l) for l in f.read_text().splitlines() if l.strip()];rows=[r for r in rows if 'dev' in r]
  for ax,key in zip(axes,['loss','accuracy','balanced_accuracy']):
   ax.plot([r['step'] for r in rows],[r['dev'][key] for r in rows],label=s['method']+' '+str(s['lr']));ax.set_xlabel('Updates');ax.set_title('Development '+key);ax.legend(fontsize=7)
 fig.tight_layout();fig.savefig(R/'curves.png',dpi=160)
 import base64
 parts.append('<img style="max-width:100%" src="data:image/png;base64,'+base64.b64encode((R/'curves.png').read_bytes()).decode()+'">')
except Exception as e:parts.append('<p>Plot generation error: '+html.escape(str(e))+'</p>')
(R/'OPDT-pilot.html').write_text(''.join(parts)+'</body></html>')
import zipfile
with zipfile.ZipFile(R/'Claude-handoff.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in [R/'summary.json',R/'OPDT-pilot.html',R/'pilot.py',R/'campaign.py',R/'README.md']+list((R/'results').rglob('*.json'))+list((R/'results').rglob('*.jsonl')):
  z.write(f,f.relative_to(R))
 if (R/'curves.png').exists():z.write(R/'curves.png','curves.png')
(R/'COMPLETE.json').write_text(json.dumps(dict(elapsed_seconds=time.time()-started,exit_codes=[p.returncode for p,_,_ in jobs])))
