"""Produce a numeric replication report, without generating figures."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from .campaign import jobs
from .train import atomic_json


REFERENCE = {
    ('dust','1M',256):6.067, ('dust','1M',1024):5.934,
    ('dust','1M',4096):5.911, ('dust','1M',16384):5.916,
    ('dust','10M',256):5.231, ('dust','10M',1024):5.111,
    ('dust','10M',4096):5.072, ('dust','10M',16384):5.049,
    ('backprop','1M',None):5.959, ('backprop','10M',None):4.989,
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    groups={};missing=[]
    for job in jobs():
        key=(job['method'],job['tokens'],job['population'])
        groups.setdefault(key,[])
        path=args.root/'runs'/job['id']/'result.json'
        if not path.exists():missing.append(job['id']);continue
        r=json.loads(path.read_text())
        if not r.get('complete'):missing.append(job['id']);continue
        groups[key].append(dict(seed=job['seed'],test_loss=r['test_at_best_val'],best_val=r['best_val']))
    rows=[]
    for key,entries in groups.items():
        values=[e['test_loss'] for e in entries]
        mean=statistics.mean(values) if values else None
        rows.append(dict(method=key[0],tokens=key[1],population=key[2],seeds=len(entries),
                         mean_test_loss=mean,std_test_loss=statistics.stdev(values) if len(values)>1 else None,
                         paper_mean=REFERENCE[key],difference=mean-REFERENCE[key] if mean is not None else None,
                         individual=entries))
    report=dict(complete=not missing,completed_runs=30-len(missing),expected_runs=30,
                reference='https://qlabs.sh/research/dust/appendix.html#app-c',
                selection='Each seed: test at minimum-validation checkpoint; aggregate three seeds.',
                rows=rows,missing=missing)
    atomic_json(args.root/'summary.json',report)
    with (args.root/'summary.csv').open('w',newline='') as stream:
        fields=[k for k in rows[0] if k!='individual']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({k:v for k,v in r.items() if k!='individual'} for r in rows)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
