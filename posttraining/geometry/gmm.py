"""Streamed full-data diagonal Gaussian mixtures with FP64 EM statistics."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import torch
from safetensors.torch import load_file

from common.flops import ComputeLedger
from ..configuration import load_config
from ..runtime import atomic_json, atomic_save, cpu_tree, source_hashes
from .cache import SiteRows, validate_cache
from .common import FORMAT, atomic_tensors, check_file, exclusive, file_record, fingerprint, site_seed


def responsibilities(x, weights, means, variances):
    """x and means use a fixed global origin internally, all arithmetic FP64."""
    inverse=variances.reciprocal()
    mahal=x.square() @ inverse.T - 2*(x @ (means*inverse).T) + (means.square()*inverse).sum(-1)
    logp=weights.log()[None]-.5*(mahal+variances.log().sum(-1)[None]+x.shape[-1]*math.log(2*math.pi))
    partition=torch.logsumexp(logp,-1)
    return (logp-partition[:,None]).exp(),partition


def global_statistics(reader, chunk_rows, device):
    count=0;mean=torch.zeros(reader.width,dtype=torch.float64,device=device);m2=torch.zeros_like(mean)
    for _,raw in reader.chunks(chunk_rows):
        x=raw.to(device=device,dtype=torch.float64)
        part_mean=x.mean(0);part_m2=(x-part_mean).square().sum(0)
        new_count=count+len(x);delta=part_mean-mean
        m2+=part_m2+delta.square()*(count*len(x)/new_count)
        mean+=delta*(len(x)/new_count);count=new_count
    if count!=reader.count:
        raise AssertionError('Global statistics did not visit the full fitting dataset')
    return mean,m2/count


def em_iteration(reader, parameters, center, floor, chunk_rows, device, initial_rows, iteration):
    weights,means,variances=parameters
    counts=torch.zeros_like(weights);wx=torch.zeros_like(means);wx2=torch.zeros_like(means)
    likelihood=torch.zeros((),dtype=torch.float64,device=device);rows=0
    for _,raw in reader.chunks(chunk_rows):
        x=raw.to(device=device,dtype=torch.float64)-center
        r,logp=responsibilities(x,weights,means,variances)
        counts+=r.sum(0);wx+=r.T@x;wx2+=r.T@x.square();likelihood+=logp.sum();rows+=len(x)
    if rows!=reader.count:
        raise AssertionError('EM iteration did not read every cached row exactly once')
    empty=counts==0
    denom=counts.clamp_min(1e-300)
    new_means=wx/denom[:,None]
    new_variances=torch.maximum(wx2/denom[:,None]-new_means.square(),floor[None])
    effective=counts.clone();reinitialized=[]
    for component in empty.nonzero().flatten().tolist():
        row=(component+iteration)%len(initial_rows)
        new_means[component]=initial_rows[row]
        new_variances[component]=variances.mean(0).clamp_min(floor)
        effective[component]=1.
        reinitialized.append(dict(component=component,initialization_sample_offset=row,reason='zero_FP64_effective_count'))
    new_weights=effective/effective.sum()
    if not all(torch.isfinite(value).all() for value in (new_weights,new_means,new_variances,likelihood)):
        raise FloatingPointError('Nonfinite streamed EM statistic')
    return (new_weights,new_means,new_variances),dict(iteration=iteration,rows=rows,
        e_step_average_log_likelihood=likelihood.item()/rows,effective_component_counts=counts.cpu().tolist(),
        reinitializations=reinitialized,minimum_variance=new_variances.min().item())


class DiagonalGMM:
    def __init__(self, weights, means, variances, device='cpu'):
        self.weights=weights.to(device=device,dtype=torch.float32)
        self.means=means.to(device=device,dtype=torch.float32)
        self.variances=variances.to(device=device,dtype=torch.float32)
        if self.means.ndim!=2 or self.variances.shape!=self.means.shape or self.weights.shape!=(len(self.means),):
            raise ValueError('Invalid mixture tensor shapes')
        if not all(torch.isfinite(t).all() for t in (self.weights,self.means,self.variances)) or (self.weights<=0).any() or (self.variances<=0).any():
            raise ValueError('Mixture parameters must be finite with positive weights/variances')
        if abs(self.weights.sum().item()-1)>1e-5:
            raise ValueError('Mixture weights must sum to one')
        self.std=self.variances.sqrt()
        self.width=self.means.shape[1]

    def sample(self, shape, component_generator, normal_generator):
        if shape[-1]!=self.width:
            raise ValueError('GMM width differs from the direct-probe coordinates')
        components=torch.multinomial(self.weights,math.prod(shape[:-1]),replacement=True,generator=component_generator).reshape(shape[:-1])
        z=torch.randn(*shape,device=self.means.device,dtype=torch.float32,generator=normal_generator)
        target=self.means[components]+self.std[components]*z
        return target,components


def fit_site(cache, manifest, site, output, settings, device, *, resume=True, stop_after_iterations=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    name=site['name'];base=output/name
    metadata_path=base.with_suffix(base.suffix+'.json')
    tensor_path=base.with_suffix(base.suffix+'.safetensors')
    checkpoint=base.with_suffix(base.suffix+'.checkpoint.pt')
    fit_settings={key:getattr(settings,key) for key in ('components','max_iter','tolerance','variance_floor_relative',
        'variance_floor_absolute','fit_chunk_rows','seed','covariance_type')}
    identity=dict(format=FORMAT,cache_fingerprint=manifest['fingerprint'],site=site,rows=manifest['expected_rows'],
        fit_settings=fit_settings,statistics_dtype='float64',coordinate_system='original_activation_coordinates',
        internal_centering='fixed_full_data_mean',variance_floor='max(relative*full_data_population_variance, absolute)',
        sources=source_hashes())
    with exclusive(base.with_suffix(base.suffix+'.lock')):
        if metadata_path.exists():
            previous=json.loads(metadata_path.read_text())
            if previous['identity']!=identity:
                raise ValueError(f'GMM fit identity changed at {name}')
            if previous.get('complete'):
                check_file(output,previous['tensor']);return previous
        reader=SiteRows(cache,manifest,name)
        ledger=ComputeLedger(device,extra_phases=('fitting',))
        started=time.monotonic();elapsed=0.;history=[];iteration=0;converged=False
        seed=site_seed(settings.seed,name,'initialization')
        if checkpoint.exists():
            if not resume:
                raise FileExistsError('Fit checkpoint exists; explicitly resume or choose another output')
            state=torch.load(checkpoint,map_location=device,weights_only=False)
            if state['identity']!=identity:
                raise ValueError('GMM checkpoint identity changed')
            center,global_variance,floor=state['center'],state['global_variance'],state['floor']
            parameters=state['parameters'];initial_rows=state['initial_rows'];initial_ids=state['initial_ids']
            history=state['history'];iteration=state['iteration'];elapsed=state['wall_seconds']
            converged=state.get('converged',False)
            ledger.load_state_dict(state['compute'])
            reader.bytes_read=state['bytes_read'];reader.io_seconds=state['io_seconds']
        else:
            if reader.count<settings.components:
                raise ValueError('The fitting set has fewer observations than configured components')
            with ledger.measure('fitting'):
                center,global_variance=global_statistics(reader,settings.fit_chunk_rows,device)
                floor=(settings.variance_floor_relative*global_variance).clamp_min(settings.variance_floor_absolute)
                initial_ids=sorted(random.Random(seed).sample(range(reader.count),min(reader.count,max(256,settings.components))))
                raw,_=reader.selected(initial_ids)
                initial_rows=raw.to(device=device,dtype=torch.float64)-center
                chosen=random.Random(seed+1).sample(range(len(initial_rows)),settings.components)
                parameters=(torch.full((settings.components,),1/settings.components,dtype=torch.float64,device=device),
                    initial_rows[chosen].clone(),torch.maximum(global_variance,floor).repeat(settings.components,1))
        while iteration<settings.max_iter and not converged:
            iteration+=1
            before=ledger.totals['fitting'];io_before=reader.bytes_read;when=time.monotonic()
            with ledger.measure('fitting'):
                parameters,record=em_iteration(reader,parameters,center,floor,settings.fit_chunk_rows,device,initial_rows,iteration)
            record.update(seconds=time.monotonic()-when,flops=ledger.totals['fitting']-before,bytes_read=reader.bytes_read-io_before)
            previous=history[-1]['e_step_average_log_likelihood'] if history else None
            improvement=record['e_step_average_log_likelihood']-previous if previous is not None else None
            record['average_log_likelihood_improvement']=improvement
            if improvement is not None and improvement < -1e-7:
                record['likelihood_decrease']=True
            converged=improvement is not None and 0<=improvement<settings.tolerance and not record['reinitializations']
            history.append(record)
            state=dict(identity=identity,center=center,global_variance=global_variance,floor=floor,
                parameters=parameters,initial_rows=initial_rows,initial_ids=initial_ids,iteration=iteration,history=history,converged=converged,
                wall_seconds=elapsed+time.monotonic()-started,compute=ledger.state_dict(),bytes_read=reader.bytes_read,io_seconds=reader.io_seconds)
            atomic_save(checkpoint,cpu_tree(state))
            atomic_json(metadata_path,dict(identity=identity,complete=False,iteration=iteration,history=history,
                wall_seconds=state['wall_seconds'],bytes_read=reader.bytes_read,io_seconds=reader.io_seconds,compute=ledger.state_dict()))
            print(json.dumps(dict(stage='gmm',site=name,iteration=iteration,log_likelihood=record['e_step_average_log_likelihood'],
                                  rows=record['rows'],converged=converged)),flush=True)
            if converged:
                break
            if stop_after_iterations is not None and iteration>=stop_after_iterations:
                return json.loads(metadata_path.read_text())
        with ledger.measure('fitting'):
            final_ll=torch.zeros((),dtype=torch.float64,device=device);final_rows=0
            for _,raw in reader.chunks(settings.fit_chunk_rows):
                _,logp=responsibilities(raw.to(device=device,dtype=torch.float64)-center,*parameters)
                final_ll+=logp.sum();final_rows+=len(raw)
            if final_rows!=reader.count:
                raise AssertionError('Final likelihood did not cover all rows')
            weights,means,variances=parameters
            original_means=means+center
        atomic_tensors(tensor_path,dict(weights=weights,means=original_means,variances=variances,
                                       global_mean=center,global_variance=global_variance,variance_floor=floor))
        result=dict(identity=identity,complete=True,covariance_type='diag',component_count=settings.components,
            width=site['width'],fit_rows=reader.count,converged=converged,termination='tolerance' if converged else 'max_iter',
            iterations=iteration,history=history,final_average_log_likelihood=final_ll.item()/final_rows,
            initialization_row_ids=initial_ids,initialization_seed=seed,tensor=file_record(tensor_path,output),
            wall_seconds=elapsed+time.monotonic()-started,bytes_read=reader.bytes_read,io_seconds=reader.io_seconds,
            compute=ledger.state_dict())
        atomic_json(metadata_path,result)
        return result


def fit_all(config, cache, output, *, worker=0, workers=1, resume=True):
    if not 0<=worker<workers:
        raise ValueError('Invalid independent site-worker partition')
    manifest=validate_cache(cache,checksums=False)
    sites=manifest['identity']['reference']['sites']
    results=[]
    for index,site in enumerate(sites):
        if index%workers==worker:
            try:
                results.append(fit_site(cache,manifest,site,output,config.geometry,config.training.device,resume=resume))
            except Exception as error:
                Path(output).mkdir(parents=True,exist_ok=True)
                atomic_json(Path(output)/(site['name']+'.failure.json'),dict(site=site['name'],
                    cache_fingerprint=manifest['fingerprint'],error_type=type(error).__name__,error=str(error),
                    numerical_failure=isinstance(error,FloatingPointError)))
                raise
    atomic_json(Path(output)/f'worker_{worker:03d}.json',dict(worker=worker,workers=workers,
        complete=all(item['complete'] for item in results),sites=[r['identity']['site']['name'] for r in results],
        wall_seconds=sum(r['wall_seconds'] for r in results),flops=sum(r['compute']['fitting_flops'] for r in results),
        cache_fingerprint=manifest['fingerprint']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--cache',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--worker',type=int,default=0);parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--set',action='append',default=[])
    args=parser.parse_args()
    fit_all(load_config(args.config,args.set),args.cache,args.output,worker=args.worker,workers=args.workers)


if __name__=='__main__':
    main()
