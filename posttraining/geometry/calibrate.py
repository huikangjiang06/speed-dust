"""Fixed per-site GMM displacement scales and a versioned, validated geometry bank."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import time

import torch
from safetensors.torch import load_file

from common.flops import ComputeLedger
from ..configuration import load_config
from ..hf_dust import HFDUST
from ..runtime import atomic_json, load_model, validate_scales
from ..sft_data import load_prepared
from .cache import SiteRows, validate_cache
from .common import FORMAT, check_file, exclusive, file_record, fingerprint, sha256, site_seed
from .gmm import DiagonalGMM
from .sites import reference_identity, resolve_sites


QUANTILES=[0.,.05,.25,.5,.75,.95,1.]


def norm_quantiles(values):
    values=values.detach().float()
    return {str(q):v for q,v in zip(QUANTILES,torch.quantile(values,torch.tensor(QUANTILES,device=values.device)).cpu().tolist())}


def calibrate_bank(config, cache, fits, output):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    with exclusive(root/'calibration.lock'):
        cache_manifest=validate_cache(cache,checksums=False)
        data,data_manifest=load_prepared(config)
        model,_=load_model(config,'dust')
        validate_scales(config,data_manifest,model)
        sites=resolve_sites(model,config.training)
        reference=reference_identity(config,data_manifest,model,sites)
        if cache_manifest['identity']['reference']!=reference:
            raise ValueError('Geometry cache does not match the initial model and training preprocessing')
        estimator=HFDUST(model,config.training,config.dust,config.training.seed)
        device=estimator.device
        g=config.geometry
        ledger=ComputeLedger(device,extra_phases=('geometry_calibration',))
        row_ids=sorted(random.Random(g.seed).sample(range(cache_manifest['expected_rows']),min(g.calibration_tokens,cache_manifest['expected_rows'])))
        row_recipe=dict(seed=g.seed,algorithm='Python Random.sample without replacement, sorted',count=len(row_ids),
                        population=cache_manifest['expected_rows'],row_ids=row_ids)
        atomic_json(root/'calibration_rows.json',row_recipe)
        identity=dict(format=FORMAT,reference=reference,cache_fingerprint=cache_manifest['fingerprint'],
            cache_manifest_sha256=sha256(Path(cache)/'manifest.json'),baseline_noise=asdict(config.dust.noise),
            baseline_scale_sha256=sha256(config.dust.noise.scales_file) if config.dust.noise.scales_file else None,
            calibration_seed=g.seed,calibration_rows=fingerprint(row_recipe),distance_guard=g.distance_guard,
            rho_cap=1.,calibration_chunk_rows=min(g.fit_chunk_rows,4096))
        started=time.monotonic();entries={};fit_flops=0;fit_seconds=0.;fit_io_bytes=0;calibration_io_bytes=0
        try:
            for site in sites:
                name=site['name'];metadata_path=Path(fits)/(name+'.json')
                fit=json.loads(metadata_path.read_text())
                if not fit['complete'] or fit['fit_rows']!=cache_manifest['expected_rows'] or fit['identity']['cache_fingerprint']!=cache_manifest['fingerprint'] or fit['identity']['site']!=site:
                    raise ValueError(f'Incomplete or incompatible full-data GMM: {name}')
                tensor_path=check_file(fits,fit['tensor'])
                tensors=load_file(str(tensor_path))
                mixture=DiagonalGMM(tensors['weights'],tensors['means'],tensors['variances'],device)
                sigma=estimator.sigma(name)
                site_identity=dict(bank_identity=fingerprint(identity),fit_metadata_sha256=sha256(metadata_path),
                    mixture_sha256=fit['tensor']['sha256'],site=name,sigma=sigma)
                out=root/'sites';out.mkdir(exist_ok=True)
                calibration_file=out/(name+'.calibration.json')
                if calibration_file.exists():
                    calibration=json.loads(calibration_file.read_text())
                    if calibration['identity']!=site_identity:
                        raise ValueError('An existing calibration belongs to different geometry/noise settings')
                    calibration_flops=calibration['flops']
                else:
                    when=time.monotonic()
                    reader=SiteRows(cache,cache_manifest,name)
                    selected,_=reader.selected(row_ids)
                    iso=[];distance=[];combined=[]
                    seeds={kind:site_seed(g.seed,name,kind) for kind in ('isotropic','component','normal')}
                    estimator.generator.manual_seed(seeds['isotropic'])
                    cat=torch.Generator(device).manual_seed(seeds['component'])
                    normal=torch.Generator(device).manual_seed(seeds['normal'])
                    before=ledger.totals['geometry_calibration']
                    with ledger.measure('geometry_calibration'):
                        for offset in range(0,len(selected),identity['calibration_chunk_rows']):
                            h=selected[offset:offset+identity['calibration_chunk_rows']].to(device=device,dtype=torch.float32)
                            epsilon=estimator.noise(*h.shape)
                            target,_=mixture.sample(h.shape,cat,normal)
                            iso.append((sigma*epsilon).square().sum(-1).sqrt().cpu())
                            distance.append((target-h).square().sum(-1).sqrt().cpu())
                        iso=torch.cat(iso);distance=torch.cat(distance)
                        b=torch.quantile(iso,.5).item();d=torch.quantile(distance,.5).item()
                        rho=min(1.,b/max(d,g.distance_guard))
                        iso_quantiles=norm_quantiles(iso);distance_quantiles=norm_quantiles(distance)
                    calibration_flops=ledger.totals['geometry_calibration']-before
                    calibration=dict(identity=site_identity,sample_count=len(row_ids),sampling_seeds=seeds,
                        calibration_rows_sha256=sha256(root/'calibration_rows.json'),baseline_sigma=sigma,
                        isotropic_norm_quantiles=iso_quantiles,target_distance_quantiles=distance_quantiles,
                        b_median=b,distance_median=d,rho_matched=rho,distance_guard=g.distance_guard,rho_cap=1.,
                        covariance_type='diag',flops=calibration_flops,wall_seconds=time.monotonic()-when,
                        bytes_read=reader.bytes_read,io_seconds=reader.io_seconds)
                    atomic_json(calibration_file,calibration)
                copied_tensor=out/(name+'.safetensors');copied_metadata=out/(name+'.gmm.json')
                shutil.copy2(tensor_path,copied_tensor);shutil.copy2(metadata_path,copied_metadata)
                entries[name]=dict(site=site,gmm=file_record(copied_tensor,root),fit_metadata=file_record(copied_metadata,root),
                    calibration=file_record(calibration_file,root),sigma=sigma,rho_matched=calibration['rho_matched'])
                fit_flops+=fit['compute']['fitting_flops'];fit_seconds+=fit['wall_seconds'];fit_io_bytes+=fit['bytes_read']
                calibration_io_bytes+=calibration.get('bytes_read',0)
                print(json.dumps(dict(stage='geometry_calibration',site=name,sigma=sigma,rho=calibration['rho_matched'],
                                      distance=calibration['distance_median'])),flush=True)
            calibration_reports=[json.loads((root/e['calibration']['path']).read_text()) for e in entries.values()]
            costs=dict(extraction_flops=cache_manifest['compute']['extraction_flops'],fitting_flops=fit_flops,
                calibration_flops=sum(c['flops'] for c in calibration_reports),extraction_seconds=cache_manifest['wall_seconds'],
                fitting_worker_seconds=fit_seconds,calibration_worker_seconds=sum(c['wall_seconds'] for c in calibration_reports),
                calibration_invocation_seconds=time.monotonic()-started,raw_cache_bytes=cache_manifest['stored_bytes'],
                fitting_bytes_read=fit_io_bytes,calibration_bytes_read=calibration_io_bytes)
            costs['total_offline_flops']=sum(costs[key] for key in ('extraction_flops','fitting_flops','calibration_flops'))
            bank=dict(format=FORMAT,complete=True,identity=identity,cache_root=str(Path(cache).resolve()),
                cache_rows=cache_manifest['expected_rows'],sites=entries,calibration_rows=file_record(root/'calibration_rows.json',root),
                costs=costs,reference_fixed=True,fit_scope='all_processed_training_positions',covariance_type='diag')
            bank['fingerprint']=fingerprint(bank)
            atomic_json(root/'manifest.json',bank)
            return bank
        finally:
            estimator.close()


def validate_bank(path, config=None, data_manifest=None, model=None, estimator=None):
    path=Path(path);path=path/'manifest.json' if path.is_dir() else path
    bank=json.loads(path.read_text());expected=bank.pop('fingerprint')
    if fingerprint(bank)!=expected or not bank['complete'] or bank['format']!=FORMAT:
        raise ValueError('Incomplete geometry bank or fingerprint mismatch')
    bank['fingerprint']=expected;root=path.parent
    check_file(root,bank['calibration_rows'])
    if set(bank['sites'])!={s['name'] for s in bank['identity']['reference']['sites']}:
        raise ValueError('Geometry bank is missing a selected site or contains an extra site')
    if sha256(Path(bank['cache_root'])/'manifest.json')!=bank['identity']['cache_manifest_sha256']:
        raise ValueError('Retained raw-cache manifest changed')
    if config is not None:
        reference=reference_identity(config,data_manifest,model,resolve_sites(model,config.training))
        if bank['identity']['reference']!=reference or bank['identity']['baseline_noise']!=asdict(config.dust.noise):
            raise ValueError('Geometry bank reference checkpoint, data, parameter selection, source or baseline noise changed')
        sigma_hash=sha256(config.dust.noise.scales_file) if config.dust.noise.scales_file else None
        if bank['identity']['baseline_scale_sha256']!=sigma_hash:
            raise ValueError('Baseline sigma artifact changed; recalibrate GMM displacement coefficients')
    for name,entry in bank['sites'].items():
        for key in ('gmm','fit_metadata','calibration'):
            check_file(root,entry[key])
        fit=json.loads((root/entry['fit_metadata']['path']).read_text())
        if not fit['complete'] or fit['fit_rows']!=bank['cache_rows'] or fit['covariance_type']!='diag':
            raise ValueError('A geometry-bank GMM is incomplete or has the wrong fitting scope')
        if config is not None and fit['component_count']!=config.geometry.components:
            raise ValueError('Geometry component count differs from the bank')
        if estimator is not None and estimator.sigma(name)!=entry['sigma']:
            raise ValueError(f'GMM calibration sigma changed at {name}')
    return bank,root


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--cache',required=True)
    parser.add_argument('--fits',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--set',action='append',default=[]);args=parser.parse_args()
    calibrate_bank(load_config(args.config,args.set),args.cache,args.fits,args.output)


if __name__=='__main__':
    main()
