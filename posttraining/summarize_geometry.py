"""Read-only aggregation of the three-method activation-geometry comparison."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


METHODS=('backprop','dust','geometry_dust')


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def summary(root):
    root=Path(root).resolve();report=dict(created_at=datetime.now(timezone.utc).isoformat(),root=str(root),
        reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),methods={},complete=False)
    data=read(root/'data/manifest.json')
    report['data']=dict(train_examples=data['splits']['train']['examples'],
        prepared_input_tokens=data['splits']['train']['input_tokens'],
        validation_input_tokens=data['splits']['validation']['input_tokens'],
        test_input_tokens=data['splits']['test']['input_tokens'],data_sha256=data['data_sha256'])
    report['completed_stage_wall_seconds']={}
    report['coordinator_observed_completion_elapsed_seconds']={}
    for path in sorted((root/'processes').glob('*.json')):
        process=read(path)
        if process.get('exit_code')==0 and process.get('finished_at'):
            elapsed=(datetime.fromisoformat(process['finished_at'])-datetime.fromisoformat(process['started_at'])).total_seconds()
            report['coordinator_observed_completion_elapsed_seconds'][path.stem]=elapsed
            # Parallel workers are joined sequentially: the coordinator can
            # observe completion later than the worker actually finished.
            if not path.stem.startswith('fit_worker_'):
                report['completed_stage_wall_seconds'][path.stem]=elapsed
    report['fitting_worker_measured_site_seconds']={path.stem:read(path)['wall_seconds']
        for path in sorted((root/'fits').glob('worker_*.json'))}
    report['fitting_time_scope']='Worker times sum measured site-fitting durations. Campaign wall time is measured around all workers; coordinator-observed completion times may include waiting to join another worker.'
    bank_path=root/'bank/manifest.json'
    if bank_path.exists():
        bank=read(bank_path);fits=[read(bank_path.parent/entry['fit_metadata']['path']) for entry in bank['sites'].values()]
        iterations=[record for fit in fits for record in fit['history']]
        report['geometry']=dict(complete=bank['complete'],fingerprint=bank['fingerprint'],cached_positions=bank['cache_rows'],
            sites=len(fits),components=sorted({fit['component_count'] for fit in fits}),covariance_type=bank['covariance_type'],
            full_rows_at_every_site=all(fit['complete'] and fit['fit_rows']==bank['cache_rows'] for fit in fits),
            full_rows_at_every_iteration=all(record['rows']==bank['cache_rows'] for record in iterations),
            finite_iteration_likelihoods=all(math.isfinite(record['e_step_average_log_likelihood']) for record in iterations),
            minimum_recorded_variance=min(record['minimum_variance'] for record in iterations),
            converged_sites=sum(fit['converged'] for fit in fits),max_iter_sites=sum(not fit['converged'] for fit in fits),
            em_iterations_total=sum(fit['iterations'] for fit in fits),
            em_iterations_range=[min(fit['iterations'] for fit in fits),max(fit['iterations'] for fit in fits)],
            reinitializations=sum(len(item['reinitializations']) for fit in fits for item in fit['history']),
            rho_range=[min(entry['rho_matched'] for entry in bank['sites'].values()),max(entry['rho_matched'] for entry in bank['sites'].values())],
            rho_median=statistics.median(entry['rho_matched'] for entry in bank['sites'].values()),
            rho_at_one_sites=[name for name,entry in bank['sites'].items() if entry['rho_matched']==1],
            sigma_range=[min(entry['sigma'] for entry in bank['sites'].values()),max(entry['sigma'] for entry in bank['sites'].values())],costs=bank['costs'])
        if (root/'fitting_wall.json').exists():
            report['geometry']['fitting_campaign_wall']=read(root/'fitting_wall.json')
    identities=[]
    for method in METHODS:
        out=root/'runs'/method
        if not (out/'result.json').exists():
            report['methods'][method]=dict(complete=False,progress=read(out/'progress.json') if (out/'progress.json').exists() else None)
            continue
        result=read(out/'result.json');metadata=read(out/'config.json');identities.append(metadata['identity'])
        steps=[r for r in result['history'] if r['step']>0]
        final=next(r['val_loss'] for r in reversed(result['history']) if 'val_loss' in r)
        item=dict(complete=result['complete'],steps=result['step'],initial_validation_loss=result['initial_val'],
            best_validation_loss=result['best_val'],best_step=result['best_step'],final_validation_loss=final,
            final_loss_change=final-result['initial_val'],train_tokens=result['train_tokens'],
            input_tokens=sum(r['input_tokens_this_step'] for r in steps),
            examples_consumed=sum(r['processed_examples_this_step'] for r in steps),
            train_flops=result['train_flops'],validation_flops=result['val_flops'],test_flops=result['test_flops'],
            training_and_evaluation_flops=result['total_flops'],offline_flops=result['offline_flops'],
            standalone_total_with_offline_flops=result['total_with_offline_flops'],
            update_seconds=sum(r['step_seconds'] for r in steps),
            peak_allocated_bytes=max((r.get('peak_allocated_bytes') or 0) for r in steps),
            gradient_l2_median=statistics.median(r['gradient_l2'] for r in steps),
            parameter_update_l2_median=statistics.median(r['parameter_update_l2'] for r in steps),
            probe_counts={key:sum(r.get('probe_counts_on_rank',{}).get(key,0) for r in steps)
                          for key in sorted(set().union(*(r.get('probe_counts_on_rank',{}) for r in steps)))},
            numerical_policy=dict(allow_bf16_reduced_precision_reduction=metadata['identity']['config']['training']['allow_bf16_reduced_precision_reduction']))
        process_path=root/'processes'/('train_'+method+'.json')
        if process_path.exists():
            launched=datetime.fromisoformat(read(process_path)['started_at']).timestamp()
            item['launch_to_result_wall_seconds']=(out/'result.json').stat().st_mtime-launched
            item['wall_time_measurement']='Process launch record to atomic result-file modification time; includes loading, evaluation, checkpoint/export I/O and any pause before resume.'
        item['last_update_elapsed_seconds']=steps[-1]['elapsed_seconds']
        if method=='geometry_dust':
            records=[site for r in steps for site in r.get('geometry',{}).values()]
            item['geometry_diagnostics']=dict(nonfinite_noise_values=sum(s['nonfinite_count'] for s in records),
                combined_to_isotropic_norm_ratio_median=statistics.median(
                    s['norm_quantiles']['combined']['0.5']/s['norm_quantiles']['isotropic']['0.5'] for s in records))
        report['methods'][method]=item
    if len(identities)==3:
        keys=('data_sha256','model_commit','plan_sha256','sources','flop_convention')
        report['matched_inputs_and_source']=all(all(identity[key]==identities[0][key] for key in keys) for identity in identities)
        assert report['matched_inputs_and_source'], 'Comparison mixes different data, schedules or implementation versions'
        controls=('model','data','training','optimizer','dust')
        report['matched_training_controls']=all(all(identity['config'][key]==identities[0]['config'][key]
            for key in controls) for identity in identities)
        assert report['matched_training_controls'], 'Comparison mixes different training or DUST controls'
    complete=all(item['complete'] for item in report['methods'].values())
    geometry=report.get('geometry',{})
    report['complete']=complete and all(geometry.get(key,False) for key in
        ('complete','full_rows_at_every_site','full_rows_at_every_iteration','finite_iteration_likelihoods'))
    if complete:
        baseline_scale=read(root/'baseline_scales/scales.json')['calibration_flops']
        report['unique_primary_campaign_flops']=sum(item['training_and_evaluation_flops'] for item in report['methods'].values())+baseline_scale+report['geometry']['costs']['total_offline_flops']
        report['cost_scope']='Per-method totals charge each method its required offline work. The unique campaign total counts shared sigma calibration once. Development overhead is separate.'
    development={}
    for name in ('real_smoke_v1','real_smoke_v2'):
        base=root.parent/name
        if (base/'smoke_complete.json').exists():
            cost=sum(read(base/'runs'/method/'result.json')['total_flops'] for method in METHODS)
            cost+=read(base/'baseline_scales/scales.json')['calibration_flops']
            cost+=read(base/'bank/manifest.json')['costs']['total_offline_flops']
            development[name]=dict(measured_flops=cost,scope='four conversations; engineering smoke only',superseded=name.endswith('v1'))
    for name in ('architecture_audit','batch_precision_audit'):
        if (root/(name+'.json')).exists():
            development[name]=dict(measured_flops=read(root/(name+'.json'))['total_flops'])
    superseded=root/'superseded_v1/activation_cache/manifest.json'
    if superseded.exists():
        old=read(superseded);development['superseded_partial_cache']=dict(measured_flops=old['compute']['total_flops'],committed_rows=old['committed_rows'],scope='durable ledger only; any uncommitted interrupted work is not reconstructed')
    report['measured_development_overhead']=development
    report['development_scope']='Separately reported measured smoke/audit/abandoned-cache work; unmetered debugging and synthetic fixture overhead is not included.'
    atomic_json(root/'comparison.json',report)
    lines=['# SmolLM2 activation-geometry comparison','',
        f"Status: {'complete' if report['complete'] else 'in progress'}. Training data: {report['data']['train_examples']} conversations; {report['data']['prepared_input_tokens']:,} retained input positions.",'',
        '| Method | Updates | Initial validation loss | Best | Final | Training PFLOPs | Total with offline PFLOPs |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method,item in report['methods'].items():
        if item['complete']:
            lines.append(f"| {method} | {item['steps']} | {item['initial_validation_loss']:.6f} | {item['best_validation_loss']:.6f} | {item['final_validation_loss']:.6f} | {item['train_flops']/1e15:.4f} | {item['standalone_total_with_offline_flops']/1e15:.4f} |")
        else:
            lines.append(f'| {method} | pending | | | | | |')
    if complete:
        baseline=report['methods']['backprop']
        lines+=['',f"Each method consumed {baseline['examples_consumed']} training conversations and {baseline['train_tokens']:,} supervised input positions in this short online schedule."]
        initial_best=[method for method,item in report['methods'].items() if item['best_step']==0]
        if initial_best:
            names=', '.join(initial_best)
            lines+=['',f"No evaluated trained checkpoint improved on initialization for: {names}. Their best.pt and best_model/ exports contain the initial model; final trained states remain in last.pt."]
        degraded=[method for method,item in report['methods'].items() if item['final_loss_change']>0]
        if degraded:
            lines+=['',f"Final validation loss increased for: {', '.join(degraded)}. These outcomes do not establish a useful geometry-guidance advantage or a stable SFT recipe for those methods."]
    if 'geometry' in report:
        g=report['geometry'];lines+=['',f"Geometry: {g['sites']} full-data diagonal GMMs, components {g['components']}, {g['cached_positions']:,} positions per site. {g['converged_sites']} fits met the tolerance; {g['max_iter_sites']} reached the configured iteration limit.",
            f"Offline geometry construction: {g['costs']['total_offline_flops']/1e15:.6f} PFLOPs. Raw cache is retained."]
    lines+=['','This is one seed at the fixed first-trial configuration. It does not establish a reliable performance advantage. The full cache covers all processed training examples; the short online training schedule may consume fewer examples.',
        '',report.get('cost_scope','Final cost accounting is pending completion.'),'',report['development_scope'],
        '', 'See comparison.json for token counts, update norms, probe counts, memory, fitting history summary, numerical policy and separate measured development costs.']
    (root/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({key:report[key] for key in ('complete','methods')},indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args();summary(args.root)


if __name__=='__main__':
    main()
