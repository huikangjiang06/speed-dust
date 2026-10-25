"""Complete processed-training activation extraction with atomic row-range commits."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from common.flops import ComputeLedger
from ..configuration import load_config
from ..hf_dust import autocast
from ..runtime import atomic_json, batch_examples, load_model, source_hashes, to_device
from ..sft_data import load_prepared
from .common import FORMAT, atomic_tensors, check_file, exclusive, file_record, fingerprint
from .sites import expected_input_tokens, reference_identity, resolve_sites


def storage_requirements(tokens, sites, settings, output):
    per_token = sum(site['width'] for site in sites) * 2
    raw = tokens * per_token
    metadata = tokens * 3 * 8
    temporary = settings.cache_shard_rows * per_token
    margin = math.ceil((raw+metadata+temporary)*settings.storage_margin_fraction) + settings.storage_margin_bytes
    parent = Path(output)
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    return dict(valid_input_tokens=tokens, site_count=len(sites), bytes_per_token=per_token,
                raw_cache_bytes=raw, token_metadata_bytes=metadata, temporary_bytes=temporary,
                free_space_margin_bytes=margin, required_bytes=raw+metadata+temporary+margin,
                available_bytes=free, fits=free>=raw+metadata+temporary+margin)


def validate_cache(path, *, require_complete=True, checksums=True):
    root = Path(path)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['format'] != FORMAT or fingerprint(manifest['identity']) != manifest['fingerprint']:
        raise ValueError('Invalid cache identity/fingerprint')
    cursor = 0
    sites = manifest['identity']['reference']['sites']
    expected_sites = {s['name']: s for s in sites}
    for entry in manifest['ranges']:
        if entry['start'] != cursor or entry['end'] <= cursor or set(entry['sites']) != set(expected_sites):
            raise ValueError('Cache contains a gap, duplicate range, or missing site')
        if checksums:
            check_file(root, entry['metadata'])
        metadata = load_file(str(root/entry['metadata']['path']))['rows']
        if metadata.shape != (entry['end']-cursor,3):
            raise ValueError('Invalid token row metadata shape')
        for name, record in entry['sites'].items():
            file = check_file(root, record) if checksums else root/record['path']
            with safe_open(str(file), framework='pt', device='cpu') as handle:
                sliced = handle.get_slice('activations')
                if sliced.get_shape() != [entry['end']-cursor,expected_sites[name]['width']] or sliced.get_dtype() != 'BF16':
                    raise ValueError(f'Wrong cache tensor shape/dtype: {file}')
        cursor = entry['end']
    if cursor != manifest['committed_rows']:
        raise ValueError('Committed row count disagrees with ranges')
    if require_complete and (not manifest['complete'] or cursor != manifest['expected_rows']):
        raise ValueError('Full-data activation cache is incomplete')
    return manifest


class SiteRows:
    """Bounded reader; every numerical iteration visits every row of a site once."""
    def __init__(self, root, manifest, site):
        self.root, self.manifest, self.site = Path(root), manifest, site
        self.count = manifest['expected_rows']
        self.width = next(s['width'] for s in manifest['identity']['reference']['sites'] if s['name']==site)
        self.bytes_read = 0
        self.io_seconds = 0.
        self.verified = set()

    def chunks(self, chunk_rows):
        for entry in self.manifest['ranges']:
            record = entry['sites'][self.site]
            started = time.monotonic()
            if record['path'] not in self.verified:
                check_file(self.root, record)
                self.bytes_read += record['bytes']
                self.verified.add(record['path'])
            self.io_seconds += time.monotonic()-started
            with safe_open(str(self.root/record['path']), framework='pt', device='cpu') as handle:
                tensor = handle.get_slice('activations')
                for start in range(0,entry['end']-entry['start'],chunk_rows):
                    started = time.monotonic()
                    values = tensor[start:start+chunk_rows].clone()
                    self.io_seconds += time.monotonic()-started
                    self.bytes_read += values.numel()*values.element_size()
                    yield entry['start']+start, values

    def selected(self, indices):
        indices = torch.tensor(sorted(indices),dtype=torch.long)
        if not len(indices) or indices[0]<0 or indices[-1]>=self.count or len(indices.unique())!=len(indices):
            raise ValueError('Selection must contain unique valid cached row IDs')
        pieces = []
        for start, chunk in self.chunks(8192):
            wanted = indices[(indices>=start)&(indices<start+len(chunk))]-start
            if len(wanted):
                pieces.append(chunk[wanted])
        return torch.cat(pieces),indices


@torch.inference_mode()
def extract(config, output, *, resume=False, stop_after_ranges=None, backbone=True, plan_only=False):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    with exclusive(output/'cache.lock'):
        data,data_manifest = load_prepared(config)
        model,_ = load_model(config,'dust')
        sites = resolve_sites(model,config.training)
        expected = expected_input_tokens(data)
        reference = reference_identity(config,data_manifest,model,sites)
        identity = dict(reference=reference, batch_size=config.geometry.cache_batch_size,
                        shard_rows=config.geometry.cache_shard_rows, backbone=backbone, data_sha256=data_manifest['data_sha256'])
        requirements = storage_requirements(expected,sites,config.geometry,output)
        atomic_json(output/'storage_plan.json',requirements)
        if plan_only:
            return requirements
        manifest_path=output/'manifest.json'
        ledger=ComputeLedger(next(model.parameters()).device,extra_phases=('extraction',))
        if manifest_path.exists():
            if not resume:
                raise FileExistsError('Cache exists; use --resume after checking the experiment identity')
            manifest=validate_cache(output,require_complete=False)
            if manifest['identity']!=identity:
                raise ValueError('Cache reference/preprocessing/source identity changed')
            if manifest['complete']:
                return manifest
            ledger.load_state_dict(manifest['compute'])
            elapsed=manifest['wall_seconds']
        else:
            if not requirements['fits']:
                atomic_json(output/'blocked.json',dict(stage='activation_extraction',reason='insufficient_capacity',**requirements))
                raise OSError(f"Full cache requires {requirements['required_bytes']} bytes including margin; only {requirements['available_bytes']} available")
            manifest=dict(format=FORMAT,identity=identity,fingerprint=fingerprint(identity),expected_rows=expected,
                          committed_rows=0,complete=False,ranges=[],compute=ledger.state_dict(),wall_seconds=0.,
                          expected_examples=len(data['train']),storage_plan=requirements,source_hashes=source_hashes())
            elapsed=0.
            atomic_json(manifest_path,manifest)
        known={entry['directory'] for entry in manifest['ranges']}
        shard_root=output/'ranges';shard_root.mkdir(exist_ok=True)
        removed=[]
        for path in shard_root.iterdir():
            if path.name not in known:
                # Only this cache's uncommitted owned ranges are removed; committed files were validated above.
                if path.name.startswith(('.partial_', 'rows_')) and path.is_dir():
                    shutil.rmtree(path);removed.append(path.name)
                else:
                    raise ValueError(f'Unexpected cache directory entry: {path}')
        if removed:
            manifest.setdefault('discarded_uncommitted_ranges',[]).extend(removed)
        resume_row=manifest['committed_rows'];seen=0
        captured={};active_mask=None
        def hook(name):
            def capture(module,args,result):
                values=result[active_mask.bool()].detach()
                if not torch.isfinite(values).all():
                    raise FloatingPointError(f'Nonfinite reference activations at {name}')
                captured[name]=values.to(dtype=torch.bfloat16,device='cpu').contiguous()
            return capture
        hooks=[model.get_submodule(s['name']).register_forward_hook(hook(s['name'])) for s in sites]
        buffers={s['name']:[] for s in sites};row_buffers=[];buffered=0;started=time.monotonic()
        committed_this_call=0
        device=next(model.parameters()).device
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)

        def commit(count):
            nonlocal buffered,committed_this_call
            start=manifest['committed_rows'];end=start+count
            dirname=f'rows_{start:012d}_{end:012d}'
            temporary=shard_root/('.partial_'+dirname);temporary.mkdir()
            rows=torch.cat(row_buffers)
            atomic_tensors(temporary/'token_rows.safetensors',dict(rows=rows[:count]))
            row_buffers[:]=[rows[count:].clone()] if len(rows)>count else []
            for site in sites:
                values=torch.cat(buffers[site['name']])
                atomic_tensors(temporary/site['filename'],dict(activations=values[:count]))
                buffers[site['name']]=[values[count:].clone()] if len(values)>count else []
            destination=shard_root/dirname
            temporary.rename(destination)
            entry=dict(start=start,end=end,directory=dirname,
                       metadata=file_record(destination/'token_rows.safetensors',output),
                       sites={s['name']:file_record(destination/s['filename'],output) for s in sites})
            manifest['ranges'].append(entry);manifest['committed_rows']=end
            manifest['compute']=ledger.state_dict();manifest['wall_seconds']=elapsed+time.monotonic()-started
            manifest['peak_allocated_bytes']=max(manifest.get('peak_allocated_bytes',0),
                torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0)
            manifest['stored_bytes']=sum(e['metadata']['bytes']+sum(f['bytes'] for f in e['sites'].values()) for e in manifest['ranges'])
            atomic_json(manifest_path,manifest)
            buffered-=count;committed_this_call+=1
            print(json.dumps(dict(stage='cache',rows=end,total_rows=expected,ranges=len(manifest['ranges']),wall_seconds=manifest['wall_seconds'])),flush=True)

        try:
            verified_backbone=manifest.get('backbone_equivalence_verified',False)
            for start in range(0,len(data['train']),config.geometry.cache_batch_size):
                indices=list(range(start,min(start+config.geometry.cache_batch_size,len(data['train']))))
                batch=to_device(batch_examples(data['train'],indices,config,data_manifest),device)
                valid=int(batch['attention_mask'].sum())
                if seen+valid<=resume_row:
                    seen+=valid;continue
                active_mask=batch['attention_mask'];captured.clear()
                with ledger.measure('extraction'),autocast(device,config.training.precision):
                    if backbone and not verified_backbone:
                        model(input_ids=batch['input_ids'],attention_mask=active_mask,use_cache=False)
                        full=captured.copy();captured.clear()
                        model.model(input_ids=batch['input_ids'],attention_mask=active_mask,use_cache=False)
                        if set(full)!=set(captured) or any(not torch.equal(full[k],captured[k]) for k in full):
                            raise AssertionError('Backbone-only extraction changed selected activations')
                        verified_backbone=True;manifest['backbone_equivalence_verified']=True
                    else:
                        target=model.model if backbone else model
                        target(input_ids=batch['input_ids'],attention_mask=active_mask,use_cache=False)
                if set(captured)!={s['name'] for s in sites}:
                    raise AssertionError('A selected activation hook did not execute')
                positions=active_mask.nonzero(as_tuple=False).cpu()
                metadata=torch.stack([torch.tensor(indices)[positions[:,0]],positions[:,1],
                    batch['input_ids'][active_mask.bool()].cpu()],-1)
                offset=max(0,resume_row-seen)
                for name,values in captured.items():
                    buffers[name].append(values[offset:])
                row_buffers.append(metadata[offset:]);buffered+=valid-offset;seen+=valid
                while buffered>=config.geometry.cache_shard_rows:
                    commit(config.geometry.cache_shard_rows)
                    if stop_after_ranges is not None and committed_this_call>=stop_after_ranges:
                        return manifest
            if buffered:
                commit(buffered)
            if manifest['committed_rows']!=expected:
                raise AssertionError('Extraction did not preserve every valid training position')
            manifest['complete']=True;manifest['wall_seconds']=elapsed+time.monotonic()-started
            manifest['compute']=ledger.state_dict();atomic_json(manifest_path,manifest)
            validate_cache(output)
            return manifest
        finally:
            for handle in hooks:
                handle.remove()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--stop-after-ranges',type=int);parser.add_argument('--full-forward',action='store_true')
    parser.add_argument('--set',action='append',default=[])
    args=parser.parse_args()
    extract(load_config(args.config,args.set),args.output,resume=args.resume,plan_only=args.plan_only,
            stop_after_ranges=args.stop_after_ranges,backbone=not args.full_forward)


if __name__=='__main__':
    main()
