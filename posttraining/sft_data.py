"""Deterministic conversation preparation, shifted targets, and padding masks."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def conversation(row, settings):
    if settings.messages_field in row:
        messages = row[settings.messages_field]
        if not isinstance(messages, list) or not messages:
            raise ValueError('messages must be a nonempty list of role/content objects')
        if any(m.get('role') not in ('system', 'user', 'assistant', 'tool') or
               not isinstance(m.get('content'), str) for m in messages):
            raise ValueError('Only text role/content conversations are supported')
        return messages
    if settings.prompt_field in row and settings.response_field in row:
        return [dict(role='user', content=str(row[settings.prompt_field])),
                dict(role='assistant', content=str(row[settings.response_field]))]
    if 'instruction' in row and 'output' in row:
        prompt = str(row['instruction'])
        if row.get('input'):
            prompt += '\n' + str(row['input'])
        return [dict(role='user', content=prompt), dict(role='assistant', content=str(row['output']))]
    return None


def encode_example(row, tokenizer, settings, audit=None):
    messages = conversation(row, settings)
    if messages is None:
        if settings.text_field not in row or settings.objective != 'full':
            raise ValueError('Supply messages, prompt/completion, instruction/output, or full-loss text')
        text = str(row[settings.text_field])
        if tokenizer.eos_token:
            text += tokenizer.eos_token
        spans = None
    else:
        if not tokenizer.chat_template:
            raise ValueError('Tokenizer has no chat template; specify data.template_file explicitly')
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        spans = []
        if settings.objective == 'assistant':
            # Require prefix-consistent templates. No guessing role boundaries from token IDs.
            for i, message in enumerate(messages):
                if message['role'] != 'assistant':
                    continue
                prefix = tokenizer.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=True)
                through = tokenizer.apply_chat_template(messages[:i+1], tokenize=False, add_generation_prompt=False)
                if not text.startswith(prefix) or not text.startswith(through) or not through.startswith(prefix):
                    raise ValueError('Assistant masking requires a prefix-consistent chat template with an assistant generation header')
                spans.append((len(prefix), len(through)))
    encoded = tokenizer(text, add_special_tokens=False,
                        return_offsets_mapping=settings.objective == 'assistant')
    ids = list(encoded['input_ids'])
    if audit is not None:
        audit['visited_examples'] += 1
        audit['visited_formatted_tokens'] += len(ids)
    if settings.objective == 'assistant':
        if not getattr(tokenizer, 'is_fast', False):
            raise ValueError('Assistant masking requires a fast tokenizer with character offsets')
        supervised = [any(end > lo and start < hi and end > start for lo, hi in spans)
                      for start, end in encoded['offset_mapping']]
    else:
        supervised = [True] * len(ids)
    if len(ids) > settings.max_length:
        if settings.long_sequences == 'drop':
            if audit is not None:
                audit['excluded_long_examples'] += 1
                audit['excluded_long_tokens'] += len(ids)
            return None
        if audit is not None:
            audit['truncated_tokens'] += len(ids) - settings.max_length
        ids, supervised = ids[:settings.max_length], supervised[:settings.max_length]
    if len(ids) < settings.min_length or not any(supervised[1:]):
        if audit is not None:
            audit['excluded_short_or_unsupervised_examples'] += 1
            audit['excluded_short_or_unsupervised_tokens'] += len(ids)
        return None
    return dict(input_ids=ids, supervised=supervised, source=str(row.get(settings.source_field, 'unknown')),
                content_sha256=digest(ids), length=len(ids))


def collate(examples, pad_token_id, *, multiple=1):
    """Already-shifted x/y: x[t] predicts y[t]; -100 masks loss, mask controls attention."""
    length = max(len(row['input_ids']) - 1 for row in examples)
    length = ((length + multiple - 1) // multiple) * multiple
    x = torch.full((len(examples), length), pad_token_id, dtype=torch.long)
    y = torch.full_like(x, -100)
    mask = torch.zeros_like(x)
    for i, row in enumerate(examples):
        ids = torch.tensor(row['input_ids'], dtype=torch.long)
        keep = torch.tensor(row['supervised'][1:], dtype=torch.bool)
        n = len(ids) - 1
        x[i, :n] = ids[:-1]
        y[i, :n] = torch.where(keep, ids[1:], -100)
        mask[i, :n] = 1
    return dict(input_ids=x, labels=y, attention_mask=mask)


def local_rows(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if path.suffix == '.json':
        value = json.loads(path.read_text())
        if not isinstance(value, list):
            raise ValueError('Local JSON dataset must contain a list of examples')
        return value
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    raise ValueError('Local datasets must be JSON, JSONL, or Parquet')


def load_tokenizer(config):
    from transformers import AutoTokenizer
    settings = config.model
    tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer_name_or_path or settings.name_or_path,
        revision=settings.tokenizer_revision or settings.revision, use_fast=True, trust_remote_code=False)
    if config.data.template_file:
        tokenizer.chat_template = Path(config.data.template_file).read_text()
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer needs an existing padding or EOS token; vocabulary is never silently resized')
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    return tokenizer


def configuration_fingerprint(config, tokenizer):
    data = asdict(config.data)
    for key in ('prepared_path', 'cache_dir', 'template_file'):
        data.pop(key)
    local_hashes = {key: hashlib.sha256(Path(value).read_bytes()).hexdigest()
                    for key, value in data.items() if key.endswith('_file') and value}
    return digest(dict(data=data, local_hashes=local_hashes, tokenizer=asdict(config.model),
                       template=tokenizer.chat_template, vocabulary=tokenizer.get_vocab(),
                       pad_token_id=tokenizer.pad_token_id))


def prepare_data(config):
    settings = config.data
    tokenizer = load_tokenizer(config)
    fingerprint = configuration_fingerprint(config, tokenizer)
    output = Path(settings.prepared_path)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['configuration_fingerprint'] != fingerprint:
            raise ValueError('Prepared data belongs to another configuration; choose another prepared_path')
        if hashlib.sha256((output / 'data.pt').read_bytes()).hexdigest() != manifest['data_sha256']:
            raise ValueError('Prepared data hash mismatch')
        return output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose an empty prepared data directory: {output}')
    if settings.train_file:
        raw = {'train': local_rows(settings.train_file)}
        if settings.validation_file:
            raw['validation'] = local_rows(settings.validation_file)
        if settings.test_file:
            raw['test'] = local_rows(settings.test_file)
    else:
        if not settings.dataset:
            raise ValueError('Supply data.dataset or data.train_file')
        from datasets import load_dataset
        dataset = load_dataset(settings.dataset, settings.subset, revision=settings.revision,
                               cache_dir=settings.cache_dir)
        raw = {'train': dataset[settings.train_split]}
        if settings.validation_split:
            raw['validation'] = dataset[settings.validation_split]
        if settings.test_split:
            raw['test'] = dataset[settings.test_split]
    seen = set()
    stats = {}

    def select(name, count, seed):
        source = raw[name]
        order = list(range(len(source)))
        random.Random(seed).shuffle(order)
        selected = []
        skipped = duplicates = 0
        audit = Counter()
        for index in order:
            encoded = encode_example(source[index], tokenizer, settings, audit=audit)
            if encoded is None:
                skipped += 1
                continue
            if encoded['content_sha256'] in seen:
                duplicates += 1
                audit['excluded_duplicate_tokens'] += len(encoded['input_ids'])
                continue
            seen.add(encoded['content_sha256'])
            encoded['source_index'] = index
            selected.append(encoded)
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError(f'{name}: requested {count} unique eligible conversations, found {len(selected)}')
        stats[name] = dict(available=len(source), selected=len(selected), filtered=skipped, duplicate=duplicates,
                           audit=dict(audit), unvisited_examples=len(source)-audit['visited_examples'])
        return selected

    holdout = 0 if 'validation' in raw else settings.validation_examples
    pool = select('train', settings.train_examples + holdout, settings.seed)
    splits = dict(train=pool[holdout:])
    splits['validation'] = select('validation', settings.validation_examples, settings.seed + 1) if not holdout else pool[:holdout]
    if settings.test_examples:
        if 'test' not in raw:
            raise ValueError('test_examples>0 requires a separate test split/file; set it to 0 to omit final test')
        splits['test'] = select('test', settings.test_examples, settings.seed + 2)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(splits, output / 'data.pt')
    tokenizer.save_pretrained(output / 'tokenizer')
    manifest = dict(configuration_fingerprint=fingerprint, data=asdict(settings), model=asdict(config.model),
                    data_sha256=hashlib.sha256((output / 'data.pt').read_bytes()).hexdigest(),
                    chat_template=tokenizer.chat_template, pad_token_id=tokenizer.pad_token_id,
                    tokenizer_size=len(tokenizer), preparation=stats,
                    splits={name: dict(examples=len(rows),
                        input_tokens=sum(len(row['input_ids'])-1 for row in rows),
                        formatted_tokens=sum(len(row['input_ids']) for row in rows),
                        final_tokens_excluded_from_inputs=len(rows),
                        supervised_tokens=sum(sum(row['supervised'][1:]) for row in rows),
                        sources=dict(Counter(row['source'] for row in rows)),
                        content_sha256=[row['content_sha256'] for row in rows]) for name, rows in splits.items()})
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return output


def load_prepared(config):
    output = Path(config.data.prepared_path)
    manifest = json.loads((output / 'manifest.json').read_text())
    if hashlib.sha256((output / 'data.pt').read_bytes()).hexdigest() != manifest['data_sha256']:
        raise ValueError('Prepared data hash mismatch')
    tokenizer = load_tokenizer(config)
    if configuration_fingerprint(config, tokenizer) != manifest['configuration_fingerprint']:
        raise ValueError('Prepared data does not match this model/tokenizer/data configuration')
    return torch.load(output / 'data.pt', map_location='cpu', weights_only=True), manifest


def main():
    import argparse
    from .configuration import load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_args()
    print(prepare_data(load_config(args.config, args.set)))


if __name__ == '__main__':
    main()
