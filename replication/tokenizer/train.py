"""Optionally retrain the BPE-4096 tokenizer used by the paper experiments."""
import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def train_tokenizer(texts):
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=4096,
        special_tokens=['<|endoftext|>'],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output', type=Path, default=Path(__file__).parent / 'retrained.json',
        help='Save a new tokenizer here; existing files are not overwritten.',
    )
    parser.add_argument(
        '--documents', type=int, default=200_000,
        help='Number of documents from the start of the stream (paper recipe: 200000).',
    )
    parser.add_argument(
        '--parquet', nargs='+',
        help='Optional local FineWeb sample-10BT shards, in original stream order.',
    )
    args = parser.parse_args()
    if args.documents < 1:
        parser.error('--documents must be positive')
    if args.output.exists():
        parser.error(f'{args.output} already exists; choose another --output path')

    from datasets import load_dataset

    if args.parquet:
        corpus = load_dataset('parquet', data_files=args.parquet, split='train', streaming=True)
    else:
        corpus = load_dataset(
            'HuggingFaceFW/fineweb', name='sample-10BT', split='train', streaming=True,
        )
    documents = iter(corpus)
    texts = [next(documents)['text'] for _ in range(args.documents)]
    tokenizer = train_tokenizer(texts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.output))
    print(
        f'Saved {args.output}: {tokenizer.get_vocab_size()} tokens, '
        f'trained on {args.documents:,} documents.',
        flush=True,
    )


if __name__ == '__main__':
    main()
