#!/usr/bin/env python3
"""
Dataset Preparation Script.

Download and process OpenProteinSet MSA data for MDLM training.

Usage:
    # Download a subset for testing
    python scripts/prepare_dataset.py download --output ./data/openproteinset --max-files 100

    # Process and filter MSAs
    python scripts/prepare_dataset.py process --input ./data/openproteinset --output ./data/processed

    # Generate preference pairs for DPO (requires a trained model)
    python scripts/prepare_dataset.py generate-preferences \
        --model-path ./checkpoints/mdlm_pretrain/last.ckpt \
        --input ./data/processed/msas.jsonl \
        --output ./data/preference_pairs.jsonl
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_command(args):
    """Download OpenProteinSet data."""
    from training.datasets.openproteinset_loader import download_openproteinset

    logger.info(f"Downloading OpenProteinSet to {args.output}")
    download_openproteinset(
        output_dir=args.output,
        subset=args.subset,
        max_files=args.max_files,
    )
    logger.info("Download complete!")


def process_command(args):
    """Process and filter MSA files."""
    from training.datasets.openproteinset_loader import OpenProteinSetLoader

    logger.info(f"Processing MSAs from {args.input}")

    loader = OpenProteinSetLoader(
        data_dir=args.input,
        min_length=args.min_length,
        max_length=args.max_length,
        min_identity=args.min_identity,
        max_gap_fraction=args.max_gap_fraction,
        min_sequences=args.min_sequences,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save filtered MSAs as JSON
    processed = []
    for i, msa_entry in enumerate(loader):
        processed.append({
            'id': msa_entry.id,
            'query': msa_entry.query_sequence,
            'sequences': msa_entry.aligned_sequences[:args.max_msa_depth],
            'depth': msa_entry.depth,
            'length': msa_entry.length,
        })

        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1} MSAs")

        if args.max_samples and len(processed) >= args.max_samples:
            break

    # Save to JSONL
    output_path = output_dir / 'msas.jsonl'
    with open(output_path, 'w') as f:
        for item in processed:
            f.write(json.dumps(item) + '\n')

    logger.info(f"Saved {len(processed)} MSAs to {output_path}")

    # Save stats
    stats = {
        'total_msas': len(processed),
        'avg_depth': sum(m['depth'] for m in processed) / len(processed) if processed else 0,
        'avg_length': sum(m['length'] for m in processed) / len(processed) if processed else 0,
    }
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Stats: {stats}")


def generate_preferences_command(args):
    """Generate preference pairs for DPO training."""
    import torch
    from training.datasets.preference_generator import PreferenceGenerator, create_preference_dataset

    logger.info("Generating preference pairs...")

    # Load processed MSAs
    msas = []
    with open(args.input, 'r') as f:
        for line in f:
            if line.strip():
                msas.append(json.loads(line))

    logger.info(f"Loaded {len(msas)} MSAs")

    # If model path provided, generate new MSAs and score them
    if args.model_path:
        logger.info(f"Loading model from {args.model_path}")

        # Import model
        from model_utils.mdlm.model_msagpt_mdlm import MSAGPT_MDLM

        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(args.model_path, map_location='cpu')

        # Create args namespace
        class Args:
            hidden_size = 1024
            num_attention_heads = 16
            num_layers = 24
            cond_dim = 256
            mlp_ratio = 4.0
            noise_type = 'loglinear'
            mask_token_id = 36
            vocab_size = 128

        model_args = Args()
        model = MSAGPT_MDLM(model_args)

        if 'state_dict' in checkpoint:
            state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()

        logger.info("Model loaded, generating MSAs for preference pairs...")

        # Generate MSAs for each query
        generator = PreferenceGenerator(min_score_difference=args.min_score_diff)
        all_pairs = []

        for i, msa_data in enumerate(msas[:args.max_queries]):
            query = msa_data['query'].replace('-', '').replace('.', '')

            # Generate multiple MSAs
            generated_msas = []
            for _ in range(args.num_samples):
                with torch.no_grad():
                    # Simple generation (would need proper implementation)
                    # For now, just perturb existing MSAs
                    perturbed = [
                        mutate_sequence(seq, rate=random.uniform(0.1, 0.3))
                        for seq in msa_data['sequences'][:8]
                    ]
                    generated_msas.append(perturbed)

            # Generate pairs
            pairs = generator.generate_pairs(query, generated_msas)
            for winner, loser, diff in pairs:
                all_pairs.append({
                    'query': query,
                    'winner': winner,
                    'loser': loser,
                    'score_difference': diff,
                })

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{min(len(msas), args.max_queries)} queries, {len(all_pairs)} pairs")

    else:
        # Without model, create pairs from existing MSAs by subsampling
        logger.info("No model provided, creating pairs from existing MSAs")

        generator = PreferenceGenerator(min_score_difference=args.min_score_diff)
        all_pairs = []

        for i, msa_data in enumerate(msas[:args.max_queries]):
            query = msa_data['query']
            sequences = msa_data['sequences']

            if len(sequences) < 10:
                continue

            # Create variants by subsampling
            variants = []
            for _ in range(args.num_samples):
                k = random.randint(3, min(len(sequences), 8))
                variant = random.sample(sequences, k)
                variants.append(variant)

            # Generate pairs
            pairs = generator.generate_pairs(query, variants)
            for winner, loser, diff in pairs:
                all_pairs.append({
                    'query': query,
                    'winner': winner,
                    'loser': loser,
                    'score_difference': diff,
                })

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1} queries, {len(all_pairs)} pairs")

    # Save pairs
    with open(args.output, 'w') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + '\n')

    logger.info(f"Saved {len(all_pairs)} preference pairs to {args.output}")


def mutate_sequence(seq: str, rate: float = 0.1) -> str:
    """Randomly mutate a sequence."""
    aa = 'ACDEFGHIKLMNPQRSTVWY'
    result = []
    for c in seq:
        if c in aa and random.random() < rate:
            result.append(random.choice(aa))
        else:
            result.append(c)
    return ''.join(result)


def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for MDLM training')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Download command
    download_parser = subparsers.add_parser('download', help='Download OpenProteinSet')
    download_parser.add_argument('--output', type=str, required=True, help='Output directory')
    download_parser.add_argument('--subset', type=str, default='uniclust30', help='Dataset subset')
    download_parser.add_argument('--max-files', type=int, default=None, help='Max files to download')

    # Process command
    process_parser = subparsers.add_parser('process', help='Process and filter MSAs')
    process_parser.add_argument('--input', type=str, required=True, help='Input directory')
    process_parser.add_argument('--output', type=str, required=True, help='Output directory')
    process_parser.add_argument('--min-length', type=int, default=25, help='Min sequence length')
    process_parser.add_argument('--max-length', type=int, default=2000, help='Max sequence length')
    process_parser.add_argument('--min-identity', type=float, default=0.30, help='Min sequence identity')
    process_parser.add_argument('--max-gap-fraction', type=float, default=0.10, help='Max gap fraction')
    process_parser.add_argument('--min-sequences', type=int, default=10, help='Min MSA depth')
    process_parser.add_argument('--max-msa-depth', type=int, default=64, help='Max MSA depth to keep')
    process_parser.add_argument('--max-samples', type=int, default=None, help='Max samples to process')

    # Generate preferences command
    pref_parser = subparsers.add_parser('generate-preferences', help='Generate preference pairs')
    pref_parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    pref_parser.add_argument('--output', type=str, required=True, help='Output JSONL file')
    pref_parser.add_argument('--model-path', type=str, default=None, help='Path to trained model')
    pref_parser.add_argument('--num-samples', type=int, default=8, help='Samples per query')
    pref_parser.add_argument('--max-queries', type=int, default=10000, help='Max queries to process')
    pref_parser.add_argument('--min-score-diff', type=float, default=0.3, help='Min score difference')

    args = parser.parse_args()

    if args.command == 'download':
        download_command(args)
    elif args.command == 'process':
        process_command(args)
    elif args.command == 'generate-preferences':
        generate_preferences_command(args)


if __name__ == '__main__':
    main()
