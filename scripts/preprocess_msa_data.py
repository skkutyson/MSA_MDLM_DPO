#!/usr/bin/env python3
"""
MSA Data Preprocessing Script for Fast Loading.

Converts raw A3M/FASTA files to fast-loading formats:
- Arrow/Parquet: Memory-mapped, columnar storage (recommended)
- LMDB: Key-value store with memory mapping
- WebDataset: Sharded tar files for distributed training

Usage:
    # Convert to Arrow format (fastest for single-node training)
    python scripts/preprocess_msa_data.py \
        --input /path/to/openfold_pdb_data \
        --output /path/to/processed/msa_data.arrow \
        --format arrow \
        --num-workers 16

    # Convert to WebDataset (best for multi-node training)
    python scripts/preprocess_msa_data.py \
        --input /path/to/openfold_pdb_data \
        --output /path/to/processed/shards \
        --format webdataset \
        --shard-size 10000

    # Convert to LMDB
    python scripts/preprocess_msa_data.py \
        --input /path/to/openfold_pdb_data \
        --output /path/to/processed/msa_data.lmdb \
        --format lmdb
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple
from dataclasses import dataclass
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pickle

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.datasets.openproteinset_loader import OpenProteinSetLoader, MSAEntry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_single_file(args: Tuple[Path, Dict]) -> Optional[Dict]:
    """Process a single MSA file. Used for parallel processing."""
    file_path, filter_config = args

    try:
        loader = OpenProteinSetLoader(
            data_dir=file_path.parent,
            min_length=filter_config.get('min_length', 25),
            max_length=filter_config.get('max_length', 2000),
            min_identity=filter_config.get('min_identity', 0.30),
            max_gap_fraction=filter_config.get('max_gap_fraction', 0.10),
            min_sequences=filter_config.get('min_sequences', 10),
        )

        entry = loader.load_msa(file_path)
        if entry is None:
            return None

        return {
            'id': entry.id,
            'query': entry.query_sequence,
            'sequences': entry.aligned_sequences,
            'depth': entry.depth,
            'length': entry.length,
        }
    except Exception as e:
        return None


def convert_to_arrow(
    input_dir: str,
    output_path: str,
    num_workers: int = 8,
    filter_config: Optional[Dict] = None,
):
    """
    Convert MSA files to Apache Arrow format.

    Arrow provides:
    - Memory-mapped access (no loading into RAM)
    - Columnar storage (fast filtering)
    - Zero-copy reads
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("PyArrow not installed. Run: pip install pyarrow")
        sys.exit(1)

    filter_config = filter_config or {}

    # Find all MSA files
    loader = OpenProteinSetLoader(input_dir)
    files = loader.get_file_list()
    logger.info(f"Found {len(files)} MSA files")

    # Process files in parallel
    records = []
    args_list = [(f, filter_config) for f in files]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_file, args): args for args in args_list}

        with tqdm(total=len(files), desc="Processing MSA files") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    records.append(result)
                pbar.update(1)

    logger.info(f"Processed {len(records)} valid MSA entries")

    if not records:
        logger.error("No valid MSA entries found!")
        sys.exit(1)

    # Create Arrow table
    # Store sequences as JSON strings to avoid nested array issues
    table = pa.table({
        'id': pa.array([r['id'] for r in records]),
        'query': pa.array([r['query'] for r in records]),
        'sequences_json': pa.array([json.dumps(r['sequences']) for r in records]),
        'depth': pa.array([r['depth'] for r in records]),
        'length': pa.array([r['length'] for r in records]),
    })

    # Write to Parquet (Arrow format with compression)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pq.write_table(
        table,
        output_path,
        compression='zstd',  # Good compression + speed balance
        row_group_size=10000,
    )

    logger.info(f"Saved Arrow dataset to {output_path}")
    logger.info(f"Total entries: {len(records)}")
    logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def convert_to_lmdb(
    input_dir: str,
    output_path: str,
    num_workers: int = 8,
    filter_config: Optional[Dict] = None,
    map_size: int = 100 * 1024 * 1024 * 1024,  # 100GB
):
    """
    Convert MSA files to LMDB format.

    LMDB provides:
    - Memory-mapped key-value store
    - Fast random access by index
    - ACID transactions
    """
    try:
        import lmdb
    except ImportError:
        logger.error("LMDB not installed. Run: pip install lmdb")
        sys.exit(1)

    filter_config = filter_config or {}

    # Find all MSA files
    loader = OpenProteinSetLoader(input_dir)
    files = loader.get_file_list()
    logger.info(f"Found {len(files)} MSA files")

    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Open LMDB environment
    env = lmdb.open(
        str(output_path),
        map_size=map_size,
        subdir=True,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    # Process files in parallel
    args_list = [(f, filter_config) for f in files]
    valid_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_file, args): i for i, args in enumerate(args_list)}

        with tqdm(total=len(files), desc="Processing MSA files") as pbar:
            with env.begin(write=True) as txn:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        # Store as pickle for fast serialization
                        key = f"{valid_count:08d}".encode()
                        value = pickle.dumps(result)
                        txn.put(key, value)
                        valid_count += 1
                    pbar.update(1)

                # Store metadata
                txn.put(b'__len__', str(valid_count).encode())

    env.close()

    logger.info(f"Saved LMDB dataset to {output_path}")
    logger.info(f"Total entries: {valid_count}")


def convert_to_webdataset(
    input_dir: str,
    output_dir: str,
    num_workers: int = 8,
    shard_size: int = 10000,
    filter_config: Optional[Dict] = None,
):
    """
    Convert MSA files to WebDataset format (sharded tar files).

    WebDataset provides:
    - Sharded storage for distributed training
    - Streaming access (no need to load all data)
    - Compatible with PyTorch DataLoader
    """
    try:
        import webdataset as wds
    except ImportError:
        logger.error("WebDataset not installed. Run: pip install webdataset")
        sys.exit(1)

    filter_config = filter_config or {}

    # Find all MSA files
    loader = OpenProteinSetLoader(input_dir)
    files = loader.get_file_list()
    logger.info(f"Found {len(files)} MSA files")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process files in parallel
    args_list = [(f, filter_config) for f in files]
    records = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_file, args): args for args in args_list}

        with tqdm(total=len(files), desc="Processing MSA files") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    records.append(result)
                pbar.update(1)

    logger.info(f"Processed {len(records)} valid MSA entries")

    # Write shards
    shard_pattern = str(output_dir / "shard-%06d.tar")

    with wds.ShardWriter(shard_pattern, maxcount=shard_size) as sink:
        for i, record in enumerate(tqdm(records, desc="Writing shards")):
            sample = {
                "__key__": f"{i:08d}",
                "json": json.dumps(record).encode(),
            }
            sink.write(sample)

    # Write metadata
    meta = {
        'total_samples': len(records),
        'shard_size': shard_size,
        'num_shards': (len(records) + shard_size - 1) // shard_size,
    }
    with open(output_dir / "meta.json", 'w') as f:
        json.dump(meta, f)

    logger.info(f"Saved WebDataset shards to {output_dir}")
    logger.info(f"Total entries: {len(records)}")
    logger.info(f"Number of shards: {meta['num_shards']}")


def convert_to_jsonl(
    input_dir: str,
    output_path: str,
    num_workers: int = 8,
    filter_config: Optional[Dict] = None,
):
    """
    Convert MSA files to JSONL format (simple but effective).

    JSONL provides:
    - Human-readable format
    - Easy to process with standard tools
    - Can be loaded line-by-line
    """
    filter_config = filter_config or {}

    # Find all MSA files
    loader = OpenProteinSetLoader(input_dir)
    files = loader.get_file_list()
    logger.info(f"Found {len(files)} MSA files")

    # Process files in parallel
    args_list = [(f, filter_config) for f in files]
    records = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_file, args): args for args in args_list}

        with tqdm(total=len(files), desc="Processing MSA files") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    records.append(result)
                pbar.update(1)

    logger.info(f"Processed {len(records)} valid MSA entries")

    # Write JSONL
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for record in tqdm(records, desc="Writing JSONL"):
            f.write(json.dumps(record) + '\n')

    logger.info(f"Saved JSONL dataset to {output_path}")
    logger.info(f"Total entries: {len(records)}")
    logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Preprocess MSA data for fast loading')

    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input directory containing MSA files')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output path for processed data')
    parser.add_argument('--format', '-f', type=str, default='arrow',
                       choices=['arrow', 'lmdb', 'webdataset', 'jsonl'],
                       help='Output format (default: arrow)')
    parser.add_argument('--num-workers', '-w', type=int, default=8,
                       help='Number of parallel workers')
    parser.add_argument('--shard-size', type=int, default=10000,
                       help='Samples per shard (for webdataset)')

    # Filtering options
    parser.add_argument('--min-length', type=int, default=25)
    parser.add_argument('--max-length', type=int, default=2000)
    parser.add_argument('--min-identity', type=float, default=0.30)
    parser.add_argument('--max-gap-fraction', type=float, default=0.10)
    parser.add_argument('--min-sequences', type=int, default=10)

    args = parser.parse_args()

    filter_config = {
        'min_length': args.min_length,
        'max_length': args.max_length,
        'min_identity': args.min_identity,
        'max_gap_fraction': args.max_gap_fraction,
        'min_sequences': args.min_sequences,
    }

    if args.format == 'arrow':
        convert_to_arrow(args.input, args.output, args.num_workers, filter_config)
    elif args.format == 'lmdb':
        convert_to_lmdb(args.input, args.output, args.num_workers, filter_config)
    elif args.format == 'webdataset':
        convert_to_webdataset(args.input, args.output, args.num_workers, args.shard_size, filter_config)
    elif args.format == 'jsonl':
        convert_to_jsonl(args.input, args.output, args.num_workers, filter_config)


if __name__ == '__main__':
    main()
