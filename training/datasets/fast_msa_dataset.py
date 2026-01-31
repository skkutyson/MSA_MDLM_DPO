"""
Fast MSA Dataset with Memory-Mapped Loading.

Supports multiple fast-loading formats:
- Arrow/Parquet: Memory-mapped columnar storage
- LMDB: Memory-mapped key-value store
- WebDataset: Sharded tar files for streaming
- Pre-tokenized: HDF5 with pre-computed tokens

These formats provide 10-100x faster loading compared to raw A3M files.
"""

import os
import json
import random
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from typing import List, Dict, Optional, Tuple, Union, Any, Iterator
from pathlib import Path
import logging

from .msa_dataset import MSATokenizer, MSACollator
from .openproteinset_loader import MSAEntry

logger = logging.getLogger(__name__)


class ArrowMSADataset(Dataset):
    """
    MSA Dataset backed by Apache Arrow/Parquet.

    Provides memory-mapped access for instant loading without RAM overhead.
    Best for single-node training with large datasets.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[MSATokenizer] = None,
        max_seq_length: int = 2048,
        max_msa_depth: int = 64,
        num_msa_sequences: int = 8,
        use_few_shot: bool = False,
        few_shot_examples: int = 2,
    ):
        """
        Args:
            data_path: Path to Arrow/Parquet file
            tokenizer: Tokenizer instance
            max_seq_length: Maximum sequence length
            max_msa_depth: Maximum MSA depth to sample from
            num_msa_sequences: Number of MSA sequences per sample
            use_few_shot: Whether to use few-shot mode
            few_shot_examples: Number of few-shot examples
        """
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("PyArrow required. Install: pip install pyarrow")

        self.data_path = Path(data_path)
        self.tokenizer = tokenizer or MSATokenizer()
        self.max_seq_length = max_seq_length
        self.max_msa_depth = max_msa_depth
        self.num_msa_sequences = num_msa_sequences
        self.use_few_shot = use_few_shot
        self.few_shot_examples = few_shot_examples if use_few_shot else 0

        # Load Arrow table (memory-mapped, not in RAM)
        self.table = pq.read_table(self.data_path, memory_map=True)
        self._length = len(self.table)

        logger.info(f"Loaded Arrow dataset with {self._length} samples")

    def __len__(self) -> int:
        return self._length

    def _get_entry(self, idx: int) -> Dict:
        """Get a single entry from the table."""
        row = self.table.slice(idx, 1).to_pydict()
        # Handle both formats: sequences_json (new) or sequences (old)
        if 'sequences_json' in row:
            sequences = json.loads(row['sequences_json'][0])
        else:
            sequences = list(row['sequences'][0])
        return {
            'id': row['id'][0],
            'query': row['query'][0],
            'sequences': sequences,
        }

    def _sample_sequences(self, sequences: List[str]) -> Tuple[List[str], List[str]]:
        """Sample MSA sequences for training."""
        sequences = sequences[:self.max_msa_depth]

        if not self.use_few_shot:
            sampled = sequences if len(sequences) <= self.num_msa_sequences else random.sample(sequences, self.num_msa_sequences)
            return [], sampled

        total_needed = self.few_shot_examples + self.num_msa_sequences
        if len(sequences) <= total_needed:
            context = sequences[:self.few_shot_examples]
            targets = sequences[self.few_shot_examples:]
        else:
            sampled = random.sample(sequences, total_needed)
            context = sampled[:self.few_shot_examples]
            targets = sampled[self.few_shot_examples:]

        return context, targets

    def _build_input(
        self,
        query: str,
        msa_sequences: List[str],
        few_shot_examples: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build input tensors (same logic as MSADataset)."""
        clean_query = query.replace('-', '').replace('.', '')
        query_ids = self.tokenizer.encode(clean_query)

        input_ids = [self.tokenizer.gmask_id, self.tokenizer.sop_id]
        input_ids.extend(query_ids)

        position_ids = [0, 1]
        position_ids.extend(range(len(query_ids)))

        block_position_ids = [0, 0]
        block_position_ids.extend([0] * len(query_ids))

        context_mask = [1, 1]
        context_mask.extend([1] * len(query_ids))

        current_block = 1

        # Add few-shot examples (context)
        if few_shot_examples:
            for seq in few_shot_examples:
                input_ids.append(self.tokenizer.msa_delimiter_id)
                position_ids.append(0)
                block_position_ids.append(current_block)
                context_mask.append(1)

                tokens = self.tokenizer.encode(seq)
                remaining = self.max_seq_length - len(input_ids) - 1
                if len(tokens) > remaining:
                    tokens = tokens[:remaining]

                input_ids.extend(tokens)
                position_ids.extend(range(len(tokens)))
                block_position_ids.extend([current_block] * len(tokens))
                context_mask.extend([1] * len(tokens))

                current_block += 1
                if len(input_ids) >= self.max_seq_length:
                    break

        # Add generation targets
        for seq in msa_sequences:
            input_ids.append(self.tokenizer.msa_delimiter_id)
            position_ids.append(0)
            block_position_ids.append(current_block)
            context_mask.append(0)

            tokens = self.tokenizer.encode(seq)
            remaining = self.max_seq_length - len(input_ids) - 1
            if len(tokens) > remaining:
                tokens = tokens[:remaining]

            input_ids.extend(tokens)
            position_ids.extend(range(len(tokens)))
            block_position_ids.extend([current_block] * len(tokens))
            context_mask.extend([0] * len(tokens))

            current_block += 1
            if len(input_ids) >= self.max_seq_length:
                break

        # Truncate
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            position_ids = position_ids[:self.max_seq_length]
            block_position_ids = block_position_ids[:self.max_seq_length]
            context_mask = context_mask[:self.max_seq_length]

        seq_len = len(input_ids)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'position_ids': torch.tensor(position_ids, dtype=torch.long),
            'block_position_ids': torch.tensor(block_position_ids, dtype=torch.long),
            'context_mask': torch.tensor(context_mask, dtype=torch.long),
            'attention_mask': torch.ones(seq_len, dtype=torch.long),
            'seq_len': seq_len,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self._get_entry(idx)
        few_shot, targets = self._sample_sequences(entry['sequences'])
        return self._build_input(
            entry['query'],
            targets,
            few_shot if self.use_few_shot else None,
        )


class LMDBMSADataset(Dataset):
    """
    MSA Dataset backed by LMDB.

    Provides memory-mapped key-value access with fast random reads.
    Good for datasets that don't fit in RAM.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[MSATokenizer] = None,
        max_seq_length: int = 2048,
        max_msa_depth: int = 64,
        num_msa_sequences: int = 8,
        use_few_shot: bool = False,
        few_shot_examples: int = 2,
    ):
        try:
            import lmdb
        except ImportError:
            raise ImportError("LMDB required. Install: pip install lmdb")

        import pickle
        self.pickle = pickle

        self.data_path = Path(data_path)
        self.tokenizer = tokenizer or MSATokenizer()
        self.max_seq_length = max_seq_length
        self.max_msa_depth = max_msa_depth
        self.num_msa_sequences = num_msa_sequences
        self.use_few_shot = use_few_shot
        self.few_shot_examples = few_shot_examples if use_few_shot else 0

        # Open LMDB environment (read-only, memory-mapped)
        self.env = lmdb.open(
            str(self.data_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        # Get length
        with self.env.begin() as txn:
            self._length = int(txn.get(b'__len__').decode())

        logger.info(f"Loaded LMDB dataset with {self._length} samples")

    def __len__(self) -> int:
        return self._length

    def _get_entry(self, idx: int) -> Dict:
        key = f"{idx:08d}".encode()
        with self.env.begin() as txn:
            value = txn.get(key)
        return self.pickle.loads(value)

    def _sample_sequences(self, sequences: List[str]) -> Tuple[List[str], List[str]]:
        sequences = sequences[:self.max_msa_depth]
        if not self.use_few_shot:
            sampled = sequences if len(sequences) <= self.num_msa_sequences else random.sample(sequences, self.num_msa_sequences)
            return [], sampled

        total_needed = self.few_shot_examples + self.num_msa_sequences
        if len(sequences) <= total_needed:
            return sequences[:self.few_shot_examples], sequences[self.few_shot_examples:]
        sampled = random.sample(sequences, total_needed)
        return sampled[:self.few_shot_examples], sampled[self.few_shot_examples:]

    def _build_input(self, query: str, msa_sequences: List[str], few_shot_examples: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        # Same implementation as ArrowMSADataset
        clean_query = query.replace('-', '').replace('.', '')
        query_ids = self.tokenizer.encode(clean_query)

        input_ids = [self.tokenizer.gmask_id, self.tokenizer.sop_id]
        input_ids.extend(query_ids)

        position_ids = [0, 1]
        position_ids.extend(range(len(query_ids)))

        block_position_ids = [0, 0]
        block_position_ids.extend([0] * len(query_ids))

        context_mask = [1, 1]
        context_mask.extend([1] * len(query_ids))

        current_block = 1

        if few_shot_examples:
            for seq in few_shot_examples:
                input_ids.append(self.tokenizer.msa_delimiter_id)
                position_ids.append(0)
                block_position_ids.append(current_block)
                context_mask.append(1)

                tokens = self.tokenizer.encode(seq)
                remaining = self.max_seq_length - len(input_ids) - 1
                if len(tokens) > remaining:
                    tokens = tokens[:remaining]

                input_ids.extend(tokens)
                position_ids.extend(range(len(tokens)))
                block_position_ids.extend([current_block] * len(tokens))
                context_mask.extend([1] * len(tokens))
                current_block += 1
                if len(input_ids) >= self.max_seq_length:
                    break

        for seq in msa_sequences:
            input_ids.append(self.tokenizer.msa_delimiter_id)
            position_ids.append(0)
            block_position_ids.append(current_block)
            context_mask.append(0)

            tokens = self.tokenizer.encode(seq)
            remaining = self.max_seq_length - len(input_ids) - 1
            if len(tokens) > remaining:
                tokens = tokens[:remaining]

            input_ids.extend(tokens)
            position_ids.extend(range(len(tokens)))
            block_position_ids.extend([current_block] * len(tokens))
            context_mask.extend([0] * len(tokens))
            current_block += 1
            if len(input_ids) >= self.max_seq_length:
                break

        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            position_ids = position_ids[:self.max_seq_length]
            block_position_ids = block_position_ids[:self.max_seq_length]
            context_mask = context_mask[:self.max_seq_length]

        seq_len = len(input_ids)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'position_ids': torch.tensor(position_ids, dtype=torch.long),
            'block_position_ids': torch.tensor(block_position_ids, dtype=torch.long),
            'context_mask': torch.tensor(context_mask, dtype=torch.long),
            'attention_mask': torch.ones(seq_len, dtype=torch.long),
            'seq_len': seq_len,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self._get_entry(idx)
        few_shot, targets = self._sample_sequences(entry['sequences'])
        return self._build_input(
            entry['query'],
            targets,
            few_shot if self.use_few_shot else None,
        )


class WebDatasetMSA(IterableDataset):
    """
    MSA Dataset using WebDataset for streaming/sharded loading.

    Best for:
    - Multi-node distributed training
    - Very large datasets
    - Streaming from cloud storage
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[MSATokenizer] = None,
        max_seq_length: int = 2048,
        max_msa_depth: int = 64,
        num_msa_sequences: int = 8,
        use_few_shot: bool = False,
        few_shot_examples: int = 2,
        shuffle: bool = True,
        epoch_shuffle: bool = True,
    ):
        try:
            import webdataset as wds
        except ImportError:
            raise ImportError("WebDataset required. Install: pip install webdataset")

        self.wds = wds
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer or MSATokenizer()
        self.max_seq_length = max_seq_length
        self.max_msa_depth = max_msa_depth
        self.num_msa_sequences = num_msa_sequences
        self.use_few_shot = use_few_shot
        self.few_shot_examples = few_shot_examples if use_few_shot else 0
        self.shuffle = shuffle
        self.epoch_shuffle = epoch_shuffle

        # Find shard files
        self.shard_pattern = str(self.data_path / "shard-{000000..999999}.tar")

        # Load metadata
        meta_path = self.data_path / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
            self._length = self.meta['total_samples']
        else:
            self._length = None

        logger.info(f"Loaded WebDataset from {self.data_path}")

    def __len__(self) -> int:
        if self._length is None:
            raise ValueError("Length unknown for WebDataset. Check meta.json")
        return self._length

    def _process_sample(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Process a single WebDataset sample."""
        data = json.loads(sample['json'])

        sequences = data['sequences'][:self.max_msa_depth]

        if not self.use_few_shot:
            sampled = sequences if len(sequences) <= self.num_msa_sequences else random.sample(sequences, self.num_msa_sequences)
            few_shot, targets = [], sampled
        else:
            total_needed = self.few_shot_examples + self.num_msa_sequences
            if len(sequences) <= total_needed:
                few_shot = sequences[:self.few_shot_examples]
                targets = sequences[self.few_shot_examples:]
            else:
                sampled = random.sample(sequences, total_needed)
                few_shot = sampled[:self.few_shot_examples]
                targets = sampled[self.few_shot_examples:]

        return self._build_input(data['query'], targets, few_shot if self.use_few_shot else None)

    def _build_input(self, query: str, msa_sequences: List[str], few_shot_examples: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        # Same implementation as other datasets
        clean_query = query.replace('-', '').replace('.', '')
        query_ids = self.tokenizer.encode(clean_query)

        input_ids = [self.tokenizer.gmask_id, self.tokenizer.sop_id]
        input_ids.extend(query_ids)

        position_ids = [0, 1]
        position_ids.extend(range(len(query_ids)))

        block_position_ids = [0, 0]
        block_position_ids.extend([0] * len(query_ids))

        context_mask = [1, 1]
        context_mask.extend([1] * len(query_ids))

        current_block = 1

        if few_shot_examples:
            for seq in few_shot_examples:
                input_ids.append(self.tokenizer.msa_delimiter_id)
                position_ids.append(0)
                block_position_ids.append(current_block)
                context_mask.append(1)

                tokens = self.tokenizer.encode(seq)
                remaining = self.max_seq_length - len(input_ids) - 1
                if len(tokens) > remaining:
                    tokens = tokens[:remaining]

                input_ids.extend(tokens)
                position_ids.extend(range(len(tokens)))
                block_position_ids.extend([current_block] * len(tokens))
                context_mask.extend([1] * len(tokens))
                current_block += 1
                if len(input_ids) >= self.max_seq_length:
                    break

        for seq in msa_sequences:
            input_ids.append(self.tokenizer.msa_delimiter_id)
            position_ids.append(0)
            block_position_ids.append(current_block)
            context_mask.append(0)

            tokens = self.tokenizer.encode(seq)
            remaining = self.max_seq_length - len(input_ids) - 1
            if len(tokens) > remaining:
                tokens = tokens[:remaining]

            input_ids.extend(tokens)
            position_ids.extend(range(len(tokens)))
            block_position_ids.extend([current_block] * len(tokens))
            context_mask.extend([0] * len(tokens))
            current_block += 1
            if len(input_ids) >= self.max_seq_length:
                break

        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            position_ids = position_ids[:self.max_seq_length]
            block_position_ids = block_position_ids[:self.max_seq_length]
            context_mask = context_mask[:self.max_seq_length]

        seq_len = len(input_ids)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'position_ids': torch.tensor(position_ids, dtype=torch.long),
            'block_position_ids': torch.tensor(block_position_ids, dtype=torch.long),
            'context_mask': torch.tensor(context_mask, dtype=torch.long),
            'attention_mask': torch.ones(seq_len, dtype=torch.long),
            'seq_len': seq_len,
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        dataset = self.wds.WebDataset(self.shard_pattern)

        if self.shuffle:
            dataset = dataset.shuffle(1000)

        dataset = dataset.decode()

        for sample in dataset:
            yield self._process_sample(sample)


def create_fast_dataloader(
    data_path: str,
    batch_size: int = 48,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    """
    Create a DataLoader with automatic format detection.

    Args:
        data_path: Path to data (Arrow, LMDB, WebDataset, or directory)
        batch_size: Batch size
        num_workers: Number of workers
        shuffle: Whether to shuffle
        **dataset_kwargs: Dataset arguments

    Returns:
        DataLoader instance
    """
    data_path = Path(data_path)

    # Detect format
    if data_path.suffix in ['.parquet', '.arrow']:
        dataset = ArrowMSADataset(str(data_path), **dataset_kwargs)
    elif data_path.is_dir() and (data_path / 'data.mdb').exists():
        dataset = LMDBMSADataset(str(data_path), **dataset_kwargs)
    elif data_path.is_dir() and list(data_path.glob('shard-*.tar')):
        dataset = WebDatasetMSA(str(data_path), shuffle=shuffle, **dataset_kwargs)
        # WebDataset handles its own batching
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=MSACollator(),
            pin_memory=True,
        )
    else:
        # Fall back to original MSADataset
        from .msa_dataset import MSADataset
        dataset = MSADataset(str(data_path), **dataset_kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=MSACollator(),
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


# For convenience, export all dataset classes
__all__ = [
    'ArrowMSADataset',
    'LMDBMSADataset',
    'WebDatasetMSA',
    'create_fast_dataloader',
]
