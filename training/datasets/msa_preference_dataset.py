"""
MSA Preference Dataset for DPO training.

Loads preference pairs (winner/loser) and prepares them for D3PO training
on masked diffusion models.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import logging

from .msa_dataset import MSATokenizer

logger = logging.getLogger(__name__)


class MSAPreferenceDataset(Dataset):
    """
    Dataset for DPO training with preference pairs.

    Each sample contains a winner and loser MSA for the same query.
    Both are tokenized with 2D position IDs for diffusion training.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[MSATokenizer] = None,
        max_seq_length: int = 2048,
        max_msa_sequences: int = 8,
    ):
        """
        Initialize the preference dataset.

        Args:
            data_path: Path to JSONL file with preference pairs
            tokenizer: Tokenizer instance
            max_seq_length: Maximum total sequence length
            max_msa_sequences: Maximum number of MSA sequences to include
        """
        self.tokenizer = tokenizer or MSATokenizer()
        self.max_seq_length = max_seq_length
        self.max_msa_sequences = max_msa_sequences

        # Load preference pairs
        self.pairs = self._load_pairs(data_path)
        logger.info(f"Loaded {len(self.pairs)} preference pairs from {data_path}")

    def _load_pairs(self, data_path: str) -> List[Dict]:
        """Load preference pairs from JSONL file."""
        pairs = []
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
        return pairs

    def _tokenize_msa(
        self,
        query: str,
        msa_sequences: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize an MSA with 2D positions.

        Format: [gMASK, sop, query..., <M>, seq1..., <M>, seq2..., ...]
        """
        # Clean query
        clean_query = query.replace('-', '').replace('.', '')
        query_ids = self.tokenizer.encode(clean_query)

        # Build sequence
        input_ids = [self.tokenizer.gmask_id, self.tokenizer.sop_id]
        input_ids.extend(query_ids)

        position_ids = [0, 1]
        position_ids.extend(range(len(query_ids)))

        block_position_ids = [0, 0]
        block_position_ids.extend([0] * len(query_ids))

        context_mask = [1, 1]
        context_mask.extend([1] * len(query_ids))

        # Add MSA sequences
        for msa_idx, msa_seq in enumerate(msa_sequences[:self.max_msa_sequences], start=1):
            # Delimiter
            input_ids.append(self.tokenizer.msa_delimiter_id)
            position_ids.append(0)
            block_position_ids.append(msa_idx)
            context_mask.append(0)

            # MSA tokens
            msa_tokens = self.tokenizer.encode(msa_seq)

            remaining = self.max_seq_length - len(input_ids) - 1
            if len(msa_tokens) > remaining:
                msa_tokens = msa_tokens[:remaining]

            input_ids.extend(msa_tokens)
            position_ids.extend(range(len(msa_tokens)))
            block_position_ids.extend([msa_idx] * len(msa_tokens))
            context_mask.extend([0] * len(msa_tokens))

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

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a preference pair.

        Returns dict with winner_* and loser_* tensors.
        """
        pair = self.pairs[idx]

        query = pair['query']
        winner_msa = pair['winner']
        loser_msa = pair['loser']

        # Tokenize both
        winner_data = self._tokenize_msa(query, winner_msa)
        loser_data = self._tokenize_msa(query, loser_msa)

        # Prefix keys
        result = {}
        for key, value in winner_data.items():
            result[f'winner_{key}'] = value
        for key, value in loser_data.items():
            result[f'loser_{key}'] = value

        return result


class PreferenceCollator:
    """
    Collator for preference pairs with separate padding for winner/loser.
    """

    def __init__(self, pad_id: int = 0, max_length: Optional[int] = None):
        self.pad_id = pad_id
        self.max_length = max_length

    def _pad_batch(
        self,
        batch: List[Dict[str, torch.Tensor]],
        prefix: str,
    ) -> Dict[str, torch.Tensor]:
        """Pad a batch for either winner or loser."""
        # Get max length
        max_len = max(item[f'{prefix}_seq_len'] for item in batch)
        if self.max_length:
            max_len = min(max_len, self.max_length)

        batch_size = len(batch)

        # Initialize
        input_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        position_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        block_position_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        context_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, item in enumerate(batch):
            seq_len = min(item[f'{prefix}_seq_len'], max_len)
            input_ids[i, :seq_len] = item[f'{prefix}_input_ids'][:seq_len]
            position_ids[i, :seq_len] = item[f'{prefix}_position_ids'][:seq_len]
            block_position_ids[i, :seq_len] = item[f'{prefix}_block_position_ids'][:seq_len]
            context_mask[i, :seq_len] = item[f'{prefix}_context_mask'][:seq_len]
            attention_mask[i, :seq_len] = item[f'{prefix}_attention_mask'][:seq_len]

        return {
            f'{prefix}_input_ids': input_ids,
            f'{prefix}_position_ids': position_ids,
            f'{prefix}_block_position_ids': block_position_ids,
            f'{prefix}_context_mask': context_mask,
            f'{prefix}_attention_mask': attention_mask,
        }

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch with separate padding for winner and loser."""
        result = {}
        result.update(self._pad_batch(batch, 'winner'))
        result.update(self._pad_batch(batch, 'loser'))
        return result


def create_preference_dataloader(
    data_path: str,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    """
    Create DataLoader for DPO training.

    Args:
        data_path: Path to preference pairs JSONL
        batch_size: Batch size (typically 1 for DPO)
        num_workers: Data loading workers
        shuffle: Whether to shuffle
        **dataset_kwargs: Additional MSAPreferenceDataset args

    Returns:
        DataLoader instance
    """
    dataset = MSAPreferenceDataset(data_path, **dataset_kwargs)
    collator = PreferenceCollator(pad_id=0)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
