"""
MSA Dataset for MDLM pre-training.

Converts MSA alignments to tokenized batches with 2D position IDs
suitable for masked diffusion language modeling.
"""

import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path
import logging

from .openproteinset_loader import OpenProteinSetLoader, MSAEntry

logger = logging.getLogger(__name__)


# Token IDs from tokenization.py
SPECIAL_TOKENS = {
    'pad': 0,
    'gMASK': 28,
    'sop': 29,
    '<M>': 30,  # MSA delimiter
    'DIFFUSION_MASK': 31,
}

# Standard amino acid tokens (indices 1-27 in vocab)
AMINO_ACIDS = 'LAGVSERTIODPKQNFYMHWCXBUZO.-'


class MSATokenizer:
    """Simple tokenizer for MSA sequences compatible with ResidueLevelTokenizer."""

    def __init__(self):
        # Build vocab matching tokenization.py
        self.pad_token = '[pad]'
        self.tokens = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
                       'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
                       'X', 'B', 'U', 'Z', 'O', '.', '-']

        self.vocab = {self.pad_token: 0}
        for i, tok in enumerate(self.tokens, start=1):
            self.vocab[tok] = i

        # Special tokens
        self.special_tokens = {
            'gMASK': 28,
            'sop': 29,
            '<M>': 30,
            'DIFFUSION_MASK': 31,
        }
        self.vocab.update(self.special_tokens)

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        # Token IDs
        self.pad_id = 0
        self.gmask_id = self.special_tokens['gMASK']
        self.sop_id = self.special_tokens['sop']
        self.msa_delimiter_id = self.special_tokens['<M>']
        self.mask_id = self.special_tokens['DIFFUSION_MASK']
        self.eos_id = None

        # Optional runtime validation against main tokenizer
        try:
            from utils import proteinglm_tokenizer
            pt = proteinglm_tokenizer()
            if len(self.vocab) != pt.vocab_size:
                raise ValueError(
                    f"MSATokenizer vocab size ({len(self.vocab)}) != proteinglm ({pt.vocab_size})"
                )
            expected = {
                'gMASK': pt.get_command('gMASK'),
                'sop': pt.get_command('sop'),
                '<M>': pt.get_command('<M>'),
                'DIFFUSION_MASK': pt.get_command('DIFFUSION_MASK'),
            }
            for key, val in expected.items():
                if self.special_tokens.get(key) != val:
                    raise ValueError(
                        f"Token ID mismatch for {key}: {self.special_tokens.get(key)} != {val}"
                    )
        except Exception as e:
            logger.warning(f"Tokenizer validation skipped or failed: {e}")

    def encode(self, sequence: str) -> List[int]:
        """Encode a protein sequence to token IDs."""
        ids = []
        for char in sequence.upper():
            if char in self.vocab:
                ids.append(self.vocab[char])
            elif char in '.-':
                ids.append(self.vocab[char])
            else:
                # Unknown amino acid -> X
                ids.append(self.vocab.get('X', self.vocab['X']))
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to sequence."""
        chars = []
        for idx in ids:
            if idx in self.inv_vocab:
                tok = self.inv_vocab[idx]
                if tok.startswith('[') or tok in self.special_tokens:
                    continue  # Skip special tokens in output
                chars.append(tok)
        return ''.join(chars)


class MSADataset(Dataset):
    """
    Dataset for MDLM pre-training on protein MSAs.

    Each sample contains:
    - input_ids: [gMASK, sop, query..., <M>, msa1..., <M>, msa2..., ...]
    - position_ids: Position within each MSA sequence
    - block_position_ids: Which MSA in the alignment (0 for query, 1+ for aligned)
    - context_mask: 1 for query (context), 0 for generated MSA sequences
    - attention_mask: 1 for valid tokens, 0 for padding
    """

    def __init__(
        self,
        data_source: Union[str, List[MSAEntry], OpenProteinSetLoader],
        tokenizer: Optional[MSATokenizer] = None,
        max_seq_length: int = 2048,
        max_msa_depth: int = 64,
        num_msa_sequences: int = 8,
        include_query_in_context: bool = True,
        cache_dir: Optional[str] = None,
        use_few_shot: bool = False,
        few_shot_examples: int = 2,
    ):
        """
        Initialize the dataset.

        Args:
            data_source: Path to data directory, list of MSAEntry, or loader
            tokenizer: Tokenizer instance (created if not provided)
            max_seq_length: Maximum total sequence length
            max_msa_depth: Maximum MSA depth to consider when sampling
            num_msa_sequences: Number of MSA sequences to sample per example
            include_query_in_context: Whether query is context (not masked)
            cache_dir: Directory for caching processed data
            use_few_shot: Whether to use few-shot learning mode
            few_shot_examples: Number of example MSA sequences in context (2-3)
        """
        self.tokenizer = tokenizer or MSATokenizer()
        self.max_seq_length = max_seq_length
        self.max_msa_depth = max_msa_depth
        self.num_msa_sequences = num_msa_sequences
        self.include_query_in_context = include_query_in_context
        self.use_few_shot = use_few_shot
        self.few_shot_examples = few_shot_examples if use_few_shot else 0

        # Load data
        if isinstance(data_source, str):
            self.loader = OpenProteinSetLoader(data_source)
            self.msa_entries = list(self.loader)
        elif isinstance(data_source, OpenProteinSetLoader):
            self.loader = data_source
            self.msa_entries = list(data_source)
        else:
            self.msa_entries = data_source

        logger.info(f"Loaded {len(self.msa_entries)} MSA entries")

        # Cache if needed
        if cache_dir:
            self._cache_data(cache_dir)

    def _cache_data(self, cache_dir: str):
        """Cache tokenized data for faster loading."""
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        # Implementation can be extended for large-scale caching
        pass

    def __len__(self) -> int:
        return len(self.msa_entries)

    def _sample_msa_sequences(self, msa_entry: MSAEntry) -> Tuple[List[str], List[str]]:
        """
        Sample MSA sequences.
        
        In few-shot mode:
        - Returns (context_examples, generation_targets)
        - context_examples: 2-3 example MSAs to condition on
        - generation_targets: remaining MSAs to generate
        
        In normal mode:
        - Returns ([], all_msa_sequences)
        """
        aligned_seqs = msa_entry.aligned_sequences[:self.max_msa_depth]
        
        if not self.use_few_shot:
            # Normal mode: all seqs are generation targets
            sampled = aligned_seqs if len(aligned_seqs) <= self.num_msa_sequences else random.sample(aligned_seqs, self.num_msa_sequences)
            return [], sampled
        
        # Few-shot mode: split context examples and generation targets
        total_needed = self.few_shot_examples + self.num_msa_sequences
        
        if len(aligned_seqs) <= total_needed:
            # Not enough seqs, use what we have
            context_examples = aligned_seqs[:self.few_shot_examples]
            generation_targets = aligned_seqs[self.few_shot_examples:]
        else:
            # Sample randomly
            sampled = random.sample(aligned_seqs, total_needed)
            context_examples = sampled[:self.few_shot_examples]
            generation_targets = sampled[self.few_shot_examples:]
        
        return context_examples, generation_targets

    def _build_input(
        self,
        query: str,
        msa_sequences: List[str],
        few_shot_examples: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Build input tensors for a single MSA example.

        Normal mode format: 
            [gMASK, sop, query..., <M>, seq1..., <M>, seq2..., ...]

        Few-shot mode format:
            [gMASK, sop, query..., <M>, example1..., <M>, example2..., <M>, seq1..., <M>, seq2..., ...]
            Context:    ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^
            Generation:                                                      ^^^^^^^^^^^^^^

        Args:
            query: Query sequence
            msa_sequences: Sequences to generate (context_mask=0)
            few_shot_examples: Optional example sequences (context_mask=1)

        Returns dictionary with:
        - input_ids: Token IDs
        - position_ids: Position within sequence
        - block_position_ids: MSA block index
        - context_mask: 1 for context, 0 for generation
        - attention_mask: 1 for valid, 0 for padding
        """
        # Remove gaps from query for clean sequence
        clean_query = query.replace('-', '').replace('.', '')

        # Tokenize query
        query_ids = self.tokenizer.encode(clean_query)
        query_len = len(query_ids)

        # Build full sequence
        input_ids = [self.tokenizer.gmask_id, self.tokenizer.sop_id]
        input_ids.extend(query_ids)

        position_ids = [0, 1]  # Special tokens get positions 0, 1
        position_ids.extend(range(len(query_ids)))

        block_position_ids = [0, 0]  # Query block = 0
        block_position_ids.extend([0] * len(query_ids))

        context_mask = [1, 1]  # Special tokens are context
        context_mask.extend([1] * len(query_ids))  # Query is context
        
        current_block = 1

        # Add few-shot examples (if provided) - CONTEXT
        if few_shot_examples:
            for example_seq in few_shot_examples:
                # Add delimiter
                input_ids.append(self.tokenizer.msa_delimiter_id)
                position_ids.append(0)
                block_position_ids.append(current_block)
                context_mask.append(1)  # ← Few-shot examples are CONTEXT!
                
                # Tokenize example
                example_tokens = self.tokenizer.encode(example_seq)
                
                # Truncate if needed
                remaining_space = self.max_seq_length - len(input_ids) - 1
                if len(example_tokens) > remaining_space:
                    example_tokens = example_tokens[:remaining_space]
                
                input_ids.extend(example_tokens)
                position_ids.extend(range(len(example_tokens)))
                block_position_ids.extend([current_block] * len(example_tokens))
                context_mask.extend([1] * len(example_tokens))  # ← Few-shot examples are CONTEXT!
                
                current_block += 1
                
                if len(input_ids) >= self.max_seq_length:
                    break

        # Add generation target MSA sequences - GENERATION
        for msa_seq in msa_sequences:
            # Add delimiter
            input_ids.append(self.tokenizer.msa_delimiter_id)
            position_ids.append(0)  # Delimiter position
            block_position_ids.append(current_block)
            context_mask.append(0)  # ← Generation targets

            # Tokenize MSA sequence
            msa_tokens = self.tokenizer.encode(msa_seq)

            # Truncate if needed
            remaining_space = self.max_seq_length - len(input_ids) - 1
            if len(msa_tokens) > remaining_space:
                msa_tokens = msa_tokens[:remaining_space]

            input_ids.extend(msa_tokens)
            position_ids.extend(range(len(msa_tokens)))
            block_position_ids.extend([current_block] * len(msa_tokens))
            context_mask.extend([0] * len(msa_tokens))  # ← Generation targets

            current_block += 1

            # Check if we've reached max length
            if len(input_ids) >= self.max_seq_length:
                break

        # Truncate to max length
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            position_ids = position_ids[:self.max_seq_length]
            block_position_ids = block_position_ids[:self.max_seq_length]
            context_mask = context_mask[:self.max_seq_length]

        # Create tensors
        seq_len = len(input_ids)
        attention_mask = [1] * seq_len

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'position_ids': torch.tensor(position_ids, dtype=torch.long),
            'block_position_ids': torch.tensor(block_position_ids, dtype=torch.long),
            'context_mask': torch.tensor(context_mask, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'seq_len': seq_len,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        msa_entry = self.msa_entries[idx]

        # Sample MSA sequences
        if self.use_few_shot:
            few_shot_examples, msa_sequences = self._sample_msa_sequences(msa_entry)
        else:
            few_shot_examples, msa_sequences = None, self._sample_msa_sequences(msa_entry)[1]

        # Build input tensors
        return self._build_input(
            msa_entry.query_sequence,
            msa_sequences,
            few_shot_examples=few_shot_examples if self.use_few_shot else None
        )


class MSACollator:
    """
    Collator for batching MSA examples with padding.
    """

    def __init__(
        self,
        pad_id: int = 0,
        max_length: Optional[int] = None,
    ):
        """
        Args:
            pad_id: Padding token ID
            max_length: Maximum sequence length (for truncation)
        """
        self.pad_id = pad_id
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch with padding.

        Returns:
            Dictionary with batched tensors
        """
        # Find max length in batch
        max_len = max(item['seq_len'] for item in batch)
        if self.max_length:
            max_len = min(max_len, self.max_length)

        batch_size = len(batch)

        # Initialize padded tensors
        input_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        position_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        block_position_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        context_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, item in enumerate(batch):
            seq_len = min(item['seq_len'], max_len)
            input_ids[i, :seq_len] = item['input_ids'][:seq_len]
            position_ids[i, :seq_len] = item['position_ids'][:seq_len]
            block_position_ids[i, :seq_len] = item['block_position_ids'][:seq_len]
            context_mask[i, :seq_len] = item['context_mask'][:seq_len]
            attention_mask[i, :seq_len] = item['attention_mask'][:seq_len]

        return {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'block_position_ids': block_position_ids,
            'context_mask': context_mask,
            'attention_mask': attention_mask,
        }


def create_dataloader(
    data_source: Union[str, List[MSAEntry]],
    batch_size: int = 48,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    """
    Create a DataLoader for MSA training.

    Args:
        data_source: Data source for MSADataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        **dataset_kwargs: Additional arguments for MSADataset

    Returns:
        DataLoader instance
    """
    dataset = MSADataset(data_source, **dataset_kwargs)
    collator = MSACollator(pad_id=0)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
