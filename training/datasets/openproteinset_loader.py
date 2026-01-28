"""
OpenProteinSet data loader for MSA files.

Parses A3M and FASTA format MSA files from OpenProteinSet.
Dataset available at: s3://openfold/uniclust30/ (no authentication required)

Reference:
- OpenProteinSet: https://registry.opendata.aws/openfold/
- MSAGPT paper filtering criteria: 25-2000 AA length, >30% identity, <10% gaps, >10 sequences
"""

import os
import gzip
import re
from typing import List, Dict, Tuple, Optional, Iterator, Union
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MSAEntry:
    """Single MSA entry with query and aligned sequences."""
    id: str
    query_sequence: str
    aligned_sequences: List[str]
    sequence_ids: List[str]

    @property
    def depth(self) -> int:
        """Number of sequences in the MSA (including query)."""
        return 1 + len(self.aligned_sequences)

    @property
    def length(self) -> int:
        """Length of the query sequence."""
        return len(self.query_sequence.replace('-', '').replace('.', ''))

    def to_matrix(self) -> List[str]:
        """Return MSA as list of sequences (query first)."""
        return [self.query_sequence] + self.aligned_sequences


def parse_a3m(content: str, remove_insertions: bool = True) -> Tuple[List[str], List[str]]:
    """
    Parse A3M format MSA content.

    A3M format uses lowercase letters for insertions and uppercase for matches.
    Gaps are represented by '-' or '.'.

    Args:
        content: A3M file content as string
        remove_insertions: If True, remove lowercase insertion characters

    Returns:
        Tuple of (sequence_ids, sequences)
    """
    sequences = []
    sequence_ids = []
    current_seq = []
    current_id = None

    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue

        if line.startswith('>'):
            # Save previous sequence
            if current_id is not None and current_seq:
                seq = ''.join(current_seq)
                if remove_insertions:
                    # Remove lowercase insertions
                    seq = re.sub(r'[a-z]', '', seq)
                sequences.append(seq)
                sequence_ids.append(current_id)

            # Parse header
            current_id = line[1:].split()[0]
            current_seq = []
        else:
            current_seq.append(line)

    # Save last sequence
    if current_id is not None and current_seq:
        seq = ''.join(current_seq)
        if remove_insertions:
            seq = re.sub(r'[a-z]', '', seq)
        sequences.append(seq)
        sequence_ids.append(current_id)

    return sequence_ids, sequences


def parse_fasta(content: str) -> Tuple[List[str], List[str]]:
    """
    Parse FASTA format content.

    Args:
        content: FASTA file content as string

    Returns:
        Tuple of (sequence_ids, sequences)
    """
    sequences = []
    sequence_ids = []
    current_seq = []
    current_id = None

    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue

        if line.startswith('>'):
            if current_id is not None and current_seq:
                sequences.append(''.join(current_seq))
                sequence_ids.append(current_id)
            current_id = line[1:].split()[0]
            current_seq = []
        else:
            current_seq.append(line)

    if current_id is not None and current_seq:
        sequences.append(''.join(current_seq))
        sequence_ids.append(current_id)

    return sequence_ids, sequences


class OpenProteinSetLoader:
    """
    Loader for OpenProteinSet MSA files.

    Supports loading from local directories with A3M or FASTA files.
    Applies MSAGPT paper filtering criteria by default.
    """

    # MSAGPT paper filtering criteria
    DEFAULT_MIN_LENGTH = 25
    DEFAULT_MAX_LENGTH = 2000
    DEFAULT_MIN_IDENTITY = 0.30
    DEFAULT_MAX_GAP_FRACTION = 0.10
    DEFAULT_MIN_SEQUENCES = 10

    def __init__(
        self,
        data_dir: str,
        min_length: int = DEFAULT_MIN_LENGTH,
        max_length: int = DEFAULT_MAX_LENGTH,
        min_identity: float = DEFAULT_MIN_IDENTITY,
        max_gap_fraction: float = DEFAULT_MAX_GAP_FRACTION,
        min_sequences: int = DEFAULT_MIN_SEQUENCES,
        file_pattern: str = "*.a3m*",
    ):
        """
        Initialize the loader.

        Args:
            data_dir: Path to directory containing MSA files
            min_length: Minimum query sequence length
            max_length: Maximum query sequence length
            min_identity: Minimum sequence identity to query
            max_gap_fraction: Maximum fraction of gaps in sequences
            min_sequences: Minimum number of sequences in MSA
            file_pattern: Glob pattern for MSA files
        """
        self.data_dir = Path(data_dir)
        self.min_length = min_length
        self.max_length = max_length
        self.min_identity = min_identity
        self.max_gap_fraction = max_gap_fraction
        self.min_sequences = min_sequences
        self.file_pattern = file_pattern

        # Find all MSA files
        self.msa_files = self._find_msa_files()
        logger.info(f"Found {len(self.msa_files)} MSA files in {data_dir}")

    def _find_msa_files(self) -> List[Path]:
        """Find all MSA files matching the pattern."""
        files = list(self.data_dir.rglob(self.file_pattern))
        return sorted(files)

    def _read_file(self, path: Path) -> str:
        """Read file content, handling gzip compression."""
        if path.suffix == '.gz':
            with gzip.open(path, 'rt') as f:
                return f.read()
        else:
            with open(path, 'r') as f:
                return f.read()

    def _compute_identity(self, query: str, seq: str) -> float:
        """Compute sequence identity between query and aligned sequence."""
        if len(query) != len(seq):
            return 0.0

        matches = 0
        aligned_positions = 0

        for q, s in zip(query, seq):
            if q not in '.-' and s not in '.-':
                aligned_positions += 1
                if q == s:
                    matches += 1

        if aligned_positions == 0:
            return 0.0
        return matches / aligned_positions

    def _compute_gap_fraction(self, seq: str) -> float:
        """Compute fraction of gaps in sequence."""
        if len(seq) == 0:
            return 1.0
        gap_count = seq.count('-') + seq.count('.')
        return gap_count / len(seq)

    def _filter_msa(self, msa_entry: MSAEntry) -> Optional[MSAEntry]:
        """
        Apply MSAGPT filtering criteria to MSA.

        Returns filtered MSA entry or None if it doesn't meet criteria.
        """
        # Check query length
        query_len = msa_entry.length
        if query_len < self.min_length or query_len > self.max_length:
            return None

        # Filter aligned sequences
        filtered_seqs = []
        filtered_ids = []
        query = msa_entry.query_sequence

        for seq, seq_id in zip(msa_entry.aligned_sequences, msa_entry.sequence_ids):
            # Check gap fraction
            gap_frac = self._compute_gap_fraction(seq)
            if gap_frac > self.max_gap_fraction:
                continue

            # Check identity
            identity = self._compute_identity(query, seq)
            if identity < self.min_identity:
                continue

            filtered_seqs.append(seq)
            filtered_ids.append(seq_id)

        # Check minimum sequences
        if len(filtered_seqs) + 1 < self.min_sequences:
            return None

        return MSAEntry(
            id=msa_entry.id,
            query_sequence=query,
            aligned_sequences=filtered_seqs,
            sequence_ids=filtered_ids,
        )

    def load_msa(self, path: Union[str, Path]) -> Optional[MSAEntry]:
        """
        Load and filter a single MSA file.

        Args:
            path: Path to MSA file

        Returns:
            Filtered MSAEntry or None if doesn't meet criteria
        """
        path = Path(path)
        content = self._read_file(path)

        # Parse based on extension
        if '.a3m' in path.name:
            seq_ids, sequences = parse_a3m(content)
        else:
            seq_ids, sequences = parse_fasta(content)

        if len(sequences) < 2:
            return None

        # Create MSA entry (first sequence is query)
        msa_entry = MSAEntry(
            id=path.stem.split('.')[0],
            query_sequence=sequences[0],
            aligned_sequences=sequences[1:],
            sequence_ids=seq_ids[1:],
        )

        # Apply filtering
        return self._filter_msa(msa_entry)

    def __iter__(self) -> Iterator[MSAEntry]:
        """Iterate over all valid MSAs."""
        for path in self.msa_files:
            try:
                msa_entry = self.load_msa(path)
                if msa_entry is not None:
                    yield msa_entry
            except Exception as e:
                logger.warning(f"Error loading {path}: {e}")
                continue

    def __len__(self) -> int:
        """Return number of MSA files (not all may pass filtering)."""
        return len(self.msa_files)

    def get_file_list(self) -> List[Path]:
        """Return list of MSA file paths."""
        return self.msa_files


def download_openproteinset(
    output_dir: str,
    subset: str = "uniclust30",
    max_files: Optional[int] = None,
) -> None:
    """
    Download OpenProteinSet from AWS S3.

    Tries AWS CLI first, then falls back to alternative methods.

    Args:
        output_dir: Directory to save files
        subset: Which subset to download (uniclust30, etc.)
        max_files: Maximum number of files to download (for testing)
    """
    import subprocess
    import shutil

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if AWS CLI is available
    aws_available = shutil.which('aws') is not None

    if not aws_available:
        logger.warning("AWS CLI not found. Please install it with: pip install awscli")
        logger.info("Alternative: Download manually from https://registry.opendata.aws/openfold/")
        logger.info("")
        logger.info("Manual download commands (using curl):")
        logger.info("  # List available files:")
        logger.info(f"  curl 'https://openfold.s3.amazonaws.com/?list-type=2&prefix={subset}/'")
        logger.info("")
        logger.info("  # Download a specific file:")
        logger.info(f"  curl -O 'https://openfold.s3.amazonaws.com/{subset}/<filename>'")
        logger.info("")
        logger.info("Or use the test dataset at ./data/test_msa for development.")
        raise FileNotFoundError(
            "AWS CLI not found. Install with 'pip install awscli' or download data manually."
        )

    s3_path = f"s3://openfold/{subset}/"

    if max_files:
        # OpenProteinSet structure: uniclust30/{protein_id}/a3m/uniclust30.a3m
        # First list protein directories
        cmd = [
            "aws", "s3", "ls",
            s3_path,
            "--no-sign-request",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to list S3 bucket: {result.stderr}")
            raise RuntimeError(f"S3 list failed: {result.stderr}")

        # Parse directory listings (format: "PRE dirname/")
        lines = result.stdout.strip().split('\n')
        protein_ids = []
        for line in lines:
            if 'PRE' in line:
                # Extract directory name
                dirname = line.split()[-1].rstrip('/')
                protein_ids.append(dirname)

        protein_ids = protein_ids[:max_files]
        logger.info(f"Downloading A3M files for {len(protein_ids)} proteins...")

        for i, protein_id in enumerate(protein_ids):
            # Download the a3m file: {protein_id}/a3m/uniclust30.a3m
            src_path = f"{s3_path}{protein_id}/a3m/uniclust30.a3m"
            dst_path = output_path / f"{protein_id}.a3m"

            file_cmd = [
                "aws", "s3", "cp",
                src_path,
                str(dst_path),
                "--no-sign-request",
            ]
            result = subprocess.run(file_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"Failed to download {protein_id}: {result.stderr.strip()}")

            if (i + 1) % 10 == 0:
                logger.info(f"Downloaded {i + 1}/{len(protein_ids)} files")
    else:
        # Full sync - download all a3m files recursively
        cmd = [
            "aws", "s3", "sync",
            s3_path,
            str(output_path),
            "--no-sign-request",
            "--exclude", "*",
            "--include", "*/a3m/*.a3m",
        ]
        subprocess.run(cmd)

    logger.info(f"Downloaded OpenProteinSet to {output_dir}")
