"""
Preference pair generator for DPO training.

Generates winner/loser pairs using proxy quality metrics when
AlphaFold2 structure prediction is not available.

Reference metrics inspired by MSAGPT paper:
- Conservation (column entropy)
- Coverage (gap fraction)
- Effective sequence count (Neff)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# Standard amino acids (20) + gap + unknown
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
GAP_CHARS = '.-'


@dataclass
class MSAQualityScore:
    """Quality scores for an MSA."""
    conservation: float  # Higher = more conserved columns
    coverage: float  # Higher = fewer gaps
    neff: float  # Effective sequence count
    combined: float  # Weighted combination


def compute_column_entropy(msa_matrix: np.ndarray, amino_acid_indices: Dict[str, int]) -> float:
    """
    Compute average column entropy of MSA.

    Lower entropy = more conserved = better quality.
    Returns negative entropy so higher is better.

    Args:
        msa_matrix: MSA as numpy array of characters (num_seqs, seq_len)
        amino_acid_indices: Mapping from AA to index

    Returns:
        Negative average entropy (higher is better)
    """
    num_seqs, seq_len = msa_matrix.shape

    if num_seqs < 2:
        return 0.0

    total_entropy = 0.0
    valid_columns = 0

    for col in range(seq_len):
        column = msa_matrix[:, col]

        # Count amino acids (excluding gaps)
        counts = {}
        total = 0
        for aa in column:
            aa = aa.upper()
            if aa in AMINO_ACIDS:
                counts[aa] = counts.get(aa, 0) + 1
                total += 1

        if total < 2:
            continue

        # Compute entropy
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)

        total_entropy += entropy
        valid_columns += 1

    if valid_columns == 0:
        return 0.0

    avg_entropy = total_entropy / valid_columns
    # Normalize and negate (max entropy for 20 AAs is log2(20) ≈ 4.32)
    max_entropy = np.log2(20)
    normalized = 1.0 - (avg_entropy / max_entropy)

    return normalized


def compute_coverage(msa_matrix: np.ndarray) -> float:
    """
    Compute coverage (fraction of non-gap positions).

    Args:
        msa_matrix: MSA as numpy array

    Returns:
        Coverage score (0-1, higher is better)
    """
    non_gap_count = 0
    total_count = msa_matrix.size

    for aa in msa_matrix.flat:
        if aa.upper() not in GAP_CHARS:
            non_gap_count += 1

    return non_gap_count / total_count if total_count > 0 else 0.0


def compute_neff(msa_matrix: np.ndarray, identity_threshold: float = 0.8) -> float:
    """
    Compute effective number of sequences (Neff).

    Sequences with >threshold identity are clustered together.

    Args:
        msa_matrix: MSA as numpy array
        identity_threshold: Threshold for clustering

    Returns:
        Normalized Neff score (0-1)
    """
    num_seqs, seq_len = msa_matrix.shape

    if num_seqs <= 1:
        return 0.0

    # Compute pairwise identities (sample if too large)
    max_seqs = min(num_seqs, 100)
    if num_seqs > max_seqs:
        indices = np.random.choice(num_seqs, max_seqs, replace=False)
        msa_subset = msa_matrix[indices]
    else:
        msa_subset = msa_matrix

    n = len(msa_subset)
    weights = np.ones(n)

    for i in range(n):
        cluster_size = 1
        for j in range(i + 1, n):
            # Compute identity
            matches = 0
            aligned = 0
            for k in range(seq_len):
                ai, aj = msa_subset[i, k].upper(), msa_subset[j, k].upper()
                if ai not in GAP_CHARS and aj not in GAP_CHARS:
                    aligned += 1
                    if ai == aj:
                        matches += 1

            if aligned > 0:
                identity = matches / aligned
                if identity > identity_threshold:
                    cluster_size += 1

        weights[i] = 1.0 / cluster_size

    neff = np.sum(weights)
    # Normalize to 0-1 range
    return min(neff / num_seqs, 1.0)


def msa_quality_score(
    msa_sequences: List[str],
    conservation_weight: float = 0.4,
    coverage_weight: float = 0.3,
    neff_weight: float = 0.3,
) -> MSAQualityScore:
    """
    Compute combined quality score for an MSA.

    Args:
        msa_sequences: List of aligned sequences
        conservation_weight: Weight for conservation score
        coverage_weight: Weight for coverage score
        neff_weight: Weight for Neff score

    Returns:
        MSAQualityScore with individual and combined scores
    """
    if len(msa_sequences) < 2:
        return MSAQualityScore(
            conservation=0.0,
            coverage=0.0,
            neff=0.0,
            combined=0.0,
        )

    # Pad sequences to same length
    max_len = max(len(s) for s in msa_sequences)
    padded = [s.ljust(max_len, '-') for s in msa_sequences]

    # Convert to numpy array
    msa_matrix = np.array([list(s) for s in padded])

    # Build AA index mapping
    aa_indices = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

    # Compute individual scores
    conservation = compute_column_entropy(msa_matrix, aa_indices)
    coverage = compute_coverage(msa_matrix)
    neff = compute_neff(msa_matrix)

    # Combined score
    combined = (
        conservation_weight * conservation +
        coverage_weight * coverage +
        neff_weight * neff
    )

    return MSAQualityScore(
        conservation=conservation,
        coverage=coverage,
        neff=neff,
        combined=combined,
    )


class PreferenceGenerator:
    """
    Generate preference pairs for DPO training.

    Creates winner/loser pairs from generated MSAs based on
    proxy quality metrics.
    """

    def __init__(
        self,
        min_score_difference: float = 0.3,
        conservation_weight: float = 0.4,
        coverage_weight: float = 0.3,
        neff_weight: float = 0.3,
    ):
        """
        Initialize preference generator.

        Args:
            min_score_difference: Minimum score difference for valid pair
            conservation_weight: Weight for conservation score
            coverage_weight: Weight for coverage score
            neff_weight: Weight for Neff score
        """
        self.min_score_difference = min_score_difference
        self.conservation_weight = conservation_weight
        self.coverage_weight = coverage_weight
        self.neff_weight = neff_weight

    def score_msa(self, msa_sequences: List[str]) -> MSAQualityScore:
        """Score a single MSA."""
        return msa_quality_score(
            msa_sequences,
            self.conservation_weight,
            self.coverage_weight,
            self.neff_weight,
        )

    def generate_pairs(
        self,
        query: str,
        generated_msas: List[List[str]],
    ) -> List[Tuple[List[str], List[str], float]]:
        """
        Generate preference pairs from multiple generated MSAs.

        Args:
            query: Query sequence
            generated_msas: List of generated MSAs (each is list of sequences)

        Returns:
            List of (winner_msa, loser_msa, score_difference) tuples
        """
        if len(generated_msas) < 2:
            return []

        # Score all MSAs
        scored = []
        for msa in generated_msas:
            # Include query in scoring
            full_msa = [query] + msa
            score = self.score_msa(full_msa)
            scored.append((msa, score))

        # Sort by combined score
        scored.sort(key=lambda x: x[1].combined, reverse=True)

        # Generate pairs
        pairs = []
        for i in range(len(scored)):
            for j in range(i + 1, len(scored)):
                winner, winner_score = scored[i]
                loser, loser_score = scored[j]

                diff = winner_score.combined - loser_score.combined
                if diff >= self.min_score_difference:
                    pairs.append((winner, loser, diff))

        logger.debug(f"Generated {len(pairs)} preference pairs from {len(generated_msas)} MSAs")
        return pairs

    def generate_from_sampling(
        self,
        model,
        query: str,
        num_samples: int = 8,
        num_pairs: int = 4,
        **sampling_kwargs,
    ) -> List[Tuple[List[str], List[str], float]]:
        """
        Generate preference pairs by sampling from a model.

        Args:
            model: Model with generate() method
            query: Query sequence
            num_samples: Number of MSAs to generate
            num_pairs: Maximum number of pairs to return
            **sampling_kwargs: Arguments for model.generate()

        Returns:
            List of preference pairs
        """
        # Generate multiple MSAs
        generated_msas = []

        for _ in range(num_samples):
            msa = model.generate(query, **sampling_kwargs)
            generated_msas.append(msa)

        # Generate pairs
        pairs = self.generate_pairs(query, generated_msas)

        # Return top pairs by score difference
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:num_pairs]


def create_preference_dataset(
    queries: List[str],
    generated_msas_per_query: List[List[List[str]]],
    min_score_difference: float = 0.3,
    output_path: Optional[str] = None,
) -> List[Dict]:
    """
    Create a preference dataset from generated MSAs.

    Args:
        queries: List of query sequences
        generated_msas_per_query: List of lists of generated MSAs per query
        min_score_difference: Minimum score difference for pairs
        output_path: Optional path to save as JSONL

    Returns:
        List of preference pair dictionaries
    """
    generator = PreferenceGenerator(min_score_difference=min_score_difference)
    dataset = []

    for query, msas in zip(queries, generated_msas_per_query):
        pairs = generator.generate_pairs(query, msas)

        for winner, loser, diff in pairs:
            dataset.append({
                'query': query,
                'winner': winner,
                'loser': loser,
                'score_difference': diff,
            })

    logger.info(f"Created preference dataset with {len(dataset)} pairs")

    if output_path:
        import json
        with open(output_path, 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Saved preference dataset to {output_path}")

    return dataset
