"""Data utilities for testing and examples."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def generate_calibrated_data(
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate perfectly calibrated scores and labels.

    The scores are uniformly distributed in (0, 1), and labels are sampled
    as Bernoulli with p = score. This gives perfect calibration by construction.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (scores, labels) tensors
    """
    if seed is not None:
        torch.manual_seed(seed)

    scores = torch.rand(n_samples)
    labels = torch.bernoulli(scores)

    return scores, labels


def generate_miscalibrated_data(
    n_samples: int = 1000,
    temperature: float = 2.0,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate miscalibrated scores and labels.

    Generates true probabilities, then distorts them with temperature scaling
    to create miscalibration. Labels are sampled from true probabilities.

    Args:
        n_samples: Number of samples to generate
        temperature: Temperature for miscalibration (>1 = overconfident, <1 = underconfident)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (miscalibrated_scores, labels) tensors
    """
    if seed is not None:
        torch.manual_seed(seed)

    # True probabilities
    true_probs = torch.rand(n_samples)

    # Miscalibrated scores (apply inverse temperature in logit space)
    logits = torch.logit(true_probs.clamp(1e-7, 1 - 1e-7))
    miscalibrated_logits = logits * temperature
    scores = torch.sigmoid(miscalibrated_logits)

    # Labels from true probabilities
    labels = torch.bernoulli(true_probs)

    return scores, labels


def generate_ranking_data(
    n_queries: int = 100,
    n_docs_per_query: int = 100,
    n_relevant_per_query: int = 10,
    score_noise: float = 0.3,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic ranking data.

    Creates queries with documents, where relevant documents have higher
    scores on average but with some noise.

    Args:
        n_queries: Number of queries
        n_docs_per_query: Documents per query
        n_relevant_per_query: Relevant documents per query
        score_noise: Standard deviation of score noise
        seed: Random seed for reproducibility

    Returns:
        Tuple of (scores, labels, query_ids) where:
        - scores: shape (n_queries * n_docs_per_query,)
        - labels: shape (n_queries * n_docs_per_query,)
        - query_ids: shape (n_queries * n_docs_per_query,)
    """
    if seed is not None:
        torch.manual_seed(seed)

    all_scores = []
    all_labels = []
    all_query_ids = []

    for q in range(n_queries):
        # Create labels (first n_relevant are relevant)
        labels = torch.zeros(n_docs_per_query)
        labels[:n_relevant_per_query] = 1

        # Shuffle
        perm = torch.randperm(n_docs_per_query)
        labels = labels[perm]

        # Generate scores: relevant docs have higher base score
        base_scores = labels * 0.6 + 0.2  # relevant: ~0.8, non-relevant: ~0.2
        scores = base_scores + torch.randn(n_docs_per_query) * score_noise
        scores = scores.clamp(0, 1)

        all_scores.append(scores)
        all_labels.append(labels)
        all_query_ids.append(torch.full((n_docs_per_query,), q))

    return (
        torch.cat(all_scores),
        torch.cat(all_labels),
        torch.cat(all_query_ids),
    )
