"""
Graph noise injection utility for spatial KNN graph robustness experiments.

For each spot, `k_noise` of its `k` spatial KNN edges are replaced by edges to
random non-neighbor spots, simulating erroneous spatial connections. The
perturbation is fully determined by (coords, k_noise, seed), so runs with the
same seed are exactly comparable.

This module is used by EMERGE's spatial-graph construction
(`EMERGE/preprocess.py`): passing `--k_noise K --noise_seed S` to `main.py`
corrupts K edges per spot; the default `--k_noise 0` disables corruption.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def generate_noise_mapping(coords, k_noise, k=6, seed=42):
    """
    Generate a deterministic mapping of which KNN edges to replace and with whom.

    This is the single source of truth for noise injection: the apply_*
    helpers below all derive their perturbation from this mapping, so the
    same (coords, k_noise, seed) always yields the same perturbed edges.

    The mapping is computed on raw (un-normalized) coordinates using the same
    KNN finder, so the result is independent of each method's internal
    preprocessing (dense kernel, sparse adjacency, etc.).

    Parameters
    ----------
    coords : array-like, shape (n_spots, 2)
        Spatial coordinates.
    k_noise : int
        Number of edges to replace per spot.  Must be > 0.
    k : int
        Number of spatial nearest neighbors (default 6).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    nbr_indices : numpy.ndarray, shape (n_spots, k)
        Original KNN neighbor indices (excluding self).
    replacements : list of list of (int, int)
        replacements[i] is a list of (old_neighbor, new_neighbor) tuples
        for spot i.  old_neighbor is from the original KNN; new_neighbor
        is a random spot not in the original KNN neighbors of i.
    """
    coords = np.asarray(coords)
    n_spots = coords.shape[0]

    # Build KNN graph
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(coords)
    nbr_indices = nbrs.kneighbors(coords, return_distance=False)
    # nbr_indices[:, 0] is self; columns 1..k are the k nearest neighbors

    rng = np.random.RandomState(seed)
    k_replace = min(k_noise, k)
    all_spots = np.arange(n_spots)

    replacements = []
    for i in range(n_spots):
        original_neighbors = set(nbr_indices[i, 1:])
        # Which positions (among the k neighbors) to replace
        positions = rng.choice(k, size=k_replace, replace=False)

        row_replacements = []
        for p in positions:
            old_j = int(nbr_indices[i, p + 1])
            # Pick a random spot NOT in the original KNN neighbors (and not self)
            candidates = np.setdiff1d(all_spots, list(original_neighbors) + [i])
            if len(candidates) > 0:
                new_j = int(rng.choice(candidates))
            else:
                new_j = old_j  # fallback: keep original (should not happen)
            row_replacements.append((old_j, new_j))
        replacements.append(row_replacements)

    return nbr_indices[:, 1:], replacements


# ---------------------------------------------------------------------------
# Helpers applying the shared noise mapping to different graph representations.
# EMERGE's preprocessing uses apply_noise_to_knn_indices(); the other two are
# kept for experiments on dense-kernel or coordinate-free graph variants.
# ---------------------------------------------------------------------------

def add_edge_noise(neighbor_indices, k_noise, n_spots, seed=42):
    """
    Apply noise to a KNN neighbor-index array WITHOUT raw coordinates.

    `neighbor_indices` has shape (n_cells, k) and does NOT include self-loops.
    The returned array has the same shape as the input, with k_noise entries
    per row replaced by random spots.

    NOTE: because no coordinates are available here, this function uses its
    own local replacement logic rather than the shared coordinate-based
    mapping, so its perturbation may differ from apply_noise_to_knn_indices()
    even with the same seed.  Prefer apply_noise_to_knn_indices() when
    coordinates are available.
    """
    if k_noise <= 0:
        return neighbor_indices

    n_cells, k = neighbor_indices.shape
    # We don't have raw coords here, so fall back to the original local logic
    # to preserve backward compatibility for callers that don't pass coords.
    # For the unified pipeline, use apply_noise_to_knn_indices() instead.
    rng = np.random.RandomState(seed)
    result = neighbor_indices.copy()
    all_spots = np.arange(n_spots)

    for i in range(n_cells):
        original_neighbors = set(neighbor_indices[i])
        positions = rng.choice(k, size=k_noise, replace=False)
        for j in positions:
            candidates = np.setdiff1d(all_spots, list(original_neighbors) + [i])
            if len(candidates) > 0:
                result[i, j] = rng.choice(candidates)

    return result


def apply_noise_to_knn_indices(coords, neighbor_indices, k_noise, k=6, seed=42):
    """
    Apply noise to a KNN neighbor-index array using the shared mapping.

    This is the entry point used by EMERGE's spatial-graph construction.

    Parameters
    ----------
    coords : array-like, shape (n_spots, 2)
        Spatial coordinates (used by generate_noise_mapping).
    neighbor_indices : numpy.ndarray, shape (n_spots, k)
        KNN neighbor indices, NOT including self-loops.
    k_noise : int
        Number of edges to replace per spot.
    k : int
        Number of spatial nearest neighbors.
    seed : int
        Random seed.

    Returns
    -------
    noisy_indices : numpy.ndarray, shape (n_spots, k)
        Noisy KNN neighbor indices.
    """
    if k_noise <= 0:
        return neighbor_indices.copy()

    _, replacements = generate_noise_mapping(coords, k_noise, k=k, seed=seed)
    result = neighbor_indices.copy()
    for i, row_reps in enumerate(replacements):
        for old_j, new_j in row_reps:
            # Find the column in neighbor_indices[i] that equals old_j
            cols = np.where(result[i] == old_j)[0]
            if len(cols) > 0:
                result[i, cols[0]] = new_j
    return result


def apply_noise_to_dense_kernel(K, coords, bandwidth, k_noise, k=6, seed=42):
    """
    Apply noise to a dense gaussian kernel matrix using the shared mapping.

    For each replaced edge, the weight to the old neighbor is set to 0 and
    the weight to the new neighbor is computed with the same gaussian formula.

    Parameters
    ----------
    K : numpy.ndarray, shape (n_spots, n_spots)
        Dense gaussian kernel matrix.
    coords : array-like, shape (n_spots, 2)
        Spatial coordinates (z-score normalized, same as used to build K).
    bandwidth : float
        Gaussian kernel bandwidth.
    k_noise : int
        Number of edges to replace per spot.
    k : int
        Number of spatial nearest neighbors.
    seed : int
        Random seed.

    Returns
    -------
    K_noisy : numpy.ndarray, shape (n_spots, n_spots)
        Noisy kernel matrix (symmetrized).
    """
    if k_noise <= 0:
        return K

    _, replacements = generate_noise_mapping(coords, k_noise, k=k, seed=seed)
    K_noisy = K.copy()

    for i, row_reps in enumerate(replacements):
        for old_j, new_j in row_reps:
            # Remove weight to old neighbor
            K_noisy[i, old_j] = 0.0
            # Add weight to new neighbor using the same gaussian formula
            dist_sq = np.sum((coords[i] - coords[new_j]) ** 2)
            K_noisy[i, new_j] = np.exp(-dist_sq / bandwidth)

    # Re-symmetrize: an edge exists if either direction has it
    K_noisy = np.maximum(K_noisy, K_noisy.T)
    return K_noisy
