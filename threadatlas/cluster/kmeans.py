"""Pure-Python k-means on sparse cosine-normalized vectors.

Deterministic: fixed seed + k-means++ init + deterministic tie-breaking.
For a single-user vault (thousands of conversations) this is fast enough
and the code is easy to audit.
"""

from __future__ import annotations

import math
import random


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Dot product of two L2-normalized sparse vectors.

    If the vectors are not L2-normalized, this is still a dot product; the
    caller is responsible for normalization.
    """
    # Iterate over the smaller dict for a minor speedup.
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(t, 0.0) for t, w in a.items())


def _centroid_mean(members: list[dict[str, float]]) -> dict[str, float]:
    """Mean of sparse vectors, L2-normalized. Zero-length input returns {}."""
    if not members:
        return {}
    summed: dict[str, float] = {}
    inv_n = 1.0 / len(members)
    for v in members:
        for t, w in v.items():
            summed[t] = summed.get(t, 0.0) + w * inv_n
    norm_sq = sum(w * w for w in summed.values())
    if norm_sq <= 0:
        return {}
    norm = math.sqrt(norm_sq)
    return {t: w / norm for t, w in summed.items()}


def _kmeans_pp_init(
    vectors: list[dict[str, float]], k: int, rng: random.Random
) -> list[dict[str, float]]:
    """Deterministic k-means++ init.

    We pick the first centroid deterministically (index 0 after a seeded
    shuffle of indices); subsequent centroids are sampled with probability
    proportional to squared distance from the nearest existing centroid.
    """
    n = len(vectors)
    indices = list(range(n))
    rng.shuffle(indices)
    first = indices[0]
    centroids = [dict(vectors[first])]

    for _ in range(1, k):
        # squared distance from each point to the nearest centroid
        dists_sq = []
        for v in vectors:
            best_sim = max(cosine_similarity(v, c) for c in centroids)
            # cosine distance in [0, 2]; we use (1 - sim) squared for weighting
            d = max(1.0 - best_sim, 0.0)
            dists_sq.append(d * d)
        total = sum(dists_sq)
        if total <= 0:
            # All points overlap existing centroids; break deterministically.
            # Fill the remaining slots by walking the seeded index shuffle.
            taken = {id(c) for c in centroids}
            for idx in indices:
                v = vectors[idx]
                if id(v) in taken:
                    continue
                centroids.append(dict(v))
                taken.add(id(v))
                if len(centroids) == k:
                    break
            break
        target = rng.random() * total
        acc = 0.0
        chosen = 0
        for i, d in enumerate(dists_sq):
            acc += d
            if acc >= target:
                chosen = i
                break
        centroids.append(dict(vectors[chosen]))
    return centroids


def kmeans(
    vectors: list[dict[str, float]],
    k: int,
    *,
    max_iter: int = 50,
    seed: int = 42,
) -> tuple[list[int], list[dict[str, float]]]:
    """Run k-means on sparse cosine-normalized vectors.

    Returns ``(assignments, centroids)`` where ``assignments[i]`` is the
    cluster index for ``vectors[i]``.

    Edge cases:
      * empty input -> ([], [])
      * k <= 1 or k >= len(vectors) -> trivial assignment
    """
    n = len(vectors)
    if n == 0:
        return [], []
    if k <= 1:
        return [0] * n, [_centroid_mean(vectors)]
    if k >= n:
        return list(range(n)), [dict(v) for v in vectors]

    rng = random.Random(seed)
    centroids = _kmeans_pp_init(vectors, k, rng)
    assignments = [-1] * n
    for _ in range(max_iter):
        new_assignments: list[int] = []
        for v in vectors:
            best_j = 0
            best_sim = -2.0
            for j, c in enumerate(centroids):
                sim = cosine_similarity(v, c)
                if sim > best_sim:
                    best_sim = sim
                    best_j = j
            new_assignments.append(best_j)
        if new_assignments == assignments:
            break
        assignments = new_assignments
        # Recompute centroids.
        new_centroids: list[dict[str, float]] = []
        for j in range(k):
            members = [vectors[i] for i, a in enumerate(assignments) if a == j]
            if not members:
                # Preserve old centroid (never disappear a cluster silently).
                new_centroids.append(centroids[j])
            else:
                new_centroids.append(_centroid_mean(members))
        centroids = new_centroids
    return assignments, centroids
