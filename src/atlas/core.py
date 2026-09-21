"""PFN Atlas core: what KIND of uncertainty is this?

TabPFN-3.5 gives U(x) (OOF entropy) + z(x) (internal test-embeddings from a
seed fit). kNN geometry over z gives support (labeled coverage) and neighbor
disagreement. Quadrants:
  UNKNOWN      high U, low support    -> label this
  AMBIGUOUS    high U, high support + high disagreement -> skip (labeling rarely helps)
  OOD-CONFIDENT low U, low support    -> flag, don't trust the confidence
  KNOWN        low U, high support    -> easy
Acquisition: UNKNOWN first (diversified by k-center), then KNOWN fill;
AMBIGUOUS deprioritized.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tabpfn import TabPFNClassifier


def seed_embeddings(X: pd.DataFrame, y: np.ndarray, frac=0.05, seed=0):
    n_seed = max(50, int(len(X) * frac))
    sidx = np.random.RandomState(seed + 1).choice(len(X), n_seed, replace=False)
    clf = TabPFNClassifier(random_state=seed)
    clf.fit(X.iloc[sidx], y[sidx])
    Z = clf.get_embeddings(X).mean(axis=0)
    return Z, sidx


def knn_stats(Z: np.ndarray, y: np.ndarray, labeled_idx: np.ndarray, k=15):
    nn = NearestNeighbors(n_neighbors=min(k, len(labeled_idx))).fit(Z[labeled_idx])
    dist, ind = nn.kneighbors(Z)
    lab = labeled_idx[ind]
    support = 1.0 / (1.0 + dist.mean(axis=1))
    dis = np.array([(y[r] != y[r[0]]).mean() if len(r) else 0.0 for r in lab])
    return support, dis


def atlas_frame(X: pd.DataFrame, y: np.ndarray, ent: np.ndarray, seed=0):
    Z, sidx = seed_embeddings(X, y, seed=seed)
    support, dis = knn_stats(Z, y, sidx)
    u_hi = ent >= np.median(ent)
    s_lo = support < np.median(support)
    d_hi = dis >= np.median(dis)
    quad = np.where(u_hi & s_lo, "UNKNOWN",
            np.where(u_hi & d_hi, "AMBIGUOUS",
             np.where(~u_hi & s_lo, "OOD-CONFIDENT", "KNOWN")))
    return pd.DataFrame({"entropy": ent, "support": support, "disagree": dis,
                         "quad": quad}), Z


def acquire(df: pd.DataFrame, Z: np.ndarray, k: int, seed=0):
    """UNKNOWN first (k-center diversified), then KNOWN, skip AMBIGUOUS last."""
    rng = np.random.RandomState(seed)
    order = {"UNKNOWN": 0, "KNOWN": 1, "OOD-CONFIDENT": 2, "AMBIGUOUS": 3}
    rank = np.array([order[q] for q in df["quad"]])
    sel = []
    for q in [0, 1, 2, 3]:
        cand = np.where(rank == q)[0]
        if len(cand) == 0:
            continue
        # k-center within tier for spread
        tier, taken, cur = [], [cand[rng.choice(len(cand))]], None
        dmin = np.linalg.norm(Z[cand] - Z[taken[0]], axis=1)
        tier.append(taken[0])
        while len(tier) < len(cand):
            i = int(np.argmax(dmin))
            tier.append(cand[i])
            dmin = np.minimum(dmin, np.linalg.norm(Z[cand] - Z[cand[i]], axis=1))
        sel += tier
        if len(sel) >= k:
            break
    return np.array(sel[:k])
