"""Atlas demo artifacts: PCA-2D, per-point quad/entropy/support, selections
(entropy/atlas/random @10%), poison overlay (corr flags + dirty entropy picks).
Saves figs/atlas_demo.npz. GPU ~7 min.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

sys.path.insert(0, "/home/dead/pfn-atlas/src")
sys.path.insert(0, "/home/dead/pfn-jepa/src")
sys.path.insert(0, "/home/dead/pfn-jepa/experiments")
sys.path.insert(0, "/home/dead/playground-series-s6e9")
from atlas.core import acquire, atlas_frame, seed_embeddings
from pfn_jepa.crossfit import oof_uncertainty
from run_matrix import openml_binary

SEED = 0


def main():
    t0 = time.time()
    Path("figs").mkdir(exist_ok=True)
    X, y = openml_binary("phoneme")
    Xpool, _, ypool, _ = train_test_split(X, y, test_size=2000, stratify=y,
                                          random_state=1)
    idx = np.random.RandomState(SEED).choice(len(Xpool), 3404, replace=False)
    Xpool, ypool = Xpool.iloc[idx].reset_index(drop=True), ypool[idx]

    u = oof_uncertainty(Xpool, ypool, seed=SEED)
    ad, Z = atlas_frame(Xpool, ypool, u["entropy"].values, seed=SEED)
    k = 340
    sels = {"random": np.random.RandomState(1).choice(len(ypool), k, replace=False),
            "entropy": np.argsort(-u["entropy"].values)[:k],
            "atlas": acquire(ad, Z, k)}
    xy = PCA(n_components=2, random_state=0).fit_transform(
        (Z - Z.mean(0)) / (Z.std(0) + 1e-9)).astype(np.float32)

    rng = np.random.RandomState(7)
    corr = np.zeros(len(ypool), bool)
    corr[rng.choice(len(ypool), int(0.10 * len(ypool)), replace=False)] = True
    out = {"xy": xy, "y": ypool.astype(np.int64), "corr": corr,
           "p_oof": u["p_oof"].values.astype(np.float32),
           "entropy": u["entropy"].values.astype(np.float32),
           "support": ad["support"].values.astype(np.float32),
           "disagree": ad["disagree"].values.astype(np.float32),
           "quad": np.array(ad["quad"].tolist())}
    for sname, sel in sels.items():
        out[f"sel_{sname}"] = sel.astype(np.int64)
    np.savez_compressed("figs/atlas_demo.npz", **out)
    print(f"saved figs/atlas_demo.npz ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
