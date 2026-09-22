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

from atlas.core import acquire, atlas_frame, seed_embeddings
from atlas.data import kcenter, openml_binary, seed_predict_entropy, seed_split, tabpfn_predict_proba
from atlas.jepa_lite import embed, prep, train_plug

SEED = 0


def main():
    t0 = time.time()
    Path("figs").mkdir(exist_ok=True)
    X, y = openml_binary("phoneme")
    Xpool, _, ypool, _ = train_test_split(X, y, test_size=2000, stratify=y,
                                          random_state=1)
    idx = np.random.RandomState(SEED).choice(len(Xpool), 3404, replace=False)
    Xpool, ypool = Xpool.iloc[idx].reset_index(drop=True), ypool[idx]

    from tabpfn import TabPFNClassifier
    sidx = seed_split(ypool)
    seed_clf = TabPFNClassifier(random_state=SEED)
    seed_clf.fit(Xpool.iloc[sidx], ypool[sidx])
    oof = np.clip(seed_clf.predict_proba(Xpool)[:, 1], 1e-6, 1 - 1e-6)
    ent = -(oof * np.log(oof) + (1 - oof) * np.log(1 - oof))
    ad, Z = atlas_frame(Xpool, ypool, ent, seed=SEED, seed_idx=sidx)
    k = 340
    sels = {"random": np.random.RandomState(1).choice(len(ypool), k, replace=False),
            "entropy": np.argsort(-ent)[:k],
            "atlas": acquire(ad, Z, k)}
    xy = PCA(n_components=2, random_state=0).fit_transform(
        (Z - Z.mean(0)) / (Z.std(0) + 1e-9)).astype(np.float32)

    rng = np.random.RandomState(7)
    corr = np.zeros(len(ypool), bool)
    corr[rng.choice(len(ypool), int(0.10 * len(ypool)), replace=False)] = True
    out = {"xy": xy, "y": ypool.astype(np.int64), "corr": corr,
           "p_oof": oof.astype(np.float32),
           "entropy": ent.astype(np.float32),
           "support": ad["support"].values.astype(np.float32),
           "disagree": ad["disagree"].values.astype(np.float32),
           "quad": np.array(ad["quad"].tolist())}
    for sname, sel in sels.items():
        out[f"sel_{sname}"] = sel.astype(np.int64)
    np.savez_compressed("figs/atlas_demo.npz", **out)
    print(f"saved figs/atlas_demo.npz ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
