"""Atlas ablation: entropy shortlist + k-center in TabPFN embedding (no quadrants).
If this equals Atlas, the taxonomy is presentation. Saves figs/ablation2.csv.
GPU ~10 min.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from atlas.core import atlas_frame
from atlas.data import kcenter, openml_binary, seed_predict_entropy, seed_split, tabpfn_predict_proba

SEED = 0


def main():
    t0 = time.time()
    Path("figs").mkdir(exist_ok=True)
    X, y = openml_binary("phoneme")
    Xpool, Xval, ypool, yval = train_test_split(X, y, test_size=2000,
                                                stratify=y, random_state=1)
    idx = np.random.RandomState(SEED).choice(len(Xpool), 3404, replace=False)
    Xpool, ypool = Xpool.iloc[idx].reset_index(drop=True), ypool[idx]
    sidx = seed_split(ypool)
    unl = np.array([i for i in range(len(ypool)) if i not in set(sidx)])
    n_seed = len(sidx)
    ent = seed_predict_entropy(Xpool, ypool, sidx)[unl]
    ad, Z = atlas_frame(Xpool, ypool, seed_predict_entropy(Xpool, ypool, sidx),
                        seed=SEED, seed_idx=sidx)
    Ze = Z[unl]
    pos = np.argsort(-ent)
    rows = []
    for b in [0.1, 0.2, 0.4]:
        k = max(n_seed + 25, int(len(ypool) * b))
        need = k - n_seed
        top = pos[:max(need, min(2000, len(unl)))]
        for sname, sel in [
                ("ent+embdiv", np.concatenate([sidx, unl[top[kcenter(Ze[top], need)]]])),
                ("atlas", None)]:
            if sname == "atlas":
                from atlas.core import acquire
                sel = np.concatenate([sidx, unl[acquire(
                    ad.iloc[unl].reset_index(drop=True), Ze, need)]])
            p = tabpfn_predict_proba(Xpool.iloc[sel], ypool[sel], Xval, seed=SEED)
            a = roc_auc_score(yval, p)
            rows.append((b, sname, a))
            print(f"b={b} {sname}: {a:.4f}", flush=True)
    import pandas as pd
    pd.DataFrame(rows, columns=["budget", "setup", "auc"]).to_csv(
        "figs/ablation2.csv", index=False)
    print(f"done {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
