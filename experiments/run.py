"""Validate Atlas under a real seed protocol (no pool-label leakage).

5% stratified seed labeled. Entropy from seed-fit TabPFN on unlabeled rows;
quadrants/disagreement from seed labels only. Budgets count total labels.
(1) clean budget curve, (2) poison phoneme. Saves figs/atlas.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from atlas.core import acquire, atlas_frame
from atlas.data import kcenter, openml_binary, seed_predict_entropy, seed_split, tabpfn_predict_proba
from atlas.jepa_lite import embed, prep, train_plug

SEED = 0


def main():
    t0 = time.time()
    Path("figs").mkdir(exist_ok=True)
    X, y = openml_binary("phoneme")
    Xpool, Xval, ypool, yval = train_test_split(X, y, test_size=2000,
                                                stratify=y, random_state=1)
    idx = np.random.RandomState(SEED).choice(len(Xpool), 3404, replace=False)
    Xpool, ypool = Xpool.iloc[idx].reset_index(drop=True), ypool[idx]
    rows = []
    seed_idx = seed_split(ypool)
    unl = np.array([i for i in range(len(ypool)) if i not in set(seed_idx)])
    n_seed = len(seed_idx)

    # clean budget curve
    ent_all = seed_predict_entropy(Xpool, ypool, seed_idx)
    ad, Z = atlas_frame(Xpool, ypool, ent_all, seed=SEED, seed_idx=seed_idx)
    print("quadrant mix:", ad["quad"].value_counts(normalize=True).round(2).to_dict(),
          flush=True)
    Xp, _ = prep(Xpool)
    net, dev = train_plug(Xp, epochs=5, d_lat=32, seed=SEED)
    Zj = embed(net, dev, Xp)[unl]
    Zr = (Xp / (np.abs(Xp).max(axis=0) + 1e-9))[unl]
    Zu, ent = Z[unl], ent_all[unl]
    pos = np.argsort(-ent)
    for b in [0.05, 0.1, 0.2, 0.4]:
        k = max(n_seed + 25, int(len(ypool) * b))
        need = k - n_seed
        take = lambda s: np.concatenate([seed_idx, unl[s]])
        sels = {"random": take(np.random.RandomState(1).choice(len(unl), need,
                                                               replace=False)),
                "entropy": take(pos[:need]),
                "jepa-kcenter": take(kcenter(Zj, need)),
                "atlas": take(acquire(ad.iloc[unl].reset_index(drop=True),
                                      Zu, need))}
        for sname, sel in sels.items():
            p = tabpfn_predict_proba(Xpool.iloc[sel], ypool[sel], Xval, seed=SEED)
            a = roc_auc_score(yval, p)
            rows.append(("clean", b, sname, a, float("nan")))
            print(f"clean b={b} {sname}: {a:.4f}", flush=True)

    # poison: quadrant distribution of entropy picks + downstream
    rng = np.random.RandomState(7)
    corr = np.zeros(len(ypool), bool)
    corr[rng.choice(len(ypool), int(0.10 * len(ypool)), replace=False)] = True
    y_dirty = ypool.copy()
    y_dirty[corr] = 1 - y_dirty[corr]
    seed_d = seed_split(y_dirty)
    unl_d = np.array([i for i in range(len(ypool)) if i not in set(seed_d)])
    ent_d = seed_predict_entropy(Xpool, y_dirty, seed_d)[unl_d]
    add, Zd = atlas_frame(Xpool, y_dirty, seed_predict_entropy(Xpool, y_dirty, seed_d),
                          seed=SEED, seed_idx=seed_d)
    Zd, entpos = Zd[unl_d], np.argsort(-ent_d)
    k = 340 - len(seed_d)
    ent_sel = unl_d[entpos[:k]]
    print("entropy picks quadrant mix:",
          add.iloc[ent_sel]["quad"].value_counts(normalize=True).round(2).to_dict(),
          flush=True)
    Xpd, _ = prep(Xpool)
    netd, devd = train_plug(Xpd, epochs=5, d_lat=32, seed=SEED)
    Zjd = embed(netd, devd, Xpd)[unl_d]
    sels = {"random": np.concatenate(
                [seed_d, unl_d[np.random.RandomState(1).choice(len(unl_d), k,
                                                               replace=False)]]),
            "entropy": np.concatenate([seed_d, ent_sel]),
            "jepa-kcenter": np.concatenate([seed_d, unl_d[kcenter(Zjd, k)]]),
            "atlas": np.concatenate(
                [seed_d, unl_d[acquire(add.iloc[unl_d].reset_index(drop=True),
                                       Zd, k)]])}
    for sname, sel in sels.items():
        p = tabpfn_predict_proba(Xpool.iloc[sel], y_dirty[sel], Xval, seed=SEED)
        a = roc_auc_score(yval, p)
        rows.append(("poison", 0.10, sname, a, float(corr[sel].mean())))
        print(f"poison {sname}: corr={corr[sel].mean():.3f} AUC={a:.4f}", flush=True)

    pd.DataFrame(rows, columns=["exp", "budget", "strategy", "auc",
                                "corr_frac"]).to_csv("figs/atlas.csv", index=False)
    print(f"saved figs/atlas.csv ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
