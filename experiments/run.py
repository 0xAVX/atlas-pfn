"""Validate Atlas: (1) poison phoneme — atlas vs entropy/random/jepa-kcenter,
(2) clean budget curve for atlas. Saves figs/atlas.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, "/home/dead/pfn-atlas/src")
sys.path.insert(0, "/home/dead/pfn-jepa/src")
sys.path.insert(0, "/home/dead/pfn-jepa/experiments")
sys.path.insert(0, "/home/dead/playground-series-s6e9")
from atlas.core import acquire, atlas_frame
from pfn_jepa.crossfit import oof_uncertainty
from pfn_jepa.jepa import embed, prep, train_plug
from selection import kcenter
from run_matrix import openml_binary
from src.ev import tabpfn_predict_proba

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

    # clean budget curve
    u = oof_uncertainty(Xpool, ypool, seed=SEED)
    ad, Z = atlas_frame(Xpool, ypool, u["entropy"].values, seed=SEED)
    print("quadrant mix:", ad["quad"].value_counts(normalize=True).round(2).to_dict(),
          flush=True)
    Xp, _ = prep(Xpool)
    net, dev = train_plug(Xp, epochs=5, d_lat=32, seed=SEED)
    Zj = embed(net, dev, Xp)
    for b in [0.05, 0.1, 0.2, 0.4]:
        k = max(50, int(len(ypool) * b))
        sels = {"random": np.random.RandomState(1).choice(len(ypool), k, replace=False),
                "entropy": np.argsort(-u["entropy"].values)[:k],
                "jepa-kcenter": kcenter(Zj, k),
                "atlas": acquire(ad, Z, k)}
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
    ud = oof_uncertainty(Xpool, y_dirty, seed=SEED)
    add, Zd = atlas_frame(Xpool, y_dirty, ud["entropy"].values, seed=SEED)
    k = 340
    ent_sel = np.argsort(-ud["entropy"].values)[:k]
    print("entropy picks quadrant mix:",
          add.iloc[ent_sel]["quad"].value_counts(normalize=True).round(2).to_dict(),
          flush=True)
    Xpd, _ = prep(Xpool)
    netd, devd = train_plug(Xpd, epochs=5, d_lat=32, seed=SEED)
    Zjd = embed(netd, devd, Xpd)
    sels = {"random": np.random.RandomState(1).choice(len(ypool), k, replace=False),
            "entropy": ent_sel,
            "jepa-kcenter": kcenter(Zjd, k),
            "atlas": acquire(add, Zd, k)}
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
