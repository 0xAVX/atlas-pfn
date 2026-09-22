"""S6E9 Atlas-context test: does atlas-selected 10k context beat random 10k
context for TabPFN val AUC? Honest gate for using Atlas in the submission.
Saves figs/s6e9_atlas_ctx.txt. GPU ~8 min.
"""
from __future__ import annotations

import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from atlas.core import acquire, atlas_frame
from atlas.data import load_s6e9, oof_entropy, stratified_subsample, tabpfn_predict_proba

SEED = 0


def main():
    t0 = time.time()
    X, y, _, _, _ = load_s6e9("data")
    Xp, yp = stratified_subsample(X, y, 20000, seed=SEED)
    Xctx_pool, Xval, yctx_pool, yval = train_test_split(
        Xp, yp, test_size=10000, stratify=yp, random_state=1)
    _, ent = oof_entropy(Xctx_pool, yctx_pool, seed=SEED)
    ad, Z = atlas_frame(Xctx_pool, yctx_pool, ent, seed=SEED)
    out = []
    for name, sel in [
            ("random", np.random.RandomState(1).choice(len(Xctx_pool), 10000,
                                                       replace=False)),
            ("atlas", acquire(ad, Z, 10000))]:
        p = tabpfn_predict_proba(Xctx_pool.iloc[sel], yctx_pool[sel], Xval,
                                 seed=SEED)
        a = roc_auc_score(yval, p)
        out.append(f"{name}: {a:.4f}")
        print(out[-1], flush=True)
    open("figs/s6e9_atlas_ctx.txt", "w").write("\n".join(out) + "\n")
    print(f"done {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
