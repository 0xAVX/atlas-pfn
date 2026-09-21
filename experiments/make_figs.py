"""Atlas README figures: quadrant scatter + budget curves. Saves figs/*.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

A = np.load("figs/atlas_demo.npz")
QCOL = {"UNKNOWN": "#e74c3c", "AMBIGUOUS": "#f39c12",
        "OOD-CONFIDENT": "#9b59b6", "KNOWN": "#2ecc71"}
fig, ax = plt.subplots(figsize=(5, 5))
for q, c in QCOL.items():
    m = A["quad"] == q
    ax.scatter(A["xy"][m, 0], A["xy"][m, 1], s=4, c=c, label=q, alpha=0.6)
ax.legend(fontsize=8, markerscale=3)
ax.set_title("PFN Atlas quadrants (phoneme pool)")
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig("figs/quadrants.png", dpi=110)

C = pd.read_csv("figs/atlas.csv")
d = C[C.exp == "clean"]
fig, ax = plt.subplots(figsize=(6, 3.5))
for s, m in [("random", "o"), ("entropy", "x"), ("jepa-kcenter", "s"),
             ("atlas", "D")]:
    dd = d[d.strategy == s].sort_values("budget")
    ax.plot(dd.budget, dd.auc, marker=m, label=s)
ax.set_title("phoneme label-budget curves")
ax.set_xlabel("budget"); ax.set_ylabel("val AUC")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figs/atlas_budget.png", dpi=110)
print("saved quadrants/atlas_budget pngs")
