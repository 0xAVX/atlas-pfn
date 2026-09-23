"""Atlas GIF: acquisition order lighting up on quadrant scatter + quad legend.
Saves figs/atlas.gif."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

A = np.load("figs/atlas_demo.npz")
xy, y = A["xy"], A["y"]
QCOL = {"UNKNOWN": "#e74c3c", "AMBIGUOUS": "#f39c12",
        "OOD-CONFIDENT": "#9b59b6", "KNOWN": "#2ecc71"}
take = np.random.RandomState(0).choice(len(xy), 2500, replace=False)
order = A["sel_atlas"]
pos = {g: r for r, g in enumerate(order.tolist())}

fig, ax = plt.subplots(figsize=(5, 5.4))
ax.scatter(xy[take, 0], xy[take, 1], s=4, c="#dddddd", alpha=0.6)
hl = ax.scatter([], [], s=28, c=[], alpha=0.9)
ax.set_xticks([])
ax.set_yticks([])
title = ax.set_title("")
N = len(order)
STEPS = 12


def draw(f):
    k = (f + 1) * N // STEPS
    sel = order[:k]
    show = [g for g in take if g in set(sel.tolist())]
    hl.set_offsets(xy[show] if show else np.zeros((0, 2)))
    hl.set_color([QCOL[str(A["quad"][g])] for g in show])
    title.set_text(f"atlas acquisition: {k}/{N} labeled")
    return (hl, title)


FuncAnimation(fig, draw, frames=STEPS, interval=600).save(
    "figs/atlas.gif", writer="pillow", dpi=90)
print("saved figs/atlas.gif")
