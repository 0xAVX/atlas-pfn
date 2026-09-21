"""PFN Atlas demo: point inspector + budget view. Deterministic: loads
figs/atlas_demo.npz + figs/atlas.csv. Run: <venv-python> demo/app.py (port 5002)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

ROOT = Path(__file__).resolve().parent.parent
A = np.load(ROOT / "figs" / "atlas_demo.npz")
CSV = pd.read_csv(ROOT / "figs" / "atlas.csv")

app = Flask(__name__)
QCOL = {"UNKNOWN": "#e74c3c", "AMBIGUOUS": "#f39c12",
        "OOD-CONFIDENT": "#9b59b6", "KNOWN": "#2ecc71"}

PAGE = """
<h1>PFN Atlas — TabPFN says uncertain. Why?</h1>
<p><a href="/?v=point">Point inspector</a> | <a href="/?v=budget">Budget view</a></p>
{% if v == 'point' %}
<form method=get><input type=hidden name=v value=point>
Point index (0-{{n}}): <input name=i value="{{i}}" size=6><input type=submit value="Inspect"></form>
<h2>Prediction: P(class 1) = {{'%.2f' % prob }}</h2>
<ul><li>TabPFN entropy: {{'%.3f' % ent}}</li><li>Support: {{'%.2f' % sup}}</li>
<li>Neighbour disagreement: {{'%.2f' % dis}}</li>
<li>Diagnosis: <b>{{quad}}</b> — {{rec}}</li></ul>
{% else %}
<table border=1 cellpadding=4><tr><th>budget</th><th>random</th><th>entropy</th>
<th>jepa-kcenter</th><th>atlas</th></tr>
{% for b, r, e, j, a in rows %}<tr><td>{{b}}</td><td>{{r}}</td><td>{{e}}</td><td>{{j}}</td><td><b>{{a}}</b></td></tr>{% endfor %}</table>
<p>Poison pool (10% flipped): random 0.902, atlas 0.886, entropy 0.365 — same ~8%
corruption selected by all. Concentration kills, not quantity.</p>
{% endif %}
<h3>Atlas</h3>{{svg|safe}}
"""


def scatter(hl=None):
    xy = A["xy"]
    take = np.random.RandomState(0).choice(len(xy), 2500, replace=False)
    xs = xy[take]
    x0, x1 = xs[:, 0].min(), xs[:, 0].max()
    y0, y1 = xs[:, 1].min(), xs[:, 1].max()
    sc = lambda a, lo, hi: 10 + (500 - 20) * (a - lo) / max(hi - lo, 1e-9)
    out = ['<svg width="500" height="500" style="border:1px solid #ccc">']
    for i, g in enumerate(take):
        c = QCOL[str(A["quad"][g])]
        r, ex = (2.2, "") if g != hl else (7, ' stroke="black" stroke-width="2"')
        out.append(f'<circle cx="{sc(xs[i,0],x0,x1):.1f}" cy="{sc(xs[i,1],y0,y1):.1f}"'
                   f' r="{r}" fill="{c}"{ex}/>')
    out.append("</svg><p>red UNKNOWN · orange AMBIGUOUS · purple OOD-CONF · green KNOWN</p>")
    return "".join(out)


REC = {"UNKNOWN": "high-value: label this sample",
       "AMBIGUOUS": "conflicting neighbourhood: labeling may not help",
       "OOD-CONFIDENT": "low support: do not trust the confidence",
       "KNOWN": "easy: low priority"}


@app.get("/")
def index():
    v = request.args.get("v", "point")
    if v == "budget":
        piv = CSV[CSV.exp == "clean"].pivot(index="budget", columns="strategy",
                                            values="auc").round(4)
        rows = [(b,) + tuple(piv.loc[b, s] for s in
                             ["random", "entropy", "jepa-kcenter", "atlas"])
                for b in piv.index]
        return render_template_string(PAGE, v=v, rows=rows, svg=scatter())
    i = int(request.args.get("i", 0)) % len(A["xy"])
    e = A["entropy"]
    p = A["p_oof"]  # cross-fit TabPFN proba: honest, never in-sample
    q = str(A["quad"][i])
    return render_template_string(PAGE, v="point", n=len(A["xy"]), i=i, prob=p[i],
                                  ent=float(e[i]), sup=float(A["support"][i]),
                                  dis=float(A["disagree"][i]), quad=q,
                                  rec=REC[q], svg=scatter(hl=i))


if __name__ == "__main__":
    app.run(debug=True, port=5002)
