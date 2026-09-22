import numpy as np
import pandas as pd

from atlas.core import acquire, knn_stats


def test_quadrants_and_acquire():
    rng = np.random.RandomState(0)
    n = 200
    Z = rng.randn(n, 8)
    y = (Z[:, 0] > 0).astype(int)
    lab = np.arange(0, n, 10)
    sup, dis = knn_stats(Z, y, lab, k=5)
    assert len(sup) == n and ((sup > 0) & (sup <= 1)).all()
    ent = np.abs(rng.randn(n))
    import pandas as pd
    df = pd.DataFrame({"entropy": ent, "support": sup, "disagree": dis,
                       "quad": np.where(ent > np.median(ent), "UNKNOWN", "KNOWN")})
    sel = acquire(df, Z, 20)
    assert len(sel) == 20 and len(set(sel.tolist())) == 20
