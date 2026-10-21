"""Calibration versus discrimination of the drought-probability forecasts (test 2005-2024).
* Murphy (1973) decomposition of the Brier score with 10 equal-width probability bins:
      BS ~ REL - RES + UNC
  REL (reliability, lower is better), RES (resolution, higher is better), UNC = o(1-o).
* Area under the ROC curve (AUC), which does not depend on calibration, pooled over points and years,
  and the AUC gain over the climatological reference (a); 95 % intervals by bootstrap of whole test years.
Usage: python skill_extra.py <pref>_prob_pred.csv <out.csv> [n_boot=1000]"""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

MODELS = ['clim', 'clim_nat', 'LOG_pers', 'LOG_lead', 'REG_lead', 'LOG_enso', 'REG_enso', 'LOG_trend', 'XGB_enso']


def murphy(p, y, bins=10):
    k = np.minimum((p * bins).astype(int), bins - 1)
    o = y.mean(); N = len(y)
    rel = res = 0.0
    for b in np.unique(k):
        i = k == b
        rel += i.sum() * (p[i].mean() - y[i].mean()) ** 2
        res += i.sum() * (y[i].mean() - o) ** 2
    return rel / N, res / N, o * (1 - o)


if __name__ == '__main__':
    src, out = sys.argv[1:3]
    nb = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    P = pd.read_csv(src)
    rows = []
    for season, g in P.groupby('season'):
        g = g.reset_index(drop=True)
        years = np.sort(g.year.unique()); idx = {y: np.where(g.year.values == y)[0] for y in years}
        y = g.y.values.astype(float)
        rng = np.random.default_rng(0)
        boots = [np.concatenate([idx[v] for v in rng.choice(years, len(years))]) for _ in range(nb)]
        auc_b = {m: np.array([roc_auc_score(y[i], g[m].values[i]) for i in boots]) for m in MODELS if g[m].nunique() > 1}
        for m in MODELS:
            p = g[m].values
            rel, res, unc = murphy(p, y)
            auc = roc_auc_score(y, p) if g[m].nunique() > 1 else 0.5
            row = dict(season=season, model=m, BS=np.mean((p - y) ** 2), REL=rel, RES=res, UNC=unc,
                       mean_prob=p.mean(), obs=y.mean(), AUC=auc)
            if m in auc_b:
                row.update(AUC_lo=np.quantile(auc_b[m], .025), AUC_hi=np.quantile(auc_b[m], .975))
                if m != 'clim':
                    d = auc_b[m] - auc_b['clim']
                    row.update(dAUC=auc - roc_auc_score(y, g.clim.values), dAUC_lo=np.quantile(d, .025),
                               dAUC_hi=np.quantile(d, .975))
            rows.append(row)
    R = pd.DataFrame(rows); R.to_csv(out, index=False)
    print(R.round(4).to_string(index=False))
