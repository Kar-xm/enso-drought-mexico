"""Uncertainty of the Brier skill score by block bootstrap of test years.
Each replicate resamples the 20 test seasons (2005-2024) with replacement, keeping all grid points
of a season together (the points of one season share the same ENSO state, so the independent units
are the seasons). BSS = 1 - sum SE(model) / sum SE(clim) over the resampled seasons.
Reports the point estimate, the 95 % percentile interval and P(BSS <= 0), nationally and by region,
plus paired differences between models (same resampled seasons).
Usage: python skill_bootstrap.py <pref>_prob_pred.csv <out.csv> [n_boot=5000]"""
import sys
import numpy as np, pandas as pd

MODELS = ['clim_nat', 'LOG_pers', 'LOG_lead', 'LOG_enso', 'LOG_trend', 'REG_lead', 'REG_enso', 'XGB_enso']
PAIRS = [('LOG_lead', 'clim_nat'), ('LOG_enso', 'XGB_enso'), ('REG_lead', 'LOG_lead'), ('REG_enso', 'LOG_enso')]


def region(lon, lat):
    if lon <= -109: return 'Noroeste'
    if lat >= 23.5: return 'Norte-Noreste'
    if lon <= -94 and lat >= 17.5: return 'Centro-Occidente'
    return 'Sur-Sureste'


def boot(se, years, n, rng):
    """se: DataFrame indexed by year with summed squared errors per model (incl. clim)."""
    idx = rng.integers(0, len(years), size=(n, len(years)))
    S = se.loc[years].values[idx].sum(axis=1)              # n x models
    return pd.DataFrame(S, columns=se.columns)


if __name__ == '__main__':
    src, out = sys.argv[1:3]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    P = pd.read_csv(src)
    P['reg'] = [region(a, b) for a, b in zip(P.lon, P.lat)]
    P['region'] = 'Nacional'
    rows = []
    for scope in ('region', 'reg'):
        for (season, rg), g in P.groupby(['season', scope]):
            cols = ['clim'] + [m for m in MODELS if m in g]
            se = pd.DataFrame({m: (g[m] - g.y) ** 2 for m in cols}).assign(year=g.year).groupby('year').sum()
            years = np.sort(g.year.unique())
            B = boot(se, years, n, np.random.default_rng(0))
            tot = se.sum()
            for m in cols[1:]:
                est = 1 - tot[m] / tot['clim']; b = 1 - B[m] / B['clim']
                rows.append(dict(season=season, scope=rg, model=m, BSS=est, lo=b.quantile(.025), hi=b.quantile(.975),
                                 p_le0=(b <= 0).mean()))
            for m1, m2 in PAIRS:
                if m1 in cols and m2 in cols:
                    est = (tot[m2] - tot[m1]) / tot['clim']; b = (B[m2] - B[m1]) / B['clim']
                    rows.append(dict(season=season, scope=rg, model=f'{m1} - {m2}', BSS=est, lo=b.quantile(.025),
                                     hi=b.quantile(.975), p_le0=(b <= 0).mean()))
    R = pd.DataFrame(rows); R.to_csv(out, index=False)
    print(R[R.scope == 'Nacional'].assign(**{c: lambda d, c=c: (d[c] * 100).round(1) for c in ('BSS', 'lo', 'hi')})
          .round(3).to_string(index=False))
