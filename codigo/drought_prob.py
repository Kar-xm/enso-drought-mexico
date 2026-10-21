"""Probability of seasonal drought (index-6 <= -1) conditioned on ENSO, Mexico 1951-2024.
Uses the dataset written by build_dataset.py (<pref>_dataset.csv). Train 1951-2004, test 2005-2024
(optional --train-start=YYYY shortens the training period, e.g. 1979 to drop the pre-satellite era).

Models
  clim      (a) per-point drought frequency in the training years (reference forecast)
  clim_nat  (b) national drought frequency in the training years (same value for every point)
  LOG_pers  (c) per-point logistic regression, persistence only (pers6, pers3)
  LOG_lead  (d) per-point logistic, persistence + ONI before the season (operational)
  LOG_enso  (e) per-point logistic, persistence + ONI before + ONI of the season
                (oni_conc is the observed ONI of the season: perfect ENSO outlook, upper bound)
  LOG_trend (f) model (e) + linear year term (non-stationarity test)
  REG_lead / REG_enso : partial pooling by macro-region: one logistic per region with slopes
                shared by the region and one L2-penalised intercept per point (the penalty shrinks
                the point intercepts towards the regional one); predictors as in (d) / (e)
  XGB_enso  (g) global XGBoost (all points) with lat, lon + predictors of (e)

Regularisation: predictors standardised with the training mean/sd of each point; the L2 strength C
of every logistic model is chosen from C_GRID by 6-fold cross-validation with blocks of years inside
the training period (all points of a year in the same fold; criterion: pooled Brier score).
XGBoost: depth 2-5 and number of trees by the same year-blocked CV (log-loss).

Outputs: <pref>_prob_pred.csv (test predictions), <pref>_prob_cv.csv (chosen C, CV Brier),
<pref>_prob_events.csv (training drought events per point), <pref>_prob_skill.csv (BSS),
<pref>_prob_shap_log.csv (exact linear SHAP of LOG_lead/LOG_enso: phi_j = beta_j * z_j),
<pref>_prob_shap.csv (TreeSHAP of XGB_enso).
Usage: python drought_prob.py <pref>_dataset.csv <out prefix> [--train-start=YYYY]"""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
import xgboost as xgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import year_folds, TRAIN, TEST

THR = -1.0
FEATS = {'LOG_pers': ['pers6', 'pers3'],
         'LOG_lead': ['pers6', 'pers3', 'oni_lead'],
         'LOG_enso': ['pers6', 'pers3', 'oni_lead', 'oni_conc'],
         'LOG_trend': ['pers6', 'pers3', 'oni_lead', 'oni_conc', 'year']}
REGF = {'REG_lead': FEATS['LOG_lead'], 'REG_enso': FEATS['LOG_enso']}
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
XF = ['lat', 'lon'] + FEATS['LOG_enso']
XP = dict(learning_rate=0.05, subsample=0.8, colsample_bytree=0.9, min_child_weight=20,
          objective='binary:logistic', eval_metric='logloss', seed=0, nthread=2)
MODELS = ['clim', 'clim_nat', 'LOG_pers', 'LOG_lead', 'LOG_enso', 'LOG_trend', 'REG_lead', 'REG_enso', 'XGB_enso']


def region(lon, lat):
    if lon <= -109: return 'Noroeste'
    if lat >= 23.5: return 'Norte-Noreste'
    if lon <= -94 and lat >= 17.5: return 'Centro-Occidente'
    return 'Sur-Sureste'


def fit_l2_logistic(X, y, C, iters=50, tol=1e-8):
    """L2-penalised logistic regression by Newton-Raphson, same objective as scikit-learn:
    min_w  C * sum log-loss + 0.5 * ||w||^2, intercept (last column) not penalised."""
    X1 = np.column_stack([X, np.ones(len(X))])
    w = np.zeros(X1.shape[1]); pen = np.ones(X1.shape[1]); pen[-1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X1 @ w, -35, 35)))
        g = C * X1.T @ (p - y) + pen * w
        H = C * (X1.T * (p * (1 - p))) @ X1 + np.diag(pen + 1e-10)
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < tol:
            break
    return w


def predict_l2(w, X):
    return 1 / (1 + np.exp(-np.clip(np.column_stack([X, np.ones(len(X))]) @ w, -35, 35)))


def std(a, b, f):
    mu, sd = a[f].mean(), a[f].std(ddof=0).replace(0, 1)
    return ((a[f] - mu) / sd).values, ((b[f] - mu) / sd).values


def logit(a, b, f, C, coef=False):
    """fit on a, predict b; constant forecast if the training years have a single class."""
    if a.y.nunique() < 2:
        p = np.full(len(b), a.y.mean())
        return (p, np.zeros((len(b), len(f)))) if coef else p
    za, zb = std(a, b, f)
    w = fit_l2_logistic(za, a.y.values.astype(float), C)
    p = predict_l2(w, zb)
    return (p, zb * w[:-1]) if coef else p


def point_cv(args):
    """pooled-CV contributions of one point: squared errors per (model, C)."""
    gp, folds_years = args
    out = {}
    for name, f in FEATS.items():
        for C in C_GRID:
            se = 0.0
            for vy in folds_years:
                a, b = gp[~gp.year.isin(vy)], gp[gp.year.isin(vy)]
                se += np.sum((logit(a, b, f, C) - b.y.values) ** 2)
            out[(name, C)] = se
    return out


def point_fit(args):
    """test predictions (and linear SHAP) of one point with the chosen C."""
    gp, Cs = args
    a, b = gp[gp.year.between(*TRAIN)], gp[gp.year.between(*TEST)]
    res = {'index': b.index, 'clim': np.full(len(b), a.y.mean())}
    shap = []
    for name, f in FEATS.items():
        p, ph = logit(a, b, f, Cs[name], coef=True)
        res[name] = p
        if name in ('LOG_lead', 'LOG_enso'):
            s = pd.DataFrame(ph, columns=['shap_' + c for c in f], index=b.index)
            shap.append(pd.concat([b[['lon', 'lat', 'year', 'season']].assign(model=name), s], axis=1))
    return res, shap


def fit_regional(k, Z, y, C, iters=50, tol=1e-8):
    """Partial pooling: logit = c + a_k + Z b, objective C*sum log-loss + 0.5(|a|^2 + |b|^2).
    Newton steps solved through the Schur complement of the diagonal intercept block."""
    n, q = k.max() + 1, Z.shape[1]
    a, b, c = np.zeros(n), np.zeros(q), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(c + a[k] + Z @ b, -35, 35))); r = p - y; w = p * (1 - p)
        ga = C * np.bincount(k, r, n) + a
        g2 = np.append(C * Z.T @ r + b, C * r.sum())
        D = C * np.bincount(k, w, n) + 1
        E = np.column_stack([C * np.column_stack([np.bincount(k, w * Z[:, j], n) for j in range(q)]),
                             C * np.bincount(k, w, n)])
        Z1 = np.column_stack([Z, np.ones(len(Z))])
        F = C * (Z1.T * w) @ Z1 + np.diag(np.append(np.ones(q), 1e-10))
        S = F - E.T @ (E / D[:, None])
        d2 = np.linalg.solve(S, g2 - E.T @ (ga / D))
        da = (ga - E @ d2) / D
        a -= da; b -= d2[:q]; c -= d2[q]
        if max(np.abs(da).max(), np.abs(d2).max()) < tol:
            break
    return a, b, c


def regional(tr, te, f, C):
    """partial pooling within a region: shared slopes, penalised point intercepts."""
    tr = tr.reset_index(drop=True); te = te.reset_index(drop=True)
    pts = sorted(set(zip(tr.lon, tr.lat))); kmap = {k: i for i, k in enumerate(pts)}
    ka = np.array([kmap[k] for k in zip(tr.lon, tr.lat)]); kb = np.array([kmap[k] for k in zip(te.lon, te.lat)])
    za = np.zeros((len(tr), len(f))); zb = np.zeros((len(te), len(f)))
    for i, k in enumerate(pts):
        ia, ib = np.where(ka == i)[0], np.where(kb == i)[0]
        x, y = std(tr.iloc[ia], te.iloc[ib], f)
        za[ia] = x; zb[ib] = y
    a, b, c = fit_regional(ka, za, tr.y.values.astype(float), C)
    return 1 / (1 + np.exp(-np.clip(c + a[kb] + zb @ b, -35, 35)))


def regional_cv(tr, f):
    folds = [np.array(b) for b in np.array_split(np.sort(tr.year.unique()), 6)]
    best = None
    for C in C_GRID:
        se = 0.0
        for vy in folds:
            a, b = tr[~tr.year.isin(vy)], tr[tr.year.isin(vy)]
            for r in tr.reg.unique():
                bb = b[b.reg == r]
                se += np.sum((regional(a[a.reg == r], bb, f, C) - bb.y.values) ** 2)
        if best is None or se < best[0]:
            best = (se, C)
    return best[1], best[0] / len(tr)


def fit_xgb(tr):
    d = xgb.DMatrix(tr[XF], tr.y)
    best = None
    for depth in (2, 3, 4, 5):
        cv = xgb.cv(dict(XP, max_depth=depth), d, 2000, folds=year_folds(tr.year.values),
                    early_stopping_rounds=50, seed=0, verbose_eval=False)
        s, n = cv['test-logloss-mean'].min(), int(cv['test-logloss-mean'].idxmin()) + 1
        if best is None or s < best[0]:
            best = (s, depth, n)
    return xgb.train(dict(XP, max_depth=best[1]), d, best[2]), best[1:]


def brier(y, p):
    return np.mean((p - y) ** 2)


if __name__ == '__main__':
    src, pref = sys.argv[1:3]
    for a in sys.argv[3:]:
        if a.startswith('--train-start='):
            TRAIN = (int(a.split('=')[1]), TRAIN[1])
    print('training period', TRAIN)
    D = pd.read_csv(src)
    D = D[(D.year >= TRAIN[0]) | (D.year >= TEST[0])]
    D['y'] = (D.target <= THR).astype(int)
    D['reg'] = [region(a, b) for a, b in zip(D.lon, D.lat)]
    preds, shap_log, shap_x, cvrows, ev = [], [], [], [], []
    for season, g in D.groupby('season'):
        tr, te = g[g.year.between(*TRAIN)], g[g.year.between(*TEST)].copy()
        ev.append(tr.groupby(['lon', 'lat']).y.sum().rename('events').reset_index().assign(season=season))
        folds_years = [np.array(b) for b in np.array_split(np.sort(tr.year.unique()), 6)]
        groups = [gp for _, gp in tr.groupby(['lon', 'lat'])]
        with Pool(2) as pool:
            parts = pool.map(point_cv, [(gp, folds_years) for gp in groups], chunksize=20)
        tot = {k: sum(p[k] for p in parts) / len(tr) for k in parts[0]}
        Cs = {}
        for name in FEATS:
            sc = {C: tot[(name, C)] for C in C_GRID}
            Cs[name] = min(sc, key=sc.get)
            cvrows += [dict(season=season, model=name, C=C, cv_brier=v, chosen=C == Cs[name]) for C, v in sc.items()]
        with Pool(2) as pool:
            fits = pool.map(point_fit, [(gp, Cs) for _, gp in g.groupby(['lon', 'lat'])], chunksize=20)
        for res, sh in fits:
            for k, v in res.items():
                if k != 'index':
                    te.loc[res['index'], k] = v
            shap_log += sh
        te['clim_nat'] = tr.y.mean()
        for name, f in REGF.items():
            C, cvb = regional_cv(tr, f)
            cvrows.append(dict(season=season, model=name, C=C, cv_brier=cvb, chosen=True))
            te[name] = np.nan
            for r in tr.reg.unique():
                te.loc[te.reg == r, name] = regional(tr[tr.reg == r], te[te.reg == r], f, C)
        bst, hp = fit_xgb(tr)
        cvrows.append(dict(season=season, model='XGB_enso', C=np.nan, cv_brier=np.nan, chosen=True,
                           depth=hp[0], trees=hp[1]))
        te['XGB_enso'] = bst.predict(xgb.DMatrix(te[XF]))
        c = bst.predict(xgb.DMatrix(te[XF]), pred_contribs=True)
        sh = pd.DataFrame(c[:, :-1], columns=['shap_' + f for f in XF], index=te.index); sh['shap_bias'] = c[:, -1]
        shap_x.append(pd.concat([te[['lon', 'lat', 'year', 'season']], sh], axis=1))
        preds.append(te)
    P = pd.concat(preds); P.to_csv(f'{pref}_prob_pred.csv', index=False)
    pd.concat(shap_log).to_csv(f'{pref}_prob_shap_log.csv', index=False)
    pd.concat(shap_x).to_csv(f'{pref}_prob_shap.csv', index=False)
    pd.DataFrame(cvrows).to_csv(f'{pref}_prob_cv.csv', index=False)
    pd.concat(ev).to_csv(f'{pref}_prob_events.csv', index=False)
    rows = []
    for season, g in P.groupby('season'):
        bc = brier(g.y, g.clim)
        for m in MODELS:
            pp = g.groupby(['lon', 'lat']).apply(
                lambda q: 1 - brier(q.y, q[m]) / brier(q.y, q.clim) if brier(q.y, q.clim) > 0 else np.nan,
                include_groups=False)
            rows.append(dict(season=season, model=m, BS=brier(g.y, g[m]), BSS=1 - brier(g.y, g[m]) / bc,
                             BSS_point_median=pp.median(), mean_prob=g[m].mean(), obs_freq=g.y.mean()))
    out = pd.DataFrame(rows); out.to_csv(f'{pref}_prob_skill.csv', index=False)
    print(pd.DataFrame(cvrows).query('chosen').to_string(index=False))
    print(out.round(3).to_string(index=False))
