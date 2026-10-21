"""Extra diagnostics requested in the fourth review (no model is refitted; everything is read from the
outputs already produced by drought_prob.py and from indices_national.pkl).
  1. Share of the annual precipitation that falls in November-April, by region.
  2. The 2023-24 dry season (index ending April 2024): forecast of (d) and observation.
  3. BSS of (d) against a more recent climatology (1975-2004 frequency of each point).
  4. (b') regional frequency of the training period, and (d') - (b').
  5. Reliability diagram of (d) and per-point BSS map of (d) (dry season)   -> fig_calibracion.png
  6. Correlation of monthly anomalies of (Tmax+Tmin)/2 and the monthly mean temperature.
  7. Fisher z test for the change of the Noroeste correlation (median values, n = 54 and 20).
Intervals: percentile bootstrap of whole test years (5000 replicates, seed 0), as in skill_bootstrap.py.
Usage: python paper_extra.py <work dir> <era5 csv dir> <fig dir>   ->  <work dir>/extra_numbers.json"""
import glob, json, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

W, E5, OUT = sys.argv[1:4]
NB = 5000
X = {}


def region(lon, lat):
    if lon <= -109: return 'Noroeste'
    if lat >= 23.5: return 'Norte-Noreste'
    if lon <= -94 and lat >= 17.5: return 'Centro-Occidente'
    return 'Sur-Sureste'


REGS = ['Noroeste', 'Norte-Noreste', 'Centro-Occidente', 'Sur-Sureste']

# ------------------------------------------------------------------ 1. Nov-Apr share of annual precipitation
r = pd.read_pickle(f'{W}/indices_national.pkl')
rows = []
for (lon, lat), v in r.items():
    s = pd.Series(np.asarray(v['P'], float), index=v['date'])
    s = s[(s.index.year >= 1951) & (s.index.year <= 2024)]
    dry = s[s.index.month.isin([11, 12, 1, 2, 3, 4])].sum()
    rows.append(dict(reg=region(lon, lat), share=dry / s.sum(), dry_mm=dry / 74, annual_mm=s.sum() / 74))
q = pd.DataFrame(rows)
X['rain_share'] = {rg: dict(share_med=float(g.share.median()), dry_mm_med=float(g.dry_mm.median()),
                            annual_mm_med=float(g.annual_mm.median())) for rg, g in q.groupby('reg')}
X['rain_share']['Nacional'] = dict(share_med=float(q.share.median()), dry_mm_med=float(q.dry_mm.median()),
                                   annual_mm_med=float(q.annual_mm.median()))

# ------------------------------------------------------------------ test predictions
P = pd.read_csv(f'{W}/model_spei_prob_pred.csv')
P['reg'] = [region(a, b) for a, b in zip(P.lon, P.lat)]
D = pd.read_csv(f'{W}/model_spei_dataset.csv'); D['y'] = (D.target <= -1).astype(float)
dry = P[P.season == 'dry'].copy()

# ------------------------------------------------------------------ 2. season ending April 2024
c = dry[dry.year == 2024]
X['case2024'] = {'Nacional': dict(p_d=float(c.LOG_lead.mean()), p_clim=float(c.clim.mean()), obs=float(c.y.mean()),
                                  spei_med=float(c.target.median()), oni_lead=float(c.oni_lead.iloc[0]),
                                  oni_conc=float(c.oni_conc.iloc[0]), n=int(len(c)))}
for rg, g in c.groupby('reg'):
    X['case2024'][rg] = dict(p_d=float(g.LOG_lead.mean()), p_clim=float(g.clim.mean()), obs=float(g.y.mean()),
                             spei_med=float(g.target.median()))
# Brier skill of (d) in that single season, against (a)
X['case2024']['Nacional']['bss_d'] = float(1 - ((c.LOG_lead - c.y) ** 2).sum() / ((c.clim - c.y) ** 2).sum())
# the preceding wet season (index ending October 2023)
cw = P[(P.season == 'wet') & (P.year == 2023)]
X['case2023wet'] = dict(obs=float(cw.y.mean()), p_d=float(cw.LOG_lead.mean()), p_clim=float(cw.clim.mean()),
                        spei_med=float(cw.target.median()))
# other dry seasons with El Nino by the prior ONI in the test period, for context
yrs = dry.drop_duplicates('year').set_index('year')
X['test_lead_nino_years'] = [int(y) for y in yrs.index[yrs.oni_lead >= 0.5]]

# ------------------------------------------------------------------ 3-4. alternative references
tr = D[(D.season == 'dry') & D.year.between(1951, 2004)].copy()
tr['reg'] = [region(a, b) for a, b in zip(tr.lon, tr.lat)]
c75 = tr[tr.year.between(1975, 2004)].groupby(['lon', 'lat']).y.mean().rename('clim75')
creg = tr.groupby('reg').y.mean().rename('clim_reg')
dry = dry.merge(c75, on=['lon', 'lat']).merge(creg, left_on='reg', right_index=True)
X['clim75_mean'] = float(dry.clim75.mean()); X['clim_mean'] = float(dry.clim.mean())
X['clim_reg'] = {k: float(v) for k, v in creg.items()}


def boot_bss(g, model, ref, n=NB):
    se = pd.DataFrame({'m': (g[model] - g.y) ** 2, 'r': (g[ref] - g.y) ** 2, 'year': g.year}).groupby('year').sum()
    yrs = se.index.values
    idx = np.random.default_rng(0).integers(0, len(yrs), size=(n, len(yrs)))
    S = se.values[idx].sum(axis=1)
    b = 1 - S[:, 0] / S[:, 1]
    return [float(1 - se.m.sum() / se.r.sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def boot_diff(g, m1, m2, ref='clim', n=NB):
    """(BS_m2 - BS_m1) / BS_ref on the same resampled years = BSS(m1) - BSS(m2)."""
    se = pd.DataFrame({'a': (g[m1] - g.y) ** 2, 'b': (g[m2] - g.y) ** 2, 'r': (g[ref] - g.y) ** 2,
                       'year': g.year}).groupby('year').sum()
    idx = np.random.default_rng(0).integers(0, len(se), size=(n, len(se)))
    S = se.values[idx].sum(axis=1)
    b = (S[:, 1] - S[:, 0]) / S[:, 2]
    return [float((se.b.sum() - se.a.sum()) / se.r.sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975))]


X['bss_ref75'] = {'clim75 vs clim': boot_bss(dry, 'clim75', 'clim'),
                  'LOG_lead vs clim75': boot_bss(dry, 'LOG_lead', 'clim75'),
                  'REG_lead vs clim75': boot_bss(dry, 'REG_lead', 'clim75'),
                  'LOG_enso vs clim75': boot_bss(dry, 'LOG_enso', 'clim75')}
X['bss_clim_reg'] = boot_bss(dry, 'clim_reg', 'clim')
X['diff_REG_lead_clim_reg'] = boot_diff(dry, 'REG_lead', 'clim_reg')
X['diff_REG_enso_clim_reg'] = boot_diff(dry, 'REG_enso', 'clim_reg')
wet = P[P.season == 'wet'].copy(); wet['reg'] = [region(a, b) for a, b in zip(wet.lon, wet.lat)]
trw = D[(D.season == 'wet') & D.year.between(1951, 2004)].copy()
trw['reg'] = [region(a, b) for a, b in zip(trw.lon, trw.lat)]
wet = wet.merge(trw.groupby('reg').y.mean().rename('clim_reg'), left_on='reg', right_index=True)
X['bss_clim_reg_wet'] = boot_bss(wet, 'clim_reg', 'clim')
# AUC of (b') with the same year-block bootstrap as skill_extra.py (1000 replicates, seed 0)
from sklearn.metrics import roc_auc_score
for nm, g in (('dry', dry), ('wet', wet)):
    g = g.reset_index(drop=True); yrs = np.sort(g.year.unique()); ix = {v: np.where(g.year.values == v)[0] for v in yrs}
    rng = np.random.default_rng(0)
    bt = [roc_auc_score(g.y.values[i], g.clim_reg.values[i]) for i in
          (np.concatenate([ix[v] for v in rng.choice(yrs, len(yrs))]) for _ in range(1000))]
    X[f'auc_clim_reg_{nm}'] = [float(roc_auc_score(g.y, g.clim_reg)), float(np.quantile(bt, .025)), float(np.quantile(bt, .975))]

# ------------------------------------------------------------------ 5. reliability and BSS map of (d), dry season
bins = np.linspace(0, 1, 11)
rel = []
for m in ('LOG_lead', 'clim'):
    k = np.minimum((dry[m].values * 10).astype(int), 9)
    for b in range(10):
        i = k == b
        if i.sum() >= 1:
            rel.append(dict(model=m, bin=b, n=int(i.sum()), p=float(dry[m].values[i].mean()), o=float(dry.y.values[i].mean())))
rel = pd.DataFrame(rel); rel.to_csv(f'{W}/reliability_dry.csv', index=False)
X['reliability_LOG_lead'] = rel[rel.model == 'LOG_lead'].to_dict('records')
pts = dry.groupby(['lon', 'lat']).apply(
    lambda z: 1 - ((z.LOG_lead - z.y) ** 2).sum() / ((z.clim - z.y) ** 2).sum() if ((z.clim - z.y) ** 2).sum() > 0 else np.nan,
    include_groups=False).rename('bss').reset_index()
pts.to_csv(f'{W}/bss_points_dry_LOG_lead.csv', index=False)
X['bss_points'] = dict(median=float(pts.bss.median()), pos=float((pts.bss > 0).mean()), n=int(pts.bss.notna().sum()))

plt.rcParams.update({'font.family': 'serif', 'font.size': 7, 'axes.titlesize': 7, 'savefig.dpi': 400,
                     'axes.edgecolor': '#c3c2b7', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'xtick.labelsize': 6, 'ytick.labelsize': 6})
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import shapefile
SHP = shapefile.Reader(f'{W}/basemap/naturalearth_lowres.shp')
FIELDS = [f[0] for f in SHP.fields[1:]]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.3, 3.75), gridspec_kw=dict(height_ratios=[0.55, 1]))
for m, col, lab in (('LOG_lead', '#2a78d6', '(d) ONI previo'), ('clim', '#eb6834', '(a) Climatología')):
    g = rel[rel.model == m]
    a1.plot(g.p, g.o, marker='o', ms=2.5, lw=1.2, color=col, label=lab, zorder=3)
a1.plot([0, 1], [0, 1], color='#b8b7b0', lw=0.6, ls='--', zorder=1)
a1.set_xlim(0, 0.8); a1.set_ylim(0, 0.8)
a1.set_xlabel('Probabilidad pronosticada'); a1.set_ylabel('Frecuencia observada')
a1.grid(color='#e1e0d9', lw=0.5, zorder=0); a1.spines[['top', 'right']].set_visible(False)
a1.legend(frameon=False, fontsize=6, loc='upper left')
ax2 = a1.twinx(); g = rel[rel.model == 'LOG_lead']
ax2.bar(g.p, g.n, width=0.05, color='#2a78d6', alpha=0.18, zorder=0); ax2.set_ylim(0, g.n.max() * 3)
ax2.set_ylabel('Casos de (d)', fontsize=6); ax2.tick_params(labelsize=5.5, length=2)
ax2.spines[['top']].set_visible(False)
a1.set_title('a) Confiabilidad, secas 2005–2024', pad=2)
for s, rec in zip(SHP.shapes(), SHP.records()):
    if dict(zip(FIELDS, rec))['iso_a3'] in ('MEX', 'USA', 'GTM', 'BLZ', 'HND', 'SLV', 'CUB'):
        parts = list(s.parts) + [len(s.points)]
        mex = dict(zip(FIELDS, rec))['iso_a3'] == 'MEX'
        for a, b in zip(parts[:-1], parts[1:]):
            xy = np.array(s.points[a:b])
            a2.plot(xy[:, 0], xy[:, 1], color='#3d3d3a' if mex else '#b8b7b0', lw=0.5 if mex else 0.35, zorder=4)
div = LinearSegmentedColormap.from_list('div', ['#b35806', '#f0efec', '#2a78d6'])
sc = a2.scatter(pts.lon, pts.lat, c=pts.bss * 100, s=7, marker='s', cmap=div, norm=TwoSlopeNorm(0, -40, 40), lw=0, zorder=2)
for xs, ys in [([-108.75, -108.75], [14.5, 33.0]), ([-108.75, -86.5], [23.25, 23.25]),
               ([-93.75, -93.75], [17.25, 23.25]), ([-108.75, -93.75], [17.25, 17.25])]:
    a2.plot(xs, ys, color='#3d3d3a', lw=0.5, ls=(0, (3, 2)), zorder=4)
a2.set_xlim(-118, -86); a2.set_ylim(14, 33); a2.set_aspect(1 / np.cos(np.radians(24)))
a2.set_xticks([-115, -105, -95]); a2.set_xticklabels(['115° O', '105° O', '95° O'])
a2.set_yticks([15, 20, 25, 30]); a2.set_yticklabels(['15° N', '20° N', '25° N', '30° N']); a2.tick_params(length=2, pad=1)
a2.set_title('b) BSS de (d) por punto, secas 2005–2024', pad=2)
cb = fig.colorbar(sc, ax=a2, fraction=0.05, pad=0.12, extend='both', orientation='horizontal', aspect=35); cb.set_label('BSS (%)'); cb.outline.set_visible(False)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_calibracion.png', bbox_inches='tight'); plt.close(fig)

# ------------------------------------------------------------------ 6. temperature: anomaly correlation and sign
tx = pd.read_csv(f'{W}/tmaxmin_all.csv')
e5 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{E5}/ERA5_Mexico_*.csv'))])
m = e5.merge(tx, on=['lon', 'lat', 'year', 'month'])
m['tmm'] = (m.tmax_c + m.tmin_c) / 2
for c_ in ('tmm', 'temp_c'):
    m[c_ + '_a'] = m[c_] - m.groupby(['lon', 'lat', 'month'])[c_].transform('mean')
X['tmed_anom_r'] = float(np.corrcoef(m.tmm_a, m.temp_c_a)[0, 1])
X['tmed_anom_r_point_med'] = float(m.groupby(['lon', 'lat']).apply(
    lambda z: np.corrcoef(z.tmm_a, z.temp_c_a)[0, 1], include_groups=False).median())
X['tmed_diff'] = float((m.tmm - m.temp_c).mean())          # > 0: (Tmax+Tmin)/2 warmer than the monthly mean
X['tmed_diff_frac_pos'] = float((m.tmm > m.temp_c).mean())

# ------------------------------------------------------------------ 7. Noroeste: Fisher z for the median correlations
q_ = D[(D.season == 'dry') & (D.lon <= -109)]
rr = {k: float(q_[q_.year.between(a, b)].groupby(['lon', 'lat'])
               .apply(lambda w: np.corrcoef(w.oni_conc, w.target)[0, 1], include_groups=False).median())
      for k, (a, b) in {'train': (1951, 2004), 'test': (2005, 2024)}.items()}
X['no_fisher_z'] = float((np.arctanh(rr['train']) - np.arctanh(rr['test'])) / np.sqrt(1 / (54 - 3) + 1 / (20 - 3)))

json.dump(X, open(f'{W}/extra_numbers.json', 'w'), indent=1, ensure_ascii=False, default=float)
print(json.dumps({k: v for k, v in X.items() if k not in ('reliability_LOG_lead',)}, indent=1, ensure_ascii=False))
