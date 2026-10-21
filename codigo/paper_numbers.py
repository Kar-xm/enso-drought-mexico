"""Computes every number quoted in the CISIMA 2026 manuscript and writes them to a JSON file, so the text
is filled from the results instead of being typed by hand.
Usage: python paper_numbers.py <work dir> <out.json>"""
import glob, json, os, sys
import numpy as np, pandas as pd

W, OUT = sys.argv[1:3]
N = {}


def reg(lon, lat):
    if lon <= -109: return 'Noroeste'
    if lat >= 23.5: return 'Norte-Noreste'
    if lon <= -94 and lat >= 17.5: return 'Centro-Occidente'
    return 'Sur-Sureste'


REGS = ['Noroeste', 'Norte-Noreste', 'Centro-Occidente', 'Sur-Sureste']
r = pd.read_pickle(f'{W}/indices_national.pkl')
k0 = next(iter(r))
N['n_points'] = len(r); N['n_months'] = len(r[k0]['date'])
N['first_month'] = str(r[k0]['date'][0].date())
lons = [k[0] for k in r]; lats = [k[1] for k in r]
N['extent'] = [min(lats), max(lats), min(lons), max(lons)]
N['n_region'] = pd.Series([reg(*k) for k in r]).value_counts().to_dict()

tx = pd.read_csv(f'{W}/tmaxmin_all.csv')
e5 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{W}/ERA5/ERA5_Mexico_*.csv'))])
m = e5.merge(tx, on=['lon', 'lat', 'year', 'month'])
N['tmed_r'] = float(np.corrcoef((m.tmax_c + m.tmin_c) / 2, m.temp_c)[0, 1])
N['tmed_diff'] = float(((m.tmax_c + m.tmin_c) / 2 - m.temp_c).mean())
N['tmax_gt_tmin'] = bool((tx.tmax_c > tx.tmin_c).all())

yrs = N['n_months'] / 12
ph = pd.Series({k: v['PET_H'].sum() / yrs for k, v in r.items()}); pt = pd.Series({k: v['PET_T'].sum() / yrs for k, v in r.items()})
P = pd.Series({k: v['P'].sum() / yrs for k, v in r.items()})
rg = pd.Series([reg(*k) for k in ph.index], index=ph.index)
N['etoH'] = [float(ph.quantile(q)) for q in (.1, .5, .9)]
N['etoT'] = [float(pt.quantile(q)) for q in (.1, .5, .9)]
N['ratio_HT'] = float((ph / pt).median())
N['ratio_HT_region'] = (ph / pt).groupby(rg).median().to_dict()
c = pd.Series({k: np.corrcoef(v['SPEI6'][12:], v['SPEIT6'][12:])[0, 1] for k, v in r.items()})
low = c[c < 0.9]
N['r_HT_median'] = float(c.median()); N['n_r_HT_low'] = int(len(low)); N['P_max_low'] = float(P[low.index].max())
N['lon_max_low'] = float(max(k[0] for k in low.index))
sd_cal = [np.nanstd(v['SPEI6'][(v['date'].year >= 1951) & (v['date'].year <= 2004)]) for v in r.values()]
N['spei_sd_cal'] = [float(min(sd_cal)), float(max(sd_cal))]

t = pd.read_csv(f'{W}/explore_corr.csv')
for s in ('dry', 'wet'):
    for i in ('SPEI6', 'SPI6', 'SPEIT6'):
        g = t[(t.season == s) & (t['index'] == i)]
        key = f'{s}_{i}'
        N[key] = dict(r_med=float(g.r_conc.median()), r_pos=float((g.r_conc > 0).mean()),
                      sig=float(g.sig_conc.mean()), sig_neg=float((g.sig_conc.astype(bool) & (g.r_conc < 0)).mean()),
                      n_sig_neg=int((g.sig_conc.astype(bool) & (g.r_conc < 0)).sum()), n_sig=int(g.sig_conc.sum()),
                      lead_sig=float(g.sig_lead.mean()), lead_r_med=float(g.r_lead.median()),
                      welch_sig=float(g.sig_comp.mean()), nino_med=float(g.mean_nino.median()),
                      nina_med=float(g.mean_nina.median()))
        gn = g[g.sig_conc.astype(bool) & (g.r_conc < 0)]
        N[key]['n_sig_neg_SC'] = int(sum(reg(a, b) in ('Sur-Sureste', 'Centro-Occidente') for a, b in zip(gn.lon, gn.lat)))

D = pd.read_csv(f'{W}/model_spei_dataset.csv'); D['y'] = D.target <= -1
D['fase'] = np.where(D.oni_conc >= 0.5, 'nino', np.where(D.oni_conc <= -0.5, 'nina', 'neutro'))
pp = D.groupby(['season', 'fase', 'lon', 'lat']).y.mean().groupby(['season', 'fase']).median()
N['prob_phase'] = {f'{s}_{f}': float(pp[(s, f)]) for s in ('dry', 'wet') for f in ('nino', 'nina')}
ny = D.groupby(['season', 'fase']).year.nunique()
N['n_phase'] = {f'{s}_{f}': int(ny[(s, f)]) for s in ('dry', 'wet') for f in ('nino', 'nina', 'neutro')}
for s in ('dry', 'wet'):
    f = pd.read_csv(f'{W}/fisher_{s}.csv'); N[f'fisher_sig_{s}'] = float(f.sig.mean())
N['n_train'] = int((D[D.season == 'dry'].year <= 2004).sum()); N['n_test'] = int((D[D.season == 'dry'].year >= 2005).sum())

# ENSO in train vs test, and expected frequency from the ENSO mix
for s, g in D.groupby('season'):
    y1 = g.drop_duplicates('year').set_index('year')
    tr, te = g[g.year <= 2004], g[g.year >= 2005]
    fph = tr.groupby('fase').y.mean(); mix = te.drop_duplicates('year').fase.value_counts(normalize=True)
    N[f'enso_{s}'] = dict(nino_tr=int((y1.loc[1951:2004].oni_conc >= .5).sum()), nina_tr=int((y1.loc[1951:2004].oni_conc <= -.5).sum()),
                          nino_te=int((y1.loc[2005:2024].oni_conc >= .5).sum()), nina_te=int((y1.loc[2005:2024].oni_conc <= -.5).sum()),
                          freq_tr=float(tr.y.mean()), freq_te=float(te.y.mean()), freq_expected=float((fph * mix).sum()),
                          nina_freq_tr=float(fph['nina']), nina_freq_te=float(te[te.fase == 'nina'].y.mean()),
                          mean_target_tr=float(tr.target.mean()), mean_target_te=float(te.target.mean()))

N['r_lead_conc'] = {s: float(g.drop_duplicates('year').oni_lead.corr(g.drop_duplicates('year').oni_conc)) for s, g in D.groupby('season')}
ev = pd.read_csv(f'{W}/model_spei_prob_events.csv')
N['events'] = {s: dict(median=float(g.events.median()), min=int(g.events.min()), max=int(g.events.max()))
               for s, g in ev.groupby('season')}
cv = pd.read_csv(f'{W}/model_spei_prob_cv.csv'); cv = cv[cv.chosen]
N['C'] = {f'{a}_{b}': (None if pd.isna(c_) else float(c_)) for a, b, c_ in zip(cv.season, cv.model, cv.C)}
N['xgb'] = {s: dict(depth=int(q.depth.iloc[0]), trees=int(q.trees.iloc[0])) for s, q in cv[cv.model == 'XGB_enso'].groupby('season')}

for ix in ('spei', 'spi', 'speit', 'speipm', 'speironi', 'spei79'):
    f = f'{W}/model_{ix}_boot.csv'
    if os.path.exists(f):
        b = pd.read_csv(f)
        N[f'boot_{ix}'] = {f'{s}|{sc}|{mo}': [float(x) for x in (bss, lo, hi)]
                           for s, sc, mo, bss, lo, hi in zip(b.season, b.scope, b.model, b.BSS, b.lo, b.hi)}
    f = f'{W}/model_{ix}_prob_skill.csv'
    if os.path.exists(f):
        k = pd.read_csv(f)
        N[f'skill_{ix}'] = {f'{s}|{mo}': dict(BSS=float(a), med=float(m_), mean_prob=float(mp), obs=float(o))
                            for s, mo, a, m_, mp, o in zip(k.season, k.model, k.BSS, k.BSS_point_median, k.mean_prob, k.obs_freq)}

Pr = pd.read_csv(f'{W}/model_spei_prob_pred.csv')
bss = lambda q, mo: 1 - ((q.y - q[mo]) ** 2).mean() / ((q.y - q.clim) ** 2).mean()
for s in ('dry', 'wet'):
    q = Pr[Pr.season == s]
    for mo in ('LOG_lead', 'LOG_enso', 'REG_lead', 'REG_enso'):
        pts = q.groupby(['lon', 'lat']).apply(lambda z: bss(z, mo) if ((z.y - z.clim) ** 2).mean() > 0 else np.nan, include_groups=False)
        N[f'frac_pos_{s}_{mo}'] = float((pts > 0).mean())
xw = Pr[Pr.season == 'wet'].XGB_enso
N['xgb_wet_range'] = [float(xw.min()), float(xw.max())]

# serial correlation (Bretherton et al., 1999) from the exploration step
for s in ('dry', 'wet'):
    g = t[(t.season == s) & (t['index'] == 'SPEI6')]
    N[f'serial_{s}'] = dict(r1_index_med=float(g.r1_index.median()), r1_oni=float(g.r1_oni.iloc[0]),
                            n_eff_med=float(g.n_eff.median()), sig_eff=float(g.sig_conc_eff.mean()))
# RONI sensitivity
if os.path.exists(f'{W}/explore_roni_corr.csv'):
    tr_ = pd.read_csv(f'{W}/explore_roni_corr.csv')
    for s in ('dry', 'wet'):
        g = tr_[(tr_.season == s) & (tr_['index'] == 'SPEI6')]
        N[f'roni_{s}'] = dict(sig=float(g.sig_conc.mean()), r_med=float(g.r_conc.median()), lead_sig=float(g.sig_lead.mean()))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from enso_explore import load_oni
    o, ro = load_oni(f'{W}/oni.ascii.txt'), load_oni(f'{W}/RONI.ascii.txt')
    j = pd.concat([o, ro], axis=1, keys=['oni', 'roni']).dropna()
    N['roni_vs_oni'] = dict(r=float(j.corr().iloc[0, 1]),
                            diff_test=float((j.roni - j.oni)[(j.index.year >= 2005) & (j.index.year <= 2024)].mean()),
                            diff_train=float((j.roni - j.oni)[(j.index.year >= 1951) & (j.index.year <= 2004)].mean()))
    Dr = pd.read_csv(f'{W}/model_speironi_dataset.csv')
    Dr['fase'] = np.where(Dr.oni_conc >= 0.5, 'nino', np.where(Dr.oni_conc <= -0.5, 'nina', 'neutro'))
    N['roni_phases_test'] = {s: dict(nino=int((g.drop_duplicates('year').query('year >= 2005').fase == 'nino').sum()),
                                     nina=int((g.drop_duplicates('year').query('year >= 2005').fase == 'nina').sum()))
                             for s, g in Dr.groupby('season')}
# calibration / discrimination (Murphy decomposition and AUC)
for ix in ('spei', 'speironi', 'speipm', 'spei79', 'spi', 'speit'):
    f = f'{W}/model_{ix}_extra.csv'
    if os.path.exists(f):
        e = pd.read_csv(f)
        N[f'extra_{ix}'] = {f'{s}|{mo}': {k: (None if pd.isna(v) else float(v)) for k, v in r_.items()
                                          if k not in ('season', 'model')}
                            for (s, mo), r_ in e.set_index(['season', 'model']).iterrows()}
# chosen C for every logistic model
for ix in ('spei', 'speironi', 'speipm', 'spei79'):
    f = f'{W}/model_{ix}_prob_cv.csv'
    if os.path.exists(f):
        cv_ = pd.read_csv(f); cv_ = cv_[cv_.chosen]
        N[f'C_{ix}'] = {f'{a}_{b}': (None if pd.isna(c_) else float(c_)) for a, b, c_ in zip(cv_.season, cv_.model, cv_.C)}

v3 = pd.read_csv(f'{W}/fig3_values.csv').set_index(['season', 'reg'])
N['shap'] = {f'{s}|{rg_}|{c_}': float(v3.loc[(s, rg_), c_]) for s, rg_ in v3.index for c_ in v3.columns}
v4 = pd.read_csv(f'{W}/fig4_values.csv')
N['blocks'] = {f'{a}|{b}|{c_}': float(f_) for a, b, c_, f_ in zip(v4['index'], v4.season, v4.start, v4.freq)}
for ix in ('spei', 'spi', 'speit', 'speipm'):
    q = pd.read_csv(f'{W}/model_{ix}_dataset.csv'); q['y'] = q.target <= -1
    for s, g in q.groupby('season'):
        N[f'freq_{ix}_{s}'] = [float(g[g.year <= 2004].y.mean()), float(g[g.year >= 2005].y.mean())]

if os.path.exists(f'{W}/pm_summary.json'):
    N['pm'] = json.load(open(f'{W}/pm_summary.json'))
if os.path.exists(f'{W}/tc_summary.json'):
    N['tc'] = json.load(open(f'{W}/tc_summary.json'))
json.dump(N, open(OUT, 'w'), indent=1, ensure_ascii=False, default=float)
print('ok', len(N))
