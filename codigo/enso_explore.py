"""ENSO - drought relationship at the 695 grid points (exploratory step).
Seasonal design (one value per year, so serial dependence is small):
  * wet season : SPEI-6 / SPI-6 ending in October (May-Oct)
  * dry season : SPEI-6 / SPI-6 ending in April   (Nov-Apr)
ENSO predictor: ONI averaged over the same 6 months (concurrent) and lead ONI: the three ONI values
whose 3-month windows end no later than the month before the season starts (JJA, JAS, ASO for the
dry season; DJF, JFM, FMA for the wet season), so the lead ONI is fully observed at season start. Pearson r, two-sided t-test, Benjamini-Hochberg FDR
(q = 0.05) across the 695 points (field significance; Wilks 2016).
Composites: mean index in El Nino (ONI >= 0.5) vs La Nina (ONI <= -0.5) seasons.
Serial correlation: lag-1 autocorrelation (between consecutive years) of the index (r1x) and the ONI (r1o);
effective sample size n_eff = n (1 - r1x r1o) / (1 + r1x r1o) (Bretherton et al., 1999), capped at n,
and the correlation test repeated with n_eff - 2 degrees of freedom (p_conc_eff, sig_conc_eff).
Usage: python enso_explore.py <indices_national.pkl> <oni.ascii.txt> <out prefix>"""
import sys
import numpy as np, pandas as pd
from scipy import stats


def lag1(z):
    return np.corrcoef(z[:-1], z[1:])[0, 1]


def p_neff(r, n_eff):
    t = r * np.sqrt((n_eff - 2) / (1 - r ** 2))
    return 2 * stats.t.sf(abs(t), n_eff - 2)

SEAS_CENTRE = {'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6, 'JJA': 7, 'JAS': 8,
               'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12}


def load_oni(path):
    o = pd.read_csv(path, sep=r'\s+')
    o['month'] = o.SEAS.map(SEAS_CENTRE)
    o['date'] = pd.to_datetime(dict(year=o.YR, month=o.month, day=1)) + pd.offsets.MonthEnd(0)
    return o.set_index('date').ANOM


def bh(p, q=0.05):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    thr = q * np.arange(1, n + 1) / n
    ok = p[o] <= thr
    k = np.where(ok)[0].max() + 1 if ok.any() else 0
    sig = np.zeros(n, bool); sig[o[:k]] = True
    return sig


SEASONS = {'wet': dict(end=10, conc=range(5, 11), lead=range(1, 4)),       # May-Oct; lead DJF, JFM, FMA
           'dry': dict(end=4, conc=[11, 12, 1, 2, 3, 4], lead=range(7, 10))}  # Nov-Apr; lead JJA, JAS, ASO


def oni_season(oni, year, months, end_month):
    """mean ONI over `months` of the season ending in `end_month` of `year`."""
    vals = []
    for m in months:
        y = year if (m <= end_month) else year - 1
        d = pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)
        vals.append(oni.get(d, np.nan))
    return np.mean(vals)


if __name__ == '__main__':
    src, oni_path, pref = sys.argv[1:4]
    r = pd.read_pickle(src)
    oni = load_oni(oni_path)
    rows = []
    for (lon, lat), v in r.items():
        dates = v['date']
        for sname, S in SEASONS.items():
            sel = (dates.month == S['end']) & (dates.year >= 1951) & (dates.year <= 2024)
            yrs = dates.year[sel]
            o_c = np.array([oni_season(oni, y, S['conc'], S['end']) for y in yrs])
            o_l = np.array([oni_season(oni, y, S['lead'], S['end']) for y in yrs])
            for idx in [i for i in ('SPI6', 'SPEI6', 'SPEIT6', 'SPEIPM6') if i in v]:
                x = v[idx][sel]
                ok = ~np.isnan(x)
                rc, pc = stats.pearsonr(o_c[ok], x[ok])
                rl, pl = stats.pearsonr(o_l[ok], x[ok])
                nino, nina = x[ok][o_c[ok] >= 0.5], x[ok][o_c[ok] <= -0.5]
                tc, pcomp = stats.ttest_ind(nino, nina, equal_var=False)
                r1x, r1o = lag1(x[ok]), lag1(o_c[ok])
                n_eff = min(ok.sum(), ok.sum() * (1 - r1x * r1o) / (1 + r1x * r1o))
                rows.append(dict(lon=lon, lat=lat, season=sname, index=idx, n=int(ok.sum()),
                                 r1_index=r1x, r1_oni=r1o, n_eff=n_eff, p_conc_eff=p_neff(rc, n_eff),
                                 r_conc=rc, p_conc=pc, r_lead=rl, p_lead=pl,
                                 mean_nino=nino.mean(), mean_nina=nina.mean(), n_nino=len(nino),
                                 n_nina=len(nina), p_comp=pcomp))
    t = pd.DataFrame(rows)
    for (s, i), g in t.groupby(['season', 'index']):
        for c in ['conc', 'lead', 'comp', 'conc_eff']:
            t.loc[g.index, f'sig_{c}'] = bh(g[f'p_{c}'].values)
    t.to_csv(f'{pref}_corr.csv', index=False)
    summ = t.groupby(['season', 'index']).agg(
        n_years=('n', 'first'), n_nino=('n_nino', 'first'), n_nina=('n_nina', 'first'),
        r_conc_median=('r_conc', 'median'), r_conc_min=('r_conc', 'min'), r_conc_max=('r_conc', 'max'),
        frac_sig_conc=('sig_conc', 'mean'), frac_sig_lead=('sig_lead', 'mean'),
        frac_sig_comp=('sig_comp', 'mean'), frac_sig_conc_eff=('sig_conc_eff', 'mean'),
        r1_index_med=('r1_index', 'median'), r1_oni=('r1_oni', 'first'), n_eff_med=('n_eff', 'median'))
    print(summ.round(3).to_string())
