"""Summary of the Penman-Monteith sensitivity test -> pm_summary.json.
Usage: python pm_summary.py <work dir>"""
import json, sys
import numpy as np, pandas as pd

W = sys.argv[1]
r = pd.read_pickle(f'{W}/indices_national.pkl')
yrs = len(next(iter(r.values()))['date']) / 12
eto = pd.Series({k: v['PET_PM'].sum() / yrs for k, v in r.items()})
rH = pd.Series({k: np.corrcoef(v['SPEI6'][12:], v['SPEIPM6'][12:])[0, 1] for k, v in r.items()})
rT = pd.Series({k: np.corrcoef(v['SPEIT6'][12:], v['SPEIPM6'][12:])[0, 1] for k, v in r.items()})
t = pd.read_csv(f'{W}/explore_corr.csv'); q = t[(t.season == 'dry') & (t['index'] == 'SPEIPM6')]
fr = []
for v in r.values():
    d = v['date']; sel = (d.month == 4) & (d.year >= 1951)
    fr += list(zip(d.year[sel], v['SPEIPM6'][sel] <= -1))
fr = pd.DataFrame(fr, columns=['year', 'y'])
out = dict(eto_med=float(eto.median()), eto_p10=float(eto.quantile(.1)), eto_p90=float(eto.quantile(.9)),
           r_H_PM=float(rH.median()), r_T_PM=float(rT.median()), sig_dry=float(q.sig_conc.mean()),
           freq_dry=[float(fr[fr.year <= 2004].y.mean()), float(fr[fr.year >= 2005].y.mean())])
json.dump(out, open(f'{W}/pm_summary.json', 'w'), indent=1)
print(out)
