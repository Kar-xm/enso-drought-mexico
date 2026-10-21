"""Seasonal dataset for the drought-probability models (one row per grid point and season), Mexico 1951-2024.
  wet season : target = index-6 ending October (May-Oct)
  dry season : target = index-6 ending April   (Nov-Apr)
Predictors:
  pers6    : index-6 of the previous, non-overlapping 6 months (ending April / October)
  pers3    : index-3 of the 3 months before the season
  oni_lead : mean of the ENSO index of the three 3-month windows that end before the season starts
             (DJF, JFM, FMA for the wet season; JJA, JAS, ASO for the dry season)
  oni_conc : mean of the ENSO index over the six months of the season (upper bound, not a forecast)
The ENSO file can be the ONI (oni.ascii.txt) or the RONI (RONI.ascii.txt): same CPC layout, column ANOM.
Split used by the models: train 1951-2004, test 2005-2024.
Usage: python build_dataset.py <indices.pkl> <oni or roni ascii> <index: SPEI|SPI|SPEIT|SPEIPM> <out prefix>"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enso_explore import load_oni

SEAS = {'wet': dict(end=10, conc=[(0, m) for m in range(5, 11)], lead=[(0, m) for m in (1, 2, 3)],
                    pers_end=(0, 4)),
        'dry': dict(end=4, conc=[(-1, 11), (-1, 12)] + [(0, m) for m in (1, 2, 3, 4)],
                    lead=[(-1, m) for m in (7, 8, 9)], pers_end=(-1, 10))}
TRAIN, TEST = (1951, 2004), (2005, 2024)


def me(y, m):
    return pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)


def build(r, oni, idx):
    rows = []
    for (lon, lat), v in r.items():
        s6 = pd.Series(v[f'{idx}6'], index=v['date']); s3 = pd.Series(v[f'{idx}3'], index=v['date'])
        for season, S in SEAS.items():
            for y in range(1951, 2025):
                py, pm = S['pers_end']
                rows.append(dict(lon=lon, lat=lat, year=y, season=season,
                                 target=s6.get(me(y, S['end']), np.nan),
                                 pers6=s6.get(me(y + py, pm), np.nan), pers3=s3.get(me(y + py, pm), np.nan),
                                 oni_lead=np.mean([oni[me(y + a, m)] for a, m in S['lead']]),
                                 oni_conc=np.mean([oni[me(y + a, m)] for a, m in S['conc']])))
    return pd.DataFrame(rows).dropna()


def year_folds(years, k=6):
    """Blocked cross-validation by year: all grid points of a year stay in the same fold
    (points of the same year are strongly correlated, so years are the real samples)."""
    uy = np.sort(np.unique(years)); blocks = np.array_split(uy, k)
    return [(np.where(~np.isin(years, b))[0], np.where(np.isin(years, b))[0]) for b in blocks]


if __name__ == '__main__':
    src, oni_path, idx, pref = sys.argv[1:5]
    d = build(pd.read_pickle(src), load_oni(oni_path), idx)
    d.to_csv(f'{pref}_dataset.csv', index=False)
    print(pref, len(d))
