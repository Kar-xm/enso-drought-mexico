"""Does the multi-decadal change in drought frequency of ERA5-Land also appear in a station-based product?
SPI-6 is computed with the same method (mixed gamma per calendar month) from ERA5-Land precipitation and
from TerraClimate precipitation (CRU TS station anomalies on a WorldClim climatology), both calibrated on
the common period 1959-2004 and evaluated 1959-2024 (first full SPI-6 in 1958 needs six months).
Outputs: tc_compare_blocks.csv (drought frequency SPI-6 <= -1 per 5-year block and season, both products),
tc_compare_points.csv (per-point correlation of the seasonal SPI-6 series).
Usage: python tc_compare.py <ERA5 folder> <TerraClimate folder with TerraClimate_pr_*.csv> <out dir>"""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indices_national import gamma_z, by_month

CAL = (1959, 2004)
SEAS_END = {'dry': 4, 'wet': 10}


def spi6(df, col):
    out = {}
    for (lon, lat), g in df.sort_values(['lon', 'lat', 'year', 'month']).groupby(['lon', 'lat']):
        x = pd.Series(g[col].values.astype(float)).rolling(6).sum().values
        yrs, mon = g.year.values, g.month.values
        out[(lon, lat)] = pd.DataFrame({'year': yrs, 'month': mon,
                                        'spi6': by_month(x, mon, (yrs >= CAL[0]) & (yrs <= CAL[1]), gamma_z)})
    return out


if __name__ == '__main__':
    e5dir, tcdir, out = sys.argv[1:4]
    e5 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(e5dir, 'ERA5_Mexico_*.csv')))])
    tc = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(tcdir, 'TerraClimate_pr_*.csv')))])
    tc['lon'] = tc.lon.round(2); tc['lat'] = tc.lat.round(2)
    e5 = e5[e5.year >= 1958]
    bad = tc.groupby(['lon', 'lat']).pr_mm.apply(lambda v: v.isna().any())
    keep = set(bad[~bad].index)
    tc = tc[[k in keep for k in zip(tc.lon, tc.lat)]]
    e5 = e5[[k in keep for k in zip(e5.lon, e5.lat)]]
    assert len(tc) == len(e5), (len(tc), len(e5))
    print('points used', len(keep), 'excluded (missing TerraClimate)', int(bad.sum()))
    A, B = spi6(e5, 'precip_mm'), spi6(tc, 'pr_mm')
    rows, pts = [], []
    for k in A:
        for s, m in SEAS_END.items():
            a = A[k][(A[k].month == m) & (A[k].year >= 1959)].set_index('year').spi6
            b = B[k][(B[k].month == m) & (B[k].year >= 1959)].set_index('year').spi6
            pts.append(dict(lon=k[0], lat=k[1], season=s, r=np.corrcoef(a, b)[0, 1]))
            for prod, v in (('ERA5-Land', a), ('TerraClimate', b)):
                rows += [dict(product=prod, season=s, lon=k[0], lat=k[1], year=y, drought=val <= -1) for y, val in v.items()]
    R = pd.DataFrame(rows)
    R['block'] = np.where(R.year <= 2004, np.minimum((R.year - 1959) // 5 * 5 + 1959, 1999), (R.year - 2005) // 5 * 5 + 2005)
    blk = R.groupby(['product', 'season', 'block']).drought.mean().mul(100).unstack('product').round(1)
    blk.to_csv(os.path.join(out, 'tc_compare_blocks.csv'))
    P = pd.DataFrame(pts); P.to_csv(os.path.join(out, 'tc_compare_points.csv'), index=False)
    print(blk.to_string())
    print(P.groupby('season').r.describe().round(2))
    summ = {f'r_{s}': float(P[P.season == s].r.median()) for s in SEAS_END}
    summ['n_points'] = len(keep)
    for s in SEAS_END:
        q = R[R.season == s]
        for prod in ('ERA5-Land', 'TerraClimate'):
            z = q[q['product'] == prod]
            summ[f'{s}|{prod}'] = dict(pre79=float(z[z.year <= 1978].drought.mean()),
                                       y79_04=float(z[(z.year >= 1979) & (z.year <= 2004)].drought.mean()),
                                       test=float(z[z.year >= 2005].drought.mean()))
        summ[f'blocks_{s}'] = {str(k): v for k, v in blk.loc[s].to_dict(orient='index').items()}
    json.dump(summ, open(os.path.join(out, 'tc_summary.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if not k.startswith('blocks')}, indent=1))
