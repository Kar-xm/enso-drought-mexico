"""SPI and SPEI (Hargreaves PET; Thornthwaite for sensitivity) at 3 and 6 months, 695 ERA5-Land grid points, Feb 1950 - Dec 2024.
Points are identified by (lon, lat) (the point_id column of the source files is truncated).
* PET (sensitivity): Thornthwaite (1948) from monthly mean 2-m temperature, heat index from the 1951-2004
  monthly climatology (calibration period), Willmott et al. (1985) formula above 26.5 C, daylength and month-length
  correction. The pet_mm column of the source files (ERA5-Land potential evaporation) is NOT used.
* PET (main): Hargreaves-Samani with monthly means of daily Tmax/Tmin (ERA5-Land DAILY_AGGR)
  and FAO-56 extraterrestrial radiation -> SPEI{s}; Thornthwaite kept for sensitivity -> SPEIT{s}.
* SPI : mixed gamma per calendar month; SPEI: generalised logistic (unbiased L-moments).
* Calibration: the distributions are fitted ONLY with the accumulations ending in 1951-2004
  (training period of the models) and then applied to every month 1950-2024, so the test period
  2005-2024 carries no information into the index definition. Values bounded to +-3.09.
* PET (sensitivity 2, optional): FAO-56 Penman-Monteith (pm_eto.py) from ERA5-Land monthly means of daily
  incoming solar radiation, dew point, 10-m wind speed and surface pressure -> SPEIPM{s}.
Usage: python indices_national.py <folder with ERA5_Mexico_*.csv> <tmaxmin_all.csv> <out.pkl> [pm_all.csv]"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy import stats
from indices_final import glo_cdf_fit, hargreaves_mm_month, P_CLIP
from pm_eto import pm_mm_month

SCALES = (3, 6)
CALIB = (1951, 2004)


def _z(p):
    return stats.norm.ppf(np.clip(p, P_CLIP, 1 - P_CLIP))


def gamma_z(x, cal):
    """mixed gamma fitted on x[cal] (non-missing), applied to all x."""
    out = np.full(len(x), np.nan); ok = ~np.isnan(x)
    v = x[cal & ok]; q = np.mean(v <= 0)
    a, _, sc = stats.gamma.fit(v[v > 0], floc=0)
    H = np.where(x[ok] > 0, q + (1 - q) * stats.gamma.cdf(x[ok], a, loc=0, scale=sc), q)
    out[ok] = _z(H)
    return out


def glo_z(x, cal):
    out = np.full(len(x), np.nan); ok = ~np.isnan(x)
    out[ok] = _z(glo_cdf_fit(x[cal & ok])(x[ok]))
    return out


def by_month(x, months, cal, func):
    out = np.full(len(x), np.nan)
    for m in range(1, 13):
        i = months == m
        out[i] = func(x[i], cal[i])
    return out


def daylength_h(lat_deg, doy):
    phi = np.radians(lat_deg)
    d = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    return 24 / np.pi * np.arccos(-np.tan(phi) * np.tan(d))


def thornthwaite(T, dates, lat):
    """T: monthly mean temperature (C); dates: month-end DatetimeIndex. Returns PET mm/month.
    The heat index I uses only the calibration years, like the index distributions."""
    Tc = np.clip(T, 0, None)
    cal = (dates.year >= CALIB[0]) & (dates.year <= CALIB[1])
    clim = pd.Series(Tc[cal]).groupby(dates.month[cal]).mean()
    I = ((clim / 5) ** 1.514).sum()
    a = 6.75e-7 * I ** 3 - 7.71e-5 * I ** 2 + 1.792e-2 * I + 0.49239
    pet = np.where(Tc < 26.5, 16 * (10 * Tc / I) ** a, -415.85 + 32.24 * Tc - 0.43 * Tc ** 2)
    pet = np.where(Tc <= 0, 0.0, pet)
    mid = dates - pd.to_timedelta(dates.day - 15, unit='D')
    N = daylength_h(lat, mid.dayofyear.values)
    return pet * (N / 12) * (dates.days_in_month.values / 30)


if __name__ == '__main__':
    src, tmm, out = sys.argv[1:4]
    pm_file = sys.argv[4] if len(sys.argv) > 4 else None
    tx = pd.read_csv(tmm)
    if pm_file:
        tx = tx.merge(pd.read_csv(pm_file)[['lon', 'lat', 'year', 'month', 'rs_mj', 'td_c', 'u10_ms', 'sp_kpa']],
                      on=['lon', 'lat', 'year', 'month'], how='left', validate='one_to_one')
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(src, 'ERA5_Mexico_*.csv')))],
                  ignore_index=True)
    cols = ['lon', 'lat', 'year', 'month', 'tmax_c', 'tmin_c'] + (['rs_mj', 'td_c', 'u10_ms', 'sp_kpa'] if pm_file else [])
    d = d.merge(tx[cols], on=['lon', 'lat', 'year', 'month'],
                how='left', validate='one_to_one')
    assert d[['tmax_c', 'tmin_c']].notna().all().all(), 'missing Tmax/Tmin'
    d['date'] = pd.to_datetime(d.date_str) + pd.offsets.MonthEnd(0)
    d = d.sort_values(['lon', 'lat', 'date'])
    res = {}
    for (lon, lat), g in d.groupby(['lon', 'lat']):
        dates = pd.DatetimeIndex(g.date.values)
        assert (np.diff(dates.year * 12 + dates.month) == 1).all()
        P = g.precip_mm.values.astype(float)
        pet_t = thornthwaite(g.temp_c.values.astype(float), dates, lat)
        pet_h = hargreaves_mm_month(dates, g.tmax_c.values.astype(float), g.tmin_c.values.astype(float), lat)
        mon = dates.month.values
        cal = (dates.year >= CALIB[0]) & (dates.year <= CALIB[1])
        row = {'date': dates, 'P': P, 'PET_H': pet_h, 'PET_T': pet_t, 'T': g.temp_c.values}
        for s in SCALES:
            Pa = pd.Series(P).rolling(s).sum().values
            row[f'SPI{s}'] = by_month(Pa, mon, cal, gamma_z)
            row[f'SPEI{s}'] = by_month(pd.Series(P - pet_h).rolling(s).sum().values, mon, cal, glo_z)
            row[f'SPEIT{s}'] = by_month(pd.Series(P - pet_t).rolling(s).sum().values, mon, cal, glo_z)
        if pm_file:
            pet_pm = pm_mm_month(dates, g.tmax_c.values.astype(float), g.tmin_c.values.astype(float),
                                 g.td_c.values.astype(float), g.rs_mj.values.astype(float),
                                 g.u10_ms.values.astype(float), g.sp_kpa.values.astype(float), lat)
            row['PET_PM'] = pet_pm
            for s in SCALES:
                row[f'SPEIPM{s}'] = by_month(pd.Series(P - pet_pm).rolling(s).sum().values, mon, cal, glo_z)
        res[(lon, lat)] = row
    pd.to_pickle(res, out)
    print('points', len(res))
