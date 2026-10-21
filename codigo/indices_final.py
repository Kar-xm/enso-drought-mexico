"""Shared functions for the drought indices (CISIMA 2026, ENSO and seasonal drought in Mexico).
  * ra_mm_day           : extraterrestrial radiation, FAO-56 eq. 21, in mm/day of evaporation equivalent
  * hargreaves_mm_month : Hargreaves-Samani reference evapotranspiration (FAO-56 eq. 52), mm/month,
                          with Ra evaluated at day 15 of each month
  * glo_cdf_fit         : generalized logistic distribution (Hosking) fitted by unbiased L-moments,
                          as in the SPEI R package (Begueria et al., 2014); returns the CDF
  * P_CLIP              : standardized values are bounded to +-3.09 (probabilities clipped at 0.001)
"""
import numpy as np
import pandas as pd
from scipy import stats

P_CLIP = 1e-3                     # standardized values bounded to +-3.09


def ra_mm_day(lat_deg, doy):
    """Extraterrestrial radiation (FAO-56 eq. 21), expressed in mm/day (x 0.408)."""
    phi = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(-np.tan(phi) * np.tan(delta))
    ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(delta)
                                          + np.cos(phi) * np.cos(delta) * np.sin(ws))
    return 0.408 * ra


def hargreaves_mm_month(dates, tmax, tmin, lat):
    """dates: month-end DatetimeIndex. Returns PET in mm/month."""
    mid = dates - pd.to_timedelta(dates.day - 15, unit='D')
    ra = ra_mm_day(lat, mid.dayofyear.values)
    tmean = (tmax + tmin) / 2.0
    return 0.0023 * ra * (tmean + 17.8) * np.sqrt(np.clip(tmax - tmin, 0, None)) * dates.days_in_month.values


def _to_z(p):
    return stats.norm.ppf(np.clip(p, P_CLIP, 1 - P_CLIP))


def glo_cdf_fit(x):
    """Generalized logistic (Hosking) fitted by unbiased L-moments. Returns CDF."""
    xs = np.sort(x)
    n = len(xs)
    i = np.arange(1, n + 1)
    b0 = xs.mean()
    b1 = np.sum((i - 1) / (n - 1) * xs) / n
    b2 = np.sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * xs) / n
    l1, l2, l3 = b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0
    k = -l3 / l2
    if abs(k) < 1e-6:
        return lambda v: 1 / (1 + np.exp(-(v - l1) / l2))
    a = l2 * np.sin(k * np.pi) / (k * np.pi)
    xi = l1 - a * (1 / k - np.pi / np.sin(k * np.pi))

    def cdf(v):
        z = 1 - k * (v - xi) / a
        with np.errstate(invalid='ignore', divide='ignore'):
            F = 1 / (1 + np.exp(np.log(z) / k))       # y = -log(z)/k ; F = 1/(1+exp(-y))
        return np.where(z > 0, F, 1.0 if k > 0 else 0.0)
    return cdf
