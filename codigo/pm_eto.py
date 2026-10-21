"""FAO-56 Penman-Monteith reference evapotranspiration (ETo) from monthly means (Allen et al., 1998).
Inputs per month: Tmax, Tmin (mean of daily extremes, C), dew point Td (C), incoming solar radiation
Rs (MJ m-2 d-1), wind speed at 10 m (m/s), surface pressure (kPa). Returns mm/month.
  * u2 = u10 * 4.87 / ln(67.8*10 - 5.42)                         (FAO-56 eq. 47)
  * es = [e0(Tmax) + e0(Tmin)] / 2 ; ea = e0(Td)                  (eqs. 11-12, 14)
  * Rn = 0.77 Rs - Rnl ; Rnl with Rs/Rso, Rso = (0.75 + 2e-5 z) Ra (eqs. 37-39)
    z from the surface pressure (inverse of eq. 7)
  * G = 0.14 (T_i - T_{i-1})                                       (eq. 44)"""
import numpy as np
import pandas as pd
from indices_final import ra_mm_day

SIGMA = 4.903e-9


def e0(t):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def pm_mm_month(dates, tmax, tmin, td, rs, u10, p_kpa, lat):
    t = (tmax + tmin) / 2
    delta = 4098 * e0(t) / (t + 237.3) ** 2
    gamma = 0.000665 * p_kpa
    u2 = u10 * 4.87 / np.log(67.8 * 10 - 5.42)
    es = (e0(tmax) + e0(tmin)) / 2
    ea = np.minimum(e0(td), es)
    mid = dates - pd.to_timedelta(dates.day - 15, unit='D')
    ra = ra_mm_day(lat, mid.dayofyear.values) / 0.408                  # MJ m-2 d-1
    z = 293 / 0.0065 * (1 - (p_kpa / 101.3) ** (1 / 5.26))
    rso = (0.75 + 2e-5 * z) * ra
    rel = np.clip(rs / rso, 0.25, 1.0)
    rnl = SIGMA * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2 * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * rel - 0.35)
    rn = 0.77 * rs - rnl
    g = 0.14 * np.diff(t, prepend=t[0])
    eto = (0.408 * delta * (rn - g) + gamma * 900 / (t + 273) * u2 * (es - ea)) / (delta + gamma * (1 + 0.34 * u2))
    return np.clip(eto, 0, None) * dates.days_in_month.values
