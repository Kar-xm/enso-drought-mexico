"""Figures for the CISIMA 2026 manuscript (Spanish labels).
Usage: python paper_figures.py <work dir> <out dir>
Needs in <work dir>: explore_corr.csv, model_{spei,spi,speit}_dataset.csv, model_spei_prob_shap_log.csv,
fisher_dry.csv / fisher_wet.csv (written here if missing) and basemap/naturalearth_lowres.shp
(Natural Earth 1:110m, public domain)."""
import sys
import numpy as np, pandas as pd
import shapefile
from scipy.stats import fisher_exact
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

W, OUT = sys.argv[1:3]
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from enso_explore import bh

plt.rcParams.update({'font.family': 'serif', 'font.size': 7, 'axes.titlesize': 7, 'savefig.dpi': 400,
                     'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e',
                     'ytick.color': '#52514e', 'xtick.labelsize': 6, 'ytick.labelsize': 6})
INK2, GRID = '#52514e', '#e1e0d9'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
SEASON = {'dry': 'Secas (nov–abr)', 'wet': 'Lluvias (may–oct)'}
ASPECT = 1 / np.cos(np.radians(24))
pct = lambda x: f'{x:.0f} %'

# ---------------------------------------------------------------- base map
SHP = shapefile.Reader(f'{W}/basemap/naturalearth_lowres.shp')
FIELDS = [f[0] for f in SHP.fields[1:]]
OUTLINES = []
for s, rec in zip(SHP.shapes(), SHP.records()):
    if dict(zip(FIELDS, rec))['iso_a3'] in ('MEX', 'USA', 'GTM', 'BLZ', 'HND', 'SLV', 'CUB'):
        parts = list(s.parts) + [len(s.points)]
        OUTLINES += [(np.array(s.points[a:b]), dict(zip(FIELDS, rec))['iso_a3'] == 'MEX')
                     for a, b in zip(parts[:-1], parts[1:])]
# macro-region limits drawn half a cell from the grid nodes (nodes on the limit belong to the region)
REG_LINES = [([-108.75, -108.75], [14.5, 33.0]), ([-108.75, -86.5], [23.25, 23.25]),
             ([-93.75, -93.75], [17.25, 23.25]), ([-108.75, -93.75], [17.25, 17.25])]
REG_LABELS = [(-114.6, 25.2, 'NO'), (-103.0, 27.5, 'N-NE'), (-108.4, 21.2, 'C-O'), (-89.0, 22.6, 'S-SE')]


def region(lon, lat):
    if lon <= -109: return 'Noroeste'
    if lat >= 23.5: return 'Norte-Noreste'
    if lon <= -94 and lat >= 17.5: return 'Centro-Occidente'
    return 'Sur-Sureste'


def mapax(ax, ticks_x=True, ticks_y=True, labels=True):
    for xy, mex in OUTLINES:
        ax.plot(xy[:, 0], xy[:, 1], color='#3d3d3a' if mex else '#b8b7b0', lw=0.5 if mex else 0.35, zorder=4)
    for xs, ys in REG_LINES:
        ax.plot(xs, ys, color='#3d3d3a', lw=0.5, ls=(0, (3, 2)), zorder=4)
    if labels:
        for x, y, tx in REG_LABELS:
            ax.text(x, y, tx, fontsize=5.5, color='#3d3d3a', ha='center', va='center', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))
    ax.set_xlim(-118, -86); ax.set_ylim(14, 33); ax.set_aspect(ASPECT)
    ax.set_xticks([-115, -105, -95] if ticks_x else []); ax.set_yticks([15, 20, 25, 30] if ticks_y else [])
    ax.set_xticklabels(['115° O', '105° O', '95° O'] if ticks_x else [])
    ax.set_yticklabels(['15° N', '20° N', '25° N', '30° N'] if ticks_y else [])
    ax.tick_params(length=2, pad=1)
    for s in ax.spines.values(): s.set_color('#c3c2b7'); s.set_linewidth(0.5)


# ---------------------------------------------------------------- Fig 1: correlation maps (SPEI-6, Hargreaves)
t = pd.read_csv(f'{W}/explore_corr.csv')
div = LinearSegmentedColormap.from_list('div', ['#b35806', '#f0efec', '#2a78d6'])
fig, axs = plt.subplots(2, 1, figsize=(3.3, 4.6))
for ax, s in zip(axs, ['dry', 'wet']):
    g = t[(t.season == s) & (t['index'] == 'SPEI6')]
    sc = ax.scatter(g.lon, g.lat, c=g.r_conc, s=8.5, marker='s', cmap=div, norm=TwoSlopeNorm(0, -0.7, 0.7), lw=0, zorder=2)
    k = g[g.sig_conc.astype(bool)]
    ax.scatter(k.lon, k.lat, s=0.5, color='k', lw=0, zorder=3)
    mapax(ax, ticks_x=(s == 'wet'))
    ax.set_title(f'{SEASON[s]}: {pct(g.sig_conc.mean() * 100)} de puntos significativos', pad=2)
cb = fig.colorbar(sc, ax=axs, fraction=0.04, pad=0.05, orientation='horizontal', aspect=35)
cb.set_label('r (ONI concurrente vs. SPEI-6)'); cb.outline.set_visible(False)
fig.savefig(f'{OUT}/fig1_correlacion.png', bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- Fig 2: drought probability by ENSO phase + Fisher test
d = pd.read_csv(f'{W}/model_spei_dataset.csv'); d['y'] = d.target <= -1
d['fase'] = np.where(d.oni_conc >= 0.5, 'nino', np.where(d.oni_conc <= -0.5, 'nina', 'neutro'))
p = d.groupby(['season', 'fase', 'lon', 'lat']).y.mean().reset_index()
fis = {}
for s, g in d.groupby('season'):
    rows = []
    for (lon, lat), q in g.groupby(['lon', 'lat']):
        a, b = q[q.fase == 'nino'].y, q[q.fase == 'nina'].y
        rows.append(dict(lon=lon, lat=lat, p=fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])[1]))
    f = pd.DataFrame(rows); f['sig'] = bh(f.p.values); f.to_csv(f'{W}/fisher_{s}.csv', index=False); fis[s] = f
seq = LinearSegmentedColormap.from_list('seq', ['#fbe9df', '#f2a27c', '#eb6834', '#a83d12', '#5c1d05'])
fig, axs = plt.subplots(2, 2, figsize=(3.4, 3.5))
for i, s in enumerate(['dry', 'wet']):
    for j, (f, nm) in enumerate([('nino', 'El Niño'), ('nina', 'La Niña')]):
        g = p[(p.season == s) & (p.fase == f)]; ax = axs[i, j]
        n = d[(d.season == s) & (d.fase == f)].year.nunique()
        sc = ax.scatter(g.lon, g.lat, c=g.y, s=4.5, marker='s', cmap=seq, vmin=0, vmax=0.6, lw=0, zorder=2)
        k = fis[s][fis[s].sig]
        ax.scatter(k.lon, k.lat, s=0.35, color='k', lw=0, zorder=3)
        mapax(ax, ticks_x=(i == 1), ticks_y=(j == 0), labels=False)
        ax.set_title(f'{SEASON[s].split(" ")[0]}, {nm} (n = {n})\nmediana {pct(g.y.median() * 100)}', pad=2)
cb = fig.colorbar(sc, ax=axs, fraction=0.04, pad=0.07, orientation='horizontal', aspect=35, extend='max')
cb.set_label('Probabilidad empírica de sequía (SPEI-6 ≤ −1)'); cb.outline.set_visible(False)
cb.set_ticks([0, .2, .4, .6]); cb.set_ticklabels(['0', '0.2', '0.4', '0.6'])
fig.savefig(f'{OUT}/fig2_probabilidad_fase.png', bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: SHAP of the per-point logistic (e), dry season
# ENSO = |phi_prev + phi_conc| (the two ONI terms share the ENSO effect under the L2 penalty, r = 0.89);
# persistence = |phi_SPEI6 + phi_SPEI3|. The wet-season contributions are all < 0.07 log-odds (text only).
s = pd.read_csv(f'{W}/model_spei_prob_shap_log.csv'); s = s[s.model == 'LOG_enso']
s['reg'] = [region(a, b) for a, b in zip(s.lon, s.lat)]
s['ENOS (ONI previo + concurrente)'] = (s.shap_oni_lead + s.shap_oni_conc).abs()
s['Persistencia (SPEI-6 + SPEI-3)'] = (s.shap_pers6 + s.shap_pers3).abs()
G = ['ENOS (ONI previo + concurrente)', 'Persistencia (SPEI-6 + SPEI-3)']
REG = ['Noroeste', 'Norte-Noreste', 'Centro-Occidente', 'Sur-Sureste']
REGL = ['Noroeste', 'Norte-\nNoreste', 'Centro-\nOccidente', 'Sur-\nSureste']
m = s.groupby(['season', 'reg'])[G].mean()
m.round(3).to_csv(f'{W}/fig3_values.csv')
fig, ax = plt.subplots(figsize=(3.3, 2.0))
x = np.arange(len(REG)); w = 0.36
for k, gname in enumerate(G):
    ax.bar(x + (k - 0.5) * w, m.loc['dry'].loc[REG, gname], w * 0.9, color=CAT[k], label=gname, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(REGL); ax.set_ylabel('|SHAP| medio (log-momios)')
ax.grid(axis='y', color=GRID, lw=0.5, zorder=0); ax.spines[['top', 'right']].set_visible(False)
ax.set_ylim(0, 1.2); ax.legend(frameon=False, fontsize=6.5, loc='upper right', ncol=1)
fig.tight_layout(); fig.savefig(f'{OUT}/fig3_shap_regiones.png', bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- Fig 4: drought frequency in 5-year blocks
BLOCKS = [(a, min(a + 4, 2004)) for a in range(1951, 2005, 5)] + [(a, a + 4) for a in range(2005, 2025, 5)]
fig, axs = plt.subplots(1, 2, figsize=(3.4, 2.0), sharey=True)
lab = {'spei': 'SPEI-6 (Hargreaves)', 'spi': 'SPI-6', 'speit': 'SPEI-6 (Thornthwaite)'}
rows = []
for k, i in enumerate(['spei', 'spi', 'speit']):
    q = pd.read_csv(f'{W}/model_{i}_dataset.csv'); q['y'] = q.target <= -1
    for ax, se in zip(axs, ['dry', 'wet']):
        qs = q[q.season == se]
        xs, ys = [], []
        for a, b in BLOCKS:
            xs.append((a + b) / 2); ys.append(qs[qs.year.between(a, b)].y.mean() * 100)
            rows.append(dict(index=i, season=se, start=a, end=b, freq=ys[-1]))
        ax.plot(xs, ys, color=CAT[k], lw=1.6 if k == 0 else 1.0, marker='o', ms=2.3, label=lab[i], zorder=3)
pd.DataFrame(rows).to_csv(f'{W}/fig4_values.csv', index=False)
for ax, se in zip(axs, ['dry', 'wet']):
    ax.axvspan(2004.5, 2024.5, color='#f0efec', zorder=0); ax.set_title(SEASON[se], pad=2)
    ax.set_xlim(1950, 2025); ax.set_ylim(0, 62); ax.set_xticks([1960, 1980, 2000, 2020]); ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(2014.5, 0.02, 'prueba', transform=ax.get_xaxis_transform(), ha='center', va='bottom', color=INK2, fontsize=6,
            bbox=dict(boxstyle='round,pad=0.1', fc='#f0efec', ec='none'), zorder=5)
axs[0].set_ylabel('Frecuencia de sequía (%)')
fig.legend(*axs[0].get_legend_handles_labels(), ncol=3, frameon=False, fontsize=6, loc='lower center',
           bbox_to_anchor=(0.5, -0.08))
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(f'{OUT}/fig4_frecuencia_quinquenios.png', bbox_inches='tight')
print('ok')
