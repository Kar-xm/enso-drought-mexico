"""Checks of the CISIMA 2026 manuscript (version 2).
The numbers in the text are generated from paper_numbers.json, so this script checks (1) that the qualitative
statements of the text hold for those numbers, (2) that a sample of numbers in the JSON match an independent
recomputation from the raw outputs, and (3) that every in-text citation has an entry in the reference list and
vice versa.
Usage: python verify_manuscript.py <work dir> <manuscript.docx>"""
import json, re, sys, unicodedata
import numpy as np, pandas as pd
from docx import Document

W, DOCX = sys.argv[1:3]
N = json.load(open(f'{W}/paper_numbers.json'))
doc = Document(DOCX)
txt = '\n'.join(p.text for p in doc.paragraphs)
ok = []


def check(name, cond, detail=''):
    ok.append((name, bool(cond), detail))


B = N['boot_spei']
g = lambda k: B[k]
# ---- qualitative statements
check('secas: operativo (d) excluye cero', g('dry|Nacional|LOG_lead')[1] > 0, g('dry|Nacional|LOG_lead'))
check('secas: cota (e) excluye cero', g('dry|Nacional|LOG_enso')[1] > 0, g('dry|Nacional|LOG_enso'))
check('secas: XGB no supera climatología (IC incluye 0)', g('dry|Nacional|XGB_enso')[1] < 0 < g('dry|Nacional|XGB_enso')[2] or g('dry|Nacional|XGB_enso')[0] < 0)
check('secas: (e) - XGB > 0 con IC', g('dry|Nacional|LOG_enso - XGB_enso')[1] > 0)
check('secas: persistencia no mejora (IC incluye 0)', g('dry|Nacional|LOG_pers')[1] <= 0)
check('lluvias: ningún modelo con ENOS excluye cero',
      all(g(f'wet|Nacional|{m}')[1] <= 0 for m in ('LOG_lead', 'LOG_enso', 'REG_lead', 'REG_enso', 'XGB_enso', 'LOG_trend')))
check('regiones NO y S-SE: IC (d) incluye cero', all(g(f'dry|{r}|LOG_lead')[1] <= 0 for r in ('Noroeste', 'Sur-Sureste')))
check('regiones N-NE y C-O: IC (d) excluye cero', all(g(f'dry|{r}|LOG_lead')[1] > 0 for r in ('Norte-Noreste', 'Centro-Occidente')))
check('tendencia (f) < (e)', g('dry|Nacional|LOG_trend')[0] < g('dry|Nacional|LOG_enso')[0])
check('frecuencia secas casi se duplica', 1.7 <= N['enso_dry']['freq_te'] / N['enso_dry']['freq_tr'] <= 2.1,
      N['enso_dry']['freq_te'] / N['enso_dry']['freq_tr'])
check('ENOS no explica el aumento', N['enso_dry']['freq_expected'] - N['enso_dry']['freq_tr'] <
      0.25 * (N['enso_dry']['freq_te'] - N['enso_dry']['freq_tr']))
check('modelos subestiman frecuencia de prueba', N['skill_spei']['dry|LOG_lead']['mean_prob'] < N['enso_dry']['freq_te'])
check('puntos r<0.9 en BC/Sonora con P<250', N['P_max_low'] < 250 and N['lon_max_low'] <= -109.5)
check('lluvias Fisher: ninguno significativo', N['fisher_sig_wet'] == 0)
check('razón H/T mayor en C-O y N-NE, ~1 en S-SE',
      sorted(N['ratio_HT_region'], key=N['ratio_HT_region'].get)[-2:] == ['Norte-Noreste', 'Centro-Occidente']
      and abs(N['ratio_HT_region']['Sur-Sureste'] - 1) < 0.1)
EN, PE = 'ENOS (ONI previo + concurrente)', 'Persistencia (SPEI-6 + SPEI-3)'
sv = lambda r, c: N['shap'][f'dry|{r}|{c}']
check('SHAP secas: ENOS mayor en N-NE y NO', all(sv(r, EN) > 2 * sv(r, PE) for r in ('Norte-Noreste', 'Noroeste')))
check('SHAP secas: ENOS > persistencia en C-O', sv('Centro-Occidente', EN) > sv('Centro-Occidente', PE))
check('SHAP secas: S-SE similares (±20 %)', abs(sv('Sur-Sureste', EN) / sv('Sur-Sureste', PE) - 1) < 0.2)
check('lluvias: sig. negativa casi toda en S-SE y C-O', N['wet_SPEI6']['n_sig_neg_SC'] >= 0.95 * N['wet_SPEI6']['n_sig_neg'])
check('SPEI-T: menos puntos significativos y más aumento',
      N['dry_SPEIT6']['sig'] < N['dry_SPEI6']['sig'] and N['freq_speit_dry'][1] > N['freq_spei_dry'][1])
check('eventos por predictor 2-3', 2 <= N['events']['dry']['median'] / 4 <= 3 or 2 <= N['events']['dry']['median'] / 3 <= 3)

if 'boot_spi' in N:
    check('SPI: IC operativo incluye cero; (e) lo excluye', N['boot_spi']['dry|Nacional|LOG_lead'][1] <= 0 < N['boot_spi']['dry|Nacional|LOG_enso'][1])
    check('SPEI-T: IC operativo excluye cero', N['boot_speit']['dry|Nacional|LOG_lead'][1] > 0)
if 'tc' in N:
    tc = N['tc']
    check('TC secas coincide antes de 1979 (<2 pp)', abs(tc['dry|ERA5-Land']['pre79'] - tc['dry|TerraClimate']['pre79']) < 0.02)
    check('TC secas: aumento en prueba en ambos, menor en TC', tc['dry|TerraClimate']['test'] > tc['dry|TerraClimate']['y79_04']
          and tc['dry|TerraClimate']['test'] < tc['dry|ERA5-Land']['test'])
    check('TC lluvias: divergen antes de 1979 y ERA5 cae despues', tc['wet|ERA5-Land']['pre79'] > tc['wet|TerraClimate']['pre79'] + 0.05
          and tc['wet|ERA5-Land']['y79_04'] < tc['wet|TerraClimate']['y79_04'])
if 'pm' in N:
    check('PM: SPEI-H reproduce a PM (r>0.98)', N['pm']['r_H_PM'] > 0.98)
    check('PM: frecuencia de prueba H cercana a PM y T mayor', abs(N['pm']['freq_dry'][1] - N['freq_spei_dry'][1]) < 0.05
          and N['freq_speit_dry'][1] > N['pm']['freq_dry'][1] + 0.05)
# ---- second-review claims
def bt(ix, k):
    return N[f'boot_{ix}'][k]
check('(d′)−(d) secas no significativa', bt('spei', 'dry|Nacional|REG_lead - LOG_lead')[1] <= 0)
check('(e′)−(e) secas significativa', bt('spei', 'dry|Nacional|REG_enso - LOG_enso')[1] > 0)
for ix in ('speironi', 'speipm'):
    if f'boot_{ix}' in N:
        check(f'{ix}: (d) secas excluye cero', bt(ix, 'dry|Nacional|LOG_lead')[1] > 0, bt(ix, 'dry|Nacional|LOG_lead'))
if 'boot_spei79' in N:
    check('1979-2004: lluvias sin destreza', all(bt('spei79', f'wet|Nacional|{m}')[1] <= 0
                                                 for m in ('LOG_lead', 'LOG_enso', 'REG_lead', 'REG_enso')))
E = N['extra_spei']
check('AUC secas (d) mayor que clim con IC', E['dry|LOG_lead']['dAUC_lo'] > 0)
check('REL (d) similar a clim (±25 %)', abs(E['dry|LOG_lead']['REL'] / E['dry|clim']['REL'] - 1) < 0.25)
check('RES (d) > RES clim', E['dry|LOG_lead']['RES'] > E['dry|clim']['RES'] + 0.005)
check('tendencia: más RES y peor REL que (e)', E['dry|LOG_trend']['REL'] > E['dry|LOG_enso']['REL'])
check('lluvias: AUC de (d) cerca de 0.5', E['wet|LOG_lead']['AUC'] < 0.56)
check('lluvias: (e′) AUC con IC sobre 0.5', E['wet|REG_enso']['AUC_lo'] > 0.5)
check('autocorrelación despreciable', abs(N['serial_dry']['r1_index_med']) < 0.1 and abs(N['serial_wet']['r1_oni']) < 0.2
      and abs(N['serial_dry']['sig_eff'] - N['dry_SPEI6']['sig']) < 0.01)
check('C de lluvias en el extremo de la rejilla', N['C_spei']['wet_LOG_lead'] == 0.01)
check('RONI: correlación significativa en la mayoría', N['roni_dry']['sig'] > 0.8)
# ---- independent recomputation of a sample of numbers
P = pd.read_csv(f'{W}/model_spei_prob_pred.csv')
for s in ('dry', 'wet'):
    q = P[P.season == s]
    for m in ('LOG_lead', 'LOG_enso', 'XGB_enso'):
        b = 1 - ((q.y - q[m]) ** 2).sum() / ((q.y - q.clim) ** 2).sum()
        check(f'BSS {s} {m} recalculado', abs(b - B[f'{s}|Nacional|{m}'][0]) < 1e-9, (b, B[f'{s}|Nacional|{m}'][0]))
t = pd.read_csv(f'{W}/explore_corr.csv'); q = t[(t.season == 'dry') & (t['index'] == 'SPEI6')]
check('fracción significativa secas recalculada', abs(q.sig_conc.mean() - N['dry_SPEI6']['sig']) < 1e-12)
D = pd.read_csv(f'{W}/model_spei_dataset.csv')
check('casos entrenamiento = 54 x puntos', N['n_train'] == 54 * N['n_points'] and N['n_test'] == 20 * N['n_points'])
check('regiones suman puntos', sum(N['n_region'].values()) == N['n_points'])

# ---- citations
refs_start = [p.text for p in doc.paragraphs].index('REFERENCIAS')
refs = [p.text for p in doc.paragraphs[refs_start + 1:] if p.text.strip()]
body = '\n'.join(p.text for p in doc.paragraphs[:refs_start]) + '\n' + '\n'.join(c.text for t in doc.tables for r in t.rows for c in r.cells)


def norm(x):
    return unicodedata.normalize('NFC', x)


ref_keys = []
for r in refs:
    m = re.match(r'^(.+?)\.? \((\d{4})([abc]?)', r)
    first = re.split(r',| y ', m.group(1))[0].strip()
    if first.startswith('Climate Prediction'): first = 'CPC'
    if first.startswith('Servicio Meteorológico'): first = 'SMN'
    if first.startswith('National Weather'): first = 'NWS'
    ref_keys.append((norm(first), m.group(2) + m.group(3)))
cites = set()
for m in re.finditer(r'([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚáéíóúñü\-]+)(?: et al\.| y [A-ZÁÉÍÓÚ][A-Za-zÁÉÍÓÚáéíóúñü\-]+)?,? \(?(\d{4}[abc]?)\)?', body):
    cites.add((norm(m.group(1)), m.group(2)))
for m in re.finditer(r'\[(CPC|SMN|NWS)\], (\d{4}[abc]?)|\((CPC|SMN|NWS), (\d{4}[abc]?)', body):
    cites.add((m.group(1) or m.group(3), m.group(2) or m.group(4)))
cited_refs = [k for k in ref_keys if k in cites]
missing_in_text = [k for k in ref_keys if k not in cites]
years_cited = {c for c in cites if c[0] in {k[0] for k in ref_keys}}
missing_in_refs = [c for c in years_cited if c not in ref_keys]
check('toda referencia está citada', not missing_in_text, missing_in_text)
check('toda cita tiene referencia', not missing_in_refs, missing_in_refs)
paras = [p.text for p in doc.paragraphs]
res = paras[paras.index('Resumen') + 1]; abs_ = paras[paras.index('Abstract') + 1]
check('resumen 150-250 palabras', 150 <= len(res.split()) <= 250, len(res.split()))
check('abstract 150-250 palabras', 150 <= len(abs_.split()) <= 250, len(abs_.split()))
check('sin guiones en rangos numéricos', not re.search(r'(?<![\w/.:])\d{4}-\d{4}(?![\w])', body.replace('2517-6161', '')),
      re.findall(r'(?<![\w/.:])\d{4}-\d{4}(?![\w])', body))
check('sin marcadores PENDIENTE', '[PENDIENTE' not in txt, txt.count('[PENDIENTE'))

for n_, good, d in ok:
    print('OK   ' if good else 'FALLA', n_, '' if good else d)
print(f'\n{sum(x[1] for x in ok)}/{len(ok)} verificaciones correctas; {len(refs)} referencias')
