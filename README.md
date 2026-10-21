# El ENOS y la probabilidad de sequía estacional en México (1951–2024)

Código del artículo *"El ENOS y la probabilidad de sequía estacional en México (1951–2024): evaluación fuera de
muestra con SPEI y modelos interpretables"*, enviado al CISIMA 2026 (revista *Tecnologías Sostenibles y
Aplicaciones Ambientales*).

Autores: Yunué Garza-Pimentel, Marcos Ángel González-Olvera y Jaime Reynaldo Santos-Reyes.

## Qué hace

1. Calcula SPI y SPEI de 3 y 6 meses en 695 puntos de ERA5-Land (1950–2024) con tres métodos de
   evapotranspiración (Hargreaves-Samani, Penman-Monteith FAO-56 y Thornthwaite), ajustados solo con 1951–2004.
2. Relaciona el ONI (y el RONI) con la sequía de secas (nov–abr) y lluvias (may–oct), con corrección de
   Benjamini-Hochberg y tamaño efectivo de muestra.
3. Pronostica retrospectivamente (*hindcast*) la probabilidad de sequía (índice-6 ≤ −1): logísticas por punto
   y con agrupamiento parcial por región, y XGBoost global; entrenamiento 1951–2004, prueba 2005–2024.
4. Evalúa con el índice de destreza de Brier, la descomposición de Murphy y el AUC, con intervalos bootstrap
   por bloques de años.
5. Compara la frecuencia de sequía con TerraClimate (precipitación basada en estaciones).

## Requisitos

Python 3.10 o superior y `pip install numpy pandas scipy scikit-learn xgboost matplotlib pyshp python-docx`.

## Datos de entrada (no incluidos; se obtienen con los scripts de `gee/`)

| Archivo | Origen |
|---|---|
| `ERA5/ERA5_Mexico_*.csv` | precipitación y temperatura mensual de ERA5-Land (colección mensual de Earth Engine) |
| `tmaxmin_all.csv` | `gee/extraer_tmax_tmin_era5land.js` (4 CSV unidos) |
| `pm_all.csv` | `gee/extraer_pm_era5land.js` (4 CSV unidos, desde 1950-02) |
| `tc_in/TerraClimate_pr_*.csv` | `gee/extraer_terraclimate_pr.js` |
| `oni.ascii.txt`, `RONI.ascii.txt` | https://www.cpc.ncep.noaa.gov/data/indices/ |

## Orden de ejecución (desde la carpeta de trabajo)

```bash
python codigo/indices_national.py ERA5 tmaxmin_all.csv indices_national.pkl pm_all.csv
python codigo/enso_explore.py indices_national.pkl oni.ascii.txt explore
python codigo/enso_explore.py indices_national.pkl RONI.ascii.txt explore_roni
# para cada índice: SPEI -> model_spei, SPI -> model_spi, SPEIT -> model_speit, SPEIPM -> model_speipm
python codigo/build_dataset.py indices_national.pkl oni.ascii.txt SPEI model_spei
python codigo/drought_prob.py model_spei_dataset.csv model_spei              # ~20 min
python codigo/skill_bootstrap.py model_spei_prob_pred.csv model_spei_boot.csv 5000
python codigo/skill_extra.py model_spei_prob_pred.csv model_spei_extra.csv 1000
# sensibilidad con RONI y con entrenamiento 1979-2004
python codigo/build_dataset.py indices_national.pkl RONI.ascii.txt SPEI model_speironi
python codigo/drought_prob.py model_speironi_dataset.csv model_speironi
python codigo/drought_prob.py model_spei_dataset.csv model_spei79 --train-start=1979
python codigo/pm_summary.py .
python codigo/tc_compare.py ERA5 tc_in .
python codigo/paper_figures.py . figuras
python codigo/xgb_base_score.py model_spei_dataset.csv model_spei xgb_base_score.json   # valor inicial de XGBoost (g)
python codigo/paper_extra.py . ERA5 figuras        # diagnósticos de la 4.ª revisión (sin reajustar modelos)
python codigo/paper_numbers.py . paper_numbers.json   # todas las cifras del manuscrito
python codigo/verify_manuscript.py . manuscrito.docx  # comprobaciones del texto
```

Módulos auxiliares: `indices_final.py` (Hargreaves, radiación extraterrestre, logística generalizada con
L-momentos) y `pm_eto.py` (FAO-56 Penman-Monteith; reproduce el ejemplo 17 de FAO-56).
`basemap/`: contorno de países de Natural Earth 1:110m (dominio público).

## Licencia

MIT (ver `LICENSE`).
