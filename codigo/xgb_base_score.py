"""Initial value (base_score) of the XGBoost models (g), requested in the fifth review.
The final booster of drought_prob.py is not saved, so it is rebuilt here with exactly the same call
(xgb.train with the depth and number of trees chosen by the year-blocked CV, read from <pref>_prob_cv.csv),
and its test predictions are compared with the saved ones (<pref>_prob_pred.csv, column XGB_enso).
Only if they coincide is the base_score reported. No hyperparameter is re-selected.
Usage: python xgb_base_score.py <pref>_dataset.csv <pref> <out.json>"""
import json, os, sys
import numpy as np, pandas as pd
import xgboost as xgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import TRAIN, TEST
from drought_prob import XF, XP, THR

src, pref, out = sys.argv[1:4]
D = pd.read_csv(src); D['y'] = (D.target <= THR).astype(int)
cv = pd.read_csv(f'{pref}_prob_cv.csv'); cv = cv[(cv.model == 'XGB_enso') & cv.chosen].set_index('season')
P = pd.read_csv(f'{pref}_prob_pred.csv')
R = {'xgboost_version': xgb.__version__}
for season, g in D.groupby('season'):
    tr, te = g[g.year.between(*TRAIN)], g[g.year.between(*TEST)]
    depth, trees = int(cv.loc[season, 'depth']), int(cv.loc[season, 'trees'])
    bst = xgb.train(dict(XP, max_depth=depth), xgb.DMatrix(tr[XF], tr.y), trees)
    p = bst.predict(xgb.DMatrix(te[XF]))
    saved = te[['lon', 'lat', 'year']].assign(p=p).merge(P[P.season == season][['lon', 'lat', 'year', 'XGB_enso']],
                                                        on=['lon', 'lat', 'year'])
    diff = float(np.abs(saved.p - saved.XGB_enso).max())
    base = float(json.loads(bst.save_config())['learner']['learner_model_param']['base_score'].strip('[]'))
    R[season] = dict(depth=depth, trees=trees, base_score=base, train_freq=float(tr.y.mean()),
                     max_abs_diff_vs_saved=diff, reproduced=diff < 1e-5,
                     pred_min=float(p.min()), pred_max=float(p.max()), pred_sd=float(p.std()))
json.dump(R, open(out, 'w'), indent=1)
print(json.dumps(R, indent=1))
