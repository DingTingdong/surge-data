"""訓練並凍結「平常時期今日發動」評分模型 v1（規格同 research/normal_model.py 事先登錄、2023 挑出的 GBT d3 lr0.1 leaf50）。
2020–2023 訓練；分數門檻用 2020–2022 模型在 2023 的預測：第 80 百分位以上＝高、第 20 百分位以下＝低。
輸出 data/model/normal_v1.pkl 與 data/model/normal_v1.json。本機執行一次即可（之後凍結不再改，給前瞻驗證）。"""
import json, os, pickle, sys
import numpy as np, pandas as pd, sklearn
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
src = open("research/relaunch.py", encoding="utf-8").read().split("# 穩健性")[0]
for a in ('json.dump({"results"', 'for per in res: print(', 'print(yearly.to_string()); print("加進雷達：", add)', 'print("恐慌期間（True）vs 平常（False）："); print(q.to_string())'):
    src = src.replace(a, "pass  # " if a.startswith(("json", "for")) else "")
exec(src)
from model_features import make_context, features_for
from sklearn.ensemble import HistGradientBoostingRegressor
ctx = make_context(P, f, S, bear)
F = B[(B.grp == "F 今日發動") & ~B.date.map(panic).fillna(False).astype(bool)]
rows = []
for r in F.itertuples():
    t, j = pos[r.date], col[r.code]; e = t + 1
    if e + 19 >= N: continue
    o = Ov[e, j]
    if not o == o or o <= 0 or locked(e, j, True): continue
    k = e + 19
    while k > e and not Cv[k, j] == Cv[k, j]: k -= 1
    rows.append({"date": r.date, "code": r.code, "y": (Cv[k, j] / o - 1) * 100 - COST, **features_for(ctx, r.date, r.code)})
D = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
FEATS = [c for c in D.columns if c not in ("date", "code", "y")]
spec = dict(max_depth=3, learning_rate=0.1, max_iter=300, min_samples_leaf=50, l2_regularization=1.0, random_state=0)
tr, va, te = D[D.date < "2023-01-01"], D[(D.date >= "2023-01-01") & (D.date < "2024-01-01")], D[D.date >= "2024-01-01"]
m_va = HistGradientBoostingRegressor(**spec).fit(tr[FEATS], tr.y.clip(-50, 100))
pv = m_va.predict(va[FEATS]); hi, lo = float(np.quantile(pv, 0.8)), float(np.quantile(pv, 0.2))
final = HistGradientBoostingRegressor(**spec).fit(pd.concat([tr, va])[FEATS], pd.concat([tr, va]).y.clip(-50, 100))
pt = final.predict(te[FEATS])
print("重現檢驗：全部 %.2f、高 %.2f（%d 筆，賺錢 %.1f%%）、低 %.2f（%d 筆）" % (te.y.mean(), te.y[pt >= hi].mean(), (pt >= hi).sum(), (te.y[pt >= hi] > 0).mean() * 100, te.y[pt <= lo].mean(), (pt <= lo).sum()))
pickle.dump({"model": final, "feats": FEATS}, open("data/model/normal_v1.pkl", "wb"))
json.dump({"version": "normal_v1", "trained": "2020-2023", "spec": spec, "hi": hi, "lo": lo, "feats": FEATS, "sklearn": sklearn.__version__,
           "test_2024_2026": {"all": round(te.y.mean(), 2), "hi": round(te.y[pt >= hi].mean(), 2), "lo": round(te.y[pt <= lo].mean(), 2)}},
          open("data/model/normal_v1.json", "w"), ensure_ascii=False, indent=1)
print("saved", len(FEATS), "features; sklearn", sklearn.__version__)
