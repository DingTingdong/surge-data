"""平常時期的今日發動：把所有特徵一起放進模型，能不能事先分出好壞。

全部在看結果之前寫下（2026/11/21 03:45），看到結果後不修改。
樣本：今日發動（v2），訊號日不在恐慌期間。隔天開盤買（開盤一價漲停買不到），持有 20 個交易日，扣 1.1%（＝research/normal_entry.py 的 P1）。
特徵（訊號日收盤後就知道的，約 40 個）：
  價量：當天漲幅、成交額、量能比、前20日均額、波動、近5／20／60日漲幅、起漲幅度、離60／120日高、站上5／20／60日線的距離、
       近60日漲停次數、上次漲停距今、當天上影線、實體、開盤跳空、股價、市值
  籌碼：當天與近5日 外資／投信／三大法人買賣超（換算成相對前20日均額）、融資5日變化
  營收：最近一月年增、近3月平均年增（只用當時已公布的）
  族群（官方產業）：產業指數相對10日線、近5日漲跌、同產業近10日其他今日發動檔數
  大盤：上市空頭家數、等權指數近5／20日、等權指數相對60日線、站上20日線比例
切分：2020–2022 訓練、2023 調參數（只用 2023 挑模型與參數）、2024–2026/11 最後檢驗一次。
模型：梯度提升樹（小網格）與邏輯迴歸，用 2023「分數前20%的平均報酬」挑一個；再用 2020–2023 重新訓練，套到 2024–2026。
分數門檻：2023 預測分數的第 80 百分位（固定，不看 2024–2026）。
判斷：2024–2026 中，分數高於門檻的那群「平均 ≥ +2%、賺錢比例 ≥ 50%、比全部今日發動平均高 ≥ 2 個百分點」→ 模型有用；否則照實報告沒有用。
"""
import glob, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
src = open("research/relaunch.py", encoding="utf-8").read().split("# 穩健性")[0]
for a in ('json.dump({"results"', 'for per in res: print(', 'print(yearly.to_string()); print("加進雷達：", add)', 'print("恐慌期間（True）vs 平常（False）："); print(q.to_string())'):
    src = src.replace(a, "pass  # " if a.startswith(("json", "for")) else "")
exec(src)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

A = P["A"]; raw = P["rawC"]
ind = pd.read_csv("data/industry.csv", dtype={"code": str}).set_index("code")
cap = pd.Series(ind["cap"]).reindex(C.columns); grp = pd.Series(ind["ind_name"]).reindex(C.columns)
inbase = S[S.base].assign(v=True).pivot(index="date", columns="code", values="v").reindex(index=days, columns=C.columns).fillna(False).astype(bool)
# 面板特徵
MA = {n: C.rolling(n, min_periods=n).mean() for n in (5, 20, 60)}
pan = {"r60": C / C.shift(60) - 1, "d_hi60": C / C.rolling(60, min_periods=30).max() - 1, "d_hi120": C / C.rolling(120, min_periods=60).max() - 1,
       "m5": C / MA[5] - 1, "m20": C / MA[20] - 1, "m60": C / MA[60] - 1, "nlim60": (f["ch"] >= 9.5).astype(float).shift(1).rolling(60, min_periods=1).sum(),
       "upper": (H - C) / (H - L).replace(0, np.nan), "body": C / O - 1, "gap": O / C.shift(1) - 1, "lprice": np.log(raw), "lmv": np.log(cap * raw / 10),
       "a20": np.log(A.rolling(20, min_periods=10).mean().shift(1))}
ch = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in sorted(glob.glob("data/chips/*.csv*"), key=lambda x: (os.path.basename(x)[:4], x.endswith(".csv")))], ignore_index=True)
ch["date"] = pd.to_datetime(ch.date); ch = ch.drop_duplicates(["date", "code"], keep="last")
a20 = A.rolling(20, min_periods=10).mean().shift(1)
for k in ("foreign", "trust", "inst"):
    x = ch.pivot(index="date", columns="code", values=k).reindex(index=days, columns=C.columns) * 1000 * raw  # 張 → 元
    pan[k + "1"] = x / a20; pan[k + "5"] = x.rolling(5, min_periods=1).sum() / a20
mg = ch.pivot(index="date", columns="code", values="margin").reindex(index=days, columns=C.columns)
pan["mg5"] = mg / mg.shift(5) - 1
rv = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in sorted(glob.glob("data/revenue/*.csv*"))], ignore_index=True)
rv["pub"] = pd.to_datetime(rv.ym + "-01") + pd.DateOffset(months=1) + pd.Timedelta(days=10)
yoy = rv.pivot_table(index="ym", columns="code", values="yoy").sort_index(); pub = rv.groupby("ym").pub.first().sort_index()
# 族群
r1 = C.pct_change(fill_method=None).where(inbase)
gidx = {}
for g, cs in grp.dropna().groupby(grp.dropna()).groups.items():
    ix = (1 + r1[list(cs)].mean(axis=1).fillna(0)).cumprod(); gidx[g] = (ix / ix.rolling(10).mean() - 1, ix / ix.shift(5) - 1)
pkd = S[S.pk][["date", "code"]].assign(g=lambda z: z.code.map(grp))
# 大盤
ewx = (1 + r1.mean(axis=1).fillna(0)).cumprod()
mkt = pd.DataFrame({"bear": bear, "ew5": ewx / ewx.shift(5) - 1, "ew20": ewx / ewx.shift(20) - 1, "ew_m60": ewx / ewx.rolling(60).mean() - 1,
                    "breadth": (C > MA[20]).where(inbase).mean(axis=1)})
Sx = S.set_index(["date", "code"])
F = B[(B.grp == "F 今日發動") & ~B.date.map(panic).fillna(False).astype(bool)]
rows = []
for r in F.itertuples():
    t, j = pos[r.date], col[r.code]; e = t + 1
    if e + 19 >= N: continue
    o = Ov[e, j]
    if not o == o or o <= 0 or locked(e, j, True): continue
    k = e + 19
    while k > e and not Cv[k, j] == Cv[k, j]: k -= 1
    y = (Cv[k, j] / o - 1) * 100 - COST
    s = Sx.loc[(r.date, r.code)]
    feat = {c: s[c] for c in ("ch", "a1", "a20p", "vr", "v", "r5", "r20", "ru", "snp", "rc")}
    feat["a1"] = np.log(max(feat["a1"], 1e-3)); feat["a20p"] = np.log(max(feat["a20p"], 1e-3))
    for kk, pv in pan.items(): feat[kk] = pv.iat[t, j]
    known = pub[pub <= r.date].index
    feat["yoy1"] = yoy.at[known[-1], r.code] if len(known) and r.code in yoy.columns else np.nan
    feat["yoy3"] = yoy.loc[known[-3:], r.code].mean() if len(known) >= 3 and r.code in yoy.columns else np.nan
    gname = grp.get(r.code)
    if isinstance(gname, str) and gname in gidx:
        feat["g_m10"], feat["g_5"] = gidx[gname][0].iat[t], gidx[gname][1].iat[t]
        feat["g_peers"] = int(((pkd.g == gname) & (pkd.code != r.code) & (pkd.date < r.date) & (pkd.date >= days[max(0, t - 10)])).sum())
    for kk in mkt: feat["mk_" + kk] = mkt[kk].iat[t]
    rows.append({"date": r.date, "code": r.code, "y": y, **feat})
D = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
FEATS = [c for c in D.columns if c not in ("date", "code", "y")]
tr = D[D.date < "2023-01-01"]; va = D[(D.date >= "2023-01-01") & (D.date < "2024-01-01")]; te = D[D.date >= "2024-01-01"]
print("樣本：訓練", len(tr), "調參", len(va), "檢驗", len(te), "特徵", len(FEATS))
def top20(pred, y): q = np.quantile(pred, 0.8); return y[pred >= q].mean()
cands = []
for depth in (3, 5):
    for lr in (0.03, 0.1):
        for leaf in (50, 200):
            m = HistGradientBoostingRegressor(max_depth=depth, learning_rate=lr, max_iter=300, min_samples_leaf=leaf, l2_regularization=1.0, random_state=0)
            m.fit(tr[FEATS], tr.y.clip(-50, 100)); cands.append((top20(m.predict(va[FEATS]), va.y.values), f"GBT d{depth} lr{lr} leaf{leaf}", ("gbt", depth, lr, leaf)))
for Cc in (0.01, 0.1, 1.0):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=Cc, max_iter=2000))
    m.fit(tr[FEATS], tr.y > 0); cands.append((top20(m.predict_proba(va[FEATS])[:, 1], va.y.values), f"Logit C{Cc}", ("lr", Cc)))
cands.sort(key=lambda x: -x[0])
print("2023 調參結果（分數前20%平均報酬）："); [print(f"  {n}: {v:+.2f}") for v, n, _ in cands]
best = cands[0][2]; trva = pd.concat([tr, va])
if best[0] == "gbt":
    model = HistGradientBoostingRegressor(max_depth=best[1], learning_rate=best[2], max_iter=300, min_samples_leaf=best[3], l2_regularization=1.0, random_state=0)
    model.fit(trva[FEATS], trva.y.clip(-50, 100)); score = lambda X: model.predict(X)
    m_va = HistGradientBoostingRegressor(max_depth=best[1], learning_rate=best[2], max_iter=300, min_samples_leaf=best[3], l2_regularization=1.0, random_state=0).fit(tr[FEATS], tr.y.clip(-50, 100))
    thr = np.quantile(m_va.predict(va[FEATS]), 0.8)
else:
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=best[1], max_iter=2000)).fit(trva[FEATS], trva.y > 0)
    score = lambda X: model.predict_proba(X)[:, 1]
    m_va = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=best[1], max_iter=2000)).fit(tr[FEATS], tr.y > 0)
    thr = np.quantile(m_va.predict_proba(va[FEATS])[:, 1], 0.8)
pt = score(te[FEATS]); sel = pt >= thr
allm, topm, topw = te.y.mean(), te.y[sel].mean(), (te.y[sel] > 0).mean() * 100
print(f"== 2024–2026 最後檢驗：全部 {len(te)} 筆平均 {allm:+.2f}%、賺錢 {(te.y > 0).mean() * 100:.1f}%；分數過門檻 {sel.sum()} 筆（{sel.mean() * 100:.0f}%）平均 {topm:+.2f}%、賺錢 {topw:.1f}%")
te = te.assign(score=pt); te["dec"] = pd.qcut(te.score.rank(method="first"), 10, labels=False) + 1
print(te.groupby("dec").y.agg(["size", "mean", "median", lambda v: (v > 0).mean() * 100]).round(2).rename(columns={"<lambda_0>": "賺錢%"}).to_string())
print("逐年（分數過門檻 vs 全部）：", te.assign(s=sel).groupby(te.date.dt.year).apply(lambda g: f"{g[g.s].y.mean():+.2f} vs {g.y.mean():+.2f}（{int(g.s.sum())}筆）", include_groups=False).to_dict())
ok = topm >= 2 and topw >= 50 and topm - allm >= 2
print("模型有用：", ok)
from sklearn.inspection import permutation_importance
pi = permutation_importance(m_va, va[FEATS], va.y.clip(-50, 100) if best[0] == "gbt" else (va.y > 0), n_repeats=5, random_state=0)
imp = pd.Series(pi.importances_mean, index=FEATS).sort_values(ascending=False).head(12).round(4)
print("2023 最有影響的特徵：", imp.to_dict())
json.dump({"best": cands[0][1], "val": [(n, round(v, 2)) for v, n, _ in cands], "test_all": round(allm, 2), "test_top": round(topm, 2), "test_top_win": round(topw, 1),
           "test_n_top": int(sel.sum()), "useful": bool(ok), "importance": imp.to_dict()}, open("research/results/normal_model.json", "w"), ensure_ascii=False, indent=1)
# 解讀：檢驗期分數最高 10% vs 最低 10% 的特徵中位數
lab = {"mk_ew_m60": "大盤等權指數離60日線", "lprice": "股價(元)", "a1": "當天成交額(百萬)", "mk_ew20": "大盤近20日", "lmv": "市值(億)", "yoy1": "最近月營收年增%",
       "d_hi120": "離120日高", "m60": "離60日線", "r20": "近20日漲幅", "g_m10": "族群離10日線", "g_5": "族群近5日", "a20p": "前20日均額(百萬)", "nlim60": "近60日漲停次數", "inst5": "法人5日買超/均額"}
hi, lo_ = te[te.dec == 10], te[te.dec == 1]
tf = lambda k, v: np.exp(v) if k in ("lprice", "a1", "a20p") else (np.exp(v) if k == "lmv" else v)
print(pd.DataFrame({lab[k]: {"分數最高10%": tf(k, hi[k].median()), "分數最低10%": tf(k, lo_[k].median()), "全部": tf(k, te[k].median())} for k in lab}).T.round(3).to_string())
