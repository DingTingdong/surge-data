"""恐慌期間：今日發動（＋再發動）和撿跌深前20，哪個比較有用（2026/11/21 使用者問；探索）。

恐慌期間＝上市空頭家數近 10 日最高 ≥ 750（同網站）。恐慌期間的每一天：
  今日發動＋再發動：當天的訊號（research/relaunch.py 的 F、R）
  撿跌深前20：當天用網站現行定義選出（自己的漲勢前30%＋近60日漲停過＋從近20日高跌 ≥20%，依漲勢取前20）
同一套買賣：隔天開盤買（開盤一價漲停買不到），① 移動停損 X6（扣 1.1%）② 固定抱 20 日（扣 1.1%）。
看：每筆平均、中位數、賺錢比例；每次恐慌（波段）的平均；每次恐慌有幾筆機會。
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
src = open("research/relaunch.py", encoding="utf-8").read().split("# 穩健性")[0]
for a in ('json.dump({"results"', 'for per in res: print(', 'print(yearly.to_string()); print("加進雷達：", add)', 'print("恐慌期間（True）vs 平常（False）："); print(q.to_string())'):
    src = src.replace(a, "pass  # " if a.startswith(("json", "for")) else "")
exec(src)
hi20 = C.rolling(20, min_periods=15).max(); lo = C.shift(20).rolling(81, min_periods=40).min()
run = hi20 / lo - 1
inbase = S[S.base].assign(v=True).pivot(index="date", columns="code", values="v").reindex(index=days, columns=C.columns).fillna(False).astype(bool)
runr = run.where(inbase).rank(axis=1, pct=True)
lim60 = (f["ch"] >= 9.5).rolling(60, min_periods=1).max().astype(bool)
deep = (C / hi20 - 1 <= -0.2)
W2 = (runr > 0.7) & lim60 & deep & inbase
wave_id = (panic != panic.shift()).cumsum()
def hold20(t, j):
    e = t + 1
    if e + 19 >= N: return None
    o = Ov[e, j]
    if not o == o or o <= 0: return None
    if locked(e, j, True): return "blocked"
    k = e + 19
    while k > e and not Cv[k, j] == Cv[k, j]: k -= 1
    return (Cv[k, j] / o - 1) * 100 - COST
rows = []
for d in days[panic.values]:
    t = pos[d]
    if t + 21 >= N: continue
    sig = B[(B.date == d) & B.grp.isin(["F 今日發動", "R 再發動"])].code
    top = run.loc[d][W2.loc[d]].sort_values(ascending=False).index[:20]
    for kind, codes in (("今日發動＋再發動", sig), ("撿跌深前20", top)):
        for c in codes:
            j = col[c]; a, b = x6(t, j), hold20(t, j)
            if a is None or b is None: continue
            rows.append({"date": d, "wave": wave_id[d], "kind": kind, "blocked": a == "blocked", "x6": np.nan if a == "blocked" else a, "h20": np.nan if b == "blocked" else b})
T = pd.DataFrame(rows)
ok = T[~T.blocked]
s = ok.groupby("kind").agg(筆數=("x6", "size"), X6平均=("x6", "mean"), X6中位數=("x6", "median"), X6賺錢比例=("x6", lambda v: (v > 0).mean() * 100),
                          抱20日平均=("h20", "mean"), 抱20日中位數=("h20", "median"), 抱20日賺錢比例=("h20", lambda v: (v > 0).mean() * 100)).round(1)
s["買不到%"] = (T.groupby("kind").blocked.mean() * 100).round(1)
print(s.to_string())
w = ok.groupby(["wave", "kind"]).agg(n=("x6", "size"), x6=("x6", "mean"), h20=("h20", "mean")).round(1).unstack()
w.index = [ok[ok.wave == i].date.min().strftime("%Y-%m") for i in w.index]
print(w.to_string())
print("每次恐慌平均（不論筆數多寡，每次權重一樣）:", ok.groupby(["wave", "kind"]).x6.mean().groupby("kind").mean().round(2).to_dict(), ok.groupby(["wave", "kind"]).h20.mean().groupby("kind").mean().round(2).to_dict())
T.to_csv("research/results/panic_compare.csv", index=False)
