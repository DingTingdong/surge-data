"""主線：「再發動」（前幾天漲停過、拉回後又漲停）要不要加進雷達。

全部在看結果之前寫下（2026/11/20 23:30），看到結果後不修改。
起因：11/20 分析範圍內 8 檔漲停（華經、資通、意藍、中華資安…），今日發動 0 檔——它們 10 日內漲停過，被「第一根漲停」規則排除。

事件（分析範圍內、當天漲 ≥ 9%、當天成交額 ≥ 2,000 萬，和今日發動相同的量能門檻）：
  F  今日發動（現行）：前 10 日沒有漲 ≥ 9%（snp > 10），前 20 日均額 ≥ 300 萬
  R  再發動：前 10 日內漲 ≥ 9% 過（snp ≤ 10），而且今天之前曾從近 20 日最高收盤拉回 ≥ 15%（前 5 日最低收盤 ≤ 近 20 日最高 × 0.85）
  K  連續強勢：前 10 日內漲 ≥ 9% 過，但沒有拉回 15%（其餘的漲停）
結果：隔天開盤買（開盤一價漲停買不到，不計），移動停損出場（research/exits.py 的 X6：收盤比買進後最高收盤低 10% 隔天開盤賣，最多 20 日），扣 1.1%；
  另報 5 日報酬、20 日內最高收盤 ≥ +20% 的比例、買不到的比例、每天平均檔數。
期間：2020–2023 與 2024–2026/11。
判斷（雷達的目的是「最快發現已經發動的股票」，所以標準是「不比今日發動差」）：
  R 在兩段期間的 X6 平均都不低於 F − 0.5 個百分點 → 加進雷達成為「再發動」；否則不加，結果照實報告。K 只報告。
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radar_core import load_prices, panels, features, tiers

COST, CAP, LOCK = 1.1, 20, 0.095
p = load_prices(); P = panels(p); C, O, H, L = P["C"], P["O"], P["H"], P["L"]
f = features(P); S = tiers(f, P["MK"])
days = C.index; N = len(days)
Cv, Ov, Hv, Lv = C.values, O.values, H.values, L.values
pos = {d: i for i, d in enumerate(days)}; col = {c: j for j, c in enumerate(C.columns)}
hi20 = C.rolling(20, min_periods=10).max().shift(1); lo5 = C.rolling(5, min_periods=3).min().shift(1)
pull = (lo5 <= hi20 * 0.85)

def locked(i, j, up):
    o, h, l = Ov[i, j], Hv[i, j], Lv[i, j]; k = i - 1
    while k >= 0 and not Cv[k, j] == Cv[k, j]: k -= 1
    if k < 0 or not (o == o and h == h and l == l): return False
    r = o / Cv[k, j] - 1
    return o == h == l and (r >= LOCK if up else r <= -LOCK)

def x6(t, j):
    e = t + 1; end = e + CAP - 1
    if end >= N: return None
    o = Ov[e, j]
    if not o == o or o <= 0: return None
    if locked(e, j, True): return "blocked"
    peak = o
    for k in range(e, end + 1):
        c = Cv[k, j]
        if not c == c: continue
        peak = max(peak, c)
        if c <= peak * 0.9 and k < end:
            for s in range(k + 1, min(k + 6, N)):
                if Ov[s, j] == Ov[s, j] and not locked(s, j, False): return (Ov[s, j] / o - 1) * 100 - COST
            return None
    k = end
    while k > e and not Cv[k, j] == Cv[k, j]: k -= 1
    return (Cv[k, j] / o - 1) * 100 - COST

B = S[S.base & (S.ch >= 9) & (S.a1 >= 20)].copy()
B["pull"] = [bool(pull.at[d, c]) if c in pull.columns else False for d, c in zip(B.date, B.code)]
B["grp"] = np.select([B.pk, (B.snp <= 10) & B.pull, B.snp <= 10], ["F 今日發動", "R 再發動", "K 連續強勢"], "")
B = B[B.grp != ""]
rows = []
for r in B.itertuples():
    t, j = pos[r.date], col[r.code]
    x = x6(t, j)
    if x is None: continue
    if x == "blocked": rows.append({"date": r.date, "grp": r.grp, "blocked": True}); continue
    o = Ov[t + 1, j]
    rows.append({"date": r.date, "grp": r.grp, "blocked": False, "x6": x, "r5": (Cv[min(t + 5, N - 1), j] / o - 1) * 100 - COST,
                 "hit20": float(np.nanmax(Cv[t + 1:t + 21, j]) / o - 1 >= 0.2)})
T = pd.DataFrame(rows); T["train"] = T.date <= "2023-12-31"
res = {}
for tr, per in ((True, "2020-2023"), (False, "2024-2026")):
    g0 = T[T.train == tr]; ndays = g0.date.nunique()
    res[per] = {}
    for k, g in g0.groupby("grp"):
        ok = g[~g.blocked]
        res[per][k] = {"n": int(len(ok)), "每天平均檔數": round(len(g) / max(1, ndays), 2), "買不到%": round(g.blocked.mean() * 100, 1),
                       "X6平均": round(ok.x6.mean(), 2), "X6中位數": round(ok.x6.median(), 2), "5日平均": round(ok.r5.mean(), 2),
                       "20日內漲20%以上%": round(ok.hit20.astype(float).mean() * 100, 1)}
add = all(res[p]["R 再發動"]["X6平均"] >= res[p]["F 今日發動"]["X6平均"] - 0.5 for p in res)
yearly = T[~T.blocked].groupby([T.date.dt.year, "grp"]).x6.mean().unstack().round(2)
import json
json.dump({"results": res, "add_relaunch": add, "yearly": yearly.to_dict()}, open("research/results/relaunch.json", "w"), ensure_ascii=False, indent=1, default=str)
for per in res: print("==", per); print(pd.DataFrame(res[per]).T.to_string())
print(yearly.to_string()); print("加進雷達：", add)

# ---- 探索（看過上面結果後才加）：恐慌期間（上市空頭家數近10日最高 ≥ 750）發生的漲停，表現是否不同 ----
alive = (C.notna().cumsum() > 0) & (C.notna()[::-1].cumsum()[::-1] > 0)
Cf = C.ffill(limit=20).where(alive)
mm = {n: Cf.rolling(n, min_periods=n).mean() for n in (5, 10, 20, 60)}
okm = Cf.notna() & mm[60].notna()
mk = P["MK"].ffill().iloc[-1]; tw = [c for c in C.columns if mk.get(c) == "TWSE"]
bear = ((mm[5] < mm[10]) & (mm[10] < mm[20]) & okm)[tw].sum(axis=1)
panic = bear.rolling(10, min_periods=1).max() >= 750
T["panic"] = T.date.map(panic)
q = T[~T.blocked].groupby(["grp", "panic"]).agg(n=("x6", "size"), X6平均=("x6", "mean"), X6中位數=("x6", "median"), 漲20比例=("hit20", "mean")).round(2)
print("恐慌期間（True）vs 平常（False）："); print(q.to_string())
# 穩健性：分期間、分「恐慌波段」
g = T[~T.blocked & T.panic].copy()
wave_id = (panic != panic.shift()).cumsum()
g["wave"] = g.date.map(wave_id)
w = g.groupby(["wave"]).agg(起=("date", "min"), 迄=("date", "max")).reset_index()
pw = g.pivot_table(index="wave", columns="grp", values="x6", aggfunc="mean").round(1)
pn = g.pivot_table(index="wave", columns="grp", values="x6", aggfunc="size")
W = w.set_index("wave").join(pw).join(pn, rsuffix="_n")
W["起"] = W["起"].dt.strftime("%Y-%m-%d"); W["迄"] = W["迄"].dt.strftime("%Y-%m-%d")
print(W.to_string())
print("F 今日發動 恐慌波段數", int(W["F 今日發動"].notna().sum()), "為正", int((W["F 今日發動"] > 0).sum()))
for tr, per in ((True, "2020-2023"), (False, "2024-2026")):
    z = T[~T.blocked & (T.train == tr)]
    print(per, z.groupby(["grp", "panic"]).x6.agg(["size", "mean", "median"]).round(2).loc["F 今日發動"].to_dict())
