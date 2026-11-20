"""平常時期的今日發動：發動後等回檔再買，會不會比隔天開盤追好。

全部在看結果之前寫下（2026/11/21 03:30），看到結果後不修改。
樣本：今日發動（v2），訊號日不在恐慌期間（上市空頭家數近10日最高 < 750）。
進場（訊號日之後 10 個交易日內；沒觸發就不買，另報觸發比例）：
  P1 隔天開盤買（現行；開盤一價漲停買不到）
  P2 收盤跌到 5 日線以下 → 隔天開盤買
  P3 收盤跌回發動當天收盤價以下 → 隔天開盤買
  P4 掛低 3% 的限價單：盤中最低價 ≤ 發動日收盤 × 0.97 → 以發動日收盤 × 0.97 成交（若當天開盤就更低，以開盤價成交）
  P5 確認後再買：發動隔天收盤仍高於發動日收盤 → 再隔天開盤買
出場：買進後持有 20 個交易日收盤賣；扣 1.1%。
期間：2020–2023、2024–2026/11。
判斷：某進場方式在兩段期間「每筆平均 > 0」而且「都比 P1 高 ≥ 1 個百分點」→ 有用。另報賺錢比例。
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
src = open("research/relaunch.py", encoding="utf-8").read().split("# 穩健性")[0]
for a in ('json.dump({"results"', 'for per in res: print(', 'print(yearly.to_string()); print("加進雷達：", add)', 'print("恐慌期間（True）vs 平常（False）："); print(q.to_string())'):
    src = src.replace(a, "pass  # " if a.startswith(("json", "for")) else "")
exec(src)
ma5 = C.rolling(5, min_periods=5).mean().values
def hold(e, j, price):
    end = e + 19
    if end >= N: return None
    k = end
    while k > e and not Cv[k, j] == Cv[k, j]: k -= 1
    return (Cv[k, j] / price - 1) * 100 - COST
rows = []
F = B[(B.grp == "F 今日發動") & ~B.date.map(panic).fillna(False).astype(bool)]
for r in F.itertuples():
    t, j = pos[r.date], col[r.code]
    if t + 31 >= N: continue
    c0 = Cv[t, j]; res = {"date": r.date, "train": r.date <= pd.Timestamp("2023-12-31")}
    o = Ov[t + 1, j]
    res["P1"] = None if (not o == o or locked(t + 1, j, True)) else hold(t + 1, j, o)
    for name, cond in (("P2", lambda k: Cv[k, j] < ma5[k, j]), ("P3", lambda k: Cv[k, j] < c0)):
        res[name] = None
        for k in range(t + 1, t + 11):
            if Cv[k, j] == Cv[k, j] and cond(k):
                e = k + 1; oo = Ov[e, j]
                if oo == oo and not locked(e, j, True): res[name] = hold(e, j, oo)
                break
    res["P4"] = None
    for k in range(t + 1, t + 11):
        lim_px = c0 * 0.97
        if Lv[k, j] == Lv[k, j] and Lv[k, j] <= lim_px:
            px = min(lim_px, Ov[k, j]) if Ov[k, j] == Ov[k, j] else lim_px
            res["P4"] = hold(k, j, px); break
    res["P5"] = None
    if Cv[t + 1, j] == Cv[t + 1, j] and Cv[t + 1, j] > c0:
        e = t + 2; oo = Ov[e, j]
        if oo == oo and not locked(e, j, True): res["P5"] = hold(e, j, oo)
    rows.append(res)
T = pd.DataFrame(rows)
out = {}
for tr, per in ((True, "2020-2023"), (False, "2024-2026")):
    g = T[T.train == tr]; tab = {}
    for k in ("P1", "P2", "P3", "P4", "P5"):
        v = pd.to_numeric(g[k], errors="coerce").dropna()
        tab[k] = {"觸發%": round(len(v) / len(g) * 100, 1), "n": len(v), "平均": round(v.mean(), 2), "中位數": round(v.median(), 2), "賺錢%": round((v > 0).mean() * 100, 1)}
    out[per] = tab; print("==", per); print(pd.DataFrame(tab).T.to_string())
good = [k for k in ("P2", "P3", "P4", "P5") if all(out[p][k]["平均"] > 0 and out[p][k]["平均"] - out[p]["P1"]["平均"] >= 1 for p in out)]
print("有用的進場方式：", good or "沒有")
import json; json.dump(out | {"good": good}, open("research/results/normal_entry.json", "w"), ensure_ascii=False, indent=1)
