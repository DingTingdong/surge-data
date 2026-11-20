"""平常時期（非恐慌）的今日發動：換一種賣法會不會變成正報酬。

全部在看結果之前寫下（2026/11/21 01:55），看到結果後不修改。
樣本：今日發動（v2），訊號日不在恐慌期間（上市空頭家數近10日最高 < 750）。隔天開盤買（開盤一價漲停買不到）。
「收盤觸發」的規則：當天收盤判斷，隔天開盤賣；賣出日開盤一價跌停則順延（最多 5 天）。每筆扣 1.1%（分批賣的每一半也各自扣）。最多 20 個交易日。
  X1 固定 5 日　X2 固定 10 日　X3 固定 20 日
  X6 移動停損 10%（現行：收盤比持有期間最高收盤低 10%）
  X7 停損 −8% 或停利 +20%（相對買進價）
  X8 漲到 +20%（收盤）先賣一半，剩下照 X6
  X9 收盤比買進價低 5% 就賣，否則照 X6
  X10 第一天收盤低於買進價就賣，否則照 X6
  X11 移動停損 7%
期間：2020–2023、2024–2026/11。
判斷：某規則在兩段期間「每筆平均 > 0」而且「都比 X6 好」→ 平常時期有可用的賣法；否則結論是平常時期換賣法也沒用。全部照實報告。
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
src = open("research/relaunch.py", encoding="utf-8").read().split("# 穩健性")[0]
for a in ('json.dump({"results"', 'for per in res: print(', 'print(yearly.to_string()); print("加進雷達：", add)', 'print("恐慌期間（True）vs 平常（False）："); print(q.to_string())'):
    src = src.replace(a, "pass  # " if a.startswith(("json", "for")) else "")
exec(src)

def sell_open(k, j):
    for s in range(k, min(k + 6, N)):
        if Ov[s, j] == Ov[s, j] and not locked(s, j, False): return Ov[s, j]
    return None

def last_close(k, e, j):
    while k > e and not Cv[k, j] == Cv[k, j]: k -= 1
    return Cv[k, j]

def run_rule(t, j, rule):
    e = t + 1; end = e + CAP - 1
    if end >= N: return None
    o = Ov[e, j]
    if not o == o or o <= 0: return None
    if locked(e, j, True): return "blocked"
    fixed = {"X1": 5, "X2": 10, "X3": 20}.get(rule)
    if fixed: return (last_close(e + fixed - 1, e, j) / o - 1) * 100 - COST
    peak, half = o, None
    trail = 0.93 if rule == "X11" else 0.9
    for k in range(e, end + 1):
        c = Cv[k, j]
        if not c == c: continue
        peak = max(peak, c)
        if k == end: break
        if rule == "X8" and half is None and c >= o * 1.2:
            px = sell_open(k + 1, j)
            if px is None: return None
            half = (px / o - 1) * 100 - COST
            continue
        hit = (c <= peak * trail) if rule in ("X6", "X8", "X9", "X10", "X11") else False
        if rule == "X7": hit = c <= o * 0.92 or c >= o * 1.2
        if rule == "X9": hit = hit or c <= o * 0.95
        if rule == "X10" and k == e: hit = hit or c < o
        if hit:
            px = sell_open(k + 1, j)
            if px is None: return None
            r = (px / o - 1) * 100 - COST
            return r if half is None else (half + r) / 2
    r = (last_close(end, e, j) / o - 1) * 100 - COST
    return r if half is None else (half + r) / 2

RULES = ["X1", "X2", "X3", "X6", "X7", "X8", "X9", "X10", "X11"]
F = B[(B.grp == "F 今日發動") & ~B.date.map(panic).fillna(False).astype(bool)]
rows = []
for r in F.itertuples():
    t, j = pos[r.date], col[r.code]
    for rule in RULES:
        x = run_rule(t, j, rule)
        if x is None or x == "blocked": continue
        rows.append({"date": r.date, "train": r.date <= pd.Timestamp("2023-12-31"), "rule": rule, "ret": x})
T = pd.DataFrame(rows)
res = {}
for tr, per in ((True, "2020-2023"), (False, "2024-2026")):
    g = T[T.train == tr]
    res[per] = g.groupby("rule").ret.agg(n="size", 平均="mean", 中位數="median", 賺錢比例=lambda v: (v > 0).mean() * 100).round(2).reindex(RULES)
    print("==", per); print(res[per].to_string())
good = [r for r in RULES if r != "X6" and all(res[p].at[r, "平均"] > 0 and res[p].at[r, "平均"] > res[p].at["X6", "平均"] for p in res)]
print("平常時期可用的賣法：", good or "沒有")
import json
json.dump({p: res[p].reset_index().to_dict("records") for p in res} | {"good": good}, open("research/results/normal_exits.json", "w"), ensure_ascii=False, indent=1)
