"""出場方式研究：買進「今日發動」之後，怎麼賣比較好。

全部在跑之前寫下（2026/11/17 18:00），看到結果後不修改規則與挑選方式。

進場：今日發動（v2）訊號隔天開盤買。開盤一價漲停（開＝高＝低且比前收漲 ≥ 9.5%）視為買不到，不計入。
出場規則（「收盤觸發」的規則：當天收盤判斷，隔天開盤賣；進場當天收盤就開始判斷；最多持有20個交易日，第20天收盤賣）：
  X1 固定持有5天：第5個交易日收盤賣
  X2 固定持有10天
  X3 固定持有20天
  X4 收盤跌破5日均線 → 隔天開盤賣
  X5 收盤跌破10日均線 → 隔天開盤賣
  X6 移動停損：收盤比持有期間最高收盤低 10% 以上 → 隔天開盤賣
  X7 停損停利：收盤比進場價低 8% 以上，或高 20% 以上 → 隔天開盤賣
賣出日開盤一價跌停（開＝高＝低且比前收跌 ≥ 9.5%）→ 順延到下一個交易日開盤，最多順延5天。
價格用除權息／減資調整後的價格。每筆扣 1.1%（手續費、證交稅、滑價）。

期間：訊號日 2020–2023 挑選，2024–2026/11 驗證。
報告：每筆平均、中位數、勝率、平均持有天數、最差5%平均、每持有一天的平均報酬、賺/賠 20% 以上比例。
挑選規則：在 2020–2023，選「每筆平均報酬」最高的規則；但如果它的「最差5%平均」比 X1 差超過 3 個百分點，
就改選次高且不超過的。驗證期照實報告全部規則。
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radar_core import load_prices, panels, features, tiers

COST, CAP, LOCK = 1.1, 20, 0.095
TRAIN_END = pd.Timestamp("2023-12-31")
p = load_prices()
P = panels(p)
C, O, H, L = P["C"], P["O"], P["H"], P["L"]
f = features(P)
S = tiers(f, P["MK"])
days = C.index; N = len(days)
Cv, Ov, Hv, Lv = C.values, O.values, H.values, L.values
ma5 = C.rolling(5, min_periods=5).mean().values
ma10 = C.rolling(10, min_periods=10).mean().values
pos = {d: i for i, d in enumerate(days)}
col = {c: j for j, c in enumerate(C.columns)}

def locked(i, j, up):
    o, h, l = Ov[i, j], Hv[i, j], Lv[i, j]
    k = i - 1
    while k >= 0 and not Cv[k, j] == Cv[k, j]: k -= 1
    if k < 0 or not (o == o and h == h and l == l): return False
    ch = o / Cv[k, j] - 1
    return o == h == l and (ch >= LOCK if up else ch <= -LOCK)

def sell_open(i, j):
    """從第 i 天開盤賣；跌停鎖死順延。回傳 (價格, 日索引)。"""
    for k in range(i, min(i + 6, N)):
        if Ov[k, j] == Ov[k, j] and not locked(k, j, False): return Ov[k, j], k
    return None, None

def last_close(i, j):
    k = i
    while k >= 0 and not Cv[k, j] == Cv[k, j]: k -= 1
    return Cv[k, j] if k >= 0 else np.nan

RULES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7"]
def trade(t, j, rule):
    e = t + 1
    if e >= N: return None
    o = Ov[e, j]
    if not o == o or o <= 0 or locked(e, j, True): return "blocked"
    fixed = {"X1": 5, "X2": 10, "X3": 20}.get(rule)
    end = e + (fixed or CAP) - 1
    if end >= N: return None  # 結果還沒出來
    if fixed: return (last_close(end, j) / o - 1) * 100 - COST, fixed
    peak = o
    for k in range(e, end + 1):
        c = Cv[k, j]
        if not c == c: continue
        peak = max(peak, c)
        hit = ((rule == "X4" and ma5[k, j] == ma5[k, j] and c < ma5[k, j]) or
               (rule == "X5" and ma10[k, j] == ma10[k, j] and c < ma10[k, j]) or
               (rule == "X6" and c <= peak * 0.9) or
               (rule == "X7" and (c <= o * 0.92 or c >= o * 1.2)))
        if hit and k < end:
            px, kk = sell_open(k + 1, j)
            if px is None: return None
            return (px / o - 1) * 100 - COST, kk - e + 1
    return (last_close(end, j) / o - 1) * 100 - COST, end - e + 1

sig = S[S.pk][["date", "code"]]
rows, blocked = [], {True: 0, False: 0}
for d, c in zip(sig.date, sig.code):
    t, j = pos[d], col[c]; tr = d <= TRAIN_END; yr = d.year
    first = trade(t, j, "X1")
    if first == "blocked": blocked[tr] += 1; continue
    if first is None: continue
    for r in RULES:
        res = trade(t, j, r)
        if isinstance(res, tuple): rows.append({"train": tr, "year": yr, "rule": r, "ret": res[0], "days": res[1]})
T = pd.DataFrame(rows)

def stats(g):
    r = g.ret; q = r.quantile(0.05)
    return {"n": int(len(r)), "avg": round(r.mean(), 2), "med": round(r.median(), 2), "win": round((r > 0).mean() * 100, 1),
            "hold": round(g.days.mean(), 1), "per_day": round(r.mean() / g.days.mean(), 3), "w5": round(r[r <= q].mean(), 1),
            "up20": round((r >= 20).mean() * 100, 1), "dn20": round((r <= -20).mean() * 100, 1)}
res = {per: {r: stats(T[(T.train == tr) & (T.rule == r)]) for r in RULES} for tr, per in ((True, "2020-2023"), (False, "2024-2026"))}
tr = res["2020-2023"]
order = sorted(RULES, key=lambda r: -tr[r]["avg"])
pick = next((r for r in order if tr[r]["w5"] >= tr["X1"]["w5"] - 3), order[0])
yearly = {str(y): {r: {"n": int(len(g[g.rule == r])), "avg": round(g[g.rule == r].ret.mean(), 2), "med": round(g[g.rule == r].ret.median(), 2)} for r in RULES} for y, g in T.groupby("year")}
out = {"yearly": yearly, "pick": pick, "order_by_train_avg": order, "blocked_entries": {"2020-2023": blocked[True], "2024-2026": blocked[False]}, "results": res}
os.makedirs("research/results", exist_ok=True)
json.dump(out, open("research/results/exits.json", "w"), ensure_ascii=False, indent=1)
for per in res:
    print("==", per); print(pd.DataFrame(res[per]).T.to_string())
print(pd.DataFrame({y: {r: v[r]["avg"] for r in RULES} for y, v in yearly.items()}).to_string())
print("挑選:", pick, "（依挑選期平均排序：", order, "）", "買不到:", out["blocked_entries"])
