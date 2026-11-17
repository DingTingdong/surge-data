"""大盤濾網：只在大盤偏多時才做「今日發動＋移動停損」，能不能避開空頭的虧損。

全部在跑之前寫下（2026/11/17 19:35），看到結果後不修改。

大盤用分析範圍內（20日均額 ≥ 1,000 萬）所有股票每天的平均漲跌幅，累乘成「等權指數」，只用訊號當天以前的資料。
候選濾網（訊號當天收盤判斷）：
  R1 等權指數收盤 > 自己的60日均線
  R2 等權指數收盤 > 自己的20日均線
  R3 站上自己20日均線的股票比例 ≥ 50%
交易：今日發動隔天開盤買，移動停損出場（research/exits.py 的 X6），扣 1.1%。
期間：2020–2023 挑選，2024–2026/11 驗證。
挑選規則：在 2020–2023，保留至少 40% 訊號的濾網中，選「每筆平均」最高的；它必須比不加濾網高 ≥ 1.0 個百分點才採用，
否則結論是大盤濾網沒有幫助。驗證期與逐年照實報告全部濾網。
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radar_core import load_prices, panels, features, tiers

COST, CAP, LOCK = 1.1, 20, 0.095
TRAIN_END = pd.Timestamp("2023-12-31")
p = load_prices(); P = panels(p)
C, O, H, L = P["C"], P["O"], P["H"], P["L"]
f = features(P); S = tiers(f, P["MK"])
days = C.index; N = len(days)
Cv, Ov, Hv, Lv = C.values, O.values, H.values, L.values
pos = {d: i for i, d in enumerate(days)}; col = {c: j for j, c in enumerate(C.columns)}

# 大盤指標
inbase = S[S.base].assign(v=True).pivot(index="date", columns="code", values="v").reindex(index=days, columns=C.columns).fillna(False).astype(bool)
r1 = C.pct_change(fill_method=None).where(inbase)
ew = (1 + r1.mean(axis=1).fillna(0)).cumprod()
above20 = (C > C.rolling(20, min_periods=20).mean()).where(inbase & C.notna()).mean(axis=1)
reg = pd.DataFrame({"R1": ew > ew.rolling(60, min_periods=60).mean(), "R2": ew > ew.rolling(20, min_periods=20).mean(), "R3": above20 >= 0.5})

def locked(i, j, up):
    o, h, l = Ov[i, j], Hv[i, j], Lv[i, j]
    k = i - 1
    while k >= 0 and not Cv[k, j] == Cv[k, j]: k -= 1
    if k < 0 or not (o == o and h == h and l == l): return False
    r = o / Cv[k, j] - 1
    return o == h == l and (r >= LOCK if up else r <= -LOCK)

def x6(t, j):
    e = t + 1; end = e + CAP - 1
    if end >= N: return None
    o = Ov[e, j]
    if not o == o or o <= 0 or locked(e, j, True): return None
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

rows = []
for d, c in zip(S[S.pk].date, S[S.pk].code):
    r = x6(pos[d], col[c])
    if r is not None: rows.append({"date": d, "year": d.year, "train": d <= TRAIN_END, "ret": r, **reg.loc[d].to_dict()})
T = pd.DataFrame(rows)
F = {"不加濾網": pd.Series(True, index=T.index), "R1 等權指數>60日線": T.R1, "R2 等權指數>20日線": T.R2, "R3 站上20日線比例≥50%": T.R3}
st = lambda g, n: {"n": int(len(g)), "keep": round(len(g) / max(1, n) * 100, 1), "avg": round(g.ret.mean(), 2), "med": round(g.ret.median(), 2),
                   "win": round((g.ret > 0).mean() * 100, 1)}
res = {}
for tr, per in ((True, "2020-2023"), (False, "2024-2026")):
    g0 = T[T.train == tr]; res[per] = {k: st(g0[m[g0.index]], len(g0)) for k, m in F.items()}
yearly = {str(y): {k: round(g[m[g.index]].ret.mean(), 2) for k, m in F.items()} for y, g in T.groupby("year")}
trn = res["2020-2023"]; base = trn["不加濾網"]["avg"]
cands = [k for k in F if k != "不加濾網" and trn[k]["keep"] >= 40]
best = max(cands, key=lambda k: trn[k]["avg"]) if cands else None
pick = best if best and trn[best]["avg"] - base >= 1.0 else None
out = {"pick": pick, "best_candidate": best, "results": res, "yearly": yearly,
       "regime_on_share": {k: round(float(reg[k].mean() * 100), 1) for k in reg}}
os.makedirs("research/results", exist_ok=True)
json.dump(out, open("research/results/regime.json", "w"), ensure_ascii=False, indent=1)
for per in res: print("==", per); print(pd.DataFrame(res[per]).T.to_string())
print(pd.DataFrame(yearly).to_string()); print("最佳候選:", best, "採用:", pick)
