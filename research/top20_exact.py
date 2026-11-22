"""分數前20（每月換股）用真實的持有方式重算報酬（2026/11/22；探索，看過 directions.py 結果後才做，不改任何判斷）。
原本用「每月形成日起 20 個交易日」的報酬接起來算年化，會有重疊和漏掉的日子，絕對報酬（年化）不準。
這裡改成：形成日隔天開盤買，下一次形成日隔天開盤賣（同時換成新名單），每個交易日剛好算一次；0050 用同一段的收盤到收盤（含息）。
成本：每次換股按換股比例扣（0.6% 與 1.1% 兩種）。開盤一價漲停買不到的那檔該月不計（其他檔等權）。
"""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from vs0050 import COMP, nw_t, per, PER
X = pd.read_pickle("/tmp/model_v2/vs0050_extra.pkl"); days = X["days"]; cols = X["cols"]; Ov, Cv, lockn, bidx = X["O"], X["C"], X["lockn"], X["bidx"]
Cff = pd.DataFrame(Cv).ffill(limit=60).values; pos = {d: i for i, d in enumerate(days)}; cj = {c: j for j, c in enumerate(cols)}
F = X["fd"].copy(); F = F[F.date >= "2011-01-01"]
F["cs"] = np.mean([F.groupby("date")[c].rank(pct=True).fillna(0.5).values for c in COMP], axis=0)
fds = sorted(F.date.unique()); rows, prev = [], set()
for a, b in zip(fds[:-1], fds[1:]):
    ta, tb = pos[a], pos[b]; g = F[F.date == a]; pk = g.nlargest(20, "cs"); cur = set(pk.code); to = 1.0 if not prev else len(cur - prev) / 20; prev = cur
    rs = []
    for c in pk.code:
        j = cj[c]; o = Ov[ta + 1, j]
        if not (o > 0) or lockn[ta, j]: continue
        ex = Ov[tb + 1, j] if tb + 1 < len(days) and Ov[tb + 1, j] > 0 else Cff[min(tb + 1, len(days) - 1), j]
        if not np.isfinite(ex): ex = Cff[tb, j]
        if np.isfinite(ex): rs.append((ex / o - 1) * 100)
    rows.append({"date": a, "g": np.mean(rs), "to": to, "b": (bidx[tb] / bidx[ta] - 1) * 100, "n": len(rs)})
T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index)
out = {}
for cost in (0.6, 1.1):
    r = T.g - cost * T.to; x = r - T.b; w, wb = (1 + r / 100).cumprod(), (1 + T.b / 100).cumprod()
    yrs = (days[pos[fds[-1]]] - fds[0]).days / 365.25
    yr = pd.DataFrame({"策略%": (1 + r / 100).groupby(T.index.year).prod() * 100 - 100, "0050%": (1 + T.b / 100).groupby(T.index.year).prod() * 100 - 100}).round(1)
    o = {"年化 策略/0050": (round((w.iloc[-1] ** (1 / yrs) - 1) * 100, 1), round((wb.iloc[-1] ** (1 / yrs) - 1) * 100, 1)),
         "每月減0050 平均/t": (round(float(x.mean()), 2), round(nw_t(x, 1), 2)), "四段 每月減0050 平均/t": {p: (round(float(x[per(T.index) == p].mean()), 2), round(nw_t(x[per(T.index) == p], 1), 2)) for p in PER},
         "樣本外（非2020–23）每月減0050 平均/t": (round(float(x[per(T.index) != "2020-23"].mean()), 2), round(nw_t(x[per(T.index) != "2020-23"], 1), 2)),
         "最大回落% 策略/0050": (round(float((w / w.cummax() - 1).min() * 100), 1), round(float((wb / wb.cummax() - 1).min() * 100), 1)),
         "贏0050的年數": f"{int((yr['策略%'] > yr['0050%']).sum())}/{len(yr)}", "逐年": yr.to_dict(orient="index")}
    out[f"成本{cost}%"] = o; print(f"\n== 成本 {cost}%", json.dumps({k: v for k, v in o.items() if k != "逐年"}, ensure_ascii=False))
    if cost == 0.6: print(yr.T.to_string())
json.dump(out, open(os.path.join(os.path.dirname(HERE), "research/results/top20_exact.json"), "w"), ensure_ascii=False, indent=1, default=str)
