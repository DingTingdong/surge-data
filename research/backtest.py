"""（此檔評估的是 v1 的精選，欄位 pk_v1；v2.0 起「今日發動」改為 research/early.py 的 E4）
長期回測：用 2020 年起的資料檢驗凍結的 v1.2 門檻。不調整任何門檻。

事先定好的檢驗（2026/11/17 寫下，資料到齊前）：
1. 每日訊號：隔天開盤買、第5個交易日收盤賣，扣 1.1%；開盤一價漲停視為買不到。
   分組：精選、仍在動的其他股票（同樣高波動的對照）、活躍度前10%、全體。逐年報告平均、中位數、勝率、賺／賠10%以上比例。
2. 不重疊每週：每週最後一個交易日的名單，下週第一天開盤買、第5個交易日收盤賣，等權。
   主要指標 = 每週「精選平均 − 仍在動其他平均」；報告平均、t 值、4週區塊 bootstrap 的 95% 區間，全期與逐年。
3. 雙向測試：活躍度排名預測 5 日 ≥ +15% 與 ≤ −15% 的 AUC，逐年。
4. 大盤漲週／跌週（全體當週平均 > 0 或 < 0，事後描述用）分開看精選與對照。
結果寫到 research/results/。
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radar_core import load_prices, panels, features, tiers

HOLD, COST, LOCK = 5, 1.1, 0.095
OUT = "research/results"
os.makedirs(OUT, exist_ok=True)

p = load_prices()
P = panels(p)
C, O, H, L = P["C"], P["O"], P["H"], P["L"]
f = features(P)
S = tiers(f, P["MK"])
S = S[S.base]  # v1 的分析範圍
days = C.index

# ---- 除權息調整檢查 ----
fac = P["fac"]
adj_events = (fac != 1.0).sum(axis=1).groupby(fac.index.year).sum()
chg_cover = p.groupby(p.date.dt.year).chg.apply(lambda s: float(s.notna().mean()))

# ---- 每筆訊號的 5 日報酬 ----
ent = O.shift(-1); prevc = C
flat = (O.shift(-1) == H.shift(-1)) & (O.shift(-1) == L.shift(-1))
blocked = flat & (O.shift(-1) / prevc - 1 >= LOCK)
ex = C.ffill(limit=HOLD).shift(-HOLD)
ret = ((ex / ent - 1) * 100 - COST).where(~blocked)
S = S.merge(ret.stack(future_stack=True).rename("ret").reset_index().rename(columns={"level_0": "date", "level_1": "code"}),
            on=["date", "code"], how="left")
S = S.merge(blocked.stack(future_stack=True).rename("blk").reset_index().rename(columns={"level_0": "date", "level_1": "code"}),
            on=["date", "code"], how="left")
S = S[S.date <= days[-HOLD - 1]]  # 只用已有完整 5 日結果的訊號
S["year"] = S.date.dt.year
S["other"] = S.hot & ~S.pk_v1
S["top10"] = S.rc >= 0.9

def desc(g):
    r = g.ret.dropna()
    if not len(r): return None
    return {"n": int(len(r)), "per_day": round(len(g) / max(1, g.date.nunique()), 1), "avg": round(r.mean(), 2), "med": round(r.median(), 2),
            "win": round((r > 0).mean() * 100, 1), "up10": round((r >= 10).mean() * 100, 1), "dn10": round((r <= -10).mean() * 100, 1),
            "blocked": int(g.blk.fillna(False).sum())}

groups = {"精選": "pk_v1", "仍在動其他": "other", "活躍度前10%": "top10", "全體": None}
daily = {}
for y, g in list(S.groupby("year")) + [("全期", S)]:
    nd = g.date.nunique()
    daily[str(y)] = {name: desc(g if col is None else g[g[col]]) for name, col in groups.items()}

# ---- 雙向 AUC ----
from scipy.stats import rankdata
def auc(x, yv):
    m = ~np.isnan(x); x, yv = x[m], yv[m]; n1 = yv.sum(); n0 = len(yv) - n1
    if n1 < 10 or n0 < 10: return None
    r = rankdata(x); return round(float((r[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), 3)
two = {}
for y, g in list(S.groupby("year")) + [("全期", S)]:
    g = g[g.ret.notna()]
    two[str(y)] = {"漲≥15%": auc(g.rc.values, (g.ret >= 15).astype(int).values), "跌≤-15%": auc(g.rc.values, (g.ret <= -15).astype(int).values),
                   "漲次數": int((g.ret >= 15).sum()), "跌次數": int((g.ret <= -15).sum())}

# ---- 不重疊每週 ----
wk = pd.Series(days, index=days).groupby(days.to_period("W-SUN")).last()  # 每週最後交易日
W = S[S.date.isin(set(wk.values))]
rows = []
for d, g in W.groupby("date"):
    pk, ot, al = g[g.pk_v1].ret.dropna(), g[g.other].ret.dropna(), g.ret.dropna()
    rows.append({"date": d, "year": d.year, "n_pk": len(pk), "pk": pk.mean() if len(pk) else np.nan,
                 "other": ot.mean() if len(ot) else np.nan, "all": al.mean() if len(al) else np.nan})
WK = pd.DataFrame(rows)
WK["diff"] = WK.pk - WK.other
rng = np.random.default_rng(20261117)
def boot(x, block=4, B=5000):
    x = x.dropna().values
    if len(x) < 8: return None
    n = len(x); k = int(np.ceil(n / block)); means = []
    for _ in range(B):
        st = rng.integers(0, n - block + 1, k)
        means.append(np.concatenate([x[s:s + block] for s in st])[:n].mean())
    return [round(float(np.percentile(means, 2.5)), 2), round(float(np.percentile(means, 97.5)), 2)]
def wsum(g):
    d = g["diff"].dropna()
    if not len(d): return None
    return {"weeks": int(len(d)), "pk_avg": round(g.pk.mean(), 2), "other_avg": round(g.other.mean(), 2), "all_avg": round(g["all"].mean(), 2),
            "diff_avg": round(d.mean(), 2), "diff_med": round(d.median(), 2), "diff_win": round((d > 0).mean() * 100, 1),
            "t": round(float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), 2) if len(d) > 2 else None, "ci95": boot(g["diff"])}
weekly = {str(y): wsum(g) for y, g in WK.groupby("year")}
weekly["全期"] = wsum(WK)
weekly["大盤漲週"] = wsum(WK[WK["all"] > 0]); weekly["大盤跌週"] = wsum(WK[WK["all"] < 0])

res = {"rule": "v1.2", "period": [str(days[0].date()), str(days[-1].date())], "trading_days": len(days),
       "chg_coverage": {str(k): round(v, 3) for k, v in chg_cover.items()},
       "adjust_events": {str(k): int(v) for k, v in adj_events.items()},
       "daily": daily, "two_sided_auc": two, "weekly": weekly}
json.dump(res, open(f"{OUT}/summary.json", "w"), ensure_ascii=False, indent=1)
WK.to_csv(f"{OUT}/weekly.csv", index=False)
print(json.dumps(res, ensure_ascii=False, indent=1))
