"""空頭家數 ≥ 800 時，撿「跌深的中小型題材股」會不會漲回最多。

全部在看結果之前寫下（2026/11/18 12:05），看到結果後不修改。
（已知：panic_pick.py — 跌深 20% 以上 20 日超額 +2.2、5/5；大公司、原本多頭等沒有再加分。）

時點：空頭家數（收盤<5<10<20<60日線）第一次衝上 800（5 次已完成）；敏感度：門檻 700（9 次）。
股票：訊號日在分析範圍（20日均額 ≥ 1,000 萬）。「跌深」＝離自己近20日最高收盤 ≥ 20%。
「中小型」＝股本 ≤ 30 億（現在的股本）。
「題材」用大跌前市場的認同來代替（訊號日前 80 到 20 個交易日之間，也就是大跌之前）：
  S0 只看跌深（對照）
  S1 跌深＋中小型
  S2 跌深＋中小型＋大跌前曾漲停（當天漲 ≥ 9.5%）
  S3 跌深＋中小型＋大跌前強（前80→20日漲幅前30%）
  S4 跌深＋中小型＋大跌前曾是「仍在動」或「今日發動」
  S5 跌深＋中小型＋大跌前強＋曾漲停
結果：隔天開盤買，20日收盤報酬、20日超額（減全部股票平均）、
  「飆最兇比例」＝20日報酬排進全部股票前10%的比例、20日內最低點、40日內收復20日高點的比例。
判斷：門檻 800 中，某組 20 日超額比 S0 高 ≥ 2 個百分點、5 次至少贏 4 次，且門檻 700 方向相同，才算更好。
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radar_core import load_prices, panels, features, tiers

p = load_prices(); P = panels(p)
C, O = P["C"], P["O"]
f = features(P); S = tiers(f, P["MK"])
days = C.index; N = len(days)
name = p.sort_values("date").drop_duplicates("code", keep="last").set_index("code").name
R = lambda n: C.rolling(n, min_periods=n).mean()
m5, m10, m20, m60 = R(5), R(10), R(20), R(60)
bear = ((C < m5) & (m5 < m10) & (m10 < m20) & (m20 < m60)).sum(axis=1)
piv = lambda col: S[S[col]].assign(v=True).pivot(index="date", columns="code", values="v").reindex(index=days, columns=C.columns).fillna(False).astype(bool)
inb, act = piv("base"), piv("hot") | piv("pk")
lim = f["ch"] >= 9.5
ind = pd.read_csv("data/industry.csv", dtype={"code": str}).set_index("code")
cap = pd.Series(ind["cap"]).reindex(C.columns); g = pd.Series(ind["ind_name"]).reindex(C.columns)
start = int(np.argmax((C.notna() & m60.notna()).sum(axis=1).values >= 1000))

def signals(th): return [i for i in range(max(start, 80), N) if bear.iloc[i] >= th and (bear.iloc[i - 20:i] < th).all()]

def groups(i):
    m = inb.iloc[i] & C.iloc[i].notna()
    hi20 = C.iloc[i - 19:i + 1].max()
    deep = (C.iloc[i] / hi20 - 1 <= -0.20) & m
    small = cap <= 30
    hadlim = lim.iloc[i - 80:i - 19].any()
    strong = (C.iloc[i - 20] / C.iloc[i - 80] - 1).where(m).rank(pct=True) > 0.7
    wasact = act.iloc[i - 80:i - 19].any()
    return m, hi20, {"S0 只看跌深": deep, "S1 跌深＋中小型": deep & small, "S2 ＋曾漲停": deep & small & hadlim,
                     "S3 ＋大跌前強": deep & small & strong, "S4 ＋曾是強勢股": deep & small & wasact, "S5 ＋大跌前強＋曾漲停": deep & small & strong & hadlim}

def run(th):
    rows = []
    for i in signals(th):
        if i + 20 >= N: continue
        m, hi20, G = groups(i)
        o = O.iloc[i + 1]
        r20 = (C.iloc[i + 20] / o - 1) * 100
        top = r20.where(m).rank(pct=True) > 0.9
        lo = (C.iloc[i + 1:i + 21].min() / o - 1) * 100
        rec = (C.iloc[i + 1:min(N, i + 41)].max() >= hi20) if i + 40 < N else pd.Series(np.nan, index=C.columns)
        b = r20[m].mean()
        for k, sel in G.items():
            sel = sel & o.notna()
            rows.append({"date": days[i].date().isoformat(), "grp": k, "n": int(sel.sum()), "r20": r20[sel].mean(), "ex20": r20[sel].mean() - b,
                         "top10": top[sel].mean() * 100, "low20": lo[sel].mean(), "rec40": rec[sel].mean() * 100 if rec.notna().any() else np.nan})
    return pd.DataFrame(rows)

out = {}
for th in (800, 700):
    T = run(th)
    pv = T.pivot(index="date", columns="grp", values="ex20")
    s = T.groupby("grp")[["n", "r20", "ex20", "top10", "low20", "rec40"]].mean().round(2)
    s["beat_S0"] = [int((pv[k] > pv["S0 只看跌深"]).sum()) for k in s.index]; s["events"] = int(pv.shape[0])
    out[str(th)] = {"summary": s.reset_index().to_dict("records"), "by_event": T.round(2).to_dict("records")}
    print("== 門檻", th); print(s.to_string()); print(pv.round(1).to_string())
s8 = pd.DataFrame(out["800"]["summary"]).set_index("grp"); s7 = pd.DataFrame(out["700"]["summary"]).set_index("grp")
b0 = "S0 只看跌深"
out["better"] = [k for k in s8.index if k != b0 and s8.at[k, "ex20"] - s8.at[b0, "ex20"] >= 2 and s8.at[k, "beat_S0"] >= 4 and s7.at[k, "ex20"] > s7.at[b0, "ex20"]]

# 歷次反彈最兇的20檔，是什麼樣的股票
ex = []
for i in signals(800):
    if i + 20 >= N: continue
    m, hi20, G = groups(i)
    r20 = ((C.iloc[i + 20] / O.iloc[i + 1] - 1) * 100).where(m).dropna().sort_values(ascending=False).head(20)
    ex.append({"date": days[i].date().isoformat(), "top": [[c, name.get(c, ""), g.get(c) if isinstance(g.get(c), str) else "", round(v, 1),
               None if cap.get(c) != cap.get(c) else round(cap[c], 1), bool(G["S0 只看跌深"][c]), bool(G["S2 ＋曾漲停"][c] or G["S4 ＋曾是強勢股"][c])] for c, v in r20.items()]})
out["top_rebounders"] = ex

# 11/17 符合的股票
i = N - 1; m, hi20, G = groups(i)
dd = (C.iloc[i] / hi20 - 1) * 100
out["now"] = {k: [[c, name.get(c, ""), g.get(c) if isinstance(g.get(c), str) else "", round(dd[c], 1), None if cap.get(c) != cap.get(c) else round(cap[c], 1)]
              for c in dd[G[k]].sort_values().index] for k in G}
json.dump(out, open("research/results/panic_small.json", "w"), ensure_ascii=False, indent=1, default=str)
print("比只看跌深更好:", out["better"] or "無")
for e in ex:
    t = e["top"]; print(e["date"], "前20名：中小型", sum(1 for x in t if x[4] is not None and x[4] <= 30), "跌深", sum(x[5] for x in t), "曾漲停或強勢", sum(x[6] for x in t))
print({k: len(v) for k, v in out["now"].items()})
