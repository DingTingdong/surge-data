"""恐慌時「大盤還在跌、它已經先漲」的逆勢股，之後表現如何（2026/11/20 使用者問；探索）。

上市空頭家數第一次到 750（7 次）／800（4 次）／700（12 次）的訊號日，分析範圍內：
  V1 逆勢上漲：近10日漲 ≥ 10%（同期等權指數是跌的）
  V2 接近新高：收盤在自己60日最高的 5% 以內
  V3 已經轉強：自己的最低點在 5 天以前，而且收盤站上10日線（大盤還在低點附近）
  V4 近20日相對強度前 5%（漲幅排名）
  V5 V1＋前波強勢（自己的漲勢前30%、近60日漲停過）
  V6 V3＋前波強勢
對照：全部股票、撿跌深前20（網站現行：自己的高點、依前波漲勢）。
隔天開盤買，持有 20 日；看平均、超額、組合最低點、檔數；各組也取「近10日漲幅前20名」看集中版。
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
exec(open("research/panic_winners.py", encoding="utf-8").read().split("out = {}")[0])
ew = (1 + C0.pct_change(fill_method=None).where(inb).mean(axis=1).fillna(0)).cumprod()
ma10 = C0.rolling(10, min_periods=10).mean()
rows = []
for th in (750, 800, 700):
    for i in signals(th):
        m = inb.iloc[i] & C0.iloc[i].notna(); o = O.iloc[i + 1]; okb = o.notna() & m
        r10 = (C0.iloc[i] / C0.iloc[i - 10] - 1); r20s = (C0.iloc[i] / C0.iloc[i - 20] - 1)
        hi20 = C0.iloc[i - 19:i + 1].max(); run = (hi20 / C0.iloc[i - 100:i - 19].min() - 1).where(m)
        strong = (run.rank(pct=True) > 0.7) & lim.iloc[i - 59:i + 1].any()
        W2 = strong & (C0.iloc[i] / hi20 - 1 <= -0.2) & okb
        top_w2 = pd.Series(C0.columns.isin(run[W2].sort_values(ascending=False).index[:20]), index=C0.columns)
        lowpos = pd.Series(19 - np.nanargmin(np.nan_to_num(C0.iloc[i - 19:i + 1].values, nan=1e12), axis=0), index=C0.columns)
        V = {"V1 逆勢上漲": (r10 >= 0.10), "V2 接近新高": (C0.iloc[i] >= C0.iloc[i - 59:i + 1].max() * 0.95),
             "V3 已經轉強": (lowpos >= 5) & (C0.iloc[i] > ma10.iloc[i]), "V4 相對強度前5%": (r20s.where(m).rank(pct=True) > 0.95)}
        V["V5 逆勢上漲＋前波強勢"] = V["V1 逆勢上漲"] & strong; V["V6 已經轉強＋前波強勢"] = V["V3 已經轉強"] & strong
        r20 = (C0.iloc[i + 20] / o - 1) * 100; b = r20[okb].mean()
        def st(sel):
            sel = sel & okb
            if sel.sum() == 0: return np.nan, np.nan, 0
            path = (C0.iloc[i + 1:i + 21].div(o)[sel.index[sel]] - 1).mean(axis=1) * 100
            return r20[sel].mean(), path.min(), int(sel.sum())
        G = {"全部股票": m, "撿跌深前20": top_w2}
        for k, v in V.items():
            G[k] = v; G[k + "（前20）"] = pd.Series(C0.columns.isin(r10[v & okb].sort_values(ascending=False).index[:20]), index=C0.columns)
        idx10 = float((ew.iloc[i] / ew.iloc[i - 10] - 1) * 100)
        for k, sel in G.items():
            a, lo, n = st(sel)
            rows.append({"th": th, "date": str(days[i].date()), "idx10": idx10, "grp": k, "n": n, "r20": a, "ex20": a - b, "low": lo})
T = pd.DataFrame(rows)
for th in (750, 800, 700):
    g = T[T.th == th]; pv = g.pivot(index="date", columns="grp", values="ex20")
    s = g.groupby("grp", sort=False)[["n", "r20", "ex20", "low"]].mean().round(2)
    s["超額為正次數"] = [int((pv[k] > 0).sum()) for k in s.index]; s["贏撿跌深前20"] = [int((pv[k] > pv["撿跌深前20"]).sum()) for k in s.index]; s["次數"] = len(pv)
    print("== 門檻", th, "（訊號前10日指數平均 %.1f%%）" % g.idx10.mean()); print(s.to_string())
T.round(2).to_json("research/results/panic_counter.json", orient="records", force_ascii=False, indent=1)
