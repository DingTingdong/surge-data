"""撿跌深：「前波強勢」改用每檔自己的漲勢與高點，不綁大盤高點（2026/11/20 使用者指出安碁資訊等被漏掉）。探索。

W  （現行）：大盤這一波起點 pk 之前 60 日漲幅前 30% 且漲停過；從 pk 前 5 日起自己最高收盤跌 ≥ 20%
W2 （自己的高點）：自己近 20 日最高收盤 ÷ 前 100～20 日的最低收盤 − 1 ＝ 自己這段漲勢，排分析範圍前 30%，且近 60 日漲停過；
     從自己近 20 日最高收盤跌 ≥ 20%
W ∪ W2：兩種都算
都依「大跌中途小反彈」排序取前 20。上市空頭家數第一次到 750／800／700，隔天開盤買，持有 20 日。
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
exec(open("research/panic_winners.py", encoding="utf-8").read().split("out = {}")[0])
ew = (1 + C0.pct_change(fill_method=None).where(inb).mean(axis=1).fillna(0)).cumprod()
rows = []
for th in (750, 800, 700):
    for i in signals(th):
        m = inb.iloc[i] & C0.iloc[i].notna(); o = O.iloc[i + 1]; okb = o.notna()
        pk = int(np.argmax(ew.values[i - 60:i + 1])) + i - 60
        pre = (C0.iloc[pk] / C0.iloc[pk - 60] - 1).where(m)
        W = (pre.rank(pct=True) > 0.7) & lim.iloc[pk - 60:pk + 1].any() & (C0.iloc[i] / C0.iloc[max(pk - 5, 0):i + 1].max() - 1 <= -0.2) & m & okb
        hi20 = C0.iloc[i - 19:i + 1].max(); run = (hi20 / C0.iloc[i - 100:i - 19].min() - 1).where(m)
        W2 = (run.rank(pct=True) > 0.7) & lim.iloc[i - 59:i + 1].any() & (C0.iloc[i] / hi20 - 1 <= -0.2) & m & okb
        lo_ = min(pk + 5, i - 1); w3 = (ew / ew.shift(3) - 1).iloc[lo_:i]
        e = days.get_loc(w3.idxmax()); bo = C0.iloc[e] / C0.iloc[e - 3] - 1
        r20 = (C0.iloc[i + 20] / o - 1) * 100; low = (C0.iloc[i + 1:i + 21].min() / o - 1) * 100; b = r20[m & okb].mean()
        def top(sel, k=20):
            idx = bo[sel].sort_values(ascending=False).index[:k]; return pd.Series(C0.columns.isin(idx), index=C0.columns)
        def top_by(score, sel, k=20):
            idx = score[sel].sort_values(ascending=False).index[:k]; return pd.Series(C0.columns.isin(idx), index=C0.columns)
        G0 = {"W2前20（依自己漲勢）": top_by(run, W2), "W2前20（依跌幅最深）": top_by(-(C0.iloc[i] / hi20 - 1), W2)}
        G = {**G0, "現行W": W, "自己高點W2": W2, "兩種都算": W | W2, "現行W前20": top(W), "自己高點W2前20": top(W2), "兩種都算前20": top(W | W2), "W2有、W沒有": W2 & ~W}
        for k, sel in G.items():
            rows.append({"th": th, "date": str(days[i].date()), "grp": k, "n": int(sel.sum()), "r20": r20[sel].mean(), "ex20": r20[sel].mean() - b, "low20": low[sel].mean()})
T = pd.DataFrame(rows)
for th in (750, 800, 700):
    g = T[T.th == th]; pv = g.pivot(index="date", columns="grp", values="ex20")
    s = g.groupby("grp", sort=False)[["n", "r20", "ex20", "low20"]].mean().round(2)
    s["贏現行W前20"] = [int((pv[k] > pv["現行W前20"]).sum()) for k in s.index]; s["次數"] = len(pv)
    print("== 門檻", th); print(s.to_string())
T.round(2).to_json("research/results/panic_own_high.json", orient="records", force_ascii=False, indent=1)
