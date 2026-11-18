"""上市空頭家數（玩股網短均線算法）到 600／650／700／750／800／850 時，買進的績效差多少。

2026/11/18 使用者問：門檻 800 是不是太嚴，700、600 難道就不會反彈？看過 800 的結果後才做，屬於探索。
算法：5日線 < 10日線 < 20日線，沒成交日沿用前一天收盤；上市股票。
訊號：第一次衝上門檻（前20個交易日都在門檻以下）。訊號日收盤看到 → 隔天開盤買（所有結果都是隔天開盤才買，已經考慮「看到時當天已收盤」）。
看：分析範圍全部股票、跌深（離20日高 ≥ 20%）、撿跌深（跌深＋市值≤100億＋大跌前強＋曾漲停）的 10／20 日報酬，
  買進後 20 日內平均最低點，以及「之後有沒有再衝上 800」。
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radar_core import load_prices, panels, features, tiers

p = load_prices(); P = panels(p); C0, O, MK, raw = P["C"], P["O"], P["MK"], P["rawC"]
f = features(P); S = tiers(f, MK)
days = C0.index; N = len(days)
alive = (C0.notna().cumsum() > 0) & (C0.notna()[::-1].cumsum()[::-1] > 0)
C = C0.ffill(limit=20).where(alive)
R = lambda n: C.rolling(n, min_periods=n).mean()
m5, m10, m20, m60 = R(5), R(10), R(20), R(60)
ok = C.notna() & m60.notna()
mk = MK.ffill().iloc[-1]; tw = [c for c in C.columns if mk.get(c) == "TWSE"]
bear = ((m5 < m10) & (m10 < m20) & ok)[tw].sum(axis=1)
inb = S[S.base].assign(v=True).pivot(index="date", columns="code", values="v").reindex(index=days, columns=C0.columns).fillna(False).astype(bool)
ind = pd.read_csv("data/industry.csv", dtype={"code": str}).set_index("code")
cap = pd.Series(ind["cap"]).reindex(C0.columns)
lim = f["ch"] >= 9.5
start = int(np.argmax(ok[tw].sum(axis=1).values >= 600)) + 80

def signals(th): return [i for i in range(start, N) if bear.iloc[i] >= th and (bear.iloc[i - 20:i] < th).all()]

rows = []
for th in (600, 650, 700, 750, 800, 850):
    for i in signals(th):
        if i + 20 >= N: rows.append({"th": th, "date": str(days[i].date()), "done": False}); continue
        m = inb.iloc[i] & C0.iloc[i].notna()
        hi20 = C0.iloc[i - 19:i + 1].max(); deep = (C0.iloc[i] / hi20 - 1 <= -0.2) & m
        strong = (C0.iloc[i - 20] / C0.iloc[i - 80] - 1).where(m).rank(pct=True) > 0.7
        pick = deep & (cap * raw.iloc[i] / 10 <= 100) & strong & lim.iloc[i - 80:i - 19].any()
        o = O.iloc[i + 1]; ok1 = o.notna()
        r10 = (C0.iloc[i + 10] / o - 1) * 100; r20 = (C0.iloc[i + 20] / o - 1) * 100
        low = (C0.iloc[i + 1:i + 21].min() / o - 1) * 100
        later800 = bool((bear.iloc[i:i + 21] >= 800).any())
        rows.append({"th": th, "date": str(days[i].date()), "done": True, "bear": int(bear.iloc[i]), "reach800": later800,
                     "all20": r20[m & ok1].mean(), "all10": r10[m & ok1].mean(), "deep20": r20[deep & ok1].mean(), "pick20": r20[pick & ok1].mean(),
                     "n_pick": int(pick.sum()), "all_low": low[m & ok1].mean(), "pick_low": low[pick & ok1].mean(), "all_up": (r20[m & ok1] > 0).mean() * 100})
T = pd.DataFrame(rows)
D = T[T.done.astype(bool)].copy(); D["reach800"] = D.reach800.astype(bool)
summ = D.groupby("th").agg(次數=("date", "size"), 全部20日=("all20", "mean"), 全部上漲比例=("all_up", "mean"), 跌深20日=("deep20", "mean"),
                           撿跌深20日=("pick20", "mean"), 撿跌深檔數=("n_pick", "mean"), 全部買後最低=("all_low", "mean"), 撿跌深買後最低=("pick_low", "mean"),
                           之後衝上800=("reach800", "sum")).round(2)
summ["撿跌深贏全部的次數"] = D.assign(w=D.pick20 > D.all20).groupby("th").w.sum()
summ["撿跌深賺錢次數"] = D.assign(w=D.pick20 > 0).groupby("th").w.sum()
# 700 到了但之後 20 天沒到 800 的那幾次
no800 = D[(D.th == 700) & ~D.reach800][["date", "bear", "all20", "deep20", "pick20"]].round(2)
yes800 = D[(D.th == 700) & D.reach800][["date", "bear", "all20", "deep20", "pick20", "all_low"]].round(2)
# 基準：所有交易日隔天開盤買、持有20日的分析範圍平均
base = np.nanmean([((C0.iloc[i + 20] / O.iloc[i + 1] - 1) * 100)[inb.iloc[i]].mean() for i in range(start, N - 21, 5)])
out = {"base_all20": round(float(base), 2), "summary": summ.reset_index().to_dict("records"), "events": T.round(2).to_dict("records"),
       "th700_no800": no800.to_dict("records"), "th700_reach800": yes800.to_dict("records")}
json.dump(out, open("research/results/bear_levels.json", "w"), ensure_ascii=False, indent=1, default=str)
print("平常任一天買、持有20日：%.2f%%" % base); print(summ.to_string())
print("700 到了、之後20天沒到800：\n", no800.to_string()); print("700 到了、之後有到800：\n", yes800.to_string())
print(T[~T.done][["th", "date"]].to_string())
