"""恐慌時什麼時候進場：一次買、分批買、等止跌確認（2026/11/20 使用者問；探索）。

名單固定在訊號日（上市空頭家數第一次到 750）選出：撿跌深前20（網站現行定義），另看正2（00631L 實際價格）。
所有方法都在「訊號日後第 30 個交易日收盤」一起結算，比較公平；還沒到 30 天的事件不算。
  E1 訊號隔天開盤一次買
  E2 等到 800 才買（之後 30 天內沒到 800 → 第 10 天收盤後隔天買，避免空手）
  E3 分三批：訊號隔天、第 5 天後、第 10 天後各買 1/3
  E4 等止跌確認：上市空頭家數從這波最高點回落 ≥ 100 家的隔天開盤買
  E5 等止跌確認：等權指數收盤站上自己5日線的隔天開盤買
  E6 等止跌確認：等權指數比這波最低收盤反彈 ≥ 5% 的隔天開盤買
看：到第 30 天的報酬、買進後到結算日的最低點、進場日比訊號日晚幾天。
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
exec(open("research/panic_winners.py", encoding="utf-8").read().split("out = {}")[0])
ew = (1 + C0.pct_change(fill_method=None).where(inb).mean(axis=1).fillna(0)).cumprod()
ew5 = ew.rolling(5).mean()
et = pd.read_csv("data/etf.csv"); et["date"] = pd.to_datetime(et.date)
x = et[et.code == "00631L"].set_index("date").reindex(days)[["open", "close"]].ffill(limit=10)
rr = x.close.pct_change(); fac = pd.Series(1.0, index=days)
for d in rr.index[rr.abs() > 0.35]: fac[days < d] *= x.close[d] / x.close.shift(1)[d]
o2, c2 = x.open * fac, x.close * fac
END = 30
rows = []
for th in (750, 700):
    for i in signals(th):
        if i + END + 1 >= N: continue
        m = inb.iloc[i] & C0.iloc[i].notna()
        hi20 = C0.iloc[i - 19:i + 1].max(); run = (hi20 / C0.iloc[i - 100:i - 19].min() - 1).where(m)
        W2 = (run.rank(pct=True) > 0.7) & lim.iloc[i - 59:i + 1].any() & (C0.iloc[i] / hi20 - 1 <= -0.2) & m & O.iloc[i + 1].notna()
        top = run[W2].sort_values(ascending=False).index[:20]
        e_end = i + END
        def entry_day(rule):
            for k in range(i, e_end - 1):
                if rule == "E1": return i + 1
                if rule == "E2" and bear.iloc[k] >= 800: return k + 1
                if rule == "E2" and k == i + 10: return k + 1
                if rule == "E4" and bear.iloc[i:k + 1].max() - bear.iloc[k] >= 100: return k + 1
                if rule == "E5" and k > i and ew.iloc[k] > ew5.iloc[k]: return k + 1
                if rule == "E6" and ew.iloc[k] >= ew.iloc[i:k + 1].min() * 1.05: return k + 1
            return None
        def perf(ent_days):  # 等權分批
            rs, lows = [], []
            for e in ent_days:
                ob = O.iloc[e][top]; rs.append(((C0.iloc[e_end][top] / ob - 1).mean()) * 100)
                lows.append(((C0.iloc[e:e_end + 1][top].div(ob) - 1).mean(axis=1).min()) * 100)
                ob2 = o2.iloc[e]; rs.append(np.nan) if False else None
            return np.mean(rs), np.mean(lows)
        def perf2(ent_days):
            rs = [(c2.iloc[e_end] / o2.iloc[e] - 1) * 100 for e in ent_days]; lows = [(c2.iloc[e:e_end + 1].min() / o2.iloc[e] - 1) * 100 for e in ent_days]
            return np.mean(rs), np.mean(lows)
        plans = {r: [entry_day(r)] for r in ("E1", "E2", "E4", "E5", "E6")}
        plans["E3"] = [i + 1, i + 6, i + 11]
        for r, ents in plans.items():
            if None in ents: continue
            a, lo = perf(ents); b, lo2 = perf2(ents)
            rows.append({"th": th, "date": str(days[i].date()), "rule": r, "delay": np.mean(ents) - (i + 1), "top20": a, "top20_low": lo, "lev2": b, "lev2_low": lo2,
                         "entry": str(days[ents[0]].date())})
T = pd.DataFrame(rows)
NM = {"E1": "訊號隔天一次買", "E2": "等到800才買", "E3": "分三批（0/5/10天）", "E4": "空頭家數從高點回落100家", "E5": "指數站上5日線", "E6": "指數從低點反彈5%"}
for th in (750, 700):
    g = T[T.th == th]
    s = g.groupby("rule").agg(次數=("date", "size"), 晚幾天=("delay", "mean"), 撿跌深前20=("top20", "mean"), 前20買後最低=("top20_low", "mean"),
                              正2=("lev2", "mean"), 正2買後最低=("lev2_low", "mean")).round(1)
    s["前20賺錢"] = g.assign(w=g.top20 > 0).groupby("rule").w.sum(); s.index = [NM[k] for k in s.index]
    print("== 門檻", th, "（都在訊號後第30天結算）"); print(s.to_string())
    if th == 750: print(g.pivot(index="date", columns="rule", values="top20").round(1).rename(columns=NM).to_string())
T.round(2).to_json("research/results/panic_entry.json", orient="records", force_ascii=False, indent=1)
# 這一波各方法的進場日
i = [k for k in signals_all if True][0] if False else None
