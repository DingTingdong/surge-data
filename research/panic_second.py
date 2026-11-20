"""同一波「第二次」碰到 800（或 750）時買，表現如何（2026/11/20 使用者問；探索）。

第一次：上市空頭家數衝上門檻，前 20 日都在門檻以下（同 bear_levels.py）。
第二次：第一次之後 40 個交易日內，家數先跌回門檻以下（中間至少 3 天低於門檻），然後再衝上門檻的那天。
看：隔天開盤買，持有 20 日；全部股票、撿跌深前20（網站現行定義，當天選）、正2（00631L 實際）。另列之後還會不會更低。
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
exec(open("research/panic_winners.py", encoding="utf-8").read().split("out = {}")[0])
ew = (1 + C0.pct_change(fill_method=None).where(inb).mean(axis=1).fillna(0)).cumprod()
et = pd.read_csv("data/etf.csv"); et["date"] = pd.to_datetime(et.date)
x = et[et.code == "00631L"].set_index("date").reindex(days)[["open", "close"]].ffill(limit=10)
rr = x.close.pct_change(); fac = pd.Series(1.0, index=days)
for d in rr.index[rr.abs() > 0.35]: fac[days < d] *= x.close[d] / x.close.shift(1)[d]
o2, c2 = x.open * fac, x.close * fac
def second(th):
    out = []
    for i0 in signals_raw(th):
        below = 0
        for k in range(i0 + 1, min(i0 + 41, N)):
            if bear.iloc[k] < th: below += 1
            elif below >= 3 and bear.iloc[k - 1] < th: out.append((i0, k)); break
    return out
def signals_raw(th): return [i for i in range(start, N) if bear.iloc[i] >= th and (bear.iloc[i - 20:i] < th).all()]
def stats(i):
    m = inb.iloc[i] & C0.iloc[i].notna(); o = O.iloc[i + 1] if i + 1 < N else None
    hi20 = C0.iloc[i - 19:i + 1].max(); run = (hi20 / C0.iloc[i - 100:i - 19].min() - 1).where(m)
    W2 = (run.rank(pct=True) > 0.7) & lim.iloc[i - 59:i + 1].any() & (C0.iloc[i] / hi20 - 1 <= -0.2) & m
    top = run[W2].sort_values(ascending=False).index[:20]
    if i + 20 >= N: return {"all": None, "top20": None, "lev2": None, "idx_low": None}
    r = (C0.iloc[i + 20] / o - 1) * 100
    return {"all": r[m & o.notna()].mean(), "top20": r[top].mean(), "lev2": (c2.iloc[i + 20] / o2.iloc[i + 1] - 1) * 100,
            "idx_low": (ew.iloc[i + 1:i + 21].min() / ew.iloc[i] - 1) * 100, "bear_max_after": int(bear.iloc[i:i + 21].max())}
rows = []
for th in (800, 750):
    for i0, k in second(th):
        a, b = stats(i0), stats(k)
        rows.append({"門檻": th, "第一次": str(days[i0].date()), "第一次家數": int(bear.iloc[i0]), "第二次": str(days[k].date()), "第二次家數": int(bear.iloc[k]),
                     "第一次後全部": a["all"], "第一次後前20": a["top20"], "第一次後正2": a["lev2"],
                     "第二次後全部": b["all"], "第二次後前20": b["top20"], "第二次後正2": b["lev2"], "第二次後指數再跌": b["idx_low"]})
T = pd.DataFrame(rows)
pd.set_option("display.width", 250); print(T.round(1).to_string(index=False))
for th in (800, 750):
    g = T[(T["門檻"] == th) & T["第二次後全部"].notna()]
    print(th, "第二次（已有結果", len(g), "次）平均：全部 %.1f 前20 %.1f 正2 %.1f；第一次同組平均：全部 %.1f 前20 %.1f 正2 %.1f" % (
        g["第二次後全部"].mean(), g["第二次後前20"].mean(), g["第二次後正2"].mean(), g["第一次後全部"].mean(), g["第一次後前20"].mean(), g["第一次後正2"].mean()))
T.round(2).to_json("research/results/panic_second.json", orient="records", force_ascii=False, indent=1)
