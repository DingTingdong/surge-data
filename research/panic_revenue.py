"""恐慌時撿跌深，加上「營收成長」會不會更好（使用者想找有未來的公司）。

全部在看結果之前寫下（2026/11/19 21:30），看到結果後不修改。
時點：上市空頭家數（玩股網短均線算法，research/bear_levels.py）第一次衝上 750（主要，網站門檻；7 次已完成），
      另報 800（4 次）與 700（12 次）。隔天開盤買、持有 20 個交易日。
營收：公開資訊觀測站月營收。某月營收當成「下個月 11 日起」才知道（法定 10 日前公布），只用訊號日當時已公布的月份。
  yoy1 = 最近一個已公布月份的年增率；yoy3 = 最近三個已公布月份年增率的平均。
組別（都在分析範圍、跌深＝離20日高 ≥ 20%）：
  R0 撿跌深（網站名單：跌深＋市值≤100億＋大跌前強＋曾漲停）
  R1 跌深＋yoy1 > 20%
  R2 跌深＋yoy3 > 20%
  R3 撿跌深＋yoy1 > 20%
  R4 撿跌深＋yoy1 ≤ 0（對照：營收衰退的）
  R5 跌深＋yoy1 > 50%
  另報：只看跌深、全部股票
看：20 日報酬、比全部股票多多少（超額）、買後 20 日內最低點、每次的檔數。
判斷：750 門檻下，R3 比 R0 的 20 日超額高 ≥ 2 個百分點、7 次中至少 5 次贏 R0，而且 800、700 方向相同 → 網站名單加上營收條件；
      否則營收只當卡片上的標示。R1／R2／R5 只報告。
"""
import glob, json, os, sys
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

rv = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in sorted(glob.glob("data/revenue/*.csv*"))], ignore_index=True)
rv["pub"] = pd.to_datetime(rv.ym + "-01") + pd.DateOffset(months=1) + pd.Timedelta(days=10)  # 下個月 11 日
yoy = rv.pivot_table(index="ym", columns="code", values="yoy").sort_index()
pub = rv.groupby("ym").pub.first().sort_index()

def rev_at(d):
    known = pub[pub <= d].index
    if len(known) == 0: return pd.Series(dtype=float), pd.Series(dtype=float)
    y1 = yoy.loc[known[-1]].reindex(C0.columns)
    y3 = yoy.loc[known[-3:]].mean(min_count=2).reindex(C0.columns)
    return y1, y3

def signals(th): return [i for i in range(start, N) if bear.iloc[i] >= th and (bear.iloc[i - 20:i] < th).all() and i + 20 < N]

out = {}
for th in (750, 800, 700):
    rows = []
    for i in signals(th):
        d = days[i]; y1, y3 = rev_at(d)
        m = inb.iloc[i] & C0.iloc[i].notna()
        hi20 = C0.iloc[i - 19:i + 1].max(); deep = (C0.iloc[i] / hi20 - 1 <= -0.2) & m
        strong = (C0.iloc[i - 20] / C0.iloc[i - 80] - 1).where(m).rank(pct=True) > 0.7
        r0 = deep & (cap * raw.iloc[i] / 10 <= 100) & strong & lim.iloc[i - 80:i - 19].any()
        G = {"全部股票": m, "只看跌深": deep, "R0 撿跌深": r0, "R1 跌深＋營收年增>20%": deep & (y1 > 20), "R2 跌深＋近3月年增>20%": deep & (y3 > 20),
             "R3 撿跌深＋營收年增>20%": r0 & (y1 > 20), "R4 撿跌深＋營收衰退": r0 & (y1 <= 0), "R5 跌深＋營收年增>50%": deep & (y1 > 50)}
        o = O.iloc[i + 1]; r20 = (C0.iloc[i + 20] / o - 1) * 100; low = (C0.iloc[i + 1:i + 21].min() / o - 1) * 100
        b = r20[m & o.notna()].mean()
        for k, sel in G.items():
            sel = sel & o.notna()
            rows.append({"date": str(d.date()), "grp": k, "n": int(sel.sum()), "r20": r20[sel].mean(), "ex20": r20[sel].mean() - b, "low20": low[sel].mean(),
                         "cov": float(y1[deep].notna().mean() * 100) if k == "只看跌深" else None})
    T = pd.DataFrame(rows)
    pv = T.pivot(index="date", columns="grp", values="ex20")
    s = T.groupby("grp")[["n", "r20", "ex20", "low20"]].mean().round(2)
    s["贏R0次數"] = [int((pv[k] > pv["R0 撿跌深"]).sum()) for k in s.index]; s["次數"] = len(pv)
    out[str(th)] = {"summary": s.reset_index().to_dict("records"), "by_event": T.round(2).to_dict("records"),
                    "coverage_deep_pct": round(float(T[T.grp == "只看跌深"]["cov"].mean()), 1)}
    print("== 門檻", th, "營收資料覆蓋（跌深股）%:", out[str(th)]["coverage_deep_pct"]); print(s.to_string()); print(pv.round(1).to_string())
s = {th: pd.DataFrame(out[th]["summary"]).set_index("grp") for th in out}
d = lambda th: s[th].at["R3 撿跌深＋營收年增>20%", "ex20"] - s[th].at["R0 撿跌深", "ex20"]
adopt = bool(d("750") >= 2 and s["750"].at["R3 撿跌深＋營收年增>20%", "贏R0次數"] >= 5 and d("800") > 0 and d("700") > 0)
out["adopt_revenue_filter"] = adopt
json.dump(out, open("research/results/panic_revenue.json", "w"), ensure_ascii=False, indent=1, default=str)
print("R3 比 R0：750 %.2f、800 %.2f、700 %.2f → 名單加營收條件: %s" % (d("750"), d("800"), d("700"), adopt))
