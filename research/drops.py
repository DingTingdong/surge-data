"""大跌的時候，哪種股票比較扛跌、跌完反彈比較快。

全部在看股票結果之前寫下（2026/11/18 09:00），看到結果後不修改。（寫之前只看過等權指數本身的走勢與大跌日期。）

大盤：分析範圍（20日均額 ≥ 1,000 萬）所有股票的等權指數，同 research/regime.py。
大跌事件：等權指數收盤比近20個交易日最高收盤低 ≥ 6% 的第一天＝「觸發日」t；之後指數回到「離高點不到3%」才算結束，
  下一次跌破 6% 才是新事件。「高點日」＝觸發日前20個交易日內指數最高的那天；「低點日」＝事件期間指數最低的那天（事後才知道）。
股票：觸發日在分析範圍內的股票。

A. 事先能知道的（觸發日收盤就能算），結果＝隔天開盤買，持有 5／10／20 個交易日收盤的報酬，減去同期間分析範圍等權平均（超額報酬）：
  F1 扛跌：高點日到觸發日的漲跌幅，前10%（跌最少） vs 後10%（跌最多）
  F2 大跌前強：高點日之前60個交易日的漲幅，前10% vs 後10%
  F3 法人逆勢買：高點日隔天到觸發日，三大法人合計買超 > 0 vs ≤ 0
  F4 投信逆勢買：同期間投信買超 > 0 vs ≤ 0
  F5 大股本：股本前10% vs 後10%（股本是現在的資料）
  F6 高波動：高點日的20日波動，前10% vs 後10%
  F7 觸發日當天是「仍在動」 vs 不是
  判斷「穩定有用」：10日超額報酬的差（前組−後組）在 2020–2023 事件與 2024–2026 事件「方向相同、平均都 ≥ 2 個百分點」，
  而且兩段各自至少 2/3 的事件方向一致。
B. 事後描述（低點只有事後才知道，不能直接拿來交易）：高點日到低點日跌幅分10組，看低點日之後10個交易日的反彈。
C. 這次（2026/10 起）的實際名單：10/28→11/06 下跌段、11/06→11/13 反彈段、11/13→11/17 下跌段，
  各列出分析範圍內表現最好的股票，並看兩段下跌都扛得住的有哪些。
"""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radar_core import load_prices, panels, features, tiers

p = load_prices(); P = panels(p)
C, O = P["C"], P["O"]
f = features(P); S = tiers(f, P["MK"])
days = C.index; N = len(days)
name = p.sort_values("date").drop_duplicates("code", keep="last").set_index("code").name
inb = S[S.base].assign(v=True).pivot(index="date", columns="code", values="v").reindex(index=days, columns=C.columns).fillna(False).astype(bool)
hot = S[S.hot].assign(v=True).pivot(index="date", columns="code", values="v").reindex(index=days, columns=C.columns).fillna(False).astype(bool)
r1 = C.pct_change(fill_method=None).where(inb)
ew = (1 + r1.mean(axis=1).fillna(0)).cumprod()
dd = ew / ew.rolling(20, min_periods=1).max() - 1
ind = pd.read_csv("data/industry.csv", dtype={"code": str}).set_index("code")
cap = pd.Series(ind["cap"]).reindex(C.columns)
g = pd.Series(ind["ind_name"]).reindex(C.columns)

ch = pd.concat([pd.read_csv(fn, dtype={"code": str}, usecols=["date", "code", "trust", "inst"]) for fn in sorted(glob.glob("data/chips/*.csv*"), key=lambda x: (os.path.basename(x)[:4], x.endswith(".csv")))], ignore_index=True)
ch["date"] = pd.to_datetime(ch.date); ch = ch.drop_duplicates(["date", "code"], keep="last")
inst = ch.pivot(index="date", columns="code", values="inst").reindex(index=days, columns=C.columns)
trust = ch.pivot(index="date", columns="code", values="trust").reindex(index=days, columns=C.columns)

# 事件
ev = []; i = 0; v = dd.values
while i < N:
    if v[i] <= -0.06:
        j = i
        while j < N and v[j] <= -0.03: j += 1
        pk = int(np.argmax(ew.values[max(0, i - 19):i + 1])) + max(0, i - 19)
        tr = int(np.argmin(ew.values[i:j])) + i
        ev.append({"t": i, "pk": pk, "tr": tr, "end": j, "done": j < N}); i = j
    else: i += 1

def fwd(t, h):
    """t+1 開盤買、第 h 天收盤的報酬（%），與同期間分析範圍等權平均的差。"""
    e, x = t + 1, t + h
    if x >= N: return None, None
    r = (C.iloc[x] / O.iloc[e] - 1) * 100
    m = inb.iloc[t]
    r = r.where(m)
    return r, r - r.mean()

A = {}
rows = []
for k, E in enumerate(ev):
    t, pk = E["t"], E["pk"]; d = days[t]
    m = inb.iloc[t] & C.iloc[t].notna() & O.iloc[min(t + 1, N - 1)].notna()
    feat = pd.DataFrame({
        "F1": (C.iloc[t] / C.iloc[pk] - 1) * 100,
        "F2": (C.iloc[pk] / C.iloc[max(0, pk - 60)] - 1) * 100,
        "F3": inst.iloc[pk + 1:t + 1].sum(min_count=1),
        "F4": trust.iloc[pk + 1:t + 1].sum(min_count=1),
        "F5": cap, "F6": f["v"].iloc[pk], "F7": hot.iloc[t].astype(float)}).where(m, axis=0)
    out = {}
    for h in (5, 10, 20):
        _, ex = fwd(t, h)
        if ex is None: continue
        for fn in feat:
            x = feat[fn]
            if fn in ("F3", "F4"): top, bot = x > 0, x <= 0
            elif fn == "F7": top, bot = x == 1, x == 0
            else:
                q = x.rank(pct=True); top, bot = q > 0.9, q <= 0.1
            a, b = ex[top & ex.notna()], ex[bot & ex.notna()]
            out[(fn, h)] = (a.mean() - b.mean(), len(a), len(b))
    rows.append({"date": d.date().isoformat(), "year": d.year, "train": d.year <= 2023, "idx_drop_at_t": round(dd.iloc[t] * 100, 1),
                 **{f"{fn}_{h}": round(val[0], 2) for (fn, h), val in out.items()}})
T = pd.DataFrame(rows)
summary = {}
for fn in ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]:
    s = {}
    for per, msk in (("2020-2023", T.train), ("2024-2026", ~T.train)):
        for h in (5, 10, 20):
            c = f"{fn}_{h}"
            x = T.loc[msk, c].dropna() if c in T else pd.Series(dtype=float)
            s[f"{per}_{h}"] = {"avg": round(x.mean(), 2) if len(x) else None, "pos_share": round((x > 0).mean() * 100) if len(x) else None, "n_ev": int(len(x))}
    a, b = s["2020-2023_10"], s["2024-2026_10"]
    ok = (a["avg"] is not None and b["avg"] is not None and np.sign(a["avg"]) == np.sign(b["avg"]) and min(abs(a["avg"]), abs(b["avg"])) >= 2
          and (a["pos_share"] >= 67 or a["pos_share"] <= 33) and (b["pos_share"] >= 67 or b["pos_share"] <= 33)
          and (a["pos_share"] >= 50) == (a["avg"] > 0) and (b["pos_share"] >= 50) == (b["avg"] > 0))
    s["stable"] = bool(ok); summary[fn] = s

# B. 事後描述：跌幅分組 vs 低點後10日反彈
B = []
for E in ev:
    pk, tr = E["pk"], E["tr"]
    if tr + 10 >= N: continue
    m = inb.iloc[tr] & C.iloc[pk].notna()
    dr = ((C.iloc[tr] / C.iloc[pk] - 1) * 100).where(m)
    rb = ((C.iloc[tr + 10] / C.iloc[tr] - 1) * 100).where(m)
    q = pd.qcut(dr.rank(method="first"), 10, labels=False)
    B.append(rb.groupby(q).mean() - rb.mean())
Bd = pd.concat(B, axis=1)
descB = {"by_drop_decile_excess_rebound10": [round(x, 2) for x in Bd.mean(axis=1)], "n_events": int(Bd.shape[1]),
         "worst_decile_beats_best_share": round(float((Bd.iloc[0] > Bd.iloc[9]).mean() * 100))}

# C. 這次的名單
def seg(a, b):
    ia, ib = days.get_loc(pd.Timestamp(a)), days.get_loc(pd.Timestamp(b))
    m = inb.iloc[ib] & C.iloc[ia].notna() & C.iloc[ib].notna()
    return ((C.iloc[ib] / C.iloc[ia] - 1) * 100).where(m).dropna()
d1, rb1, d2 = seg("2026-10-28", "2026-11-06"), seg("2026-11-06", "2026-11-13"), seg("2026-11-13", "2026-11-17")
def top(x, n=15): return [[c, name.get(c, ""), g.get(c) if isinstance(g.get(c), str) else "", round(v, 1)] for c, v in x.sort_values(ascending=False).head(n).items()]
both = pd.concat([d1.rank(pct=True), d2.rank(pct=True)], axis=1).dropna()
both = both[(both[0] >= 0.8) & (both[1] >= 0.8)]
recent = {"idx": {"drop1": round((ew["2026-11-06"] / ew["2026-10-28"] - 1) * 100, 1), "rebound": round((ew["2026-11-13"] / ew["2026-11-06"] - 1) * 100, 1),
                  "drop2": round((ew["2026-11-17"] / ew["2026-11-13"] - 1) * 100, 1)},
          "drop1_best": top(d1), "rebound_best": top(rb1), "drop2_best": top(d2),
          "both_drops_top20pct": [[c, name.get(c, ""), g.get(c) if isinstance(g.get(c), str) else "", round(d1[c], 1), round(rb1.get(c, np.nan), 1), round(d2[c], 1)] for c in both.index],
          "corr_drop1_vs_rebound": round(float(pd.concat([d1, rb1], axis=1).dropna().corr(method="spearman").iloc[0, 1]), 2),
          "corr_rebound_vs_drop2": round(float(pd.concat([rb1, d2], axis=1).dropna().corr(method="spearman").iloc[0, 1]), 2),
          "corr_drop1_vs_drop2": round(float(pd.concat([d1, d2], axis=1).dropna().corr(method="spearman").iloc[0, 1]), 2)}
gi = pd.DataFrame({"g": g, "d1": d1, "rb": rb1, "d2": d2}).dropna(subset=["g", "d1"])
gg = gi.groupby("g").agg(n=("d1", "size"), d1=("d1", "median"), rb=("rb", "median"), d2=("d2", "median"))
recent["by_industry"] = gg[gg.n >= 10].round(1).sort_values("d2", ascending=False).reset_index().values.tolist()

out = {"events": T.to_dict("records"), "summary": summary, "hindsight": descB, "recent": recent}
json.dump(out, open("research/results/drops.json", "w"), ensure_ascii=False, indent=1, default=str)
print(T[["date", "idx_drop_at_t"] + [c for c in T.columns if c.endswith("_10")]].to_string())
for fn, s in summary.items(): print(fn, "stable" if s["stable"] else "-", {k: v for k, v in s.items() if k != "stable"})
print("B", descB)
print(json.dumps({k: v for k, v in recent.items() if k.startswith(("idx", "corr"))}, ensure_ascii=False))
