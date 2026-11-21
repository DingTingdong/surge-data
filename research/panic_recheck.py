"""外部檢查後的重算（2026/11/21；探索）：恐慌期間「今日發動＋再發動」vs「撿跌深前20」、族群仍弱 vs 已反彈，換加權方式結論會不會變。
檢查報告指出：① 逐筆平均被少數訊號特別多的日子主導（2025/04/10 一天 631 筆）；② 沒有去重（同一檔每天都算一筆）；
③ 再發動的「拉回」沒要求高點在低點之前（一路漲也會被算進去）。這裡同時算：
  a 逐筆平均；b 每波內逐筆、再七波等權；c 只用兩組都有訊號的日子，每天先平均、波內日期等權、再七波等權；
  d 去重：同一檔每一波只算第一次訊號，再七波等權。
再發動分成舊定義（網站現行）與修正定義：前 5 日內某天的收盤，相對「到那天為止的近 20 日最高」跌了 ≥ 15%（高點一定在低點之前）。
"""
import os, sys, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
src = open("research/relaunch.py", encoding="utf-8").read().split("# 穩健性")[0]
for a in ('json.dump({"results"', 'for per in res: print(', 'print(yearly.to_string()); print("加進雷達：", add)', 'print("恐慌期間（True）vs 平常（False）："); print(q.to_string())'):
    src = src.replace(a, "pass  # " if a.startswith(("json", "for")) else "")
exec(src)
peak = C.rolling(20, min_periods=10).max(); dd = C / peak - 1
pull_new = dd.rolling(5, min_periods=3).min().shift(1) <= -0.15
B["pull_new"] = [bool(pull_new.at[d, c]) if c in pull_new.columns else False for d, c in zip(B.date, B.code)]
hi20d = C.rolling(20, min_periods=15).max(); lo = C.shift(20).rolling(81, min_periods=40).min(); run = hi20d / lo - 1
inbase = S[S.base].assign(v=True).pivot(index="date", columns="code", values="v").reindex(index=days, columns=C.columns).fillna(False).astype(bool)
runr = run.where(inbase).rank(axis=1, pct=True); lim60 = (f["ch"] >= 9.5).rolling(60, min_periods=1).max().astype(bool)
W2 = (runr > 0.7) & lim60 & (C / hi20d - 1 <= -0.2) & inbase
wave_id = (panic != panic.shift()).cumsum()
def hold20(t, j):
    e = t + 1
    if e + 19 >= N: return None
    o = Ov[e, j]
    if not o == o or o <= 0: return None
    if locked(e, j, True): return "blocked"
    k = e + 19
    while k > e and not Cv[k, j] == Cv[k, j]: k -= 1
    return (Cv[k, j] / o - 1) * 100 - COST
gm10, g5 = {}, {}
ind = pd.read_csv("data/industry.csv", dtype={"code": str}).set_index("code"); grp = ind["ind_name"].reindex(C.columns)
r1 = C.pct_change(fill_method=None).where(inbase)
for gname, cs in grp.dropna().groupby(grp.dropna()).groups.items():
    ix = (1 + r1[list(cs)].mean(axis=1).fillna(0)).cumprod()
    for c in cs: gm10[c] = ix / ix.rolling(10).mean() - 1; g5[c] = ix / ix.shift(5) - 1
rows = []
for d in days[panic.values]:
    t = pos[d]
    if t + 21 >= N: continue
    bd = B[B.date == d]
    sets = {"今日發動": bd[bd.grp == "F 今日發動"].code, "再發動(舊)": bd[bd.grp == "R 再發動"].code,
            "再發動(修正)": bd[(bd.snp <= 10) & ~bd.pk & bd.pull_new].code, "撿跌深前20": run.loc[d][W2.loc[d]].sort_values(ascending=False).index[:20]}
    for kind, codes in sets.items():
        for c in codes:
            j = col[c]; b = hold20(t, j)
            if b is None or b == "blocked": continue
            g = gm10.get(c); gw = np.nan if g is None else (1 if g.iat[t] < 0 else (2 if (g.iat[t] > 0 and g5[c].iat[t] > 0) else 0))
            rows.append({"date": d, "wave": int(wave_id[d]), "kind": kind, "code": c, "h20": b, "gw": gw})
T = pd.DataFrame(rows); T["wl"] = T.wave.map({w: T[T.wave == w].date.min().strftime("%Y-%m") for w in T.wave.unique()})
def metrics(z, common_days=None):
    a = z.h20.mean(); b = z.groupby("wave").h20.mean().mean()
    zc = z if common_days is None else z[z.date.isin(common_days)]
    c = zc.groupby(["wave", "date"]).h20.mean().groupby("wave").mean().mean()
    dz = z.sort_values("date").drop_duplicates(["wave", "code"]); d_ = dz.groupby("wave").h20.mean().mean()
    lossc = zc.assign(l=zc.h20 < -10).groupby(["wave", "date"]).l.mean().groupby("wave").mean().mean() * 100
    return {"筆數": len(z), "a逐筆": round(a, 2), "b波等權": round(b, 2), "c同日·日期·波等權": round(c, 2), "d去重·波等權": round(d_, 2), "c口徑虧損≥10%比例": round(lossc, 1)}
out = {}
for k1 in ("今日發動＋再發動(舊)", "今日發動＋再發動(修正)", "今日發動"):
    parts = {"今日發動＋再發動(舊)": ["今日發動", "再發動(舊)"], "今日發動＋再發動(修正)": ["今日發動", "再發動(修正)"], "今日發動": ["今日發動"]}[k1]
    A = T[T.kind.isin(parts)].drop_duplicates(["date", "code"]); Dd = T[T.kind == "撿跌深前20"]
    common = set(A.date) & set(Dd.date)
    out[k1] = {"發動組": metrics(A, common), "撿跌深前20": metrics(Dd, common), "共同日數": len(common)}
    perw = pd.DataFrame({"發動": A[A.date.isin(common)].groupby(["wl", "date"]).h20.mean().groupby("wl").mean(), "跌深": Dd[Dd.date.isin(common)].groupby(["wl", "date"]).h20.mean().groupby("wl").mean()}).round(2)
    out[k1]["各波(c口徑)"] = perw.to_dict("index")
    print(f"\n== {k1}（共同 {len(common)} 天）\n  發動組 {out[k1]['發動組']}\n  撿跌深 {out[k1]['撿跌深前20']}\n{perw.to_string()}")
for k in ("再發動(舊)", "再發動(修正)"):
    z = T[T.kind == k]; print(k, "筆數", len(z), metrics(z))
old_only = set(zip(T[T.kind == "再發動(舊)"].date, T[T.kind == "再發動(舊)"].code)) - set(zip(T[T.kind == "再發動(修正)"].date, T[T.kind == "再發動(修正)"].code))
print("舊定義有、修正後沒有（沒有真正拉回）的筆數：", len(old_only))
# 族群仍弱 vs 已反彈（今日發動＋再發動修正）
A = T[T.kind.isin(["今日發動", "再發動(修正)"])].drop_duplicates(["date", "code"])
for lab, z in (("族群仍弱", A[A.gw == 1]), ("族群已反彈", A[A.gw == 2])): print(lab, metrics(z))
wv = pd.DataFrame({"仍弱": A[A.gw == 1].groupby(["wl", "date"]).h20.mean().groupby("wl").mean(), "已反彈": A[A.gw == 2].groupby(["wl", "date"]).h20.mean().groupby("wl").mean()}).round(2)
print("各波（每天先平均、波內日期等權）：\n", wv.to_string(), "\n仍弱較好的波數：", int((wv.仍弱 > wv.已反彈).sum()), "/", int(wv.dropna().shape[0]))
json.dump({"compare": out}, open("research/results/panic_recheck.json", "w"), ensure_ascii=False, indent=1, default=str)
T.to_csv("research/results/panic_recheck_trades.csv", index=False)
