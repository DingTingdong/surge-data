"""雷達的名單，能不能贏「直接買 0050」？（新高突破＝主要檢驗；今日發動＝描述，回答「是不是賠錢」）

全部在看結果之前寫下（2026/11/22 14:05），看到結果後不修改。
資料：data/older ＋ data/old ＋ data/history ＋ prices.csv 接成 2010-01～2026-11 一段（之前「2020–23」只有 2023 起，因為新資料不夠 750 天歷史）。
恐慌：2020 前＝空頭家數 ÷ 有60日線上市檔數 近10日最高 ≥ 0.7499；2020 起＝空頭家數近10日最高 ≥ 750。主要檢驗只用平常時期。
0050：含息（除權息還原）。期間：2011–15（新高突破約 2013 起）、2016–19、2020–23、2024–26。

新高突破（定義同網站與 O3）：收盤首次高於此前所有收盤（至少 750 個交易日歷史）、前 60 日沒創過、當天漲幅 < 9%（2015/6 前 < 6.5%）、
  在分析範圍內（20 日均額 ≥ 1,000 萬）。隔天開盤買（開盤一價漲停買不到不計），抱 60 個交易日收盤賣，扣 0.6%。
  分數 cs：12 個「趨勢＋營收」特性，各自對「事件日當天或之前最近一次每月快照（每月 11 日後第一個交易日的分析範圍）」算百分位，缺值給 0.5，等權平均。
  彙總：每天先平均，月內日期等權，每月一筆；t＝Newey-West（落後 3）。
N1（主要）：全部新高突破，60 日報酬扣 0.6% 減 0050 同期。
N2（主要）：新高突破且 cs ≥ 0.5（網站不標「趨勢營收弱」的那些），同上。
  兩項各自判斷，三項都要符合才成立：① 全期平均 > 0 且 NW t ≥ 2.9；② 四段期間平均都 > 0；③ 同一家公司一年只算一次，平均仍 > 0。
網站怎麼改（事先決定）：N2 成立 → 「新高突破（分數 ≥ 0.5）」改成第一個分頁、標成可考慮買進的名單；只有 N1 成立 → 全部新高突破當第一個分頁；
  都不成立 → 網站明寫「沒有任何分頁證明能贏 0050，名單只當觀察」，分頁順序不動。
另報（不判定）：新高突破 抱 20／120 日、移動停損（收盤比買進後最高收盤低 10% → 隔天開盤賣，最多 60 日）、cs ≥ 0.8、cs < 0.5、成本 1.1%、
  減等權分析範圍、恐慌期間的事件；絕對報酬的平均、中位數、賺錢比例。
今日發動（只描述，不判定）：平常時期、分析範圍內，20 日扣 1.1% 的絕對報酬（平均、中位數、賺錢比例）與減 0050，
  依四段期間與分數 <0.5／0.5–0.8／≥0.8 分開；另報網站建議的賣法（移動停損 10%、最多 20 日）。
"""
import glob, json, os, shutil, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, REPO)
COMP = ["r20", "mom6_1", "d_hi250", "m60", "am", "yoy1", "yoy3", "rev_acc", "rev_hi12", "mom1", "mg20", "rs_g20"]
LOCK = 0.095; CACHE = "/tmp/model_v2/vs0050.pkl"
FDX = os.environ.get("FD_OFFSETS")  # 11/22 加：例如 "0,2,4"＝每月 11 日後第 1、3、5 個交易日都存快照（給 top20_goal 第 4 輪用）；不設時結果完全不變
if FDX: CACHE = CACHE.replace(".pkl", "_fdx.pkl")
OLDEST = os.environ.get("INCLUDE_OLDEST") == "1"  # 11/22 加：多讀 data/oldest（2005–2009，給 oos_2006.py）；不設時結果完全不變
if OLDEST: CACHE = CACHE.replace(".pkl", "_oldest.pkl")
OLDEST_ONLY = os.environ.get("OLDEST_ONLY") == "1"  # 11/23 加：只讀 2005–2010（data/oldest＋data/older 的 2010），整段 2005–2026 會超過記憶體
if OLDEST_ONLY: OLDEST = True; CACHE = "/tmp/model_v2/vs0050_oldestonly.pkl"
PER = ("2011-15", "2016-19", "2020-23", "2024-26")

def make_root():
    r = "/tmp/root_all" + ("_oldestonly" if OLDEST_ONLY else "_oldest" if OLDEST else ""); d = r + "/data"
    if os.path.exists(r): shutil.rmtree(r)
    for sub in ("history", "chips", "revenue"): os.makedirs(f"{d}/{sub}")
    hist, chips, rev = {}, {}, {}
    keep_y = (lambda fn: os.path.basename(fn)[:4] <= "2010") if OLDEST_ONLY else (lambda fn: True)
    for s in (("oldest/",) if OLDEST else ()) + ((("older/",)) if OLDEST_ONLY else ("older/", "old/", "")):
        for fn in sorted(glob.glob(f"{REPO}/data/{s}history/*.csv.gz")):
            if keep_y(fn): hist.setdefault(os.path.basename(fn), []).append(fn)
        for fn in sorted(glob.glob(f"{REPO}/data/{s}chips/*.csv*")):
            if keep_y(fn): chips.setdefault(os.path.basename(fn), []).append(fn)
    for s in ((("oldest/", "older/")) if OLDEST_ONLY else (("",) + (("oldest/",) if OLDEST else ()) + ("older/", "old/"))):  # 營收同年份（oldest 的 2009 是整年，older 的 2009 只有 12 月）：新資料優先，其次 older（同 offense.py／older_era.py）
        for fn in sorted(glob.glob(f"{REPO}/data/{s}revenue/*.csv*")):
            if keep_y(fn): rev.setdefault(os.path.basename(fn), fn)
    for dic, sub in ((hist, "history"), (chips, "chips")):
        for name, fns in dic.items():
            if len(fns) == 1: os.symlink(fns[0], f"{d}/{sub}/{name}")
            else: pd.concat([pd.read_csv(x, dtype={"code": str}) for x in fns]).drop_duplicates(["date", "market", "code"], keep="last").to_csv(f"{d}/{sub}/{name}", index=False)
    for name, fn in rev.items(): os.symlink(fn, f"{d}/revenue/{name}")
    for name in ("adjust.csv", "disposal.csv"):
        pd.concat([pd.read_csv(f"{REPO}/data/{s}{name}", dtype={"code": str}) for s in (("oldest/",) if OLDEST else ()) + ("older/", "old/", "") if os.path.exists(f"{REPO}/data/{s}{name}")]).drop_duplicates().to_csv(f"{d}/{name}", index=False)
    for name in (("industry.csv",) if OLDEST_ONLY else ("industry.csv", "prices.csv")): os.symlink(f"{REPO}/data/{name}", f"{d}/{name}")
    print("合併的年份檔：", {k: len(v) for k, v in hist.items() if len(v) > 1}, {k: len(v) for k, v in chips.items() if len(v) > 1}, flush=True)
    return r

def bench(days):
    fs = sorted(glob.glob("data/history/*.csv.gz")) + (["data/prices.csv"] if os.path.exists("data/prices.csv") else [])
    s = pd.concat([pd.read_csv(f, usecols=["date", "code", "close"], dtype={"code": str}).query("code == '0050'") for f in fs])
    s = s.assign(date=lambda z: pd.to_datetime(z.date)).drop_duplicates("date", keep="last").set_index("date").close.sort_index()
    adj = pd.read_csv("data/adjust.csv", dtype={"code": str}); adj = adj[adj.code == "0050"].assign(date=lambda z: pd.to_datetime(z.date)).groupby("date").ratio.prod()
    r = s / (s.shift(1) * adj.reindex(s.index).fillna(1.0)); r = r.where((r > 0.45) & (r < 2.2), 1.0).fillna(1.0)
    b = r.cumprod().reindex(days); print("0050 缺的交易日：", int(b.isna().sum()), "每年報酬%：", (b.ffill().groupby(b.index.year).last().pct_change() * 100).round(1).to_dict(), flush=True)
    return b.ffill()

def build():
    if os.path.exists(CACHE): return pd.read_pickle(CACHE)
    os.chdir(make_root())
    from radar_core import load_prices, panels, features, tiers
    p = load_prices(); P = panels(p); del p
    C, O, H, L, A = P["C"], P["O"], P["H"], P["L"], P["A"]
    f = features(P); S = tiers(f, P["MK"]); days, cols = C.index, C.columns; N, J = len(days), len(cols)
    alive = (C.notna().cumsum() > 0) & (C.notna()[::-1].cumsum()[::-1] > 0); Cf = C.ffill(limit=20).where(alive)
    mm = {n: Cf.rolling(n, min_periods=n).mean() for n in (5, 10, 20, 60)}; okm = Cf.notna() & mm[60].notna()
    mk = P["MK"].ffill().iloc[-1]; tw = [c for c in cols if mk.get(c) == "TWSE"]
    bear = ((mm[5] < mm[10]) & (mm[10] < mm[20]) & okm)[tw].sum(axis=1); okn = okm[tw].sum(axis=1); del Cf, mm, alive
    panic = pd.Series(np.where(days < pd.Timestamp("2020-01-01"), (bear / okn).rolling(10, min_periods=1).max() >= 0.7499,
                               bear.rolling(10, min_periods=1).max() >= 750), index=days)
    lim = pd.Series(np.where(days < pd.Timestamp("2015-06-01"), 6.5, 9.0), index=days)
    ch = f["ch"]; islu = ch.ge(lim, axis=0).values
    idx = np.arange(N)[:, None]; pos = pd.DataFrame(np.where(islu, idx, np.nan)).ffill().shift(1).values
    snp = np.where(np.isnan(idx - pos), 99, idx - pos)
    a1 = (A / 1e6).values; a20p = (A.shift(1).rolling(20, min_periods=20).mean() / 1e6).values
    pkm = islu & (snp > 10) & (a1 >= 20) & (a20p >= 3)
    inb = S[S.base].assign(v=True).pivot(index="date", columns="code", values="v").reindex(index=days, columns=cols).fillna(False).astype(bool).values
    Ov, Hv, Lv, Cv = O.values, H.values, L.values, C.values; prevc = C.ffill().shift(1).values
    Onext = np.vstack([Ov[1:], np.full((1, J), np.nan)])
    lockn = np.vstack([((Ov == Hv) & (Hv == Lv) & (Ov / prevc - 1 >= LOCK - (0.03 * (days < pd.Timestamp("2015-06-01")))[:, None]))[1:], np.zeros((1, J), bool)])
    bidx = bench(days).values
    cnt = np.cumsum(~np.isnan(Cv), axis=0); prior = np.vstack([np.zeros((1, J)), cnt[:-1]])
    pm = pd.DataFrame(Cv).shift(1).expanding().max().values; isath = (Cv > pm) & (prior >= 750)
    recent = pd.DataFrame(isath.astype(float)).shift(1).rolling(60, min_periods=1).max().fillna(0).values > 0
    nhm = isath & ~recent & inb & (ch.values < lim.values[:, None])
    dser = pd.Series(days); _gm = dser[dser.dt.day >= 11].groupby(dser.dt.to_period("M")); fd = set(_gm.first()); fdx = set(fd)
    if FDX:
        for _k in [int(x) for x in FDX.split(",") if int(x) > 0]: fdx |= set(_gm.nth(_k))
    dpos = {d: i for i, d in enumerate(days)}; cpos = {c: j for j, c in enumerate(cols)}
    B = S[S.base & S.date.isin(fdx)]; parts = []
    for kind, m in (("pk", pkm), ("nh", nhm)):
        t_, j_ = np.nonzero(m); parts.append(pd.DataFrame({"ti": t_, "ji": j_, "kind": kind}))
    parts.append(pd.DataFrame({"ti": B.date.map(dpos).values, "ji": B.code.map(cpos).values, "kind": "fd"}))
    R = pd.concat(parts, ignore_index=True); ti, ji = R.ti.values.astype(int), R.ji.values.astype(int)
    R["date"] = days[ti]; R["code"] = cols[ji]; R["base"] = inb[ti, ji]; R["panic"] = panic.values[ti]
    for h in (20, 60, 120):
        Cff = C.ffill(limit=min(h, 60)).values; fut = np.vstack([Cff[h:], np.full((h, J), np.nan)])
        y = (fut / Onext - 1) * 100; y[~(Onext > 0) | lockn] = np.nan
        R[f"y{h}"] = y[ti, ji]; R[f"u{h}"] = np.nanmean(np.where(inb, y, np.nan), axis=1)[ti]
        R[f"b{h}"] = np.concatenate([(bidx[h:] / bidx[:-h] - 1) * 100, np.full(h, np.nan)])[ti]; del y, fut
    # 移動停損：收盤比買進後最高收盤低 10% → 隔天開盤賣；最多 Hd 日
    C60 = C.ffill(limit=60).values; yt, bt = np.full(len(R), np.nan), np.full(len(R), np.nan)
    for i, (t, j, k) in enumerate(zip(ti, ji, R.kind.values)):
        if k == "fd": continue
        Hd = 20 if k == "pk" else 60; o = Onext[t, j]
        if not (o > 0) or lockn[t, j] or t + Hd + 1 >= N: continue
        cl = C60[t + 1:t + Hd + 1, j]
        if np.isnan(cl).any(): continue
        hit = np.nonzero(cl <= 0.9 * np.maximum.accumulate(cl))[0]
        if len(hit) and hit[0] < Hd - 1:
            q = t + 1 + hit[0]; ex = Ov[q + 1, j]; ex = ex if ex > 0 else C60[q + 1, j]; bb = bidx[q] / bidx[t]
        else: ex = cl[-1]; bb = bidx[t + Hd] / bidx[t]
        yt[i] = (ex / o - 1) * 100; bt[i] = (bb - 1) * 100
    R["yt"], R["bt"] = yt, bt
    with np.errstate(divide="ignore", invalid="ignore"):
        take = lambda X: (X.values if isinstance(X, pd.DataFrame) else X)[ti, ji]
        r1 = C.pct_change(fill_method=None)
        R["r20"] = take(C / C.shift(20) - 1); R["mom6_1"] = take(C.shift(20) / C.shift(125) - 1)
        R["d_hi250"] = take(C / C.rolling(250, min_periods=120).max() - 1); R["m60"] = take(C / C.rolling(60, min_periods=60).mean() - 1)
        R["am"] = take(A.rolling(5, min_periods=5).mean() / A.rolling(20, min_periods=10).mean())
        chp = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in sorted(glob.glob("data/chips/*.csv*"))], ignore_index=True)
        chp["date"] = pd.to_datetime(chp.date); chp = chp.drop_duplicates(["date", "code"], keep="last")
        mg = chp.pivot(index="date", columns="code", values="margin").reindex(index=days, columns=cols); R["mg20"] = take(mg / mg.shift(20) - 1); del mg, chp
        rv = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in sorted(glob.glob("data/revenue/*.csv*"))], ignore_index=True)
        rv["pub"] = pd.to_datetime(rv.ym + "-01") + pd.DateOffset(months=1) + pd.Timedelta(days=10)
        piv = lambda v: rv.pivot_table(index="ym", columns="code", values=v).sort_index().reindex(columns=cols)
        yoy = piv("yoy"); rev = piv("rev").reindex(yoy.index); mom = piv("mom").reindex(yoy.index); pub = rv.groupby("ym").pub.first().sort_index().reindex(yoy.index)
        kn = np.searchsorted(pub.values, R.date.values, side="right") - 1
        g = lambda M, k_: np.where(k_ >= 0, M[np.maximum(k_, 0), ji], np.nan)
        R["yoy1"] = g(yoy.values, kn)
        y3 = np.vstack([np.where(kn - s >= 0, g(yoy.values, kn - s), np.nan) for s in range(3)]); R["yoy3"] = np.where(kn >= 2, np.nanmean(y3, axis=0), np.nan)
        p12 = np.vstack([np.where(kn - s >= 0, g(rev.values, kn - s), np.nan) for s in range(1, 13)])
        R["rev_hi12"] = np.where(np.isfinite(p12).sum(axis=0) >= 6, g(rev.values, kn) / np.nanmax(p12, axis=0) - 1, np.nan)
        R["rev_acc"] = R.yoy1 - R.yoy3; R["mom1"] = g(mom.values, kn)
        ind = pd.read_csv("data/industry.csv", dtype={"code": str}).set_index("code"); grp = pd.Series(ind["ind_name"]).reindex(cols)
        rb = r1.where(inb); g20 = np.full((N, J), np.nan)
        for gname, cs_ in grp.dropna().groupby(grp.dropna()).groups.items():
            ix = (1 + rb[list(cs_)].mean(axis=1).fillna(0)).cumprod(); jj = [cols.get_loc(c) for c in cs_]
            g20[:, jj] = (ix / ix.shift(20) - 1).values[:, None]
        R["rs_g20"] = R.r20 - g20[ti, ji]
        R["a20"] = take(A.rolling(20, min_periods=10).mean()); R["r120"] = take(C / C.shift(120) - 1)  # 11/22 加：給 directions.py 用，不影響本檔結果
    R = R.replace([np.inf, -np.inf], np.nan)
    pd.to_pickle({"fd": R[R.kind == "fd"].drop(columns=["ti", "ji"]), "days": days, "cols": cols, "C": C.values, "O": Ov, "lockn": lockn, "bidx": bidx, "inb": inb, "panic": panic.values},
                 CACHE.replace(".pkl", "_extra.pkl"))  # 11/22 加：給 directions.py 用
    # 分數：對「事件日當天或之前最近一次每月快照」算百分位
    U = R[(R.kind == "fd") & R.date.isin(fd)]; fds = np.array(sorted(U.date.unique())); K = R[R.kind != "fd"].copy()
    ri = np.searchsorted(fds, K.date.values, side="right") - 1; K["ref"] = np.where(ri >= 0, fds[np.maximum(ri, 0)], np.datetime64("NaT"))
    sc = np.zeros(len(K)); refs = {d: g_ for d, g_ in U.groupby("date")}
    for fc in COMP:
        pr = np.full(len(K), 0.5)
        for d, gg in K.groupby("ref"):
            ref = np.sort(refs[d][fc].dropna().values)
            if len(ref) < 50: continue
            v = gg[fc].values; pr[K.index.get_indexer(gg.index)] = np.where(np.isnan(v), 0.5, np.searchsorted(ref, v, side="right") / len(ref))
        sc += pr
    K["cs"] = sc / len(COMP); os.chdir(REPO)
    K = K[["date", "code", "kind", "base", "panic", "y20", "y60", "y120", "u20", "u60", "u120", "b20", "b60", "b120", "yt", "bt", "cs"]]
    pd.to_pickle(K, CACHE); return K

def nw_t(s, L):
    s = pd.Series(s).dropna().values; n = len(s)
    if n < 5: return np.nan
    x = s - s.mean(); v = (x @ x) / n
    for k in range(1, L + 1): v += 2 * (1 - k / (L + 1)) * (x[k:] @ x[:-k]) / n
    return float(s.mean() / np.sqrt(v / n))
def per(d): return np.where(d < pd.Timestamp("2016-01-01"), "2011-15", np.where(d < pd.Timestamp("2020-01-01"), "2016-19", np.where(d < pd.Timestamp("2024-01-01"), "2020-23", "2024-26")))
def cal(z, col, L):
    z = z[z[col].notna()]
    if len(z) == 0: return {"平均": None, "t": None, "月數": 0, "筆數": 0}
    c = z.groupby([z.date.dt.to_period("M"), "date"])[col].mean().groupby(level=0).mean()
    return {"平均": round(float(c.mean()), 2), "t": round(nw_t(c, L), 2), "月數": int(len(c)), "筆數": int(len(z))}
def absol(z, col):
    v = z[col].dropna(); return {"平均": round(float(v.mean()), 2), "中位數": round(float(v.median()), 2), "賺錢%": round(float((v > 0).mean() * 100), 1), "筆數": int(len(v))}
def dedup_year(z):
    z = z.sort_values("date"); keep, last = [], {}
    for i, d, c in zip(z.index, z.date, z.code):
        if c in last and (d - last[c]).days < 365: continue
        last[c] = d; keep.append(i)
    return z.loc[keep]

if __name__ == "__main__":
    K = build(); K["per"] = per(K.date); K = K[K.date >= "2011-01-01"]
    out = {}
    E = K[(K.kind == "nh") & ~K.panic].copy()
    E["x"] = E.y60 - 0.6 - E.b60; E["a"] = E.y60 - 0.6; E["x11"] = E.y60 - 1.1 - E.b60; E["xu"] = E.y60 - 0.6 - E.u60
    E["x20"] = E.y20 - 0.6 - E.b20; E["x120"] = E.y120 - 0.6 - E.b120; E["xt"] = E.yt - 0.6 - E.bt; E["abt"] = E.yt - 0.6; E["b"] = E.b60
    for lab, z in (("N1 全部新高突破", E), ("N2 新高突破且分數≥0.5", E[E.cs >= 0.5])):
        z = z[z.x.notna()]; a_ = cal(z, "x", 3); ps = {p: cal(z[z.per == p], "x", 3) for p in PER}; dd = cal(dedup_year(z), "x", 3)
        ok = bool(a_["平均"] > 0 and (a_["t"] or 0) >= 2.9 and all((ps[p]["平均"] or -1) > 0 for p in PER) and dd["平均"] > 0)
        out[lab] = {"減0050 全期": a_, "四段": ps, "一年只算一次": dd, "成立": ok,
                    "絕對報酬(逐筆)": absol(z, "a"), "絕對報酬 四段": {p: absol(z[z.per == p], "a") for p in PER},
                    "同期0050(每月)": {p: cal(z[z.per == p], "b", 3)["平均"] for p in PER},
                    "另報": {"20日減0050": cal(z, "x20", 1), "120日減0050": cal(z, "x120", 6), "移動停損減0050": cal(z, "xt", 3),
                            "移動停損 四段": {p: cal(z[z.per == p], "xt", 3) for p in PER}, "移動停損 絕對": absol(z, "abt"),
                            "成本1.1%": cal(z, "x11", 3), "減等權": cal(z, "xu", 3), "減等權 四段": {p: cal(z[z.per == p], "xu", 3) for p in PER}}}
        print("\n==", lab, json.dumps(out[lab], ensure_ascii=False), flush=True)
    out["新高突破 分數分組 減0050(60日)"] = {"cs≥0.8": cal(E[E.cs >= 0.8], "x", 3), "0.5–0.8": cal(E[(E.cs >= 0.5) & (E.cs < 0.8)], "x", 3), "cs<0.5": cal(E[E.cs < 0.5], "x", 3),
                                        "cs≥0.8 四段": {p: cal(E[(E.cs >= 0.8) & (E.per == p)], "x", 3) for p in PER}}
    Ep = K[(K.kind == "nh") & K.panic]; out["新高突破 恐慌期間"] = {"減0050": cal(Ep.assign(x=Ep.y60 - 0.6 - Ep.b60), "x", 3)}
    print("\n==", json.dumps({k: out[k] for k in ("新高突破 分數分組 減0050(60日)", "新高突破 恐慌期間")}, ensure_ascii=False), flush=True)
    # 今日發動（描述）
    P_ = K[(K.kind == "pk") & ~K.panic].copy(); P_["a"] = P_.y20 - 1.1; P_["x"] = P_.a - P_.b20; P_["abt"] = P_.yt - 1.1; P_["xt"] = P_.abt - P_.bt; P_["b"] = P_.b20
    band = lambda z: {"<0.5": z[z.cs < 0.5], "0.5–0.8": z[(z.cs >= 0.5) & (z.cs < 0.8)], "≥0.8": z[z.cs >= 0.8]}
    o = {}
    for scope, z0 in (("分析範圍內", P_[P_.base]), ("全部（含量小）", P_)):
        for p in PER:
            z = z0[z0.per == p]
            o[f"{scope} {p}"] = {"抱20日 絕對": absol(z, "a"), "抱20日 絕對(每月)": cal(z, "a", 1)["平均"], "減0050": cal(z, "x", 1), "同期0050(每月)": cal(z, "b", 1)["平均"],
                                 "移動停損 絕對": absol(z, "abt"), "移動停損 減0050": cal(z, "xt", 1),
                                 "分數分組 絕對/減0050": {k: (absol(v, "a")["平均"], absol(v, "a")["中位數"], cal(v, "x", 1)["平均"], int(len(v))) for k, v in band(z).items()}}
            print(scope, p, json.dumps(o[f"{scope} {p}"], ensure_ascii=False), flush=True)
    out["今日發動（描述）"] = o
    json.dump(out, open(f"{REPO}/research/results/vs0050.json", "w"), ensure_ascii=False, indent=1)
