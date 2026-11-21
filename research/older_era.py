"""更早的全新期間 2011-01～2015-05：之前的主要發現在這段從沒看過的資料上是否也成立。

全部在資料下載完成之前寫下（2026/11/21 23:10），看到結果後不修改。
資料：data/older（2010-01～2015-05；2010 年只當暖身）接上 data/old，讓 2015 年上半的訊號能算到之後的報酬。
這段期間的不同：漲跌幅限制 7%（2015/6/1 起才是 10%），所以「漲停」改成當天漲 ≥ 6.5%（10% 時代用 ≥ 9%）；
  上櫃法人資料抓不到、上市法人從 2012/5 起才有；月營收 2013 年改採 IFRS，營收金額前後基礎不同（年增率用官方數字）；
  市場成交額比現在小很多，分析範圍（20 日均額 ≥ 1,000 萬）和今日發動（當天 ≥ 2,000 萬、前20日平均 ≥ 300 萬）照原本名目門檻。
恐慌：空頭家數 ÷ 有60日線上市檔數，近10日最高 ≥ 0.7499（同 2016–2019 的做法）。成本：今日發動 1.1%，月選股 0.6%×換股比例。

O1 「趨勢＋營收」綜合分數（research/deep_normal.py 的 12 個特性、每月 11 日後第一個交易日、只算平常月份）：
   每月前20檔扣成本後勝等權分析範圍：每月平均 > 0 且 t ≥ 2 → 成立。另報相對 0050、五等分差。
O2 防守：平常時期今日發動（7% 時代用 ≥ 6.5%）中，綜合分數 < 0.5 者抱20日扣1.1%後減同期 0050：
   以每月平均算，平均 < 0 且 t ≤ −2 → 成立。另報減等權分析範圍、分數 ≥ 0.8 組。
O3 非漲停越過 3 年以上新高（至少 750 個交易日歷史、前60日沒創過、當天漲幅 < 6.5%（2015/6 起 < 9%）、在分析範圍內、平常時期）：
   60 日扣 0.6% 後減同一進場日等權分析範圍：以每月平均算，平均 > 0 且 t ≥ 2 → 成立（事件約從 2013 年開始）。
O4 恐慌：恐慌期間今日發動抱20日扣1.1%：平均 > 0 且比同期平常今日發動高 ≥ 3 個百分點 → 成立；
   恐慌期間「族群仍弱」平均 > 「族群已反彈」 → 成立（定義同 research/panic_old.py）。另報每一波。
"""
import glob, json, os, shutil, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO)
COMP = ["r20", "mom6_1", "d_hi250", "m60", "am", "yoy1", "yoy3", "rev_acc", "rev_hi12", "mom1", "mg20", "rs_g20"]
LOCK = 0.095

def make_root(kind):
    """kind = older：data/older ＋ data/old；check：只用 data/old（拿來和 2016–2019 已知結果對照，檢查程式）。"""
    r = f"/tmp/root_{kind}"; d = r + "/data"
    if os.path.exists(r): shutil.rmtree(r)
    for sub in ("history", "chips", "revenue"): os.makedirs(f"{d}/{sub}")
    if kind == "new":  # 2020–2026（網站用的資料）
        for sub in ("history", "chips", "revenue"):
            for fn in glob.glob(f"{REPO}/data/{sub}/*.csv*"): os.symlink(fn, f"{d}/{sub}/{os.path.basename(fn)}")
        for name in ("adjust.csv", "disposal.csv", "industry.csv", "prices.csv"): os.symlink(f"{REPO}/data/{name}", f"{d}/{name}")
        return r
    srcs = (["older"] if kind == "older" else []) + ["old"]
    hist = {}; chips = {}; rev = {}
    for s in srcs:
        for fn in glob.glob(f"{REPO}/data/{s}/history/*.csv.gz"): hist.setdefault(os.path.basename(fn), []).append(fn)
        for fn in glob.glob(f"{REPO}/data/{s}/chips/*.csv.gz"): chips.setdefault(os.path.basename(fn), []).append(fn)
        for fn in glob.glob(f"{REPO}/data/{s}/revenue/*.csv.gz"): rev.setdefault(os.path.basename(fn), fn)  # 同年份以先列者（older）為準
    for name, fns in hist.items():
        if len(fns) == 1: os.symlink(fns[0], f"{d}/history/{name}")
        else: pd.concat([pd.read_csv(x, dtype={"code": str}) for x in fns]).drop_duplicates(["date", "market", "code"], keep="last").to_csv(f"{d}/history/{name}", index=False)
    for name, fns in chips.items():
        if len(fns) == 1: os.symlink(fns[0], f"{d}/chips/{name}")
        else: pd.concat([pd.read_csv(x, dtype={"code": str}) for x in fns]).drop_duplicates(["date", "market", "code"], keep="last").to_csv(f"{d}/chips/{name}", index=False)
    for name, fn in rev.items(): os.symlink(fn, f"{d}/revenue/{name}")
    for name in ("adjust.csv", "disposal.csv"):
        pd.concat([pd.read_csv(f"{REPO}/data/{s}/{name}", dtype={"code": str}) for s in srcs]).drop_duplicates().to_csv(f"{d}/{name}", index=False)
    os.symlink(f"{REPO}/data/industry.csv", f"{d}/industry.csv")
    return r

def bench_from_history(days):
    """0050 含息指數：從歷史行情原始檔（含 0050）＋除權息比例。"""
    fs = sorted(glob.glob("data/history/*.csv.gz"))
    s = pd.concat([pd.read_csv(f, usecols=["date", "code", "close"], dtype={"code": str}).query("code == '0050'") for f in fs])
    s = s.assign(date=lambda z: pd.to_datetime(z.date)).drop_duplicates("date").set_index("date").close.sort_index()
    adj = pd.read_csv("data/adjust.csv", dtype={"code": str}); adj = adj[adj.code == "0050"].assign(date=lambda z: pd.to_datetime(z.date)).groupby("date").ratio.prod()
    r = s / (s.shift(1) * adj.reindex(s.index).fillna(1.0)); r = r.where((r > 0.45) & (r < 2.2), 1.0).fillna(1.0)
    return r.cumprod().reindex(days).ffill()

def build(kind):
    os.chdir(make_root(kind))
    from radar_core import load_prices, panels, features, tiers
    p = load_prices(); P = panels(p); del p
    C, O, H, L, A, raw = P["C"], P["O"], P["H"], P["L"], P["A"], P["rawC"]
    f = features(P); S = tiers(f, P["MK"]); days, cols = C.index, C.columns; N = len(days)
    alive = (C.notna().cumsum() > 0) & (C.notna()[::-1].cumsum()[::-1] > 0); Cf = C.ffill(limit=20).where(alive)
    mm = {n: Cf.rolling(n, min_periods=n).mean() for n in (5, 10, 20, 60)}; okm = Cf.notna() & mm[60].notna()
    mk = P["MK"].ffill().iloc[-1]; tw = [c for c in cols if mk.get(c) == "TWSE"]
    bear = ((mm[5] < mm[10]) & (mm[10] < mm[20]) & okm)[tw].sum(axis=1); okn = okm[tw].sum(axis=1); del Cf, mm, alive
    panic = ((bear / okn).rolling(10, min_periods=1).max() >= 0.7499)
    # 漲停門檻：2015/6/1 前 6.5%，之後 9%
    lim = pd.Series(np.where(days < pd.Timestamp("2015-06-01"), 6.5, 9.0), index=days)
    ch = f["ch"]; islu = ch.ge(lim, axis=0)
    idx = np.arange(N)[:, None]; pos = pd.DataFrame(np.where(islu.values, idx, np.nan)).ffill().shift(1).values
    snp = pd.DataFrame(idx - pos, index=days, columns=cols).fillna(99)
    a1 = A / 1e6; a20p = A.shift(1).rolling(20, min_periods=20).mean() / 1e6
    pkm = islu & (snp > 10) & (a1 >= 20) & (a20p >= 3)
    inb = S[S.base].assign(v=True).pivot(index="date", columns="code", values="v").reindex(index=days, columns=cols).fillna(False).astype(bool)
    Ov, Hv, Lv = O.values, H.values, L.values; prevc = C.ffill().shift(1).values
    Onext = np.vstack([Ov[1:], np.full((1, len(cols)), np.nan)])
    lockn = np.vstack([((Ov == Hv) & (Hv == Lv) & (Ov / prevc - 1 >= LOCK - (0.03 * (days < pd.Timestamp("2015-06-01")))[:, None]))[1:], np.zeros((1, len(cols)), bool)])
    YH, FW = {}, {}; bidx = bench_from_history(days).values
    for h in (20, 60):
        Cff = C.ffill(limit=h).values; fut = np.vstack([Cff[h:], np.full((h, len(cols)), np.nan)])
        y = (fut / Onext - 1) * 100; y[~(Onext > 0) | lockn] = np.nan; YH[h] = y
        FW[h] = np.nanmean(np.where(inb.values, y, np.nan), axis=1); FW[f"b{h}"] = np.concatenate([(bidx[h:] / bidx[:-h] - 1) * 100, np.full(h, np.nan)])
    dser = pd.Series(days); fd = set(dser[dser.dt.day >= 11].groupby(dser.dt.to_period("M")).first())
    # 列：每月快照（分析範圍）＋ 今日發動（7%／10% 門檻）
    ti_pk, ji_pk = np.nonzero(pkm.values)
    rows = pd.DataFrame({"ti": ti_pk, "ji": ji_pk, "pk": True})
    B = S[S.base & S.date.isin(fd)]; dpos = {d: i for i, d in enumerate(days)}; cpos = {c: j for j, c in enumerate(cols)}
    rows = pd.concat([rows, pd.DataFrame({"ti": B.date.map(dpos).values, "ji": B.code.map(cpos).values, "pk": False})], ignore_index=True)
    rows["fd"] = ~rows.pk; R = rows; ti, ji = R.ti.values, R.ji.values
    R["date"] = days[ti]; R["code"] = cols[ji]; R["base"] = inb.values[ti, ji]; R["panic"] = panic.values[ti]
    take = lambda X: (X.values if isinstance(X, pd.DataFrame) else X)[ti, ji]
    for h in (20, 60): R[f"y{h}"] = YH[h][ti, ji]; R[f"u{h}"] = FW[h][ti]; R[f"b{h}"] = FW[f"b{h}"][ti]
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = C.pct_change(fill_method=None)
        R["r20"] = take(C / C.shift(20) - 1); R["mom6_1"] = take(C.shift(20) / C.shift(125) - 1)
        R["d_hi250"] = take(C / C.rolling(250, min_periods=120).max() - 1); R["m60"] = take(C / C.rolling(60, min_periods=60).mean() - 1)
        a20 = A.rolling(20, min_periods=10).mean(); R["am"] = take(A.rolling(5, min_periods=5).mean() / a20)
        fns = sorted(glob.glob("data/chips/*.csv*"))
        chp = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in fns], ignore_index=True)
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
        rb = r1.where(inb); g20, gm10, g5 = (np.full((N, len(cols)), np.nan) for _ in range(3))
        for gname, cs in grp.dropna().groupby(grp.dropna()).groups.items():
            ix = (1 + rb[list(cs)].mean(axis=1).fillna(0)).cumprod(); jj = [cols.get_loc(c) for c in cs]
            g20[:, jj] = (ix / ix.shift(20) - 1).values[:, None]; gm10[:, jj] = (ix / ix.rolling(10).mean() - 1).values[:, None]; g5[:, jj] = (ix / ix.shift(5) - 1).values[:, None]
        R["g20"] = g20[ti, ji]; R["rs_g20"] = R.r20 - R.g20; R["g_m10"] = gm10[ti, ji]; R["g_5"] = g5[ti, ji]
        # 非漲停越過 3 年以上新高
        Cv = C.values; cnt = np.cumsum(~np.isnan(Cv), axis=0); prior = np.vstack([np.zeros((1, len(cols))), cnt[:-1]])
        pm = pd.DataFrame(Cv).shift(1).expanding().max().values; isath = (Cv > pm) & (prior >= 750)
        recent = pd.DataFrame(isath.astype(float)).shift(1).rolling(60, min_periods=1).max().fillna(0).values > 0
        ev = isath & ~recent & inb.values & ~panic.values[:, None] & (ch.values < lim.values[:, None])
        t_, j_ = np.nonzero(ev); E = pd.DataFrame({"date": days[t_], "code": cols[j_], "y": YH[60][t_, j_], "u": FW[60][t_]})
    R = R.replace([np.inf, -np.inf], np.nan); os.chdir(REPO)
    return R, E

def tser(s):
    s = pd.Series(s).dropna(); return (round(float(s.mean()), 2), round(float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))), 2) if len(s) > 2 else np.nan, int(len(s)))

def analyze(R, E, a, b):
    out = {}
    M = R[R.fd & R.base & ~R.panic & R.y20.notna() & (R.date >= a) & (R.date <= b)].copy()
    M["cs"] = np.mean([M.groupby("date")[c].rank(pct=True).fillna(0.5).values for c in COMP], axis=0)
    prev, rows = set(), {}
    for d, g in M.groupby("date"):
        pk = g.sort_values("cs", ascending=False).head(20); cur = set(pk.code); to = 1.0 if not prev else len(cur - prev) / 20; prev = cur
        rows[d] = (pk.y20.mean() - 0.6 * to - g.y20.mean(), pk.y20.mean() - 0.6 * to - g.b20.iloc[0])
    T = pd.DataFrame(rows, index=["u", "b"]).T
    q = M.assign(q=np.minimum(5, np.floor(M.groupby("date").cs.rank(pct=True, method="first") * 5 - 1e-9) + 1)).groupby(["date", "q"]).y20.mean().unstack()
    out["O1 綜合前20 勝等權（每月）"] = tser(T.u); out["O1 相對0050（每月）"] = tser(T.b); out["O1 五等分差"] = tser(q[5.0] - q[1.0])
    out["O1 成立"] = bool(out["O1 綜合前20 勝等權（每月）"][0] > 0 and out["O1 綜合前20 勝等權（每月）"][1] >= 2)
    U = R[R.fd & R.base & R.y20.notna()].copy(); U["m"] = U.date.dt.to_period("M")
    K = R[R.pk & R.base & R.y20.notna() & (R.date >= a) & (R.date <= b)].copy(); K["m"] = K.date.dt.to_period("M")  # 同之前研究：今日發動限分析範圍內
    sc = np.zeros(len(K))
    for fc in COMP:
        pr = np.full(len(K), 0.5)
        for m, gg in K.groupby("m"):
            ref = np.sort(U.loc[U.m == m, fc].dropna().values)
            if len(ref) < 50: continue
            v = gg[fc].values; pr[K.index.get_indexer(gg.index)] = np.where(np.isnan(v), 0.5, np.searchsorted(ref, v, side="right") / len(ref))
        sc += pr
    K["cs"] = sc / len(COMP); K["yn"] = K.y20 - 1.1
    Kn = K[~K.panic]
    s_lo_b = Kn[Kn.cs < 0.5].assign(x=lambda z: z.yn - z.b20).groupby("m").x.mean(); s_lo_u = Kn[Kn.cs < 0.5].assign(x=lambda z: z.yn - z.u20).groupby("m").x.mean()
    s_hi_b = Kn[Kn.cs >= 0.8].assign(x=lambda z: z.yn - z.b20).groupby("m").x.mean()
    out["O2 分數<0.5 減0050（每月）"] = tser(s_lo_b); out["O2 分數<0.5 減等權"] = tser(s_lo_u); out["O2 分數≥0.8 減0050"] = tser(s_hi_b)
    out["O2 平常今日發動 筆數"] = int(len(Kn)); out["O2 成立"] = bool(out["O2 分數<0.5 減0050（每月）"][0] < 0 and out["O2 分數<0.5 減0050（每月）"][1] <= -2)
    Ee = E[(E.date >= a) & (E.date <= b)].assign(x=lambda z: z.y - 0.6 - z.u); se = Ee.groupby(Ee.date.dt.to_period("M")).x.mean()
    out["O3 新高突破60日 勝等權（每月）"] = tser(se); out["O3 事件數"] = int(Ee.x.notna().sum()); out["O3 成立"] = bool(out["O3 新高突破60日 勝等權（每月）"][0] > 0 and out["O3 新高突破60日 勝等權（每月）"][1] >= 2)
    Kp = K[K.panic]; weak = Kp[Kp.g_m10 < 0]; reb = Kp[(Kp.g_m10 > 0) & (Kp.g_5 > 0)]
    st = lambda y: {"n": int(len(y)), "平均": round(float(y.mean()), 2), "賺錢%": round(float((y > 0).mean() * 100), 1)}
    out["O4 恐慌今日發動"] = st(Kp.yn); out["O4 平常今日發動"] = st(Kn.yn); out["O4 族群仍弱"] = st(weak.yn); out["O4 族群已反彈"] = st(reb.yn)
    out["O4 恐慌成立"] = bool(Kp.yn.mean() > 0 and Kp.yn.mean() - Kn.yn.mean() >= 3); out["O4 族群成立"] = bool(weak.yn.mean() > reb.yn.mean())
    # 補充（11/21 外部檢查後加，不影響事先登錄的判斷）：每天先平均、波內日期等權、波等權；平常用每月等權
    _pd = sorted(R[R.panic & (R.date >= a) & (R.date <= b)].date.unique())
    if _pd:
        _ds = pd.Series(_pd); _all = pd.Series(sorted(R.date.unique())); _pos = {x: i for i, x in enumerate(_all)}
        _w = dict(zip(_ds, (_ds.map(_pos).diff().fillna(99) > 10).cumsum()))
        bal = lambda z: round(float(z.assign(w=z.date.map(_w)).groupby(["w", "date"]).yn.mean().groupby("w").mean().mean()), 2) if len(z) else None
        out["補充 平衡加權"] = {"恐慌今日發動": bal(Kp), "平常今日發動(每月等權)": round(float(Kn.groupby("m").yn.mean().mean()), 2), "族群仍弱": bal(weak), "族群已反彈": bal(reb)}
    pdays = sorted(R[R.panic & (R.date >= a) & (R.date <= b)].date.unique())
    if pdays:
        ds = pd.Series(pdays); dd = pd.Series(sorted(R.date.unique())); pos_ = {x: i for i, x in enumerate(dd)}
        w = (ds.map(pos_).diff().fillna(99) > 10).cumsum(); wm = dict(zip(ds, w)); Kp = Kp.assign(w=Kp.date.map(wm))
        out["O4 各波"] = {f"{g.date.min():%Y-%m-%d}": {**st(g.yn), "仍弱": round(float(g[g.g_m10 < 0].yn.mean()), 2), "已反彈": round(float(g[(g.g_m10 > 0) & (g.g_5 > 0)].yn.mean()), 2)} for _, g in Kp.groupby("w")}
    return out

if __name__ == "__main__":
    kind = sys.argv[1]
    R, E = build(kind)
    a, b = ("2011-01-01", "2015-05-31") if kind == "older" else ("2016-01-01", "2019-12-31")
    out = analyze(R, E, a, b)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if kind == "older": json.dump(out, open(f"{REPO}/research/results/older_era.json", "w"), ensure_ascii=False, indent=1)
