"""攻擊型選股假說檢驗（依 reports/台股攻擊型選股訊號文獻回顧.md 的 H1–H7、H10）。

全部在看結果之前寫下（2026/11/21 22:40），看到結果後不修改。
資料：2015-06～2026-11 接成一段（data/old ＋ data/history）。恐慌：2020 前＝空頭家數 ÷ 有60日線上市檔數 近10日最高 ≥ 0.7499；
      2020 起＝空頭家數近10日最高 ≥ 750。只用平常時期。三段期間：2016–19、2020–23、2024–26。
共同設定：每月第一個 11 日以後的交易日收盤後形成（上月營收已公布），範圍＝分析範圍內股票（20日均額 ≥ 1,000 萬），
  隔天開盤買（開盤一價漲停買不到不計）；綜合分數 cs＝research/deep_normal.py 的 12 個特性當月百分位等權平均。
  Fama-MacBeth：每月把報酬對「新訊號百分位、cs」做迴歸，新訊號係數＝從最低到最高百分位的報酬差（%）；t＝係數月序列的 t。
  營收一律用「下個月 11 日起才知道」（沒有逐家實際公告日，偏保守）。融券只有融券，沒有借券賣出。
共同判斷（全部符合才算成立）：① 全期 t 值在假說方向 ≥ 2.9（8 個假說的多重檢定校正）；② 三段期間係數同方向；
  ③ 實際可做：見各假說的「做多檢驗」，扣成本後相對「等權分析範圍」為正（另報相對 0050）。

H1 標準化營收驚奇 SUR：X＝當月營收 − 去年同月營收；SUR＝(X − 前24個月 X 平均) ÷ 前24個月 X 標準差（至少18個月）。方向＋，20 日。
   做多檢驗：cs 前20% 之中 SUR 前三分之一，每月等權、扣 0.6%×換股比例，減等權分析範圍。
H2 營收創歷史新高：最新月營收 ≥ 2014 年以來所有月份最高（至少要有 36 個月），是＝1、否＝0。方向＋，20 日。
   做多檢驗：cs 前20% 之中營收創歷史新高者，同上。另報：第 2–6 個月（20 日到 120 日）報酬係數，看會不會反轉。
H3 營收支撐的今日發動：平常時期今日發動中，「營收支撐」＝最新月 SUR > 0 且年增 > 0。方向＋。
   以今日發動為樣本（每天的訊號，每月平均成一筆），20 日報酬對「營收支撐、cs」迴歸；做多檢驗：營收支撐且 cs ≥ 0.5 的今日發動，
   20 日扣 1.1% 後減同一進場日的等權分析範圍報酬。另報 60 日。
H4 拉長持有：cs 能否預測第 2–6 個月（20 日收盤到 120 日收盤）報酬。方向＋（不反轉，表示可以抱久、少換股）。
   做多檢驗：cs 前20檔每 3 個月換一次（抱 60 日），扣 0.6%×換股比例，與每月換股相比、並減等權分析範圍。
H5 非漲停越過歷史高點：收盤首次高於「資料內此前所有收盤」（至少 750 個交易日歷史），而且前 60 日沒有創過歷史新高、當天漲幅 < 9%。
   每天的事件，隔天開盤買，60 日報酬扣 0.6% 減同一進場日的等權分析範圍。方向＋。判斷改為：全期 t ≥ 2.9、至少兩段為正。
   另報：漲停型越過歷史高點、20 日。
H6 融券流量：SF＝(今天融券餘額 − 5 天前) × 1000 × 收盤價 ÷ (5 × 過去250日平均成交額)。方向 −，20 日。
   做多檢驗：cs 前20% 剔除 SF 最高的 10% 後，比剔除前每月多賺多少（要 > 0）。
H7 投信買超短線：每天依「投信買超金額 ÷ 市值」取分析範圍內前 5 檔（買超 > 0），隔天開盤買、抱 10 日，扣 0.6% 減同一進場日等權分析範圍。
   方向＋。每天的組合每月平均成一筆算 t。另報抱 5／20／60 日的衰減。
H10 持續贏家：過去 12 個 20 日區間中，有幾個區間的漲幅落在當時分析範圍前三分之一。方向＋，20 日。
   做多檢驗：cs 前20% 之中持續贏家前三分之一，同 H1。
"""
import glob, json, os, shutil, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO)
CACHE = "/tmp/model_v2"; LOCK = 0.095
COMP = ["r20", "mom6_1", "d_hi250", "m60", "am", "yoy1", "yoy3", "rev_acc", "rev_hi12", "mom1", "mg20", "rs_g20"]

def full_root():
    r = "/tmp/fullroot"; d = r + "/data"
    if os.path.exists(r): shutil.rmtree(r)
    for sub in ("history", "chips", "revenue"): os.makedirs(f"{d}/{sub}")
    for fn in sorted(glob.glob(f"{REPO}/data/old/history/*.csv.gz")) + sorted(glob.glob(f"{REPO}/data/history/*.csv.gz")): os.symlink(fn, f"{d}/history/{os.path.basename(fn)}")
    os.symlink(f"{REPO}/data/prices.csv", f"{d}/prices.csv")
    for fn in sorted(glob.glob(f"{REPO}/data/old/chips/*.csv.gz")) + sorted(glob.glob(f"{REPO}/data/chips/*.csv*")): os.symlink(fn, f"{d}/chips/{os.path.basename(fn)}")
    for fn in sorted(glob.glob(f"{REPO}/data/old/revenue/*.csv.gz")):
        if os.path.basename(fn)[:4] < "2019": os.symlink(fn, f"{d}/revenue/{os.path.basename(fn)}")
    for fn in sorted(glob.glob(f"{REPO}/data/revenue/*.csv*")): os.symlink(fn, f"{d}/revenue/{os.path.basename(fn)}")
    for name in ("adjust.csv", "disposal.csv"):
        pd.concat([pd.read_csv(f"{REPO}/data/old/{name}", dtype={"code": str}), pd.read_csv(f"{REPO}/data/{name}", dtype={"code": str})]).drop_duplicates().to_csv(f"{d}/{name}", index=False)
    os.symlink(f"{REPO}/data/industry.csv", f"{d}/industry.csv")
    return r

def bench_index(days):
    e = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in (f"{REPO}/data/old/etf.csv", f"{REPO}/data/etf.csv")])
    s = e[e.code == "0050"].drop_duplicates("date", keep="last").assign(date=lambda z: pd.to_datetime(z.date)).set_index("date").close.sort_index()
    adj = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in (f"{REPO}/data/old/adjust.csv", f"{REPO}/data/adjust.csv")])
    adj = adj[adj.code == "0050"].assign(date=lambda z: pd.to_datetime(z.date)).groupby("date").ratio.prod()
    r = s / (s.shift(1) * adj.reindex(s.index).fillna(1.0)); r = r.where((r > 0.45) & (r < 2.2), 1.0).fillna(1.0)
    return r.cumprod().reindex(days).ffill()

def build():
    out = f"{CACHE}/offense.pkl"
    if os.path.exists(out): return pd.read_pickle(out)
    os.chdir(full_root())
    from radar_core import load_prices, panels, features, tiers
    p = load_prices(); P = panels(p); del p
    C, O, H, L, A, raw = P["C"], P["O"], P["H"], P["L"], P["A"], P["rawC"]
    f = features(P); S = tiers(f, P["MK"]); days, cols = C.index, C.columns; N = len(days)
    # 恐慌
    alive = (C.notna().cumsum() > 0) & (C.notna()[::-1].cumsum()[::-1] > 0); Cf = C.ffill(limit=20).where(alive)
    mm = {n: Cf.rolling(n, min_periods=n).mean() for n in (5, 10, 20, 60)}; okm = Cf.notna() & mm[60].notna()
    mk = P["MK"].ffill().iloc[-1]; tw = [c for c in cols if mk.get(c) == "TWSE"]
    bear = ((mm[5] < mm[10]) & (mm[10] < mm[20]) & okm)[tw].sum(axis=1); okn = okm[tw].sum(axis=1); del Cf, mm, alive
    old = days < pd.Timestamp("2020-01-01")
    panic = pd.Series(np.where(old, (bear / okn).rolling(10, min_periods=1).max() >= 0.7499, bear.rolling(10, min_periods=1).max() >= 750), index=days)
    # 等權分析範圍的未來報酬（同一進場日）與 0050
    inb = S[S.base].assign(v=True).pivot(index="date", columns="code", values="v").reindex(index=days, columns=cols).fillna(False).astype(bool)
    Ov, Hv, Lv = O.values, H.values, L.values; prevc = C.ffill().shift(1).values
    Onext = np.vstack([Ov[1:], np.full((1, len(cols)), np.nan)])
    lockn = np.vstack([((Ov == Hv) & (Hv == Lv) & (Ov / prevc - 1 >= LOCK))[1:], np.zeros((1, len(cols)), bool)])
    FW, YH = {}, {}
    bidx = bench_index(days).values
    for h in (5, 10, 20, 60, 120):
        Cff = C.ffill(limit=h).values; fut = np.vstack([Cff[h:], np.full((h, len(cols)), np.nan)])
        y = (fut / Onext - 1) * 100; y[~(Onext > 0) | lockn] = np.nan; YH[h] = y
        FW[h] = np.nanmean(np.where(inb.values, y, np.nan), axis=1)
        FW[f"b{h}"] = np.concatenate([(bidx[h:] / bidx[:-h] - 1) * 100, np.full(h, np.nan)])
        del Cff, fut
    dser = pd.Series(days); fd = set(dser[dser.dt.day >= 11].groupby(dser.dt.to_period("M")).first())
    R = S[S.base & (S.date.isin(fd) | S.pk)][["date", "code", "pk", "hot", "r20", "amt"]].copy(); R["fd"] = R.date.isin(fd)
    R["ti"] = R.date.map({d: i for i, d in enumerate(days)}).values; R["ji"] = R.code.map({c: j for j, c in enumerate(cols)}).values
    ti, ji = R.ti.values, R.ji.values; take = lambda X: (X.values if isinstance(X, pd.DataFrame) else X)[ti, ji]
    R["panic"] = panic.values[ti]
    for h in (20, 60, 120): R[f"y{h}"] = YH[h][ti, ji]; R[f"u{h}"] = FW[h][ti]; R[f"b{h}"] = FW[f"b{h}"][ti]
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = C.pct_change(fill_method=None)
        R["r20"] = R.r20 / 100; R["mom6_1"] = take(C.shift(20) / C.shift(125) - 1)
        R["d_hi250"] = take(C / C.rolling(250, min_periods=120).max() - 1); R["m60"] = take(C / C.rolling(60, min_periods=60).mean() - 1)
        a20 = A.rolling(20, min_periods=10).mean(); R["am"] = take(A.rolling(5, min_periods=5).mean() / a20)
        for k in range(1, 13): R[f"w{k}"] = take(C.shift(20 * (k - 1)) / C.shift(20 * k) - 1)
        fns = sorted(glob.glob("data/chips/*.csv*"), key=lambda x: (os.path.basename(x)[:4], x.endswith(".csv")))
        ch = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in fns], ignore_index=True)
        ch["date"] = pd.to_datetime(ch.date); ch = ch.drop_duplicates(["date", "code"], keep="last")
        mg = ch.pivot(index="date", columns="code", values="margin").reindex(index=days, columns=cols); R["mg20"] = take(mg / mg.shift(20) - 1); del mg
        sh = ch.pivot(index="date", columns="code", values="short").reindex(index=days, columns=cols)
        SF = (sh - sh.shift(5)) * 1000 * raw / (5 * A.rolling(250, min_periods=120).mean()); R["sf"] = take(SF); del sh
        tr = ch.pivot(index="date", columns="code", values="trust").reindex(index=days, columns=cols); del ch
        rv = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in sorted(glob.glob("data/revenue/*.csv*"))], ignore_index=True)
        rv["pub"] = pd.to_datetime(rv.ym + "-01") + pd.DateOffset(months=1) + pd.Timedelta(days=10)
        piv = lambda v: rv.pivot_table(index="ym", columns="code", values=v).sort_index().reindex(columns=cols)
        yoy = piv("yoy"); rev = piv("rev").reindex(yoy.index); mom = piv("mom").reindex(yoy.index)
        pub = rv.groupby("ym").pub.first().sort_index().reindex(yoy.index)
        X = rev - rev.shift(12); mu = X.shift(1).rolling(24, min_periods=18).mean(); sd = X.shift(1).rolling(24, min_periods=18).std()
        SUR = (X - mu) / sd.replace(0, np.nan)
        pmax = rev.shift(1).expanding().max(); pcnt = rev.shift(1).notna().cumsum()
        ATH = ((rev >= pmax).astype(float)).where((pcnt >= 36) & rev.notna())
        kn = np.searchsorted(pub.values, R.date.values, side="right") - 1
        g = lambda M, k_: np.where(k_ >= 0, M[np.maximum(k_, 0), ji], np.nan)
        R["yoy1"] = g(yoy.values, kn)
        y3 = np.vstack([np.where(kn - s >= 0, g(yoy.values, kn - s), np.nan) for s in range(3)]); R["yoy3"] = np.where(kn >= 2, np.nanmean(y3, axis=0), np.nan)
        p12 = np.vstack([np.where(kn - s >= 0, g(rev.values, kn - s), np.nan) for s in range(1, 13)])
        R["rev_hi12"] = np.where(np.isfinite(p12).sum(axis=0) >= 6, g(rev.values, kn) / np.nanmax(p12, axis=0) - 1, np.nan)
        R["rev_acc"] = R.yoy1 - R.yoy3; R["mom1"] = g(mom.values, kn); R["sur"] = g(SUR.values, kn); R["ath"] = g(ATH.values, kn)
        ind = pd.read_csv("data/industry.csv", dtype={"code": str}).set_index("code")
        grp = pd.Series(ind["ind_name"]).reindex(cols); cap = pd.Series(ind["cap"]).reindex(cols)
        rb = r1.where(inb); g20 = np.full((N, len(cols)), np.nan)
        for gname, cs in grp.dropna().groupby(grp.dropna()).groups.items():
            ix = (1 + rb[list(cs)].mean(axis=1).fillna(0)).cumprod(); g20[:, [cols.get_loc(c) for c in cs]] = (ix / ix.shift(20) - 1).values[:, None]
        R["g20"] = g20[ti, ji]; R["rs_g20"] = R.r20 - R.g20; del g20, rb
        # ---- 事件：H5 越過歷史高點 ----
        Cv = C.values; cnt = np.cumsum(~np.isnan(Cv), axis=0)
        pm = pd.DataFrame(Cv).shift(1).expanding().max().values
        isath = (Cv > pm) & (np.vstack([np.zeros((1, len(cols))), cnt[:-1]]) >= 750)
        recent = pd.DataFrame(isath.astype(float)).shift(1).rolling(60, min_periods=1).max().fillna(0).values > 0
        chv = f["ch"].reindex(index=days, columns=cols).values
        ev = isath & ~recent & inb.values & ~panic.values[:, None]
        E5 = []
        for lab, m in (("非漲停", ev & (chv < 9)), ("漲停", ev & (chv >= 9))):
            t_, j_ = np.nonzero(m)
            for h in (20, 60):
                E5.append(pd.DataFrame({"date": days[t_], "code": cols[j_], "type": lab, "h": h, "y": YH[h][t_, j_], "u": FW[h][t_], "b": FW[f"b{h}"][t_]}))
        E5 = pd.concat(E5, ignore_index=True)
        # ---- 事件：H7 投信買超前5檔（÷市值；另報 ÷20日均額，避免只用現存公司的偏誤）----
        E7 = []
        for lab, den in (("÷市值", cap.values[None, :] / 1000), ("÷20日均額", (a20.values / (1000 * raw.values)))):
            sc = np.where(inb.values & (tr.values > 0), tr.values / den, np.nan)
            for t in range(N):
                if panic.values[t]: continue
                row = sc[t]; ok = np.isfinite(row)
                if ok.sum() < 5: continue
                jj = np.argsort(np.where(ok, -row, np.inf))[:5]
                for h in (5, 10, 20, 60):
                    E7.append((days[t], lab, h, np.nanmean(YH[h][t, jj]), FW[h][t], FW[f"b{h}"][t]))
        E7 = pd.DataFrame(E7, columns=["date", "type", "h", "y", "u", "b"])
    R = R.replace([np.inf, -np.inf], np.nan)
    pd.to_pickle({"R": R, "E5": E5, "E7": E7}, out); os.chdir(REPO)
    return {"R": R, "E5": E5, "E7": E7}

if __name__ == "__main__" and sys.argv[1:2] == ["build"]:
    D = build(); R = D["R"]
    print(len(R), R.date.min(), R.date.max(), "每月快照", R[R.fd].date.nunique(), "今日發動", int(R.pk.sum()), "；H5 事件", len(D["E5"]) // 2, "；H7 筆", len(D["E7"]))
    print("缺值%", (R[R.fd][COMP + ["sur", "ath", "sf"]].isna().mean() * 100).round(0).astype(int).to_dict())

def per(d):
    d = pd.to_datetime(pd.Series(d)).values
    return np.where(d < np.datetime64("2016-01-01"), "pre", np.where(d < np.datetime64("2020-01-01"), "2016-19", np.where(d < np.datetime64("2024-01-01"), "2020-23", "2024-26")))

def tser(s):
    s = pd.Series(s).dropna(); return (round(float(s.mean()), 3), round(float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))), 2) if len(s) > 2 else np.nan, int(len(s)))

def by_period(s):
    s = pd.Series(s).dropna(); p = per(s.index)
    return {k: tser(s[p == k]) for k in ("2016-19", "2020-23", "2024-26")}

def fm(M, y, x, ctrl=("cs",)):
    out = {}
    for d, g in M.groupby("date"):
        g = g[g[y].notna() & g[x].notna()]
        if len(g) < 30 or g[x].nunique() < 2: continue
        xs = g[x].rank(pct=True) if g[x].nunique() > 2 else g[x].astype(float)
        Z = np.column_stack([np.ones(len(g)), xs.values] + [g[c].values for c in ctrl])
        b = np.linalg.lstsq(Z, g[y].values, rcond=None)[0]; out[d] = b[1]
    return pd.Series(out)

def longonly(M, sel, cost=0.6):
    rows, prev = {}, set()
    for d, g in M.groupby("date"):
        pk = g[sel(g)]; cur = set(pk.code)
        if not len(pk): prev = set(); continue
        to = 1.0 if not prev else len(cur - prev) / len(cur); prev = cur
        rows[d] = (pk.y20.mean() - cost * to - g.y20.mean(), pk.y20.mean() - cost * to - g.b20.iloc[0], len(pk))
    T = pd.DataFrame(rows, index=["ex_u", "ex_b", "n"]).T
    return T

if __name__ == "__main__" and sys.argv[1:2] == ["run"]:
    D = build(); R = D["R"]; R = R[R.date >= "2016-01-01"]
    M = R[R.fd & ~R.panic & R.y20.notna()].copy()
    M["cs"] = np.mean([M.groupby("date")[c].rank(pct=True).fillna(0.5).values for c in COMP], axis=0)
    M["csr"] = M.groupby("date").cs.rank(pct=True)
    W = M[[f"w{k}" for k in range(1, 13)]]
    tops = np.column_stack([(M.groupby("date")[f"w{k}"].rank(pct=True) >= 2 / 3).values & M[f"w{k}"].notna().values for k in range(1, 13)])
    M["pers"] = np.where(W.notna().sum(axis=1) >= 9, tops.sum(axis=1), np.nan)
    M["ysk"] = ((1 + M.y120 / 100) / (1 + M.y20 / 100) - 1) * 100
    res = {}
    def judge(name, coef, sign, lo_series=None, lo_ok=None, need_all=True):
        pooled = tser(coef); bp = by_period(coef)
        c1 = pooled[1] * sign >= 2.9; c2 = all(v[0] * sign > 0 for v in bp.values()) if need_all else sum(v[0] * sign > 0 for v in bp.values()) >= 2
        c3 = lo_ok if lo_ok is not None else True
        res[name] = {"全期(平均,t,月數)": pooled, "分段": bp, "做多檢驗": lo_series, "①t": bool(c1), "②同向": bool(c2), "③做多": bool(c3), "成立": bool(c1 and c2 and c3)}
        print(f"\n== {name}：成立={res[name]['成立']}（①{c1} ②{c2} ③{c3}）\n   係數 全期 {pooled}；分段 {bp}\n   做多 {lo_series}", flush=True)
    top = lambda g: g.csr >= 0.8
    # H1
    c = fm(M, "y20", "sur"); T = longonly(M, lambda g: top(g) & (g.sur.rank(pct=True) >= 2 / 3))
    judge("H1 營收驚奇SUR", c, +1, {"相對等權": tser(T.ex_u), "分段": by_period(T.ex_u), "相對0050": tser(T.ex_b)}, T.ex_u.mean() > 0)
    # H2
    c = fm(M, "y20", "ath"); T = longonly(M, lambda g: top(g) & (g.ath == 1)); c_sk = fm(M, "ysk", "ath")
    judge("H2 營收創歷史新高", c, +1, {"相對等權": tser(T.ex_u), "分段": by_period(T.ex_u), "相對0050": tser(T.ex_b), "第2-6月係數": tser(c_sk)}, T.ex_u.mean() > 0)
    # H3 今日發動
    K = R[R.pk & ~R.panic & R.y20.notna()].copy(); K["m"] = K.date.dt.to_period("M"); U = R[R.fd & R.y20.notna()].copy(); U["m"] = U.date.dt.to_period("M")
    sc = np.zeros(len(K))
    for fc in COMP:
        pr = np.full(len(K), 0.5)
        for m, g in K.groupby("m"):
            ref = np.sort(U.loc[U.m == m, fc].dropna().values)
            if len(ref) < 50: continue
            v = g[fc].values; pr[K.index.get_indexer(g.index)] = np.where(np.isnan(v), 0.5, np.searchsorted(ref, v, side="right") / len(ref))
        sc += pr
    K["cs"] = sc / len(COMP); K["sup"] = ((K.sur > 0) & (K.yoy1 > 0)).astype(float); K["yn"] = K.y20 - 1.1; K["yn60"] = K.y60 - 1.1
    co = {}
    for m, g in K.groupby("m"):
        if len(g) < 10 or g.sup.nunique() < 2: continue
        Z = np.column_stack([np.ones(len(g)), g.sup, g.cs]); co[g.date.iloc[0]] = np.linalg.lstsq(Z, g.yn.values, rcond=None)[0][1]
    co = pd.Series(co)
    L3 = K[(K.sup == 1) & (K.cs >= 0.5)].assign(x=lambda z: z.yn - z.u20, x60=lambda z: z.yn60 - z.u60, xb=lambda z: z.yn - z.b20)
    s3 = L3.groupby("m").x.mean(); s3.index = L3.groupby("m").date.first().values
    s360 = L3.groupby("m").x60.mean(); s360.index = s3.index
    judge("H3 營收支撐的今日發動", co, +1, {"相對等權(20日)": tser(s3), "分段": by_period(s3), "相對等權(60日)": tser(s360), "筆數": len(L3),
          "非支撐且cs<0.5相對等權": tser(K[(K.sup == 0) & (K.cs < 0.5)].assign(x=lambda z: z.yn - z.u20).groupby("m").x.mean())}, s3.mean() > 0)
    # H4
    c = fm(M, "ysk", "cs", ctrl=())
    Q = M.copy(); months = sorted(Q.date.unique()); qm = set(months[::3]); prev = set(); rows = {}
    for d in months:
        if d not in qm: continue
        g = Q[Q.date == d]; pk = g.sort_values("cs", ascending=False).head(20); cur = set(pk.code); to = 1.0 if not prev else len(cur - prev) / 20; prev = cur
        rows[d] = pk.y60.mean() - 0.6 * to - g.u60.iloc[0]
    q = pd.Series(rows); mo = longonly(M, lambda g: g.cs.rank(ascending=False, method="first") <= 20)
    judge("H4 綜合分數不反轉（可抱久）", c, +1, {"每季換股 前20 相對等權(每季)": tser(q), "分段": by_period(q), "每月換股 前20 相對等權(每月)": tser(mo.ex_u),
          "折成每季比較：每月×3": round(float(mo.ex_u.mean() * 3), 2)}, q.mean() > 0)
    # H5
    E5 = D["E5"]; E5 = E5[E5.date >= "2016-01-01"].copy(); E5["x"] = E5.y - 0.6 - E5.u; E5["m"] = E5.date.dt.to_period("M")
    def evs(t, h):
        z = E5[(E5.type == t) & (E5.h == h)]; s = z.groupby("m").x.mean(); s.index = z.groupby("m").date.first().values; return s, len(z)
    s5, n5 = evs("非漲停", 60); s5b, _ = evs("非漲停", 20); s5l, n5l = evs("漲停", 60)
    judge("H5 非漲停越過歷史高點（60日）", s5, +1, {"事件數": n5, "20日": tser(s5b), "漲停型60日": tser(s5l), "漲停型事件數": n5l}, s5.mean() > 0, need_all=False)
    # H6
    c = fm(M, "y20", "sf")
    imp = {}
    for d, g in M.groupby("date"):
        gg = g[top(g)]
        if len(gg) < 20: continue
        cut = gg.sf.quantile(0.9); keep = gg[~(gg.sf > cut)]; imp[d] = keep.y20.mean() - gg.y20.mean()
    imp = pd.Series(imp)
    judge("H6 融券流量（越多越差）", c, -1, {"剔除後多賺(每月)": tser(imp), "分段": by_period(imp)}, imp.mean() > 0)
    # H7
    E7 = D["E7"]; E7 = E7[E7.date >= "2016-01-01"].copy(); E7["x"] = E7.y - 0.6 - E7.u; E7["xb"] = E7.y - 0.6 - E7.b; E7["m"] = E7.date.dt.to_period("M")
    def e7(t, h, col="x"):
        z = E7[(E7.type == t) & (E7.h == h)]; s = z.groupby("m")[col].mean(); s.index = z.groupby("m").date.first().values; return s
    s7 = e7("÷市值", 10)
    judge("H7 投信買超前5檔抱10日", s7, +1, {"÷20日均額（無存活偏誤）": tser(e7("÷20日均額", 10)), "分段(÷均額)": by_period(e7("÷20日均額", 10)),
          "衰減(÷市值) 5/10/20/60": [tser(e7("÷市值", h))[0] for h in (5, 10, 20, 60)], "相對0050(10日)": tser(e7("÷市值", 10, "xb"))}, s7.mean() > 0 and e7("÷20日均額", 10).mean() > 0)
    # H10
    c = fm(M, "y20", "pers"); T = longonly(M, lambda g: top(g) & (g.pers.rank(pct=True) >= 2 / 3))
    judge("H10 持續贏家", c, +1, {"相對等權": tser(T.ex_u), "分段": by_period(T.ex_u), "相對0050": tser(T.ex_b)}, T.ex_u.mean() > 0)
    json.dump(res, open(f"{REPO}/research/results/offense.json", "w"), ensure_ascii=False, indent=1, default=str)
