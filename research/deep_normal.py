"""平常時期深入研究：台股到底哪一類股票之後會漲？強勢股要抱多久？恐慌結束後效果延續多久？

全部在看結果之前寫下（2026/11/21 19:35），看到結果後不修改。
資料：2016–2019（data/old）與 2020–2026/11（data/history）。恐慌：2020 之後＝上市短均線空頭家數近10日最高 ≥ 750；
      2016–2019＝同比例（空頭家數 ÷ 有60日線上市檔數 ≥ 0.7499）。「平常」＝不在恐慌期間。
三段期間：2016–2019（之前只用來考今日發動模型與恐慌，這裡的問題全新）、2020–2023（發現期）、2024–2026（看過今日發動，這裡的問題全新）。

A 全市場因子研究
  每個月第一個「11 日以後」的交易日（上月營收已公布）收盤後排序，範圍＝當天分析範圍內全部股票（20 日均額 ≥ 1,000 萬），
  隔天開盤買（開盤一價漲停買不到不計），持有 20 個交易日（另報 60 日），等權。只用平常月份。
  特性（約 22 個）：近20日漲幅、近6個月漲幅（跳過最近1個月）、近12個月漲幅（跳過最近1個月）、離250日高、離60日線、
     20日波動、近20日單日最大漲幅、近60日漲停次數、市值、成交額占全市場比例、量能（5日/20日均額）、
     最近月營收年增、近3月營收年增、營收加速、營收比前12個月最高、營收月增、
     投信近5／20日買超（÷均額）、外資近20日買超、融資20日變化、產業指數近20日、相對產業強度。
  每個特性每月分五等分，看「第5組 − 第1組」的 20 日報酬差（每月一筆，算平均與 t 值）、第5組比全部平均、第5組比 0050 同期。
  方向與入選只看 2020–2023：五等分差的 t 值絕對值 ≥ 2 的特性入選，方向＝正負號。
  單一特性「全新期間也成立」：2016–2019 與 2024–2026 兩段的五等分差都和 2020–2023 同方向且 > 0（換成方向後），
     而且逐年同方向的年數 ≥ 2016–19 四年中 3 年、2024–26 三年中 2 年。
  綜合分數：入選特性的百分位排名（依方向，缺值當 0.5）等權平均，不另外擬合；每月買分數最高 20 檔，換股成本每次來回 0.6%
     （只算換掉的比例；另報 1.1%）。比較 0050 含息同期。判斷：2016–2019 與 2024–2026 兩段的綜合前20檔都贏 0050（年化）→ 平常時期有可用的選股方法；
     否則照實報告。另報：同樣的綜合分數只在「仍在動＋今日發動」裡挑（強勢股版本）。
B 抱多久：平常時期今日發動（隔天開盤買）抱 5／20／60／120 日，平均、賺錢比例、減去 0050 同期報酬的超額。
  判斷：60 或 120 日的超額在三段期間都 > 0 → 「強勢股要抱久」成立。另報 v1 模型「高」。
C 恐慌結束後：今日發動依「距離最後一個恐慌日幾個交易日」分組：恐慌中、1–20、21–60、61–120、>120，抱 20 日。
  判斷：1–20 或 21–60 組在 2016–2019 與 2020–2026 都 > >120 組而且 > 0 → 恐慌後的反彈效果會延續，雷達可以延長「恐慌模式」。
"""
import glob, json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/research")
from model_v2 import old_root, CACHE, LOCK
RATIO = 0.7499
HS = (5, 20, 60, 120)
FACT = ["r20", "mom6_1", "mom12_1", "d_hi250", "m60", "vol20", "max20", "nlim60", "lmv", "a20r", "am", "yoy1", "yoy3", "rev_acc", "rev_hi12", "mom1",
        "trust5", "trust20", "foreign20", "mg20", "g20", "rs_g20"]
NAME = {"r20": "近20日漲幅", "mom6_1": "近6個月漲幅(跳過近1月)", "mom12_1": "近12個月漲幅(跳過近1月)", "d_hi250": "離250日高", "m60": "離60日線",
        "vol20": "20日波動", "max20": "近20日最大單日漲幅", "nlim60": "近60日漲停次數", "lmv": "市值", "a20r": "成交額占比", "am": "量能(5日/20日)",
        "yoy1": "最近月營收年增", "yoy3": "近3月營收年增", "rev_acc": "營收加速", "rev_hi12": "營收比前12月最高", "mom1": "營收月增",
        "trust5": "投信5日買超", "trust20": "投信20日買超", "foreign20": "外資20日買超", "mg20": "融資20日變化", "g20": "產業指數近20日", "rs_g20": "相對產業強度"}

def bench_index(days):
    e = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in (f"{REPO}/data/old/etf.csv", f"{REPO}/data/etf.csv")])
    s = e[e.code == "0050"].drop_duplicates("date", keep="last").assign(date=lambda z: pd.to_datetime(z.date)).set_index("date").close.sort_index()
    adj = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in (f"{REPO}/data/old/adjust.csv", f"{REPO}/data/adjust.csv")])
    adj = adj[adj.code == "0050"].assign(date=lambda z: pd.to_datetime(z.date)).groupby("date").ratio.prod()
    r = s / (s.shift(1) * adj.reindex(s.index).fillna(1.0)); r = r.where((r > 0.45) & (r < 2.2), 1.0).fillna(1.0)
    return r.cumprod().reindex(days).ffill()

def snap(era):
    out = f"{CACHE}/deep_{era}.pkl"
    if os.path.exists(out): return pd.read_pickle(out)
    os.chdir(old_root() if era == "old" else REPO)
    from radar_core import load_prices, panels, features, tiers
    p = load_prices(); P = panels(p); del p
    C, O, H, L, A, raw = P["C"], P["O"], P["H"], P["L"], P["A"], P["rawC"]
    f = features(P); S = tiers(f, P["MK"]); days, cols = C.index, C.columns; N = len(days)
    alive = (C.notna().cumsum() > 0) & (C.notna()[::-1].cumsum()[::-1] > 0); Cf = C.ffill(limit=20).where(alive)
    mm = {n: Cf.rolling(n, min_periods=n).mean() for n in (5, 10, 20, 60)}; okm = Cf.notna() & mm[60].notna()
    mk = P["MK"].ffill().iloc[-1]; tw = [c for c in cols if mk.get(c) == "TWSE"]
    bear = ((mm[5] < mm[10]) & (mm[10] < mm[20]) & okm)[tw].sum(axis=1); okn = okm[tw].sum(axis=1); del Cf, mm, alive
    panic = (bear.rolling(10, min_periods=1).max() >= 750) if era == "new" else ((bear / okn).rolling(10, min_periods=1).max() >= RATIO)
    idx = np.arange(N); lastp = pd.Series(np.where(panic.values, idx, np.nan)).ffill().values; dsp = idx - lastp
    dser = pd.Series(days)
    fd = set(dser[dser.dt.day >= 11].groupby(dser.dt.to_period("M")).first())
    sel = S.base & (S.date.isin(fd) | S.pk)
    R = S[sel][["date", "code", "pk", "hot", "r20"]].copy()
    R["fd"] = R.date.isin(fd)
    R["ti"] = R.date.map({d: i for i, d in enumerate(days)}).values; R["ji"] = R.code.map({c: j for j, c in enumerate(cols)}).values
    ti, ji = R.ti.values, R.ji.values
    take = lambda X: (X.values if isinstance(X, pd.DataFrame) else X)[ti, ji]
    R["panic"] = panic.values[ti]; R["dsp"] = dsp[ti]
    # 報酬：隔天開盤買，抱 H 日
    Ov, Hv, Lv = O.values, H.values, L.values; prevc = C.ffill().shift(1).values
    e = np.minimum(ti + 1, N - 1); o = Ov[e, ji]
    locked = (o == Hv[e, ji]) & (Hv[e, ji] == Lv[e, ji]) & (o / prevc[e, ji] - 1 >= LOCK)
    bidx = bench_index(days).values
    for h in HS:
        Cff = C.ffill(limit=h).values; k = ti + h; ok = (k < N) & (o > 0) & ~locked
        kk = np.minimum(k, N - 1)
        R[f"y{h}"] = np.where(ok, (Cff[kk, ji] / o - 1) * 100, np.nan)
        R[f"b{h}"] = np.where(k < N, (bidx[np.minimum(k, N - 1)] / bidx[ti] - 1) * 100, np.nan)
        del Cff
    R["blocked"] = locked
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = C.pct_change(fill_method=None)
        R["mom6_1"] = take(C.shift(20) / C.shift(125) - 1); R["mom12_1"] = take(C.shift(20) / C.shift(250) - 1)
        R["d_hi250"] = take(C / C.rolling(250, min_periods=120).max() - 1); R["m60"] = take(C / C.rolling(60, min_periods=60).mean() - 1)
        R["vol20"] = take(r1.rolling(20, min_periods=15).std()); R["max20"] = take(r1.rolling(20, min_periods=15).max())
        R["nlim60"] = take((f["ch"] >= 9.5).astype(float).rolling(60, min_periods=20).sum())
        ind = pd.read_csv("data/industry.csv", dtype={"code": str}).set_index("code")
        cap = pd.Series(ind["cap"]).reindex(cols); grp = pd.Series(ind["ind_name"]).reindex(cols)
        R["lmv"] = take(np.log(raw * (cap / 10).values[None, :]))
        a20 = A.rolling(20, min_periods=10).mean(); R["a20r"] = take(np.log(a20.div(A.sum(axis=1).rolling(20, min_periods=10).mean(), axis=0)))
        R["am"] = take(A.rolling(5, min_periods=5).mean() / a20)
        fns = sorted(glob.glob("data/chips/*.csv*"), key=lambda x: (os.path.basename(x)[:4], x.endswith(".csv")))
        ch = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in fns], ignore_index=True)
        ch["date"] = pd.to_datetime(ch.date); ch = ch.drop_duplicates(["date", "code"], keep="last")
        for kk_, nn in (("trust", 5), ("trust", 20), ("foreign", 20)):
            x = ch.pivot(index="date", columns="code", values=kk_).reindex(index=days, columns=cols) * 1000 * raw
            R[f"{kk_}{nn}"] = take(x.rolling(nn, min_periods=1).sum() / a20); del x
        mg = ch.pivot(index="date", columns="code", values="margin").reindex(index=days, columns=cols)
        R["mg20"] = take(mg / mg.shift(20) - 1); del mg, ch
        rv = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in sorted(glob.glob("data/revenue/*.csv*"))], ignore_index=True)
        rv["pub"] = pd.to_datetime(rv.ym + "-01") + pd.DateOffset(months=1) + pd.Timedelta(days=10)
        yoy = rv.pivot_table(index="ym", columns="code", values="yoy").sort_index().reindex(columns=cols)
        rev = rv.pivot_table(index="ym", columns="code", values="rev").sort_index().reindex(index=yoy.index, columns=cols)
        mom = rv.pivot_table(index="ym", columns="code", values="mom").sort_index().reindex(index=yoy.index, columns=cols)
        pub = rv.groupby("ym").pub.first().sort_index().reindex(yoy.index)
        kn = np.searchsorted(pub.values, R.date.values, side="right") - 1
        g = lambda M, k_: np.where(k_ >= 0, M[np.maximum(k_, 0), ji], np.nan)
        R["yoy1"] = g(yoy.values, kn)
        y3 = np.vstack([np.where(kn - s >= 0, g(yoy.values, kn - s), np.nan) for s in range(3)]); R["yoy3"] = np.where(kn >= 2, np.nanmean(y3, axis=0), np.nan)
        p12 = np.vstack([np.where(kn - s >= 0, g(rev.values, kn - s), np.nan) for s in range(1, 13)])
        R["rev_hi12"] = np.where(np.isfinite(p12).sum(axis=0) >= 6, g(rev.values, kn) / np.nanmax(p12, axis=0) - 1, np.nan)
        R["rev_acc"] = R.yoy1 - R.yoy3; R["mom1"] = g(mom.values, kn); del y3, p12
        inb = S[S.base].assign(v=True).pivot(index="date", columns="code", values="v").reindex(index=days, columns=cols).fillna(False).astype(bool)
        rb = r1.where(inb); g20 = np.full((N, len(cols)), np.nan)
        for gname, cs in grp.dropna().groupby(grp.dropna()).groups.items():
            ix = (1 + rb[list(cs)].mean(axis=1).fillna(0)).cumprod(); g20[:, [cols.get_loc(c) for c in cs]] = (ix / ix.shift(20) - 1).values[:, None]
        R["g20"] = g20[ti, ji]; R["r20"] = R.r20 / 100; R["rs_g20"] = R.r20 - R.g20
    R = R.replace([np.inf, -np.inf], np.nan)
    R.to_pickle(out); os.chdir(REPO)
    return R

if __name__ == "__main__" and sys.argv[1:2] == ["build"]:
    for era in ("new", "old"):
        R = snap(era); m = R[R.fd]
        print(era, len(R), "列；每月快照", m.date.nunique(), "個月，平均每月", round(len(m) / m.date.nunique()), "檔；今日發動", int(R.pk.sum()),
              "；缺值%", (m[FACT].isna().mean() * 100).round(0).astype(int).to_dict(), flush=True)

def period(d):
    return np.where(d < pd.Timestamp("2020-01-01"), "2016-19", np.where(d < pd.Timestamp("2024-01-01"), "2020-23", "2024-26"))

def tstat(s):
    s = s.dropna(); return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))) if len(s) > 2 else np.nan

def load():
    Rn, Ro = snap("new"), snap("old")
    Ro = Ro[(Ro.date >= "2016-01-01") & (Ro.date <= "2019-12-31")]
    R = pd.concat([Ro.assign(era="old"), Rn.assign(era="new")], ignore_index=True); R["per"] = period(R.date)
    return R

def study_a(R, cost=0.6):
    M = R[R.fd & ~R.panic & R.y20.notna()].copy()
    res, ser = {}, {}
    for fc in FACT:
        z = M[M[fc].notna()]
        q = z.groupby("date")[fc].rank(pct=True, method="first")
        z = z.assign(q=np.minimum(5, np.floor(q * 5 - 1e-9) + 1))
        tab = z.groupby(["date", "q"]).y20.mean().unstack()
        sp = (tab[5.0] - tab[1.0]); ser[fc] = tab
        res[fc] = {}
        for per in ("2016-19", "2020-23", "2024-26"):
            pm = period(tab.index) == per; s = sp[pm]
            yrs = s.groupby(s.index.year).mean()
            res[fc][per] = {"差": round(float(s.mean()), 2), "t": round(tstat(s), 2), "第5組": round(float(tab[5.0][pm].mean()), 2), "第1組": round(float(tab[1.0][pm].mean()), 2),
                            "逐年差": yrs.round(2).to_dict(), "月數": int(pm.sum())}
    sel = {fc: int(np.sign(res[fc]["2020-23"]["差"])) for fc in FACT if abs(res[fc]["2020-23"]["t"]) >= 2}
    for fc in FACT:
        d = sel.get(fc, int(np.sign(res[fc]["2020-23"]["差"])))
        ok = all(res[fc][p]["差"] * d > 0 for p in ("2016-19", "2024-26"))
        ok = ok and sum(v * d > 0 for v in res[fc]["2016-19"]["逐年差"].values()) >= 3 and sum(v * d > 0 for v in res[fc]["2024-26"]["逐年差"].values()) >= 2
        res[fc]["入選"] = fc in sel; res[fc]["方向"] = d; res[fc]["全新期間也成立"] = bool(ok)
    # 綜合分數
    def composite(z):
        sc = np.zeros(len(z))
        for fc, d in sel.items():
            pr = z.groupby("date")[fc].rank(pct=True).fillna(0.5).values; sc += pr if d > 0 else 1 - pr
        return sc / max(1, len(sel))
    ports = {}
    U = M.assign(s=composite(M))
    pool = R[R.fd & ~R.panic & R.y20.notna() & (R.hot | R.pk)]
    P2 = U.merge(pool[["date", "code"]], on=["date", "code"])  # 用全市場排名的分數，只在強勢股裡挑
    for nm, z, K in (("全市場前20", U, 20), ("全市場前50", U, 50), ("強勢股裡前10", P2, 10), ("全部分析範圍（等權）", U, None)):
        z = z.sort_values(["date", "s"], ascending=[True, False]); pk = z if K is None else z.groupby("date").head(K)
        months = sorted(pk.date.unique()); prev = set(); rows = []
        for d in months:
            g = pk[pk.date == d]; cur = set(g.code); to = 1.0 if not prev else len(cur - prev) / len(cur); prev = cur
            rows.append({"date": d, "gross": g.y20.mean(), "to": to if K is not None else 0.0, "b": g.b20.iloc[0]})
        T = pd.DataFrame(rows); T["net"] = T.gross - cost * T.to; T["net11"] = T.gross - 1.1 * T.to; T["per"] = period(T.date)
        ports[nm] = {}
        for per, g in T.groupby("per"):
            n = len(g); cagr = lambda r: round(float((np.prod(1 + r / 100) ** (12 / n) - 1) * 100), 1)
            ports[nm][per] = {"月數": n, "年化（成本0.6%）": cagr(g.net), "年化（成本1.1%）": cagr(g.net11), "0050年化": cagr(g.b), "贏0050的月份%": round(float((g.net > g.b).mean() * 100), 0),
                              "平均換股比例%": round(float(g.to.mean() * 100), 0), "每月平均（未扣成本）": round(float(g.gross.mean()), 2)}
    return res, sel, ports

def study_b(R, v1):
    K = R[R.pk & ~R.panic].merge(v1, on=["date", "code"], how="left")
    out = {}
    for per, g in K.groupby("per"):
        out[per] = {}
        for lab, z in (("全部", g), ("v1高", g[g.v1 >= 2.70])):
            out[per][lab] = {}
            for h in HS:
                ok = z[z[f"y{h}"].notna() & z[f"b{h}"].notna()]; y = ok[f"y{h}"] - 1.1
                out[per][lab][f"{h}日"] = {"n": int(len(ok)), "平均": round(float(y.mean()), 2), "賺錢%": round(float((y > 0).mean() * 100), 1),
                                           "減0050": round(float((y - ok[f"b{h}"]).mean()), 2), "贏0050%": round(float((y > ok[f"b{h}"]).mean() * 100), 1)}
    return out

def study_c(R):
    K = R[R.pk & R.y20.notna() & ~((R.era == "new") & (R.date < "2020-07-01"))].copy()
    K["grp"] = np.where(K.panic, "恐慌中", pd.cut(K.dsp.fillna(9999), [0, 20, 60, 120, 1e9], labels=["1–20", "21–60", "61–120", ">120"]).astype(str))
    K["y"] = K.y20 - 1.1; K["eraP"] = np.where(K.era == "old", "2016-19", "2020-26")
    t = K.groupby(["eraP", "grp"]).agg(n=("y", "size"), 平均=("y", "mean"), 中位=("y", "median"), 賺錢=("y", lambda v: (v > 0).mean() * 100), 減0050=("b20", lambda v: 0)).round(2)
    t["減0050"] = K.assign(x=K.y - K.b20).groupby(["eraP", "grp"]).x.mean().round(2)
    yr = K[~K.panic].assign(g=K.grp).groupby([K.date.dt.year, "g"]).y.mean().unstack().round(2)
    return t, yr

if __name__ == "__main__" and sys.argv[1:2] == ["run"]:
    R = load()
    import pickle
    m1 = pickle.load(open(f"{REPO}/data/model/normal_v1.pkl", "rb"))
    v1 = []
    for era in ("new", "old"):
        rr = pd.read_pickle(f"{CACHE}/rows_{era}.pkl"); rr = rr[rr.pk]
        v1.append(pd.DataFrame({"date": rr.date.values, "code": rr.code.values, "v1": m1["model"].predict(rr[m1["feats"]].astype(float))}))
    v1 = pd.concat(v1).drop_duplicates(["date", "code"], keep="last")
    res, sel, ports = study_a(R)
    print("== A 因子（平常月份，五等分差＝第5組−第1組，20日報酬%，未扣成本）")
    rows = []
    for fc in FACT:
        r = res[fc]; rows.append([NAME[fc], r["2020-23"]["差"], r["2020-23"]["t"], r["2016-19"]["差"], r["2016-19"]["t"], r["2024-26"]["差"], r["2024-26"]["t"],
                                  "入選" if r["入選"] else "", ("+" if r["方向"] > 0 else "−"), "成立" if r["全新期間也成立"] else ""])
    print(pd.DataFrame(rows, columns=["特性", "20-23差", "t", "16-19差", "t", "24-26差", "t", "入選", "方向", "全新期間"]).to_string(index=False))
    print("入選（依 2020–23）：", {NAME[k]: v for k, v in sel.items()})
    print("== 綜合分數組合（只算平常月份）")
    for k, v in ports.items(): print(k, json.dumps(v, ensure_ascii=False))
    b = study_b(R, v1)
    print("== B 抱多久（平常今日發動，扣1.1%）")
    for per, v in b.items():
        for lab, w in v.items(): print(per, lab, {h: (x["平均"], x["賺錢%"], x["減0050"], x["n"]) for h, x in w.items()})
    t, yr = study_c(R)
    print("== C 恐慌後窗口（今日發動抱20日、扣1.1%）"); print(t.to_string()); print(yr.to_string())
    json.dump({"A": res, "A_sel": sel, "A_ports": ports, "B": b, "C": t.reset_index().to_dict("records")}, open(f"{REPO}/research/results/deep_normal.json", "w"), ensure_ascii=False, indent=1, default=str)
