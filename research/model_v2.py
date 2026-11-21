"""平常時期選股模型 v2：找出能改善 normal_v1 的做法，最後在從沒用過的 2016–2019 檢驗。

全部在看結果之前寫下（2026/11/21 17:05），看到結果後不修改。

為什麼：normal_v1 在 2024–2026 有排序能力（高 +3.22%、低 −3.72%），但賺錢比例 46.8% 未達 50%。
2024–2026 已經看過很多次，不能再拿來挑方法；所以另外補抓 2015-06～2019-12 的資料（data/old/），當作全新的最後檢驗期。

共同設定（同 research/normal_model.py）：今日發動（分析範圍內）、訊號日不在恐慌期間；隔天開盤買（開盤一價漲停買不到不計）、
持有 20 個交易日、扣 1.1%。GBT 參數固定同 v1（max_depth 3、learning_rate 0.1、300 次、min_samples_leaf 50、l2 1.0），不再調參數。

候選（開發只用 2020–2023）：
  B  基準：v1 的 40 個特徵，同方法重訓（用來比較）
  N  新特徵：B ＋ 離250日高、營收比前12個月最高、營收年增加速（最近月 − 近3月平均）、營收月增、
     發動前20日收盤區間（越窄＝整理越緊）、短長期波動比（5日／60日真實波幅）、相對產業強度（近20日漲幅 − 產業指數近20日）、
     上市天數（最多500）、是否處置中、投信連買天數、外資近20日買超／均額；
     另外把「名目」特徵換成「跨年代可比」：成交額改成占全市場成交額比例、上市空頭家數改成占有60日線上市檔數比例
     （2016–2019 市場成交額、上市檔數都比較少）
  X  N 的特徵但不放大盤特徵；目標改成超額報酬（減去同一天買進全部分析範圍股票的平均 20 日報酬）
  W  N 的特徵；目標改成「20 日後賺錢與否」（分類，排序用機率）
  Q  N 的特徵；訓練樣本擴大到「仍在動 ＋ 今日發動」（同樣排除恐慌期間），目標同 B
開發方法：兩折時間序列驗證——2020–21 訓練→2022 驗證；2020–22 訓練→2023 驗證。
開發指標：驗證年的今日發動中分數前 20% 的平均 20 日報酬，兩折平均。指標最高者＝v2（只挑一個；B 勝出就表示新做法沒有用）。

最後檢驗（各只做一次）：
  ① 2016–2019（全新）：v1 用凍結的 data/model/normal_v1（不重訓），門檻 hi=2.70、lo=−6.96；
     v2 用 2020–2023 全部重訓，門檻＝「2020–22 訓練的 v2 在 2023 今日發動預測分數」第 80／20 百分位。
  ② 2024–2026/11（已看過，只當次要參考）：同樣套用。
  恐慌期間排除：2020 之後照原定義（上市空頭家數近10日最高 ≥ 750）；2016–2019 上市檔數較少，用同比例：
     「空頭家數 ÷ 有60日線的上市檔數」近10日最高 ≥ 750 ÷（2020–2026 有60日線上市檔數的日平均）。
判斷：
  v1 在全新期間是否仍有用：2016–2019 中 v1「高」平均 > 全部今日發動平均、「高」平均 > 「低」平均、4 年中至少 3 年「高」> 全部。
  網站換成 v2：2016–2019 中 v2 過門檻組平均 ≥ v1「高」+ 1 個百分點、賺錢比例不低於 v1「高」、4 年中至少 3 年 v2 過門檻組 > 全部，
     而且 2024–2026 不比 v1「高」差超過 0.5 個百分點。否則網站維持 v1。
另外報告（不影響判斷）：每天只買今日發動中分數最高的 1 檔／3 檔（資金分 20 份輪流進場、各抱 20 日）的資金曲線、最大回落，對比 0050；
     Q 另報從「仍在動＋今日發動」每天挑前 3 檔。
"""
import glob, json, os, pickle, shutil, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO)
COST, LOCK, HOLD = 1.1, 0.095, 20
GBT = dict(max_depth=3, learning_rate=0.1, max_iter=300, min_samples_leaf=50, l2_regularization=1.0, random_state=0)
CACHE = "/tmp/model_v2"; os.makedirs(CACHE, exist_ok=True)
V1 = ["ch", "a1", "a20p", "vr", "v", "r5", "r20", "ru", "snp", "rc", "r60", "d_hi60", "d_hi120", "m5", "m20", "m60", "nlim60", "upper", "body", "gap",
      "lprice", "lmv", "a20", "foreign1", "foreign5", "trust1", "trust5", "inst1", "inst5", "mg5", "yoy1", "yoy3", "g_m10", "g_5", "g_peers",
      "mk_bear", "mk_ew5", "mk_ew20", "mk_ew_m60", "mk_breadth"]
NEW = ["d_hi250", "rev_hi12", "rev_acc", "mom1", "tight20", "atr_ratio", "rs_g20", "age", "disp", "trust_days", "foreign20", "a1r", "a20pr", "a20r", "mk_bearf"]
MKT = ["mk_bear", "mk_ew5", "mk_ew20", "mk_ew_m60", "mk_breadth", "mk_bearf"]
FEATS = {"B": V1, "N": [c for c in V1 if c not in ("a1", "a20p", "a20", "mk_bear")] + NEW}
FEATS["X"] = [c for c in FEATS["N"] if c not in MKT]; FEATS["W"] = FEATS["N"]; FEATS["Q"] = FEATS["N"]

def old_root():
    """2015-06～2019-12 的資料（＋2020 年，讓 2019 年底的訊號能算到 20 日後）組成一個暫時的資料夾，沿用同一套程式。"""
    r = "/tmp/oldroot"; d = r + "/data"
    if os.path.exists(r): shutil.rmtree(r)
    for sub in ("history", "chips", "revenue"): os.makedirs(f"{d}/{sub}")
    ln = lambda a, b: os.symlink(a, b)
    for fn in sorted(glob.glob(f"{REPO}/data/old/history/*.csv.gz")) + [f"{REPO}/data/history/2020.csv.gz"]: ln(fn, f"{d}/history/{os.path.basename(fn)}")
    for fn in sorted(glob.glob(f"{REPO}/data/old/chips/*.csv.gz")) + [f"{REPO}/data/chips/2020.csv.gz"]: ln(fn, f"{d}/chips/{os.path.basename(fn)}")
    for fn in sorted(glob.glob(f"{REPO}/data/old/revenue/*.csv.gz")):
        if os.path.basename(fn)[:4] < "2019": ln(fn, f"{d}/revenue/{os.path.basename(fn)}")
    for y in ("2019", "2020"): ln(f"{REPO}/data/revenue/{y}.csv.gz", f"{d}/revenue/{y}.csv.gz")
    for name in ("adjust.csv", "disposal.csv"):
        a = pd.read_csv(f"{REPO}/data/old/{name}", dtype={"code": str}); b = pd.read_csv(f"{REPO}/data/{name}", dtype={"code": str})
        pd.concat([a, b]).drop_duplicates().to_csv(f"{d}/{name}", index=False)
    ln(f"{REPO}/data/industry.csv", f"{d}/industry.csv")
    return r

def build(era):
    """era = new（2020–2026）或 old（2015-06～2019）。回傳「仍在動＋今日發動」每列的特徵、報酬、旗標。"""
    out = f"{CACHE}/rows_{era}.pkl"
    if os.path.exists(out): return pd.read_pickle(out)
    os.chdir(old_root() if era == "old" else REPO)
    from radar_core import load_prices, panels, features, tiers
    p = load_prices(); P = panels(p); del p
    C, O, H, L, A, raw = P["C"], P["O"], P["H"], P["L"], P["A"], P["rawC"]
    f = features(P); S = tiers(f, P["MK"])
    days, cols = C.index, C.columns; N = len(days)
    # 恐慌（上市短均線空頭排列家數；同 build_site / research/relaunch.py）
    alive = (C.notna().cumsum() > 0) & (C.notna()[::-1].cumsum()[::-1] > 0)
    Cf = C.ffill(limit=20).where(alive)
    mm = {n: Cf.rolling(n, min_periods=n).mean() for n in (5, 10, 20, 60)}
    okm = Cf.notna() & mm[60].notna()
    mk = P["MK"].ffill().iloc[-1]; tw = [c for c in cols if mk.get(c) == "TWSE"]
    bear = ((mm[5] < mm[10]) & (mm[10] < mm[20]) & okm)[tw].sum(axis=1); okn = okm[tw].sum(axis=1)
    del Cf, mm, alive
    # 樣本：分析範圍內的仍在動 ∪ 今日發動
    R = S[S.base & (S.hot | S.pk)].copy()
    R["ti"] = R.date.map({d: i for i, d in enumerate(days)}).values; R["ji"] = R.code.map({c: j for j, c in enumerate(cols)}).values
    ti, ji = R.ti.values, R.ji.values
    take = lambda X: (X.values if isinstance(X, pd.DataFrame) else X)[ti, ji]
    R["a1"] = np.log(np.maximum(R.a1, 1e-3)); R["a20p"] = np.log(np.maximum(R.a20p, 1e-3))
    # 報酬：隔天開盤買，抱 20 日（同 normal_model.py）
    Cv, Ov, Hv, Lv = C.values, O.values, H.values, L.values
    prevc = C.ffill().shift(1).values
    e = ti + 1; ok = e + HOLD - 1 < N
    ec = np.where(ok, e, N - 1)
    o = Ov[ec, ji]; locked = (Ov[ec, ji] == Hv[ec, ji]) & (Hv[ec, ji] == Lv[ec, ji]) & (Ov[ec, ji] / prevc[ec, ji] - 1 >= LOCK)
    Cff = C.ffill(limit=HOLD).values
    k = np.where(ok, e + HOLD - 1, N - 1)
    y = (Cff[k, ji] / o - 1) * 100 - COST
    R["y"] = np.where(ok & (o > 0) & ~locked, y, np.nan); R["blocked"] = locked & ok
    # 同一天買進全部分析範圍股票的平均 20 日報酬（超額報酬用）
    inb = S[S.base].assign(v=True).pivot(index="date", columns="code", values="v").reindex(index=days, columns=cols).fillna(False).astype(bool)
    fwd = (C.ffill(limit=HOLD).shift(-HOLD) / O.shift(-1) - 1) * 100
    R["yx"] = R.y - fwd.where(inb).mean(axis=1).values[ti]
    del fwd
    # 恐慌旗標
    R["bear"] = bear.values[ti]; R["bearf"] = (bear / okn).values[ti]
    R["panic750"] = (bear.rolling(10, min_periods=1).max() >= 750).values[ti]
    R["bearf10"] = (bear / okn).rolling(10, min_periods=1).max().values[ti]
    R["okn"] = okn.values[ti]
    # ---- v1 的面板特徵（同 model_features.make_context）----
    MA = {n: C.rolling(n, min_periods=n).mean() for n in (5, 20, 60)}
    with np.errstate(divide="ignore", invalid="ignore"):
        R["r60"] = take(C / C.shift(60) - 1); R["d_hi60"] = take(C / C.rolling(60, min_periods=30).max() - 1)
        R["d_hi120"] = take(C / C.rolling(120, min_periods=60).max() - 1); R["d_hi250"] = take(C / C.rolling(250, min_periods=120).max() - 1)
        for n in (5, 20, 60): R[f"m{n}"] = take(C / MA[n] - 1)
        R["nlim60"] = take((f["ch"] >= 9.5).astype(float).shift(1).rolling(60, min_periods=1).sum())
        R["upper"] = take((H - C) / (H - L).replace(0, np.nan)); R["body"] = take(C / O - 1); R["gap"] = take(O / C.shift(1) - 1)
        R["lprice"] = take(np.log(raw))
        ind = pd.read_csv("data/industry.csv", dtype={"code": str}).set_index("code")
        cap = pd.Series(ind["cap"]).reindex(cols); grp = pd.Series(ind["ind_name"]).reindex(cols)
        R["lmv"] = take(np.log(raw * (cap / 10).values[None, :]))
        a20 = A.rolling(20, min_periods=10).mean().shift(1)
        R["a20"] = take(np.log(a20))
        tot = A.sum(axis=1)
        R["a1r"] = take(np.log(A.div(tot, axis=0))); R["a20pr"] = take(np.log((A.shift(1).rolling(20, min_periods=20).mean()).div(tot.shift(1).rolling(20).mean(), axis=0)))
        R["a20r"] = take(np.log(a20.div(tot.rolling(20, min_periods=10).mean().shift(1), axis=0)))
        tr = pd.concat([(H - L), (H - C.shift()).abs(), (C.shift() - L).abs()]).groupby(level=0).max()
        R["atr_ratio"] = take(tr.rolling(5, min_periods=5).mean() / tr.rolling(60, min_periods=40).mean()); del tr
        c1 = C.shift(1); R["tight20"] = take((c1.rolling(20, min_periods=15).max() - c1.rolling(20, min_periods=15).min()) / c1); del c1
        first = C.notna().values.argmax(axis=0); age = np.arange(N)[:, None] - first[None, :]; age = np.where(first[None, :] == 0, 500, np.minimum(age, 500))
        R["age"] = age[ti, ji]; del age
    del MA
    # 籌碼
    fns = sorted(glob.glob("data/chips/*.csv*"), key=lambda x: (os.path.basename(x)[:4], x.endswith(".csv")))
    ch = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in fns], ignore_index=True)
    ch["date"] = pd.to_datetime(ch.date); ch = ch.drop_duplicates(["date", "code"], keep="last")
    for kk in ("foreign", "trust", "inst"):
        x = ch.pivot(index="date", columns="code", values=kk).reindex(index=days, columns=cols) * 1000 * raw
        R[kk + "1"] = take(x / a20); R[kk + "5"] = take(x.rolling(5, min_periods=1).sum() / a20)
        if kk == "foreign": R["foreign20"] = take(x.rolling(20, min_periods=1).sum() / a20)
        if kk == "trust":
            pos_ = (x > 0).astype(int).values; run = np.zeros_like(pos_)
            for i in range(N): run[i] = (run[i - 1] + 1) * pos_[i] if i else pos_[i]
            R["trust_days"] = np.minimum(run[ti, ji], 20)
        del x
    mg = ch.pivot(index="date", columns="code", values="margin").reindex(index=days, columns=cols)
    R["mg5"] = take(mg / mg.shift(5) - 1); del mg, ch
    # 營收（YYYY-MM 的營收當成下個月 11 日起才知道）
    rv = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in sorted(glob.glob("data/revenue/*.csv*"))], ignore_index=True)
    rv["pub"] = pd.to_datetime(rv.ym + "-01") + pd.DateOffset(months=1) + pd.Timedelta(days=10)
    yoy = rv.pivot_table(index="ym", columns="code", values="yoy").sort_index().reindex(columns=cols)
    rev = rv.pivot_table(index="ym", columns="code", values="rev").sort_index().reindex(index=yoy.index, columns=cols)
    mom = rv.pivot_table(index="ym", columns="code", values="mom").sort_index().reindex(index=yoy.index, columns=cols)
    pub = rv.groupby("ym").pub.first().sort_index().reindex(yoy.index)
    kn = np.searchsorted(pub.values, R.date.values, side="right") - 1  # 最新已公布月份
    Y, RV, MO = yoy.values, rev.values, mom.values; nan = np.full(len(R), np.nan)
    g = lambda M, kk: np.where(kk >= 0, M[np.maximum(kk, 0), ji], np.nan)
    R["yoy1"] = g(Y, kn)
    y3 = np.vstack([g(Y, kn - s) for s in range(3)]); y3 = np.where(np.vstack([kn - s >= 0 for s in range(3)]), y3, np.nan)
    R["yoy3"] = np.where(kn >= 2, np.nanmean(y3, axis=0), np.nan)
    prev12 = np.vstack([g(RV, kn - s) for s in range(1, 13)]); prev12 = np.where(np.vstack([kn - s >= 0 for s in range(1, 13)]), prev12, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        R["rev_hi12"] = np.where(np.isfinite(prev12).sum(axis=0) >= 6, g(RV, kn) / np.nanmax(prev12, axis=0) - 1, np.nan)
    R["rev_acc"] = R.yoy1 - R.yoy3; R["mom1"] = g(MO, kn)
    del y3, prev12
    # 族群（官方產業）與大盤
    r1 = C.pct_change(fill_method=None).where(inb)
    gm10, g5, g20 = (np.full((N, len(cols)), np.nan) for _ in range(3))
    pkc = S[S.pk].assign(g=lambda z: z.code.map(grp)).groupby(["date", "g"]).size().unstack().reindex(index=days).fillna(0)
    gpeer = np.full((N, len(cols)), np.nan)
    for gname, cs in grp.dropna().groupby(grp.dropna()).groups.items():
        ix = (1 + r1[list(cs)].mean(axis=1).fillna(0)).cumprod(); jj = [cols.get_loc(c) for c in cs]
        gm10[:, jj] = (ix / ix.rolling(10).mean() - 1).values[:, None]; g5[:, jj] = (ix / ix.shift(5) - 1).values[:, None]
        g20[:, jj] = (ix / ix.shift(20) - 1).values[:, None]
        if gname in pkc.columns: gpeer[:, jj] = pkc[gname].shift(1).rolling(10, min_periods=1).sum().fillna(0).values[:, None]
        else: gpeer[:, jj] = 0
    R["g_m10"], R["g_5"], R["g_peers"] = gm10[ti, ji], g5[ti, ji], gpeer[ti, ji]
    R["rs_g20"] = R.r20 / 100 - g20[ti, ji]
    del gm10, g5, g20, gpeer
    ewx = (1 + r1.mean(axis=1).fillna(0)).cumprod()
    breadth = (C > C.rolling(20, min_periods=20).mean()).where(inb).mean(axis=1)
    R["mk_bear"] = bear.values[ti]; R["mk_ew5"] = (ewx / ewx.shift(5) - 1).values[ti]; R["mk_ew20"] = (ewx / ewx.shift(20) - 1).values[ti]
    R["mk_ew_m60"] = (ewx / ewx.rolling(60).mean() - 1).values[ti]; R["mk_breadth"] = breadth.values[ti]; R["mk_bearf"] = R.bearf
    # 處置中
    dp = pd.read_csv("data/disposal.csv", dtype={"code": str}); dp["start"] = pd.to_datetime(dp.start); dp["end"] = pd.to_datetime(dp.end)
    dmat = np.zeros((N, len(cols)), dtype=bool)
    for r in dp.dropna(subset=["start", "end"]).itertuples():
        if r.code in cols: dmat[(days >= r.start) & (days <= r.end), cols.get_loc(r.code)] = True
    R["disp"] = dmat[ti, ji].astype(float)
    R = R.replace([np.inf, -np.inf], np.nan)
    R.to_pickle(out)
    # 組合模擬要用的收盤與開盤（只留有用到的股票）
    used = sorted(set(R.code)); pd.to_pickle({"C": C[used].ffill(limit=HOLD), "O": O[used], "days": days}, f"{CACHE}/px_{era}.pkl")
    os.chdir(REPO)
    return R

if __name__ == "__main__" and sys.argv[1:2] == ["build"]:
    R = build(sys.argv[2]); print(sys.argv[2], len(R), "rows", R.date.min(), R.date.max(), "pk", int(R.pk.sum()), "hot", int(R.hot.sum()))

def sample(R, kind, panic_col="panic750"):
    base = R[R.y.notna() & ~R[panic_col].astype(bool)]
    return base[base.hot | base.pk] if kind == "Q" else base[base.pk]

def fit(kind, tr):
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    fs = FEATS[kind]; X = tr[fs].astype(float)
    if kind == "W":
        m = HistGradientBoostingClassifier(**GBT).fit(X, tr.y > 0); return lambda Z: m.predict_proba(Z[fs].astype(float))[:, 1], m
    m = HistGradientBoostingRegressor(**GBT).fit(X, (tr.yx if kind == "X" else tr.y).clip(-50, 100))
    return (lambda Z: m.predict(Z[fs].astype(float))), m

def grp_stats(y):
    return {"n": int(len(y)), "平均": round(float(y.mean()), 2), "中位": round(float(y.median()), 2), "賺錢%": round(float((y > 0).mean() * 100), 1)}

if __name__ == "__main__" and sys.argv[1:2] == ["dev"]:
    R = build("new"); R = R[R.date < "2024-01-01"]
    res = {}
    for kind in ("B", "N", "X", "W", "Q"):
        res[kind] = {}
        for tr_end, va_year in (("2022-01-01", 2022), ("2023-01-01", 2023)):
            tr = sample(R[R.date < tr_end], kind); va = sample(R[R.date.dt.year == va_year], "F")
            score, _ = fit(kind, tr); s = score(va); q8, q2 = np.quantile(s, 0.8), np.quantile(s, 0.2)
            res[kind][va_year] = {"訓練筆數": len(tr), "前20%": grp_stats(va.y[s >= q8]), "後20%": grp_stats(va.y[s <= q2]), "全部": grp_stats(va.y)}
        res[kind]["指標"] = round(np.mean([res[kind][y]["前20%"]["平均"] for y in (2022, 2023)]), 2)
        print(kind, "指標", res[kind]["指標"], {y: (res[kind][y]["前20%"]["平均"], res[kind][y]["前20%"]["賺錢%"], res[kind][y]["後20%"]["平均"], res[kind][y]["全部"]["平均"]) for y in (2022, 2023)}, flush=True)
    best = max(res, key=lambda k: res[k]["指標"])
    print("v2 ＝", best)
    json.dump({"dev": res, "v2": best}, open(f"{REPO}/research/results/model_v2_dev.json", "w"), ensure_ascii=False, indent=1)
