"""「趨勢＋營收」綜合分數（research/deep_normal.py 的 12 個入選特性，等權百分位，不擬合）。

每個特性換算成「當天分析範圍內全部股票」裡的百分位（缺值當 0.5），12 個平均＝分數（0～1）。
研究結果（平常時期今日發動，抱20日）：分數 < 0.5 的三段期間都比 0050 少賺約 5%（每月為單位 t −5～−8），所以網站拿來當防守標籤「趨勢營收弱」。
定義與 research/deep_normal.py snap() 相同；只用當天收盤後就知道的資料（營收：YYYY-MM 從下個月 11 日起才算知道）。
"""
import glob, os
import numpy as np, pandas as pd

FEATS = ["r20", "mom6_1", "d_hi250", "m60", "am", "yoy1", "yoy3", "rev_acc", "rev_hi12", "mom1", "mg20", "rs_g20"]
WEAK = 0.5

def factors_on(P, f, S, date):
    """回傳 DataFrame（index＝全部股票代號）：當天 12 個特性的值，以及是否在分析範圍內。"""
    C, A, raw = P["C"], P["A"], P["rawC"]; days, cols = C.index, C.columns; t = days.get_loc(date)
    at = lambda X, k=0: (X.iloc[t - k] if t - k >= 0 else pd.Series(np.nan, index=cols))
    Sd = S[S.date == date].set_index("code")
    out = pd.DataFrame(index=cols)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["r20"] = (Sd.r20 / 100).reindex(cols)
        out["mom6_1"] = at(C, 20) / at(C, 125) - 1
        w = C.iloc[max(0, t - 249):t + 1]; out["d_hi250"] = np.where(w.notna().sum() >= 120, at(C) / w.max() - 1, np.nan)
        w = C.iloc[max(0, t - 59):t + 1]; out["m60"] = np.where(w.notna().sum() >= 60, at(C) / w.mean() - 1, np.nan)
        a5 = A.iloc[max(0, t - 4):t + 1]; a20 = A.iloc[max(0, t - 19):t + 1]
        out["am"] = np.where((a5.notna().sum() >= 5) & (a20.notna().sum() >= 10), a5.mean() / a20.mean(), np.nan)
        fns = sorted(glob.glob("data/chips/*.csv*"), key=lambda x: (os.path.basename(x)[:4], x.endswith(".csv")))[-2:]
        ch = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in fns], ignore_index=True)
        ch["date"] = pd.to_datetime(ch.date); ch = ch.drop_duplicates(["date", "code"], keep="last")
        mg = ch.pivot(index="date", columns="code", values="margin").reindex(index=days, columns=cols)
        # 11/27：融資餘額晚上約 21:30～22:00 才公布；當天還沒有時先用最近一天有資料的融資（之後到齊會自動換成當天）
        tm = t
        while tm > 0 and not mg.iloc[tm].notna().any(): tm -= 1
        out["mg20"] = (mg.iloc[tm] / mg.iloc[tm - 20] - 1) if tm >= 20 else np.nan
        rv = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in sorted(glob.glob("data/revenue/*.csv*"))[-3:]], ignore_index=True)
        rv["pub"] = pd.to_datetime(rv.ym + "-01") + pd.DateOffset(months=1) + pd.Timedelta(days=10)
        rv = rv[rv.pub <= date]
        piv = lambda v: rv.pivot_table(index="ym", columns="code", values=v).sort_index().reindex(columns=cols)
        yoy, rev, mom = piv("yoy"), piv("rev"), piv("mom")
        out["yoy1"] = yoy.iloc[-1]; out["yoy3"] = yoy.iloc[-3:].mean() if len(yoy) >= 3 else np.nan
        p12 = rev.iloc[-13:-1]
        out["rev_hi12"] = np.where(p12.notna().sum() >= 6, rev.iloc[-1] / p12.max() - 1, np.nan)
        out["rev_acc"] = out.yoy1 - out.yoy3; out["mom1"] = mom.iloc[-1]
        ind = pd.read_csv("data/industry.csv", dtype={"code": str}).set_index("code"); grp = pd.Series(ind["ind_name"]).reindex(cols)
        lo = max(0, t - 21)
        Cw = C.iloc[lo:t + 1]; r1 = Cw.pct_change(fill_method=None)
        inb = S[S.base & (S.date >= days[lo]) & (S.date <= date)].assign(v=True).pivot(index="date", columns="code", values="v").reindex(index=Cw.index, columns=cols).fillna(False).astype(bool)
        rb = r1.where(inb); g20 = pd.Series(np.nan, index=cols)
        for g, cs in grp.dropna().groupby(grp.dropna()).groups.items():
            ix = (1 + rb[list(cs)].mean(axis=1).fillna(0)).cumprod()
            if len(ix) > 20: g20[list(cs)] = ix.iloc[-1] / ix.iloc[-21] - 1
        out["rs_g20"] = out.r20 - g20
    out = out.replace([np.inf, -np.inf], np.nan)
    out["base"] = out.index.isin(Sd.index[Sd.base])
    return out

def is_tdr(codes):
    """91xx 臺灣存託憑證（第 64 輪：月選與今日分數都排除）。"""
    return pd.Index(codes).astype(str).str.match(r"^91\d\d$")

def margin_ok(date, cover=0.98):
    """當天的融資餘額是否到齊（11/27 第八、九份覆核）：去重後，上市、上櫃兩個市場都要有資料，
    而且「前一個兩市場都有資料的交易日」有融資的股票，今天各市場至少 98% 也有融資。"""
    fns = sorted(glob.glob("data/chips/*.csv*"), key=lambda x: (os.path.basename(x)[:4], x.endswith(".csv")))[-2:]
    if not fns: return False
    ch = pd.concat([pd.read_csv(fn, dtype={"code": str}, usecols=["date", "market", "code", "margin"]) for fn in fns], ignore_index=True)
    ch["date"] = pd.to_datetime(ch.date); ch = ch[ch.margin.notna()].drop_duplicates(["date", "market", "code"], keep="last")
    d = pd.Timestamp(date); mk = ("TWSE", "TPEX")
    have = ch.groupby("date").market.agg(lambda z: set(z)); both = [x for x, v in have.items() if all(m in v for m in mk)]
    if d not in both: return False
    prev = [x for x in both if x < d]
    if not prev: return True
    p = prev[-1]
    for m in mk:
        a, b = set(ch[(ch.date == p) & (ch.market == m)].code), set(ch[(ch.date == d) & (ch.market == m)].code)
        if not a or len(a & b) / len(a) < cover: return False
    return True

def table(P, f, S, date, codes):
    """codes 的 12 個百分位與綜合分數（DataFrame）。百分位＝平均名次 ÷ 檔數，同值取平均（與研究端 pandas rank(pct=True) 相同；11/27 第八份覆核）。"""
    X = factors_on(P, f, S, date); ref = X[X.base & ~is_tdr(X.index)]   # 11/27 第 64 輪：TDR 不列入百分位基準
    R = {k: np.sort(ref[k].dropna().values) for k in FEATS}; rows = {}
    for c in codes:
        if c not in X.index: continue
        ps = {}
        for k in FEATS:
            v, r = X.at[c, k], R[k]
            if pd.isna(v) or len(r) < 50: ps[k] = 0.5
            else:
                lo, hi = np.searchsorted(r, v, side="left"), np.searchsorted(r, v, side="right")
                ps[k] = float((lo + (hi - lo + 1) / 2) / len(r)) if hi > lo else float(lo / len(r))
        rows[c] = ps
    T = pd.DataFrame.from_dict(rows, orient="index", columns=FEATS); T["score"] = T[FEATS].mean(axis=1)
    for k in FEATS: T[k + "_v"] = X.loc[T.index, k].astype(float).values   # 原始值（11/27 第九份覆核）
    return T

def scores(P, f, S, date, codes):
    """codes 的綜合分數：每個特性在「當天分析範圍內全部股票」的百分位（缺值 0.5）等權平均。"""
    T = table(P, f, S, date, codes); return {c: float(v) for c, v in T.score.items()}
