"""雷達的共用計算：讀行情、除權息調整、每日指標、分級。網站（build_site.py）與長期回測（research/）都用這一份。

除權息與減資：官方行情的 chg 是對當天「參考價」的漲跌，參考價 = close − chg。
參考價和前一天收盤不同的那天就是除權息或減資，比例 = 參考價 ÷ 前收。把之前的價格乘上這些比例（向後調整），
漲跌幅、振幅、波動就不會被除息缺口或減資恢復交易的假漲跌干擾。沒有 chg 的舊資料不調整。
"""
import glob, os, numpy as np, pandas as pd

AM_MIN = 2.0

def load_prices(years_back=None):
    """data/history/*.csv.gz ＋ data/prices.csv。years_back=1 → 只讀今年和去年（網站用，夠算60日指標）。"""
    files = sorted(glob.glob("data/history/*.csv.gz"))
    if years_back is not None and os.path.exists("data/prices.csv"):
        last_year = int(pd.read_csv("data/prices.csv", usecols=["date"]).date.max()[:4])
        files = [f for f in files if int(os.path.basename(f)[:4]) >= last_year - years_back]
    parts = [pd.read_csv(f, dtype={"code": str}) for f in files]
    if os.path.exists("data/prices.csv"):
        parts.append(pd.read_csv("data/prices.csv", dtype={"code": str}))
    p = pd.concat(parts, ignore_index=True)
    if "chg" not in p.columns: p["chg"] = np.nan
    p["date"] = pd.to_datetime(p.date)
    p = p[~p.code.str.startswith("0")].drop_duplicates(["date", "code"], keep="last")
    tw = set(p.loc[p.market == "TWSE", "date"])
    p = p[(p.market == "TWSE") | p.date.isin(tw)]  # 證交所休市的日子，櫃買的資料不採用
    return p

def panels(p):
    pv = lambda c: p.pivot(index="date", columns="code", values=c).sort_index()
    C, O, H, L, A, MK, CH = pv("close"), pv("open"), pv("high"), pv("low"), pv("amount"), pv("market"), pv("chg")
    prev = C.ffill().shift()                       # 最後可得的前一個收盤（跨停牌）
    ref = C - CH                                   # 官方參考價
    fac = (ref / prev).where(CH.notna() & prev.notna() & (ref > 0) & ((ref - prev).abs() > 0.005), 1.0)
    fac = fac.where((fac >= 0.05) & (fac <= 20), 1.0)  # 異常值不採用
    M = fac.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)  # 該日之後所有調整的乘積
    adj = lambda X: X * M
    return {"C": adj(C), "O": adj(O), "H": adj(H), "L": adj(L), "A": A, "MK": MK, "rawC": C, "fac": fac}

def features(P):
    C, O, H, L, A = P["C"], P["O"], P["H"], P["L"], P["A"]
    days = C.index
    r1 = C.pct_change(fill_method=None)            # 停牌中斷不補價
    R = lambda df, n, f: getattr(df.rolling(n, min_periods=n), f)()
    tr = pd.concat([(H - L), (H - C.shift()).abs(), (C.shift() - L).abs()]).groupby(level=0).max()
    f = {}
    f["v"] = R(r1, 20, "std") * 100
    f["rg"] = (R(H, 10, "max") - R(L, 10, "min")) / C * 100
    f["atr"] = R(tr, 14, "mean") / C * 100
    f["lu"] = R((r1 >= 0.09).astype(float).where(r1.notna()), 20, "sum")
    f["mx"] = (C / C.shift(5) - 1).rolling(40, min_periods=40).max() * 100
    f["am"] = R(A, 5, "mean") / R(A, 20, "mean"); f["amt"] = R(A, 20, "mean") / 1e6
    f["ch"] = r1 * 100; f["r5"] = (C / C.shift(5) - 1) * 100; f["r20"] = (C / C.shift(20) - 1) * 100
    f["dh"] = (C / R(C, 60, "max") - 1) * 100; f["c"] = P["rawC"]
    f["n60"] = C.rolling(60, min_periods=60).count(); f["ma20"] = R(C, 20, "mean")
    f["bk"] = pd.concat([(C.shift(5 * k) / C.shift(5 * k + 5) - 1) * 100 for k in range(4)], keys=range(4)).groupby(level=1).max().reindex(days)
    idx = np.arange(len(days))[:, None]
    pos = pd.DataFrame(np.where((r1 >= 0.09).values, idx, np.nan), index=days, columns=C.columns).ffill()
    f["sn"] = pd.DataFrame(idx - pos.values, index=days, columns=C.columns).fillna(99)  # 0 = 今天漲停
    return f

def tiers(f, MK, since=None):
    """每日橫斷面排名與分級。since：只回傳這天以後（指標仍用完整歷史算）。"""
    keep = ["v", "rg", "atr", "lu", "mx", "am", "amt", "ch", "r5", "r20", "dh", "c", "n60", "bk", "sn"]
    sub = {k: (f[k] if since is None else f[k].loc[since:]) for k in keep}
    S = pd.DataFrame({k: v.stack(future_stack=True) for k, v in sub.items()})
    S.index.names = ["date", "code"]
    S = S.dropna(subset=["v", "rg", "atr", "lu", "mx", "am", "bk", "c"])
    S = S[(S.n60 >= 55) & (S.amt >= 10)].reset_index()  # 近60日至少55天有行情、20日均額≥1,000萬
    S["mk"] = MK.stack().reindex(pd.MultiIndex.from_frame(S[["date", "code"]])).map({"TPEX": "櫃", "TWSE": "市"}).values
    g = S.groupby(["date", "mk"])
    S["rv"] = g.v.rank(pct=True)
    S["rc"] = pd.concat([g[c].rank(pct=True) for c in ["v", "rg", "atr", "lu", "mx"]], axis=1).mean(axis=1)
    S["rc"] = S.groupby(["date", "mk"]).rc.rank(pct=True)
    # 仍在動 = 活躍度前10% 且 今天或前10個交易日內有漲停；精選再加 當天上漲、量能比≥2；觀察 = 量能比 1～2
    S["hot"] = (S.rc >= 0.9) & (S.sn <= 10)
    S["pk"] = S.hot & (S.ch > 0) & (S.am >= AM_MIN)
    S["wt"] = S.hot & (S.ch > 0) & (S.am >= 1) & ~S.pk
    S["tier"] = np.select([S.pk, S.wt, S.hot], ["精選", "觀察", "仍在動"], "")
    return S
