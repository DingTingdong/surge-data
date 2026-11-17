"""雷達的共用計算：讀行情、除權息調整、每日指標、分級。網站（build_site.py）與長期回測（research/）都用這一份。

除權息與減資：用證交所、櫃買中心公布的除權息計算結果與減資恢復買賣參考價（data/adjust.csv，由 fetch_adjust.py 抓），
比例 = 參考價 ÷ 前一交易日收盤。把該日之前的價格乘上這些比例（向後調整），漲跌幅、振幅、波動、報酬
就不會被除息缺口或減資恢復交易的假漲跌干擾。沒有 adjust.csv 時不調整。
"""
import glob, os, numpy as np, pandas as pd

AM_MIN = 2.0          # 舊版精選（v1）用
RULE_VERSION = "v2.0"

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
    C, O, H, L, A, MK = pv("close"), pv("open"), pv("high"), pv("low"), pv("amount"), pv("market")
    # 官方除權息／減資比例（data/adjust.csv）；同一天多筆（例如除權又除息）相乘
    fac = pd.DataFrame(1.0, index=C.index, columns=C.columns)
    if os.path.exists("data/adjust.csv"):
        a = pd.read_csv("data/adjust.csv", dtype={"code": str})
        a["date"] = pd.to_datetime(a.date)
        a = a[a.code.isin(C.columns) & a.date.isin(C.index) & (a.ratio > 0.05) & (a.ratio < 20)]
        r = a.groupby(["date", "code"]).ratio.prod().unstack().reindex(index=C.index, columns=C.columns)
        fac = r.fillna(1.0)
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
    f["snp"] = pd.DataFrame(idx - pos.shift(1).values, index=days, columns=C.columns).fillna(99)  # 不含今天，上一次漲停距今幾天
    f["a1"] = A / 1e6                                        # 當天成交額（百萬）
    f["a20p"] = A.shift(1).rolling(20, min_periods=20).mean() / 1e6  # 前20日平均成交額，不含當天
    f["vr"] = A / (f["a20p"] * 1e6)                          # 當天量比
    f["ru"] = (C / R(C, 20, "min") - 1) * 100                # 距近20日最低收盤的漲幅
    return f

def tiers(f, MK, since=None):
    """每日橫斷面排名與分級（規則 v2.0）。since：只回傳這天以後（指標仍用完整歷史算）。

    今日發動 = 當天漲 ≥ 9%，前10個交易日都沒有漲 ≥ 9%（第一根漲停），當天成交額 ≥ 2,000 萬、
               前20日平均成交額 ≥ 300 萬。不需要先在活躍度前10%。（research/early.py 的 E4）
    仍在動   = 活躍度前10%（上市、上櫃分開排名，範圍：20日均額 ≥ 1,000 萬）且 今天或前10個交易日內有漲停。
    pk_v1    = 舊版精選（仍在動＋當天上漲＋量能比≥2），只給舊研究重現用。
    """
    keep = ["v", "rg", "atr", "lu", "mx", "am", "amt", "ch", "r5", "r20", "dh", "c", "n60", "bk", "sn", "snp", "a1", "a20p", "vr", "ru"]
    sub = {k: (f[k] if since is None else f[k].loc[since:]) for k in keep}
    S = pd.DataFrame({k: v.stack(future_stack=True) for k, v in sub.items()})
    S.index.names = ["date", "code"]
    S = S.dropna(subset=["c", "n60"])
    S = S[S.n60 >= 55]
    base = S[["v", "rg", "atr", "lu", "mx", "am", "bk"]].notna().all(axis=1) & (S.amt >= 10)
    early = (S.ch >= 9) & (S.snp > 10) & (S.a1 >= 20) & (S.a20p >= 3)
    S = S[base | early].assign(base=base[base | early]).reset_index()
    S["mk"] = MK.stack().reindex(pd.MultiIndex.from_frame(S[["date", "code"]])).map({"TPEX": "櫃", "TWSE": "市"}).values
    B = S[S.base]
    g = B.groupby(["date", "mk"])
    rc = pd.concat([g[c].rank(pct=True) for c in ["v", "rg", "atr", "lu", "mx"]], axis=1).mean(axis=1)
    S["rv"] = g.v.rank(pct=True).reindex(S.index)
    S["rc"] = rc.groupby([B.date, B.mk]).rank(pct=True).reindex(S.index)
    S["hot"] = S.base & (S.rc >= 0.9) & (S.sn <= 10)
    S["pk"] = (S.ch >= 9) & (S.snp > 10) & (S.a1 >= 20) & (S.a20p >= 3)
    S["pk_v1"] = S.hot & (S.ch > 0) & (S.am >= AM_MIN)
    S["tier"] = np.select([S.pk, S.hot], ["今日發動", "仍在動"], "")
    return S
