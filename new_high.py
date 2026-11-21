"""新高突破（research/offense.py 的 H5，驗證中）：
收盤首次高於「資料內此前所有收盤」（至少 750 個交易日歷史，資料從 2015/6 起）、前 60 個交易日沒有創過這種新高、當天漲幅 < 9%（不是漲停）。
研究結果（平常時期、2018–2026）：隔天開盤買、抱 60 日扣 0.6% 後，比同一天買全部分析範圍股票平均多 3.9%（t 5.6，每年都正）；
但相對 0050 不顯著（2024–26 輸約 3%）。網站列出最近 10 個交易日的突破，只當觀察名單。
需要讀全部歷史（data/old/history ＋ data/history ＋ data/prices.csv），還原方式與 radar_core.panels 相同（除權息／減資比例，含 data/old/adjust.csv）。
"""
import glob, os
import numpy as np, pandas as pd

def full_close():
    files = sorted(glob.glob("data/old/history/*.csv.gz")) + sorted(glob.glob("data/history/*.csv.gz"))
    cols = ["date", "market", "code", "close"]
    parts = [pd.read_csv(f, usecols=cols, dtype={"code": str}) for f in files]
    if os.path.exists("data/prices.csv"): parts.append(pd.read_csv("data/prices.csv", usecols=cols, dtype={"code": str}))
    p = pd.concat(parts, ignore_index=True); p["date"] = pd.to_datetime(p.date)
    p = p[~p.code.str.startswith("0")].drop_duplicates(["date", "code"], keep="last")
    tw = set(p.loc[p.market == "TWSE", "date"]); p = p[(p.market == "TWSE") | p.date.isin(tw)]
    C = p.pivot(index="date", columns="code", values="close").sort_index(); del p
    adj = [pd.read_csv(f, dtype={"code": str}) for f in ("data/old/adjust.csv", "data/adjust.csv") if os.path.exists(f)]
    fac = pd.DataFrame(1.0, index=C.index, columns=C.columns)
    if adj:
        a = pd.concat(adj).drop_duplicates(); a["date"] = pd.to_datetime(a.date)
        a = a[a.code.isin(C.columns) & a.date.isin(C.index) & (a.ratio > 0.05) & (a.ratio < 20)]
        fac = a.groupby(["date", "code"]).ratio.prod().unstack().reindex(index=C.index, columns=C.columns).fillna(1.0)
    M = fac.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)
    return C * M, C

def events(lookback=10):
    """最近 lookback 個交易日的新高突破：DataFrame(date, code, close（原始價）, ch)。"""
    C, raw = full_close(); Cv = C.values
    cnt = np.cumsum(~np.isnan(Cv), axis=0); prior = np.vstack([np.zeros((1, Cv.shape[1])), cnt[:-1]])
    pm = pd.DataFrame(Cv).shift(1).expanding().max().values
    with np.errstate(invalid="ignore"):
        isath = (Cv > pm) & (prior >= 750)
        ch = (C / C.shift(1) - 1).values * 100
    recent = pd.DataFrame(isath.astype(float)).shift(1).rolling(60, min_periods=1).max().fillna(0).values > 0
    ev = isath & ~recent & (ch < 9)
    t0 = len(C) - lookback; t_, j_ = np.nonzero(ev[t0:]); t_ = t_ + t0
    return pd.DataFrame({"date": C.index[t_], "code": C.columns[j_], "close": raw.values[t_, j_], "ch": ch[t_, j_]})
