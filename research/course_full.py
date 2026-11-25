"""課程選股完整版：線型＋年營收＋ROE（＋大盤）（第 54 輪；2026/11/25 16:45 事先登錄，看結果前寫下，看到結果後不修改）

起因：使用者指出前兩輪比較像進出場；課程的選股是「大盤 20／120 日線 → 週線布林中軸轉向＋跌不破／上碰上 → 題材 → 基本面排除」。
  題材無法客觀回測；這輪把可量化的基本面加上去，算是完整檢驗課程選股。
線型（用第 50 輪敏感度裡「抓得到老師例子」的寫法，固定不再調）：每週最後交易日收盤後，
  中軸（20 週均線）本週 > 上週，且 8 週內曾經往下（剛轉向）；本週最高價 ≥ 上軌（中軸＋2 倍 20 週標準差）；同一檔前 26 週沒出現過。
  T2＝不要求低檔（國巨 2020-11-20、宏達電 2020-12-04 都會被抓到）；T1＝另外要求週收盤 ≤ 一年最高週收盤 × 0.85（低檔起漲）。
基本面（訊號日當時已公布的資料）：
  年營收：最近 12 個月營收合計比前 12 個月成長（> 0），且 3 個月前算的也 > 0（穩定成長；課程「看年營收、以 YOY 為主」）。
  ROE：最近 4 季歸屬母公司淨利合計 ÷ 最近一季歸屬母公司權益 ≥ 8%（課程「ROE 太低的不要」）；季報以法定期限（5/15、8/14、11/14、3/31）當可用日。
    季報從 2013Q1 起，所以 ROE 從 2014 年 4 月起才有；有 ROE 的檢驗期間＝2014-04～2026。
  大盤：等權指數 20 日均線 > 120 日均線。
組合：主要＝T2＋年營收＋ROE；另報 T2、T1、T2＋年營收、T2＋ROE、T2＋年營收＋ROE＋大盤、T1＋年營收＋ROE。
進場：下一個交易日開盤（開盤一價漲停買不到就放棄）；出場：H20 主要，另報 H60、H120；扣 0.6%。比較 0050（含息）與同日全部股票平均。
判斷（主要組合）：2014-04 起，H20 減 0050 在 2014–15、2016–19、2020–23、2024–26 四段都 > 0，以進場月份分群 NW t ≥ 2.9，贏 0050 ≥ 50%，
  減全部股票 > 0 → 網站加「課程選股（線型＋基本面）」觀察名單；否則照實報告。
實作修正（跑出結果前）：季報欄位名稱 eq 和 pandas 方法衝突，改名 equity，規則不變。
結果（11/25 17:10）：不通過。2014-04～2026，20 日減 0050／減全部股票：
  T2 線型 8,104 筆 −2.04（t −5.75）／−0.05；T1 加低檔 −3.58／−1.29；T2＋年營收 −1.53／+0.28；T2＋ROE −1.66／+0.04；
  主要 T2＋年營收＋ROE 2,816 筆 −1.39（t −3.66，贏 39%，四段 −0.62／−1.58／−0.45／−3.00）／+0.18；60 天 −3.37、120 天 −6.38；
  ＋大盤強 −1.98／+0.03；T1＋年營收＋ROE 737 筆 −3.01／−1.28。→ 基本面讓它從「比一般股票略差」變「和一般股票差不多」，但仍明顯輸 0050；低檔、大盤濾網都讓它更差。

"""
import json, os, sys, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
from vs0050 import nw_t
COST = 0.6
SEGS = ["2014-15", "2016-19", "2020-23", "2024-26"]
def seg(d): return "2014-15" if d.year < 2016 else "2016-19" if d.year < 2020 else "2020-23" if d.year < 2024 else "2024-26"

def roe_table():
    rows = []
    for fn in sorted(glob.glob(f"{REPO}/data/fin/*.csv.gz")):
        b = os.path.basename(fn)
        if not b[:4].isdigit(): continue
        y, s = int(b[:4]), int(b[5]); d = pd.read_csv(fn, dtype={"code": str}); d["code"] = d.code.str[:4]
        p = d.pivot_table(index="code", columns="field", values="value", aggfunc="first")
        pick = lambda names: p[[n for n in names if n in p.columns]].bfill(axis=1).iloc[:, 0] if any(n in p.columns for n in names) else pd.Series(np.nan, index=p.index)
        ni = pick(["淨利（淨損）歸屬於母公司業主", "本期淨利（淨損）", "本期稅後淨利（淨損）"]); eq = pick(["歸屬於母公司業主之權益合計", "權益總計", "權益總額"])
        rows.append(pd.DataFrame({"code": p.index, "y": y, "s": s, "ni": ni.values, "equity": eq.values}))
    Q = pd.concat(rows, ignore_index=True).sort_values(["code", "y", "s"])
    prev = Q.groupby(["code", "y"]).ni.shift(1); Q["ni_q"] = np.where(Q.s == 1, Q.ni, Q.ni - prev)   # 累計 → 單季
    Q["ttm"] = Q.groupby("code").ni_q.transform(lambda x: x.rolling(4, min_periods=4).sum())
    Q["roe"] = Q.ttm / Q.equity * 100; Q.loc[Q.equity <= 0, "roe"] = np.nan
    dl = {1: (5, 15), 2: (8, 14), 3: (11, 14), 4: (3, 31)}; Q["avail"] = [pd.Timestamp(y + (s == 4), *dl[s]) for y, s in zip(Q.y, Q.s)]
    return Q.dropna(subset=["roe"])[["code", "avail", "roe"]]

def rev_table():
    fs = sorted(glob.glob(f"{REPO}/data/revenue/*.csv*") + glob.glob(f"{REPO}/data/old/revenue/*.csv*") + glob.glob(f"{REPO}/data/older/revenue/*.csv*") + glob.glob(f"{REPO}/data/oldest/revenue/*.csv*"))
    R = pd.concat([pd.read_csv(f, dtype={"code": str}, usecols=lambda c: c in ("ym", "code", "rev")) for f in fs], ignore_index=True).drop_duplicates(["ym", "code"], keep="last").sort_values(["code", "ym"])
    g = R.groupby("code").rev; R["t12"] = g.transform(lambda s: s.rolling(12, min_periods=12).sum()); R["p12"] = R.groupby("code").t12.shift(12)
    R["ttmyoy"] = (R.t12 / R.p12 - 1) * 100; R["ttmyoy3"] = R.groupby("code").ttmyoy.shift(3)
    R["pub"] = pd.to_datetime(R.ym + "-01") + pd.DateOffset(months=1) + pd.Timedelta(days=10)
    R["revok"] = (R.ttmyoy > 0) & (R.ttmyoy3 > 0); return R.dropna(subset=["ttmyoy"])[["code", "pub", "revok", "ttmyoy"]]

def signals():
    X = pd.read_pickle("/tmp/model_v2/vs0050_extra.pkl"); days = pd.DatetimeIndex(X["days"]); N = len(days); cols = list(X["cols"])
    Cf = pd.DataFrame(X["C"], index=days).ffill(limit=5); Cff = Cf.values; O = np.asarray(X["O"], float); inb = np.asarray(X["inb"]).astype(bool); lk = X["lockn"]; b = np.asarray(X["bidx"], float)
    wk = pd.Series(days.to_period("W-FRI"), index=days); wi = np.where(~wk.duplicated(keep="last").values)[0]; W = Cf.iloc[wi]
    Hh = pd.DataFrame(np.asarray(X["H"], float), index=days); WH = Hh.groupby(wk.values).max().reindex(pd.PeriodIndex(wk.values[wi])); WH.index = W.index
    mid = W.rolling(20, min_periods=20).mean(); up = mid + 2 * W.rolling(20, min_periods=20).std(); hi52 = W.rolling(52, min_periods=40).max(); md = mid.values
    down = np.zeros(md.shape, bool)
    for q in range(1, 9): down |= np.nan_to_num(np.roll(md, q, 0) < np.roll(md, q + 1, 0)).astype(bool)
    sig = np.nan_to_num((mid > mid.shift(1)).values & down & (WH >= up).values).astype(bool)
    pr = pd.DataFrame(sig.astype(float)).shift(1).rolling(26, min_periods=1).max().fillna(0).values > 0; sig &= ~pr
    low = np.nan_to_num((W <= 0.85 * hi52).values).astype(bool)
    r = Cf.pct_change(fill_method=None).where(inb); ew = (1 + r.mean(axis=1).fillna(0)).cumprod(); strong = (ew.rolling(20).mean() > ew.rolling(120).mean()).values
    okO = O > 0; RU = {}
    for H in (20, 60, 120):
        Rr = np.full(N, np.nan)
        for ie in range(1, N - H + 1):
            rr = Cff[ie + H - 1] / np.where(okO[ie], O[ie], np.nan) - 1; m = inb[ie - 1] & np.isfinite(rr)
            if m.sum() > 50: Rr[ie] = np.nanmean(rr[m]) * 100 - COST
        RU[H] = Rr
    t0 = int(np.searchsorted(days.values, np.datetime64("2011-01-01"))); out = []
    for k, t in enumerate(wi):
        if t < t0: continue
        for j in np.where(sig[k] & inb[t])[0]:
            ie = t + 1
            if not (O[ie, j] > 0) if ie < N else True: continue
            if lk[ie - 1, j] or ie + 19 >= N: continue
            rec = {"date": days[t], "code": cols[j], "low": bool(low[k, j]), "strong": bool(strong[t])}
            for H in (20, 60, 120):
                ix = ie + H - 1
                if ix >= N: rec[f"x{H}"] = rec[f"u{H}"] = np.nan; continue
                rH = (Cff[ix, j] / O[ie, j] - 1) * 100 - COST; rec[f"x{H}"] = rH - (b[ix] / b[ie - 1] - 1) * 100; rec[f"u{H}"] = rH - RU[H][ie]
            out.append(rec)
    return pd.DataFrame(out)

def summ(Z, col="x20"):
    Z = Z.dropna(subset=[col])
    if not len(Z): return {"筆數": 0}
    m = Z.groupby(Z.date.dt.to_period("M"))[col].mean()
    return {"筆數": int(len(Z)), "平均": round(float(Z[col].mean()), 2), "中位": round(float(Z[col].median()), 2), "贏%": round(float((Z[col] > 0).mean() * 100)),
            "t": round(nw_t(m.reset_index(drop=True), 3), 2) if len(m) > 10 else None, "四段": {s: (int((Z.seg == s).sum()), round(float(Z[Z.seg == s][col].mean()), 2) if (Z.seg == s).any() else None) for s in SEGS}}

if __name__ == "__main__":
    E = signals(); print("線型訊號（2011 起）", len(E), flush=True)
    RO, RV = roe_table(), rev_table()
    E = pd.merge_asof(E.sort_values("date"), RO.sort_values("avail"), left_on="date", right_on="avail", by="code", direction="backward")
    E.loc[(E.date - E.avail).dt.days > 200, "roe"] = np.nan
    E = pd.merge_asof(E.sort_values("date"), RV.sort_values("pub"), left_on="date", right_on="pub", by="code", direction="backward")
    E.loc[(E.date - E.pub).dt.days > 70, "revok"] = np.nan
    E = E[E.date >= "2014-04-01"].copy(); E["seg"] = E.date.map(seg)
    E["rv"] = E.revok.fillna(False).astype(bool); E["ro"] = E.roe >= 8
    print("2014-04 起", len(E), "有 ROE", int(E.roe.notna().sum()), "有年營收", int(E.revok.notna().sum()), flush=True)
    combos = {"T2 線型": E, "T1 線型＋低檔": E[E.low], "T2＋年營收": E[E.rv], "T2＋ROE≥8%": E[E.ro], "主要 T2＋年營收＋ROE": E[E.rv & E.ro],
              "T2＋年營收＋ROE＋大盤強": E[E.rv & E.ro & E.strong], "T1＋年營收＋ROE": E[E.low & E.rv & E.ro]}
    res = {}
    for k, Z in combos.items():
        res[k] = {"20天減0050": summ(Z, "x20"), "20天減全部": summ(Z, "u20")["平均"] if len(Z) else None, "60天減0050": summ(Z, "x60")["平均"] if len(Z) else None,
                  "60天減全部": summ(Z, "u60")["平均"] if len(Z) else None, "120天減0050": summ(Z, "x120")["平均"] if len(Z) else None}
        print(k, json.dumps(res[k], ensure_ascii=False), flush=True)
    m = res["主要 T2＋年營收＋ROE"]["20天減0050"]
    res["通過"] = bool(m["筆數"] and all(v[1] is not None and v[1] > 0 for v in m["四段"].values()) and (m["t"] or 0) >= 2.9 and m["贏%"] >= 50 and (res["主要 T2＋年營收＋ROE"]["20天減全部"] or 0) > 0)
    print("通過", res["通過"]); json.dump(res, open(f"{REPO}/research/results/course_full.json", "w"), ensure_ascii=False, indent=1, default=str)
