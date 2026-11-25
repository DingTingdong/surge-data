"""課程（風潮）規則檢驗（第 50 輪；2026/11/25 11:00 事先登錄，看結果前寫下，看到結果後不修改）

來源：使用者提供的課程投影片與 6 支影片逐字稿（可轉債 cb1～cb4、選股 1～2）。課程的可量化規則：
  選股：大盤用 20 日、120 日均線看強弱；個股看週線布林（20 週、2 倍標準差）—「中軸轉向（往上）」＋「上碰上（週收盤碰到上軌）」＝轉強，
       「低檔起漲千萬要買」；中軸向下時的反彈是「死貓跳」。賣出：布林高檔爆大量、亞當滿足區（主觀，不測）。
  可轉債：發可轉債目的多是炒股；「定價以後容易出行情」；「轉換日小於 60 日的通常有行情」；可轉債決議後股價打底、「布林上碰上」要買。
  無法客觀量化而不測：綠線（大量 K 的價位）、跌不破的點、亞當理論、黃金切割、CBAS 第 6 天爆量（缺可轉債掛牌日與成交量）。
資料：2006–2009（vs0050_oldestonly_extra.pkl）＋2011–2026（vs0050_extra.pkl），還原價；分析範圍內股票；成本 0.6%；2010 不算。
  可轉債：觀測站重大訊息（data/mopsnews）。決議＝第 36 輪的 CB 事件（主旨含轉換公司債＋董事會決議／決議發行／募集，排除私募、子公司、初次上市櫃）；
  發行時點＝主旨含「轉換公司債」且含「代收」的公告（代收價款及存儲專戶，法規要求，約在詢圈／定價前後；每檔債取第一則）。2010 年前涵蓋率低（6%～40%），所以 H4 只看 2011–2026。
假說與訊號：
  H1 週線布林轉強（全部股票）：訊號週（每週最後一個交易日收盤後）同時符合
     ① 中軸（20 週均線）本週 > 上週（往上），且 4 週前的中軸 < 5 週前（四週內才由下轉上＝轉向）；
     ② 週收盤 ≥ 上軌（中軸＋2 倍 20 週標準差）＝上碰上；
     ③ 低檔：週收盤 ≤ 52 週最高週收盤 × 0.85（還沒回到前高＝低檔起漲，不是追新高）；
     ④ 同一檔前 26 週沒出現過這個訊號。
     下一個交易日開盤買（開盤一價漲停買不到就放棄）。
  H2 H1＋大盤強：訊號日等權指數 20 日均線 > 120 日均線。
  H3 H1＋可轉債：訊號日前 180 個日曆天內有該公司 CB 決議事件。另和「同期沒有 CB 的 H1」比。
  H4 可轉債發行後行情：發行時點（代收公告日）隔天開盤買，抱 20／60 個交易日（60 日約到轉換期開始，對應「轉換日小於 60 日」）。另報公告前 20 日（打低？）。
出場：主要 H20（抱 20 個交易日收盤賣）；另報 H60、S（週收盤跌破中軸→下個交易日開盤賣，最多 120 天）。
比較：0050（含息）；同日進場、同樣持有天數的全部股票平均（扣同樣成本）。
判斷（多重檢定，門檻用 t ≥ 2.9；H3 對照差與 H4 用 t ≥ 2）：
  H1、H2：H20 每筆減 0050 平均五段都 > 0、以進場月份分群 NW t ≥ 2.9、贏 0050 比例 ≥ 50%，且減全部股票平均 > 0 → 網站加「週線轉強」觀察名單（不當買進訊號）。
  H3：H3 減 0050 符合 H1 的條件，且比同期沒有 CB 的 H1 好（差的 t ≥ 2）→ 可轉債加回網站當「可轉債＋週線轉強」。
  H4：60 日減全部股票平均 > 0 且 t ≥ 2，2011–2015、2016–2019、2020–2026 三段都 > 0 →「可轉債發行後」提醒。
實作修正（11/25 11:30，跑出結果前的程式錯誤）：讀公告時索引重複、陣列唯讀，修正後才產生結果，規則沒變。
結果（11/25 11:40）：H1、H2、H3 不通過（明顯是負的），H4 通過（只對「全部股票平均」）。
  H1 週線布林低檔上碰上：2,380 筆，H20 減 0050 −2.55%（中位 −4.39，贏 35%，t −4.84），五段 −0.22／−3.40／−2.95／−2.02／−4.77；
     減全部股票 −1.45（t −2.99）；H60 −5.26；週收盤跌破中軸才賣 −4.09（平均抱 60 天，贏 23%）。
  H2 加大盤 20 日 > 120 日：−2.69（t −4.39），沒變好。
  H3 加可轉債決議 180 天內：201 筆 −3.69（t −2.61），比沒可轉債的 H1（−2.45）更差，差 −1.2。
  H4 可轉債發行時點（代收公告，2011–2026，828 檔次）：60 日減全部股票 +2.46（t 2.16），三段 +0.69／+3.05／+3.06 → 通過；
     但減 0050 −1.20、中位數減全部 −2.25（少數大漲拉高平均）；20 日減全部 +2.90、減 0050 +0.95。公告前 20 日減 0050 +1.90（不是「打低」）。
  → 網站：個股卡片加「剛發可轉債」提醒（發行日 60 個交易日內，依櫃買中心流通可轉債清單），說明只是比一般股票好、沒有贏 0050；不加週線轉強名單。

"""
import json, os, sys, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
from vs0050 import nw_t
import seo_cb_filter as SC
SEGS = ["2006-09", "2011-15", "2016-19", "2020-23", "2024-26"]
def seg(d): return np.where(d < pd.Timestamp("2010-01-01"), "2006-09", np.where(d < pd.Timestamp("2011-01-01"), "2010", np.where(d < pd.Timestamp("2016-01-01"), "2011-15", np.where(d < pd.Timestamp("2020-01-01"), "2016-19", np.where(d < pd.Timestamp("2024-01-01"), "2020-23", "2024-26")))))
COST = 0.6

def cb_dates():
    _, CB = SC.events(); CB = CB.copy()
    N = pd.concat([pd.read_csv(f, dtype={"code": str, "date": str, "time": str}) for f in sorted(glob.glob(f"{REPO}/data/mopsnews/*.csv.gz"))]).drop_duplicates(["date", "time", "code", "subject"]).reset_index(drop=True)
    s = N.subject.fillna("").str.replace(r"\s+", "", regex=True)
    k = N[s.str.contains("轉換公司債") & s.str.contains("代收") & ~s.str.contains("私募|海外|外國|子公司")].copy()
    k["n"] = s[k.index].str.extract(r"第([一二三四五六七八九十\d]+)次")[0].fillna("?")
    k["d"] = pd.to_datetime(k.date, format="%Y%m%d", errors="coerce")
    iss = k.dropna(subset=["d"]).sort_values("d").drop_duplicates(["code", "n"])[["d", "code"]]
    return CB, iss

def run(path, start, end, CB, ISS):
    X = pd.read_pickle(path); days = pd.DatetimeIndex(X["days"]); N = len(days); cols = list(X["cols"]); cj = {c: j for j, c in enumerate(cols)}
    C = pd.DataFrame(X["C"], index=days); Cf = C.ffill(limit=5); O = X["O"]; inb = np.asarray(X["inb"]).astype(bool); lk = X["lockn"]; b = np.asarray(X["bidx"], float)
    Cff = Cf.values
    # 週線（每週最後一個交易日）
    wk = pd.Series(days.to_period("W-FRI"), index=days); last = ~wk.duplicated(keep="last").values; wi = np.where(last)[0]
    W = Cf.iloc[wi]; mid = W.rolling(20, min_periods=20).mean(); sd = W.rolling(20, min_periods=20).std(); up = mid + 2 * sd
    hi52 = W.rolling(52, min_periods=40).max()
    turn = (mid > mid.shift(1)) & (mid.shift(4) < mid.shift(5))
    sigW = turn & (W >= up) & (W <= 0.85 * hi52)
    sigW = sigW.fillna(False).values.astype(bool).copy()
    prior = pd.DataFrame(sigW.astype(float)).shift(1).rolling(26, min_periods=1).max().fillna(0).values > 0
    sigW &= ~prior
    below_mid = (W < mid).values  # 週收盤跌破中軸
    # 大盤：等權指數 20 日 > 120 日均線
    r = Cf.pct_change(fill_method=None).where(inb); ew = (1 + r.mean(axis=1).fillna(0)).cumprod(); strong = (ew.rolling(20).mean() > ew.rolling(120).mean()).values
    # 全部股票同日進場平均（減 0050）
    okO = O > 0; RU = {}
    for H in (20, 60):
        R = np.full(N, np.nan)
        for ie in range(1, N - H + 1):
            rr = Cff[ie + H - 1] / np.where(okO[ie], O[ie], np.nan) - 1; m = inb[ie - 1] & np.isfinite(rr)
            if m.sum() > 50: R[ie] = np.nanmean(rr[m]) * 100 - COST
        RU[H] = R
    bret = lambda ie, ix: (b[ix] / b[ie - 1] - 1) * 100
    t0 = int(np.searchsorted(days.values, np.datetime64(start))); t1 = int(np.searchsorted(days.values, np.datetime64(end)))
    cbmap = {}
    for d, c in zip(CB.d, CB.code): cbmap.setdefault(c, []).append(np.datetime64(d))
    out = []
    for k, t in enumerate(wi):
        if not (t0 <= t < t1): continue
        for j in np.where(sigW[k] & inb[t])[0]:
            ie = t + 1
            if ie + 59 >= N or not (O[ie, j] > 0) or lk[ie - 1, j]: continue
            o = O[ie, j]; rec = {"date": days[t], "code": cols[j], "strong": bool(strong[t])}
            for H in (20, 60):
                ix = ie + H - 1; rH = (Cff[ix, j] / o - 1) * 100 - COST
                rec[f"x{H}"] = rH - bret(ie, ix); rec[f"u{H}"] = rH - RU[H][ie]
            ixS = min(ie + 119, N - 1)
            for kk in range(k + 1, len(wi)):
                if wi[kk] >= ie + 119: break
                if wi[kk] >= ie and below_mid[kk, j]: ixS = min(wi[kk] + 1, N - 1); break
            exS = O[ixS, j] if O[ixS, j] > 0 else Cff[ixS, j]; rec["xS"] = (exS / o - 1) * 100 - COST - bret(ie, ixS); rec["holdS"] = ixS - ie
            ds = cbmap.get(cols[j], [])
            rec["cb"] = any((np.datetime64(days[t]) - d) / np.timedelta64(1, "D") <= 180 and d <= np.datetime64(days[t]) for d in ds)
            out.append(rec)
    E = pd.DataFrame(out)
    # H4：可轉債發行時點事件
    ev = []
    for d, c in zip(ISS.d, ISS.code):
        if c not in cj: continue
        i = int(np.searchsorted(days.values, np.datetime64(d)))
        if i < 21 or i + 61 >= N or not (t0 <= i < t1): continue
        j = cj[c]; ie = i + 1
        if not (O[ie, j] > 0) or lk[ie - 1, j] or not inb[i, j]: continue
        o = O[ie, j]; rec = {"date": days[i], "code": c, "pre20": (Cff[i, j] / Cff[i - 20, j] - 1) * 100 - (b[i] / b[i - 20] - 1) * 100}
        for H in (20, 60):
            ix = ie + H - 1; rH = (Cff[ix, j] / o - 1) * 100 - COST; rec[f"x{H}"] = rH - bret(ie, ix); rec[f"u{H}"] = rH - RU[H][ie]
        ev.append(rec)
    return E, pd.DataFrame(ev)

def summ(E, col, by_seg=True):
    E = E[E.seg != "2010"]
    if not len(E): return {"筆數": 0}
    m = E.groupby(E.date.dt.to_period("M"))[col].mean()
    res = {"筆數": int(len(E)), "平均": round(float(E[col].mean()), 2), "中位": round(float(E[col].median()), 2), "贏%": round(float((E[col] > 0).mean() * 100)),
           "t": round(nw_t(m.reset_index(drop=True), 3), 2) if len(m) > 10 else None}
    if by_seg: res["五段"] = {s: (int((E.seg == s).sum()), round(float(E[E.seg == s][col].mean()), 2) if (E.seg == s).any() else None) for s in SEGS}
    return res

def pass_main(Ex, Eu):
    s = Ex["五段"]
    return bool(all(v[1] is not None and v[1] > 0 for v in s.values()) and (Ex["t"] or 0) >= 2.9 and Ex["贏%"] >= 50 and Eu["平均"] > 0)

if __name__ == "__main__":
    CB, ISS = cb_dates(); print("CB 決議事件", len(CB), "發行時點（代收公告）", len(ISS), flush=True)
    Eo, Vo = run("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01", CB, ISS)
    En, Vn = run("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01", CB, ISS)
    E = pd.concat([Eo, En], ignore_index=True); E["seg"] = seg(E.date); V = pd.concat([Vo, Vn], ignore_index=True); V["seg"] = seg(V.date)
    res = {}
    for lab, Z in (("H1 週線轉強", E), ("H2 ＋大盤強", E[E.strong]), ("H3 ＋可轉債決議", E[E.cb]), ("H1 沒有可轉債", E[~E.cb])):
        res[lab] = {"H20減0050": summ(Z, "x20"), "H20減全部": summ(Z, "u20", False), "H60減0050": summ(Z, "x60", False), "H60減全部": summ(Z, "u60", False), "S減0050": summ(Z, "xS", False),
                    "S平均持有天數": round(float(Z.holdS.mean()), 1) if len(Z) else None, "每年訊號": Z.date.dt.year.value_counts().sort_index().to_dict()}
        print(lab, json.dumps(res[lab], ensure_ascii=False, default=str), flush=True)
    a, c = E[E.cb & (E.seg != "2010")], E[~E.cb & (E.seg != "2010")]
    diff = a.x20.mean() - c.x20.mean(); se = np.sqrt(a.x20.var() / max(len(a), 1) + c.x20.var() / max(len(c), 1)); res["H3 減 沒CB（H20）"] = {"差": round(float(diff), 2), "t": round(float(diff / se), 2) if se > 0 else None}
    V11 = V[V.date >= "2011-01-01"].copy(); V11["seg3"] = np.where(V11.date < "2016-01-01", "2011-15", np.where(V11.date < "2020-01-01", "2016-19", "2020-26"))
    m60 = V11.groupby(V11.date.dt.to_period("M")).u60.mean()
    res["H4 發行後"] = {"筆數": int(len(V11)), "公告前20日減0050": round(float(V11.pre20.mean()), 2), "20日減0050": round(float(V11.x20.mean()), 2), "20日減全部": round(float(V11.u20.mean()), 2),
                       "60日減0050": round(float(V11.x60.mean()), 2), "60日減全部": round(float(V11.u60.mean()), 2), "60日減全部 中位": round(float(V11.u60.median()), 2),
                       "60日減全部 t": round(nw_t(m60.reset_index(drop=True), 3), 2), "三段 60日減全部": {s: round(float(g.u60.mean()), 2) for s, g in V11.groupby("seg3")},
                       "2006–09（涵蓋率低，只供參考）60日減全部": round(float(V[V.seg == "2006-09"].u60.mean()), 2) if (V.seg == "2006-09").any() else None}
    print("H4", json.dumps(res["H4 發行後"], ensure_ascii=False), flush=True)
    j = {"H1": pass_main(res["H1 週線轉強"]["H20減0050"], res["H1 週線轉強"]["H20減全部"]), "H2": pass_main(res["H2 ＋大盤強"]["H20減0050"], res["H2 ＋大盤強"]["H20減全部"]),
         "H3": pass_main(res["H3 ＋可轉債決議"]["H20減0050"], res["H3 ＋可轉債決議"]["H20減全部"]) and (res["H3 減 沒CB（H20）"]["t"] or 0) >= 2,
         "H4": res["H4 發行後"]["60日減全部"] > 0 and res["H4 發行後"]["60日減全部 t"] >= 2 and all(v > 0 for v in res["H4 發行後"]["三段 60日減全部"].values())}
    res["判斷"] = j; print("判斷", j)
    json.dump(res, open(f"{REPO}/research/results/course_rules.json", "w"), ensure_ascii=False, indent=1, default=str)
