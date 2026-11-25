"""課程完整流程（照外部 AI 覆核提的順序）＋可轉債短期行情描述（第 56 輪；2026/11/25 17:55 事先登錄，看結果前寫下，看到結果後不修改）

起因：使用者把課程整理給兩個外部 AI 覆核，兩份都指出：第 54 輪「完整版」其實是突破直接買＋基本面，沒有「突破後等拉回、不破、再轉強才買」；
  第 51 輪的「不破」允許大幅跌破（週低 ≤ 中軸×1.02 就算）、綠線可能在回測之後才形成；0050 用訊號日收盤起算、沒扣成本；
  「全部股票」其實是雷達分析範圍（20 日均成交值 ≥ 1 千萬等）；第 50 輪中軸轉向寫成「恰好 4 週前向下」。
  參數照覆核提的（1～8 週、跌破不超過 2%、收盤站回），不另外調。這輪只跑一次。
A. 課程完整流程（每週最後交易日判斷，資料 2014-04～2026，因 ROE 從那時起才有）：
  ① 轉強週 S：中軸（20 週均線）本週 > 上週，且 8 週內曾向下；本週最高 ≥ 上軌（中軸＋2 倍 20 週標準差）；同一檔前 26 週沒出現過（與第 54 輪相同）。
  ② 等拉回：S 之後第 1～8 週，依序檢查每週；支撐＝當週中軸（主要）或綠線（另報：S 週以前 60 個交易日內最近一根大量紅 K 的收盤，
     大量紅 K＝成交值 ≥ 前 20 日平均 3 倍且收 > 開；必須在 S 週結束前已經出現）。
     某週最低 < 支撐×0.98 → 跌破，這次作廢；最低 ≤ 支撐×1.02、≥ 支撐×0.98、週收盤 ≥ 支撐、中軸仍向上 → 拉回週 P。
  ③ 再轉強：P 之後 2 週內第一個「週收盤 > P 週最高」的週 R（期間沒跌破）；R 週最後交易日的隔天開盤買。另報「不等再轉強」（P 週隔天買）。
  ④ 基本面（買進訊號當時已公布）：近 12 個月營收比前 12 個月成長、3 個月前算的也成長；ROE（近 4 季淨利÷最近一季權益）≥ 8%（與第 54 輪相同）。
  ⑤ 大盤：0050 含息指數 20 日均線 > 120 日均線，且 20 日均線比 5 天前高（覆核建議的「加權指數＋斜率」，用 0050 代替加權指數）。
  出場：固定 20、60 個交易日收盤；另報「週收盤跌破中軸，隔天開盤賣」（最多 120 天）。開盤一價漲停買不到就放棄。
  比較（修正覆核指出的問題）：0050 一律從「同一天開盤」買起（用 0050 當天開盤／收盤比例換算含息指數），扣 0050 來回 0.18%；
    個股扣 0.6%（2.8 折手續費＋證交稅 0.38%，加滑價約 0.22%）；雷達分析範圍等權也排除開盤鎖漲停、用同一天開盤。
  統計：每筆平均、每月平均（先月內平均、月份等權）都報；t 用月份平均（NW 落後 3）；勝率只算有報酬的筆數。
  主要＝①②③（中軸）＋④＋⑤，H20。判斷：每月平均減 0050 > 0、t ≥ 2.9、四段（2014–15、2016–19、2020–23、2024–26）每月平均都 > 0、
    贏 0050 ≥ 50%、減分析範圍 > 0 → 網站加「課程完整流程」觀察名單；否則照實報告，不再改參數。
B. 可轉債代收公告後（2011–2026，描述性；20 日結果第 50 輪已看過，所以這部分不算獨立驗證，網站不因此改）：
  發行時點去重改成：抓得到「第 N 次」照次數；抓不到的同一公司 90 天內只算一次（第 50 輪會把所有「?」併成一檔）。
  報 5／10／20／40／60 日減 0050（同上公平算法）與減分析範圍、中位數、前 5% 筆數占總超額的比例、60 日內最高浮盈平均。
"""
import json, os, sys, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
from vs0050 import nw_t
from course_full import roe_table, rev_table
COST, BCOST = 0.6, 0.18
SEGS = ["2014-15", "2016-19", "2020-23", "2024-26"]
def seg(d): return "2014-15" if d.year < 2016 else "2016-19" if d.year < 2020 else "2020-23" if d.year < 2024 else "2024-26"

def base():
    X = pd.read_pickle("/tmp/model_v2/vs0050_extra.pkl"); days = pd.DatetimeIndex(X["days"]); N = len(days); cols = list(X["cols"])
    Cf = pd.DataFrame(X["C"], index=days).ffill(limit=5); Cff = Cf.values; O = np.asarray(X["O"], float); inb = np.asarray(X["inb"]).astype(bool); lk = np.asarray(X["lockn"]).astype(bool)
    b = np.asarray(X["bidx"], float)
    fs = sorted(glob.glob(f"{REPO}/data/*/history/*.csv.gz") + glob.glob(f"{REPO}/data/history/*.csv.gz"))
    E5 = pd.concat([pd.read_csv(f, dtype={"code": str}, usecols=["date", "code", "open", "close"]).query("code in ['50','0050']") for f in fs]).drop_duplicates("date")
    E5["date"] = pd.to_datetime(E5.date); E5 = E5.set_index("date").reindex(days)
    ratio = (E5.open / E5.close).where((E5.open > 0) & (E5.close > 0)).fillna(1.0).values
    bO = b * ratio                                                     # 0050 含息指數的「當天開盤」值
    okO = O > 0; RU = {}
    for H in (5, 10, 20, 40, 60):
        R = np.full(N, np.nan)
        for ie in range(1, N - H + 1):
            rr = Cff[ie + H - 1] / np.where(okO[ie], O[ie], np.nan) - 1; m = inb[ie - 1] & np.isfinite(rr) & ~lk[ie - 1]
            if m.sum() > 50: R[ie] = np.nanmean(rr[m]) * 100 - COST
        RU[H] = R
    bs = pd.Series(b); ma20 = bs.rolling(20).mean(); strong = ((ma20 > bs.rolling(120).mean()) & (ma20 > ma20.shift(5))).values
    return dict(X=X, days=days, N=N, cols=cols, Cf=Cf, Cff=Cff, O=O, inb=inb, lk=lk, b=b, bO=bO, RU=RU, strong=strong)

def bx(B, ie, ix): return (B["b"][ix] / B["bO"][ie] - 1) * 100 - BCOST

def seq_signals(B):
    X, days, N, cols, Cf, Cff, O, inb, lk = (B[k] for k in ("X", "days", "N", "cols", "Cf", "Cff", "O", "inb", "lk"))
    wk = pd.Series(days.to_period("W-FRI"), index=days); wi = np.where(~wk.duplicated(keep="last").values)[0]; W = Cf.iloc[wi]
    per = pd.PeriodIndex(wk.values[wi])
    WH = pd.DataFrame(np.asarray(X["H"], float), index=days).groupby(wk.values).max().reindex(per).values
    WL = pd.DataFrame(np.where(np.asarray(X["L"], float) > 0, np.asarray(X["L"], float), np.nan), index=days).groupby(wk.values).min().reindex(per).values
    mid = W.rolling(20, min_periods=20).mean(); up = mid + 2 * W.rolling(20, min_periods=20).std(); md = mid.values; Wv = W.values
    down = np.zeros(md.shape, bool)
    for q in range(1, 9): down |= np.nan_to_num(np.roll(md, q, 0) < np.roll(md, q + 1, 0)).astype(bool)
    rising = np.nan_to_num(md > np.roll(md, 1, 0)).astype(bool)
    S = rising & down & np.nan_to_num(WH >= up.values).astype(bool)
    pr = pd.DataFrame(S.astype(float)).shift(1).rolling(26, min_periods=1).max().fillna(0).values > 0; S &= ~pr
    # 綠線：大量紅 K（每日），取到第 t 日為止 60 日內最近一根的收盤
    A = pd.DataFrame(np.asarray(X["A"], float), index=days); Cr = pd.DataFrame(np.asarray(X["C"], float), index=days)
    big = (A >= 3 * A.shift(1).rolling(20, min_periods=15).mean()) & (Cr > pd.DataFrame(O, index=days))
    glv = Cr.where(big).ffill(limit=60).values                      # 第 t 日看得到的最近大量紅 K 收盤（60 日內）
    t0 = int(np.searchsorted(days.values, np.datetime64("2014-04-01"))); out = []
    nW = len(wi)
    for k in range(nW):
        t = wi[k]
        if t < t0: continue
        for j in np.where(S[k] & inb[t])[0]:
            for sup in ("mid", "green"):
                g0 = glv[t, j] if sup == "green" else np.nan
                if sup == "green" and not np.isfinite(g0): continue
                P = None
                for q in range(k + 1, min(k + 9, nW)):
                    s = md[q, j] if sup == "mid" else g0
                    if not np.isfinite(s) or not np.isfinite(WL[q, j]): continue
                    if WL[q, j] < 0.98 * s: break
                    if WL[q, j] <= 1.02 * s and Wv[q, j] >= s and rising[q, j]: P = q; break
                if P is None: continue
                Rk = None
                for q in range(P + 1, min(P + 3, nW)):
                    s = md[q, j] if sup == "mid" else g0
                    if np.isfinite(WL[q, j]) and np.isfinite(s) and WL[q, j] < 0.98 * s: break
                    if Wv[q, j] > WH[P, j]: Rk = q; break
                for conf, kk in (("等再轉強", Rk), ("不等", P)):
                    if kk is None: continue
                    ts = wi[kk]; ie = ts + 1
                    if ie + 19 >= N or not (O[ie, j] > 0) or lk[ie - 1, j] or not inb[ts, j]: continue
                    rec = {"date": days[ts], "code": cols[j], "sup": sup, "conf": conf, "strong": bool(B["strong"][ts])}
                    for H in (20, 60):
                        ix = ie + H - 1
                        if ix >= N: rec[f"x{H}"] = rec[f"u{H}"] = np.nan; continue
                        rH = (Cff[ix, j] / O[ie, j] - 1) * 100 - COST; rec[f"x{H}"] = rH - bx(B, ie, ix); rec[f"u{H}"] = rH - B["RU"][H][ie]
                    # 週收盤跌破中軸 → 隔天開盤賣（最多 120 天）
                    ex = None
                    for q in range(kk + 1, nW):
                        if wi[q] - ie >= 119: break
                        if np.isfinite(md[q, j]) and Wv[q, j] < md[q, j]: ex = wi[q] + 1; break
                    if ex is not None and ex < N and O[ex, j] > 0: rS = (O[ex, j] / O[ie, j] - 1) * 100 - COST; bS = (B["bO"][ex] / B["bO"][ie] - 1) * 100 - BCOST
                    else:
                        ix = min(ie + 119, N - 1); rS = (Cff[ix, j] / O[ie, j] - 1) * 100 - COST; bS = bx(B, ie, ix)
                    rec["xS"] = rS - bS; out.append(rec)
    return pd.DataFrame(out)

def summ(Z, col):
    Z = Z.dropna(subset=[col])
    if not len(Z): return {"筆數": 0}
    m = Z.groupby(Z.date.dt.to_period("M"))[col].mean()
    sm = {s: round(float(Z[Z.seg == s].groupby(Z[Z.seg == s].date.dt.to_period("M"))[col].mean().mean()), 2) if (Z.seg == s).any() else None for s in SEGS}
    return {"筆數": int(len(Z)), "每筆平均": round(float(Z[col].mean()), 2), "每月平均": round(float(m.mean()), 2), "中位": round(float(Z[col].median()), 2),
            "贏%": round(float((Z[col] > 0).mean() * 100)), "t": round(nw_t(m.reset_index(drop=True), 3), 2) if len(m) > 10 else None, "四段每月平均": sm}

def cb_part(B):
    from course_rules import cb_dates
    N_ = pd.concat([pd.read_csv(f, dtype={"code": str, "date": str, "time": str}) for f in sorted(glob.glob(f"{REPO}/data/mopsnews/*.csv.gz"))]).drop_duplicates(["date", "time", "code", "subject"]).reset_index(drop=True)
    s = N_.subject.fillna("").str.replace(r"\s+", "", regex=True)
    k = N_[s.str.contains("轉換公司債") & s.str.contains("代收") & ~s.str.contains("私募|海外|外國|子公司")].copy()
    k["n"] = s[k.index].str.extract(r"第([一二三四五六七八九十\d]+)次")[0]; k["d"] = pd.to_datetime(k.date, format="%Y%m%d", errors="coerce")
    k = k.dropna(subset=["d"]).sort_values("d"); a = k[k.n.notna()].drop_duplicates(["code", "n"]); rows = []; last = {}
    for d, c in zip(k[k.n.isna()].d, k[k.n.isna()].code):
        if c in last and (d - last[c]).days < 90: continue
        last[c] = d; rows.append((d, c))
    iss = pd.concat([a[["d", "code"]], pd.DataFrame(rows, columns=["d", "code"])]).sort_values("d")
    # 有次數與無次數可能是同一檔：同公司 30 天內只留一筆
    keep, lastc = [], {}
    for d, c in zip(iss.d, iss.code):
        if c in lastc and (d - lastc[c]).days < 30: continue
        lastc[c] = d; keep.append((d, c))
    iss = pd.DataFrame(keep, columns=["d", "code"]); _, old = cb_dates()
    days, N, cj, O, Cff, inb, lk = B["days"], B["N"], {c: j for j, c in enumerate(B["cols"])}, B["O"], B["Cff"], B["inb"], B["lk"]
    Hh = np.asarray(B["X"]["H"], float); t0 = int(np.searchsorted(days.values, np.datetime64("2011-01-01"))); ev = []
    for d, c in zip(iss.d, iss.code):
        j = cj.get(c); i = int(np.searchsorted(days.values, np.datetime64(d)))
        if j is None or not (t0 <= i) or i + 61 >= N: continue
        ie = i + 1
        if not (O[ie, j] > 0) or lk[ie - 1, j] or not inb[i, j]: continue
        r = {"date": days[i], "code": c}
        for H in (5, 10, 20, 40, 60):
            ix = ie + H - 1; rH = (Cff[ix, j] / O[ie, j] - 1) * 100 - COST; r[f"x{H}"] = rH - bx(B, ie, ix); r[f"u{H}"] = rH - B["RU"][H][ie]
        r["mfe60"] = (np.nanmax(Hh[ie:ie + 60, j]) / O[ie, j] - 1) * 100
        ev.append(r)
    V = pd.DataFrame(ev); V["seg"] = V.date.map(lambda d: "2011-15" if d.year < 2016 else "2016-19" if d.year < 2020 else "2020-26")
    out = {"事件數（新去重）": int(len(V)), "第50輪去重事件數（同期全部）": int(len(old[old.d >= "2011-01-01"]))}
    for H in (5, 10, 20, 40, 60):
        m = V.groupby(V.date.dt.to_period("M"))[f"x{H}"].mean(); mu = V.groupby(V.date.dt.to_period("M"))[f"u{H}"].mean()
        srt = V[f"u{H}"].sort_values(ascending=False); top = srt.iloc[:max(1, len(srt) // 20)].sum() / srt.sum() * 100 if srt.sum() > 0 else None
        out[f"{H}日"] = {"減0050 每筆": round(float(V[f'x{H}'].mean()), 2), "減0050 中位": round(float(V[f'x{H}'].median()), 2), "減0050 t": round(nw_t(m.reset_index(drop=True), 3), 2),
                        "減分析範圍 每筆": round(float(V[f'u{H}'].mean()), 2), "減分析範圍 中位": round(float(V[f'u{H}'].median()), 2), "減分析範圍 t": round(nw_t(mu.reset_index(drop=True), 3), 2),
                        "前5%占總超額%": round(float(top), 0) if top is not None else None, "三段減0050": {s: round(float(V[V.seg == s][f"x{H}"].mean()), 2) for s in ("2011-15", "2016-19", "2020-26")}}
    out["60日內最高浮盈 平均／中位"] = [round(float(V.mfe60.mean()), 1), round(float(V.mfe60.median()), 1)]
    return out

if __name__ == "__main__":
    B = base(); print("base ok", flush=True)
    E = seq_signals(B); print("流程訊號", len(E), flush=True)
    RO, RV = roe_table(), rev_table()
    E = pd.merge_asof(E.sort_values("date"), RO.sort_values("avail"), left_on="date", right_on="avail", by="code", direction="backward"); E.loc[(E.date - E.avail).dt.days > 200, "roe"] = np.nan
    E = pd.merge_asof(E.sort_values("date"), RV.sort_values("pub"), left_on="date", right_on="pub", by="code", direction="backward"); E.loc[(E.date - E.pub).dt.days > 70, "revok"] = np.nan
    E["seg"] = E.date.map(seg); E["fund"] = E.revok.fillna(False).astype(bool) & (E.roe >= 8)
    res = {}
    for sup in ("mid", "green"):
        for conf in ("等再轉強", "不等"):
            Z0 = E[(E.sup == sup) & (E.conf == conf)]
            for lab, Z in (("線型", Z0), ("＋基本面", Z0[Z0.fund]), ("＋基本面＋大盤", Z0[Z0.fund & Z0.strong])):
                key = f"{'中軸' if sup == 'mid' else '綠線'}｜{conf}｜{lab}"
                res[key] = {"H20減0050": summ(Z, "x20"), "H20減分析範圍": summ(Z, "u20").get("每月平均"), "H60減0050": summ(Z, "x60").get("每月平均"),
                            "H60減分析範圍": summ(Z, "u60").get("每月平均"), "跌破中軸賣 減0050": summ(Z, "xS").get("每月平均")}
                print(key, json.dumps(res[key], ensure_ascii=False), flush=True)
    m = res["中軸｜等再轉強｜＋基本面＋大盤"]
    h = m["H20減0050"]
    res["通過"] = bool(h.get("筆數") and h["每月平均"] > 0 and (h["t"] or 0) >= 2.9 and all(v is not None and v > 0 for v in h["四段每月平均"].values()) and h["贏%"] >= 50 and (m["H20減分析範圍"] or 0) > 0)
    print("通過", res["通過"], flush=True)
    res["B 可轉債"] = cb_part(B); print("CB", json.dumps(res["B 可轉債"], ensure_ascii=False), flush=True)
    json.dump(res, open(f"{REPO}/research/results/course_seq.json", "w"), ensure_ascii=False, indent=1, default=str)
