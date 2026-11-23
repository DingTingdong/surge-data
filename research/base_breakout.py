"""低基期第一段：整理後第一根放量紅 K 突破（第 29 輪；2026/11/23 14:40 登錄，看結果前寫下，看到結果後不修改）

起因：使用者過去的可轉債選股法，核心是「低基期 → 籌碼催化 → 第一根放量紅 K → 突破確認 → 三日法則」。
  這輪先測不需要可轉債資料的部分（全部股票），可轉債的部分等歷史名單抓到再測。
訊號（第 t 天收盤後判斷，分析範圍內股票）：
  ① 低基期：t−1 的 20 日漲幅 < 25%，且 t 的 20 日漲幅 < 30%（排除已上漲超過 30%）；
  ② 整理：t−40～t−1 收盤最高 ÷ 最低 − 1 ≤ 20%（約 2 個月橫盤）；
  ③ 突破：t 收盤 > t−40～t−1 最高收盤（關鍵價位＝整理區間高點），且收紅（收盤 > 開盤）；
  ④ 放量：t 成交值 ≥ 前 20 日平均成交值 × 2；
  ⑤ 第一段：同一檔前 60 個交易日內沒有出現過這個訊號。
進場：
  E1 隔天開盤買（開盤一價漲停買不到就放棄）；
  E3 三日法則（使用者原本的用法）：t+1、t+2、t+3 收盤都沒跌回關鍵價位以下，才在 t+4 開盤買；有一天跌回就放棄。
出場：
  H 買進後抱 20 個交易日（收盤賣）；
  S 抱 20 個交易日，但期間收盤跌回關鍵價位以下 → 隔天開盤賣。
成本：每筆 0.6%。比較：同期間 0050（含息）。資料：2006–2009（vs0050_oldestonly_extra.pkl）＋2011–2026（vs0050_extra.pkl）。
判斷（E3＋H 為主，全部符合才在網站加「低基期突破」分頁）：
  ① 每筆減 0050 的平均，五段（2006–09、2011–15、2016–19、2020–23、2024–26）都 > 0；
  ② 全部平均 > +1%，且以進場月份分群的每月平均 Newey-West t ≥ 2.9；③ 贏 0050 的比例 ≥ 50%。
另報：E1、S 的結果；每年訊號數；放量改成 1.5 倍的版本；跟同期全部股票平均比。
結果（11/23 14:50）：不通過 → 網站不加。E3（三日法則）抱 20 天：每筆減 0050 平均 −1.29%（中位 −2.59%），贏 0050 38%，
  五段 +1.09／−1.12／−1.21／−1.51／−3.13；和同一天進場的全部股票平均差不多（−0.4～+0.4）。E1 隔天就買 −1.39%；跌回關鍵價就賣 −0.9%（贏 0050 比例更低）。
  放量 1.5 倍結果相同。訊號一年 400～1,800 個，太多 → 單靠價量型態沒有優勢；使用者原本的方法還多了可轉債結構與題材篩選，下一輪加營收催化再測。
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
from vs0050 import nw_t
SEGS = ["2006-09", "2011-15", "2016-19", "2020-23", "2024-26"]
def seg(d): return np.where(d < pd.Timestamp("2010-01-01"), "2006-09", np.where(d < pd.Timestamp("2011-01-01"), "2010", np.where(d < pd.Timestamp("2016-01-01"), "2011-15", np.where(d < pd.Timestamp("2020-01-01"), "2016-19", np.where(d < pd.Timestamp("2024-01-01"), "2020-23", "2024-26")))))

def events(path, start, end, volx=2.0, cost=0.6):
    X = pd.read_pickle(path); days = X["days"]; N = len(days)
    C = pd.DataFrame(X["C"], index=days); Cf = C.ffill(limit=5); O, H, L = X["O"], X["H"], X["L"]; A = pd.DataFrame(X["A"], index=days)
    inb = np.asarray(X["inb"]).astype(bool); lk = X["lockn"]; b = X["bidx"]
    r20 = (Cf / Cf.shift(20) - 1).values; hi40 = Cf.shift(1).rolling(40, min_periods=35).max().values; lo40 = Cf.shift(1).rolling(40, min_periods=35).min().values
    a20 = A.shift(1).rolling(20, min_periods=15).mean().values; Cv = C.values
    with np.errstate(invalid="ignore", divide="ignore"):
        sig = inb & (np.vstack([np.full((1, Cv.shape[1]), np.nan), r20[:-1]]) < 0.25) & (r20 < 0.30) & (hi40 / lo40 - 1 <= 0.20) & (Cv > hi40) & (Cv > O) & (A.values >= volx * a20)
    sig = np.nan_to_num(sig).astype(bool)
    # 第一段：前 60 天沒有訊號
    prior = pd.DataFrame(sig.astype(float)).shift(1).rolling(60, min_periods=1).max().fillna(0).values > 0
    sig &= ~prior
    t0 = int(np.searchsorted(days.values, np.datetime64(start))); t1 = int(np.searchsorted(days.values, np.datetime64(end)))
    Cff = Cf.values; out = []
    # 同一天進場、抱 20 天的全部股票平均（扣同樣成本、減 0050），用來和訊號比
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.full(N, np.nan); okO = O > 0
        for ie in range(1, N - 19):
            r = Cff[ie + 19] / np.where(okO[ie], O[ie], np.nan) - 1; m = inb[ie - 1] & np.isfinite(r)
            if m.sum() > 50: R[ie] = (np.nanmean(r[m]) * 100 - cost) - (b[ie + 19] / b[ie - 1] - 1) * 100
    for t, j in zip(*np.where(sig[t0:t1])):
        t += t0; lev = hi40[t, j]
        for mode in ("E1", "E3"):
            if mode == "E1": ie = t + 1
            else:
                if t + 4 >= N or not all(Cff[t + k, j] >= lev for k in (1, 2, 3)): continue
                ie = t + 4
            if ie + 19 >= N or not (O[ie, j] > 0) or lk[ie - 1, j]: continue
            o = O[ie, j]; ixH = ie + 19
            rH = (Cff[ixH, j] / o - 1) * 100 - cost; bH = (b[ixH] / b[ie - 1] - 1) * 100
            # S：收盤跌回關鍵價位 → 隔天開盤賣
            ixS, exS = ixH, Cff[ixH, j]
            for k in range(ie, ie + 19):
                if Cff[k, j] < lev:
                    ixS = k + 1; exS = O[k + 1, j] if O[k + 1, j] > 0 else Cff[k + 1, j]; break
            rS = (exS / o - 1) * 100 - cost; bS = (b[ixS] / b[ie - 1] - 1) * 100 if ixS < ixH else bH
            out.append({"date": days[t], "code": X["cols"][j], "mode": mode, "xH": rH - bH, "xS": rS - bS, "uH": rH - bH - R[ie]})
    return pd.DataFrame(out)

def summarize(E, col):
    res = {}
    for s in SEGS:
        z = E[E.seg == s][col]
        if len(z): res[s] = {"筆數": int(len(z)), "平均": round(float(z.mean()), 2), "中位": round(float(z.median()), 2), "贏0050%": round(float((z > 0).mean() * 100))}
    z = E[E.seg != "2010"]; m = z.groupby(z.date.dt.to_period("M"))[col].mean()
    res["全部"] = {"筆數": int(len(z)), "平均": round(float(z[col].mean()), 2), "中位": round(float(z[col].median()), 2), "贏0050%": round(float((z[col] > 0).mean() * 100)),
                 "每月平均 t": round(nw_t(m.reset_index(drop=True), 3), 2), "有訊號的月份": int(len(m))}
    return res

if __name__ == "__main__":
    out = {}
    for volx in (2.0, 1.5):
        E = pd.concat([events("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01", volx), events("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01", volx)], ignore_index=True)
        E["seg"] = seg(E.date)
        for mode in ("E3", "E1"):
            for col, lab in (("xH", "抱20天"), ("xS", "跌回關鍵價就賣")):
                k = f"放量{volx}倍 {mode} {lab}"; out[k] = summarize(E[E["mode"] == mode], col); print(k, json.dumps(out[k], ensure_ascii=False), flush=True)
        for mode in ("E3", "E1"):
            z = E[(E["mode"] == mode)].dropna(subset=["uH"]); out[f"放量{volx}倍 {mode} 抱20天 減全部股票"] = {s: round(float(z[z.seg == s].uH.mean()), 2) for s in SEGS if (z.seg == s).any()}
            print(f"  {mode} 減同日全部股票平均", out[f"放量{volx}倍 {mode} 抱20天 減全部股票"], flush=True)
        if volx == 2.0: out["每年訊號數（E1）"] = E[E["mode"] == "E1"].groupby(E.date.dt.year).size().to_dict()
    m = out["放量2.0倍 E3 抱20天"]
    c = {"①": all(m[s]["平均"] > 0 for s in SEGS if s in m), "②": m["全部"]["平均"] > 1 and m["全部"]["每月平均 t"] >= 2.9, "③": m["全部"]["贏0050%"] >= 50}
    out["各項"] = c; out["通過"] = all(c.values()); print("每年訊號數", out["每年訊號數（E1）"], "\n各項", c, "通過" if all(c.values()) else "不通過")
    json.dump(out, open(f"{REPO}/research/results/base_breakout.json", "w"), ensure_ascii=False, indent=1, default=str)
