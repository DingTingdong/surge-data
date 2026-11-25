"""課程選股影片：中軸轉向後「拉回跌不破」再買（第 51 輪；2026/11/25 11:30 事先登錄，看結果前寫下，看到結果後不修改）

來源：選股 1、2 逐字稿。老師的兩個條件：「第一，跌不破的點；第二，中軸轉向」。買點是轉向後「往下測一下再上，站穩中軸」、
  「等回到綠線（大量 K 的價位）才買」，不是追上軌；中軸向下時的反彈是死貓跳（給空）。大盤用 20 日、120 日均線看強弱。
（第 50 輪測的是「上碰上」追強，來自可轉債影片；這輪測選股影片的拉回買法。）
資料、期間、成本、比較同第 50 輪（course_rules.py）。
訊號（週線布林 20 週、2 倍標準差；每週最後一個交易日收盤後判斷）：
  P1 中軸拉回：① 中軸本週 > 上週，且 8 週內曾經是往下（過去第 2～8 週中有任一週中軸 < 前一週）＝剛轉向；
     ② 本週最低價 ≤ 中軸 × 1.02（回測到中軸附近）且週收盤 ≥ 中軸（跌不破）；③ 前 13 週沒出現過 P1。
  P2 綠線拉回：① 同 P1 的中軸剛轉向；② 過去 60 個交易日內有「大量紅 K」（成交值 ≥ 前 20 日平均 3 倍且收紅），綠線＝那天收盤價（取最近一根）；
     ③ 本週最低價 ≤ 綠線 × 1.02 且週收盤 ≥ 綠線（回到綠線、跌不破）；④ 前 13 週沒出現過 P2。
  P3 P1＋大盤強（等權指數 20 日均線 > 120 日均線）。
  另報（診斷）D：只有 ② 拉回跌不破、但中軸往下（死貓跳）的版本，看老師說的「中軸向下的反彈不要做」對不對。
進場：下個交易日開盤（開盤一價漲停買不到就放棄）。出場：H20 主要；另報 H60、S（週收盤跌破中軸→下個交易日開盤賣，最多 120 天）。
判斷（同第 50 輪 H1）：H20 減 0050 五段都 > 0、NW t ≥ 2.9、贏 0050 ≥ 50%，且減全部股票平均 > 0 → 網站加「中軸轉向拉回」觀察名單。
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
from course_rules import summ, pass_main, seg, COST

def run(path, start, end):
    X = pd.read_pickle(path); days = pd.DatetimeIndex(X["days"]); N = len(days); cols = list(X["cols"])
    C = pd.DataFrame(X["C"], index=days); Cf = C.ffill(limit=5); O = np.asarray(X["O"], float); Hh = np.asarray(X["H"], float); L = pd.DataFrame(np.asarray(X["L"], float), index=days)
    A = pd.DataFrame(np.asarray(X["A"], float), index=days); inb = np.asarray(X["inb"]).astype(bool); lk = X["lockn"]; b = np.asarray(X["bidx"], float); Cff = Cf.values
    # 每日的「最近一根大量紅 K 收盤價」（60 日內）
    a20 = A.shift(1).rolling(20, min_periods=15).mean(); big = ((A >= 3 * a20) & (C > pd.DataFrame(O, index=days))).values
    gl = np.where(big, Cff, np.nan); gl = pd.DataFrame(gl, index=days).ffill(limit=60).values
    wk = pd.Series(days.to_period("W-FRI"), index=days); last = ~wk.duplicated(keep="last").values; wi = np.where(last)[0]
    W = Cf.iloc[wi]; mid = W.rolling(20, min_periods=20).mean()
    wlow = L.groupby(wk.values).min().reindex(pd.PeriodIndex(wk.values[wi])).values  # 週最低
    up_now = (mid > mid.shift(1)).values
    down_recent = np.zeros_like(up_now, dtype=bool)
    md = mid.values
    for k in range(2, 9): down_recent |= np.nan_to_num(np.roll(md, k, axis=0) < np.roll(md, k + 1, axis=0)).astype(bool)
    turned = up_now & down_recent
    Wv, Mv = W.values, md
    pull = (wlow <= Mv * 1.02) & (Wv >= Mv)
    glw = gl[wi]; green = (wlow <= glw * 1.02) & (Wv >= glw)
    def first(sig, gap=13):
        sig = np.nan_to_num(sig).astype(bool).copy(); pr = pd.DataFrame(sig.astype(float)).shift(1).rolling(gap, min_periods=1).max().fillna(0).values > 0; return sig & ~pr
    S = {"P1": first(turned & pull), "P2": first(turned & green), "D": first(~up_now & pull)}
    below_mid = (Wv < Mv)
    r = Cf.pct_change(fill_method=None).where(inb); ew = (1 + r.mean(axis=1).fillna(0)).cumprod(); strong = (ew.rolling(20).mean() > ew.rolling(120).mean()).values
    okO = O > 0; RU = {}
    for H in (20, 60):
        R = np.full(N, np.nan)
        for ie in range(1, N - H + 1):
            rr = Cff[ie + H - 1] / np.where(okO[ie], O[ie], np.nan) - 1; m = inb[ie - 1] & np.isfinite(rr)
            if m.sum() > 50: R[ie] = np.nanmean(rr[m]) * 100 - COST
        RU[H] = R
    bret = lambda ie, ix: (b[ix] / b[ie - 1] - 1) * 100
    t0 = int(np.searchsorted(days.values, np.datetime64(start))); t1 = int(np.searchsorted(days.values, np.datetime64(end))); out = []
    for nm, sig in S.items():
        for k, t in enumerate(wi):
            if not (t0 <= t < t1): continue
            for j in np.where(sig[k] & inb[t])[0]:
                ie = t + 1
                if ie + 59 >= N or not (O[ie, j] > 0) or lk[ie - 1, j]: continue
                o = O[ie, j]; rec = {"sig": nm, "date": days[t], "code": cols[j], "strong": bool(strong[t])}
                for H in (20, 60):
                    ix = ie + H - 1; rH = (Cff[ix, j] / o - 1) * 100 - COST; rec[f"x{H}"] = rH - bret(ie, ix); rec[f"u{H}"] = rH - RU[H][ie]
                ixS = min(ie + 119, N - 1)
                for kk in range(k + 1, len(wi)):
                    if wi[kk] >= ie + 119: break
                    if wi[kk] >= ie and below_mid[kk, j]: ixS = min(wi[kk] + 1, N - 1); break
                exS = O[ixS, j] if O[ixS, j] > 0 else Cff[ixS, j]; rec["xS"] = (exS / o - 1) * 100 - COST - bret(ie, ixS); rec["holdS"] = ixS - ie
                out.append(rec)
    return pd.DataFrame(out)

if __name__ == "__main__":
    E = pd.concat([run("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01"), run("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01")], ignore_index=True)
    E["seg"] = seg(E.date); res = {}
    for lab, Z in (("P1 中軸轉向後拉回不破", E[E.sig == "P1"]), ("P2 回到綠線不破", E[E.sig == "P2"]), ("P3 P1＋大盤強", E[(E.sig == "P1") & E.strong]), ("D 中軸向下的拉回（死貓跳）", E[E.sig == "D"])):
        res[lab] = {"H20減0050": summ(Z, "x20"), "H20減全部": summ(Z, "u20", False), "H60減0050": summ(Z, "x60", False), "H60減全部": summ(Z, "u60", False), "S減0050": summ(Z, "xS", False),
                    "S平均持有天數": round(float(Z.holdS.mean()), 1) if len(Z) else None}
        print(lab, json.dumps(res[lab], ensure_ascii=False, default=str), flush=True)
    res["判斷"] = {k: pass_main(res[k]["H20減0050"], res[k]["H20減全部"]) for k in ("P1 中軸轉向後拉回不破", "P2 回到綠線不破", "P3 P1＋大盤強")}
    print("判斷", res["判斷"]); json.dump(res, open(f"{REPO}/research/results/course_pullback.json", "w"), ensure_ascii=False, indent=1, default=str)
