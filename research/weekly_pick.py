"""週選：專門抓「接下來一兩週會漲」的短線選股（第 18 輪；2026/11/23 11:10 登錄，看結果前寫下，看到結果後不修改）

起因：使用者指出每個月贏大盤的股票很多，雷達應該更常換股去抓它們。第 6 輪每天換股用的是月選分數（半年動能、月營收等慢變數），
  不是專門預測短期的訊號；專門的短線排名還沒測過。
作法（同一套模擬，只差換股頻率與分數）：
  換股日：每週最後一個交易日收盤後排名，下一個交易日開盤買賣；仍在前 20 的續抱，跌出前 20 的換成排名最高、還沒持有的。
  W0 月選（對照）：同一套模擬，但換股日是每月第一個 ≥ 11 號的交易日、分數用月選 12 個特性（daily_rank 每日分數，已核對和月選一致）。
  W1 週選短分：每週、分數用 5 個短線特性的當天百分位等權平均（分析範圍內）：
      r5 五日漲幅；am5 五日均成交值 ÷ 六十日均成交值（量能突增）；d_hi60 收盤距 60 日最高收盤；
      lu10 近 10 日收漲停次數（漲幅 ≥ 9.5%，2015/6 以前 6.5%）；clv5 近 5 日收盤在當日高低區間的位置平均（收得強不強）。
  W2 週選月選分：每週、分數用月選分數（只改頻率）。
  W3 週選合併：每週、分數 = 月選分數百分位與短分百分位平均。
  開盤一價漲停買不到 → 那格放 0050 到下次換股日；下市或停牌超過 60 天 → 以最後價出場。
  成本：每次賣股扣 0.6%（使用者 2.8 折），0050 進出各扣 0.1%。每日 20 格等權，按月複利，減 0050（含息）同月報酬。
資料：/tmp/model_v2/vs0050_extra.pkl 與 daily_cs.pkl（2010-01～2026-11），2011-01 起算；只用價量，三大法人 2020 以前沒有資料所以不用。
採用條件（W1 或 W3 全部符合才算「比月選好」）：
  ① 每月減 0050 平均 > W0，且 Newey-West t（落後 1）≥ 2.9；② 四段（2011–15、2016–19、2020–23、2024–26）都 > 0，且 2024–26 > W0；
  ③ 每月「減 W0」平均 > 0、t ≥ 2；④ 最大回落不比 W0 深 5 個百分點以上；⑤ 成本 1.1% 時每月減 0050 仍 > 0、t ≥ 2。
  累計檢驗約 35 個規則，門檻維持 t ≥ 2.9。通過的還要 2006–2009 年也 > 0 才放上網站。
另報（不影響判斷）：平均持有天數、每格每年換手；每次換股的入選股「下 5 個交易日贏 0050 超過 5%」比例 vs 全部股票（抓大漲股的能力）；
  W1 各特性單獨的每週橫斷面 Fama-MacBeth 係數（下 5 日報酬，t 值）。
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
from vs0050 import nw_t, per, PER, LOCK

def load():
    X = pd.read_pickle("/tmp/model_v2/vs0050_extra.pkl"); D = pd.read_pickle("/tmp/model_v2/daily_cs.pkl")
    assert len(X["days"]) == len(D["days"]) and (X["cols"] == D["cols"]).all()
    days, cols = X["days"], X["cols"]
    C = pd.DataFrame(X["C"], index=days, columns=cols); Cf = C.ffill(limit=60)
    H = pd.DataFrame(X["H"], index=days, columns=cols); L = pd.DataFrame(X["L"], index=days, columns=cols); A = pd.DataFrame(X["A"], index=days, columns=cols)
    inb = X["inb"].astype(bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = {"r5": Cf / Cf.shift(5) - 1,
             "am5": A.rolling(5, min_periods=5).mean() / A.rolling(60, min_periods=40).mean(),
             "d_hi60": Cf / Cf.rolling(60, min_periods=40).max() - 1}
        thr = LOCK - 0.03 * (days < pd.Timestamp("2015-06-01"))
        up = (C / C.shift(1) - 1).ge(pd.Series(thr, index=days), axis=0) & C.notna()
        F["lu10"] = up.astype(float).rolling(10, min_periods=10).sum().where(C.notna())
        rng = (H - L).where(H > L); F["clv5"] = ((C - L) / rng).rolling(5, min_periods=3).mean()
    pct = lambda M: pd.DataFrame(np.where(inb, M, np.nan), index=days, columns=cols).rank(axis=1, pct=True)
    P = {k: pct(v.replace([np.inf, -np.inf], np.nan).values) for k, v in F.items()}
    short = np.where(inb, np.mean([P[k].fillna(0.5).values for k in P], axis=0), np.nan)
    cs = D["cs"].astype(float)
    comb = np.where(inb, (pct(cs).fillna(0.5).values + pct(short).fillna(0.5).values) / 2, np.nan)
    return {"days": days, "cols": cols, "C": Cf.values, "O": X["O"], "lockn": X["lockn"], "bidx": X["bidx"], "inb": inb,
            "S": {"月選分": cs, "短分": short, "合併": comb}, "P": P}

def formation_days(days, kind):
    d = pd.Series(days)
    if kind == "W":
        wk = d.dt.isocalendar().week.astype(int) + d.dt.isocalendar().year.astype(int) * 100
        return np.where(wk.values != np.roll(wk.values, -1))[0][:-1]
    ym = d.dt.year * 100 + d.dt.month
    ok = d.dt.day >= 11; idx = []
    for m in ym.unique():
        q = np.where((ym == m) & ok)[0]
        if len(q): idx.append(q[0])
    return np.array(idx)

def simulate(D, score, kind, cost=0.6, start="2011-01-01", n=20):
    days, Cf, Ov, lk, bidx, inb = D["days"], D["C"], D["O"], D["lockn"], D["bidx"], D["inb"]; N = len(days)
    t0 = int(np.searchsorted(days.values, np.datetime64(start))); fds = set(int(i) for i in formation_days(days, kind) if i >= t0 - 1)
    slots = [None] * n; daily = np.zeros(N); nexit = 0; hold = []; orders = None; hits = []
    for d in range(t0, N):
        b1 = bidx[d] / bidx[d - 1] - 1; r = np.zeros(n); sell, buy = orders if orders is not None else (set(), []); orders = None
        sold, empty = [], []
        for i, s in enumerate(slots):
            if s is None: r[i] = b1; empty.append(i); continue
            j = s["j"]
            if j in sell:
                op = Ov[d, j] if Ov[d, j] > 0 else Cf[d, j]
                r[i] = (op / Cf[d - 1, j] - 1 if np.isfinite(op) and Cf[d - 1, j] > 0 else 0.0) - cost / 100 - 0.001
                hold.append(d - s["t"]); nexit += 1; slots[i] = None; sold.append(i); continue
            c1, c0 = Cf[d, j], Cf[d - 1, j]
            if not np.isfinite(c1): r[i] = -cost / 100; slots[i] = None; nexit += 1; hold.append(d - s["t"])
            else: r[i] = c1 / c0 - 1
        free = sold + empty
        for j in buy:
            if not free: break
            o = Ov[d, j]
            if not (o > 0) or lk[d - 1, j] or not np.isfinite(Cf[d, j]): continue  # 買不到：該格留 0050
            i = free.pop(0); g = Cf[d, j] / o - 1
            r[i] = (1 + r[i]) * (1 + g) - 1 if i in sold else g - 0.001
            slots[i] = {"j": j, "px": o, "t": d}
        daily[d] = r.mean()
        if d in fds and d + 1 < N:
            sc = np.where(inb[d] & np.isfinite(score[d]), score[d], -np.inf); order = np.argsort(-sc); top = [j for j in order[:n] if sc[j] > -np.inf]
            held = {s["j"] for s in slots if isinstance(s, dict)}; topset = set(top)
            sell = {j for j in held if j not in topset}; nfree = n - (len(held) - len(sell))
            buy = [j for j in top if j not in held][:nfree]; orders = (sell, buy)
            if d + 6 < N:
                o1 = Ov[d + 1]; f5 = Cf[d + 6] / np.where(o1 > 0, o1, np.nan) - 1; b5 = bidx[d + 6] / bidx[d + 1] - 1
                ok = inb[d] & np.isfinite(f5); hits.append(((f5[top] - b5 > 0.05).mean() if top else np.nan, (f5[ok] - b5 > 0.05).mean()))
    s = pd.Series(daily[t0:], index=days[t0:]); bd = pd.Series(np.concatenate([[0], bidx[1:] / bidx[:-1] - 1])[t0:], index=days[t0:])
    M = pd.DataFrame({"p": (1 + s).groupby(s.index.to_period("M")).prod() * 100 - 100, "b": (1 + bd).groupby(bd.index.to_period("M")).prod() * 100 - 100})
    M.index = M.index.to_timestamp(); yrs = (days[-1] - days[t0]).days / 365.25; h = np.array(hits, float)
    return M, {"平均持有天數": round(float(np.mean(hold)), 1) if hold else None, "每格每年換手": round(nexit / n / yrs, 1),
               "入選股下5日贏0050超過5%": round(float(np.nanmean(h[:, 0]) * 100), 1), "全部股票": round(float(np.nanmean(h[:, 1]) * 100), 1)}

def stats(M):
    x = M.p - M.b; pr = per(x.index); w = (1 + M.p / 100).cumprod(); yrs = len(x) / 12
    return {"全期": (round(float(x.mean()), 2), round(nw_t(x, 1), 2)), "四段": {p: round(float(x[pr == p].mean()), 2) for p in PER},
            "最大回落": round(float((w / w.cummax() - 1).min() * 100), 1), "年化": round(float(w.iloc[-1] ** (1 / yrs) - 1) * 100, 1)}

if __name__ == "__main__":
    D = load(); out = {}
    M0, i0 = simulate(D, D["S"]["月選分"], "M"); s0 = stats(M0); out["W0 月選"] = {**s0, **i0}; print("W0 月選", s0, i0, flush=True)
    for lab, key in (("W1 週選短分", "短分"), ("W2 週選月選分", "月選分"), ("W3 週選合併", "合併")):
        M, info = simulate(D, D["S"][key], "W"); s = stats(M); M11, _ = simulate(D, D["S"][key], "W", cost=1.1); s11 = stats(M11)
        dx = (M.p - M.b) - (M0.p - M0.b)
        c = {"①": s["全期"][0] > s0["全期"][0] and s["全期"][1] >= 2.9, "②": all(v > 0 for v in s["四段"].values()) and s["四段"]["2024-26"] > s0["四段"]["2024-26"],
             "③": dx.mean() > 0 and nw_t(dx, 1) >= 2, "④": s["最大回落"] >= s0["最大回落"] - 5, "⑤": s11["全期"][0] > 0 and s11["全期"][1] >= 2}
        out[lab] = {**s, **info, "減W0": (round(float(dx.mean()), 2), round(nw_t(dx, 1), 2)), "成本1.1%": s11, "各項": c, "通過": all(c.values()) and lab != "W2 週選月選分"}
        print(lab, out[lab], flush=True)
    # W1 各特性單獨：每週橫斷面 FM（下 5 日、開盤買）
    days, Cf, Ov, inb = D["days"], D["C"], D["O"], D["inb"]; fds = [i for i in formation_days(days, "W") if days[i] >= pd.Timestamp("2011-01-01") and i + 6 < len(days)]
    fm = {k: [] for k in D["P"]}
    for d in fds:
        o1 = Ov[d + 1]; y = Cf[d + 6] / np.where(o1 > 0, o1, np.nan) - 1; ok = inb[d] & np.isfinite(y)
        for k, P in D["P"].items():
            z = P.values[d]; m = ok & np.isfinite(z)
            if m.sum() > 50: zz = (z[m] - z[m].mean()) / z[m].std(); fm[k].append(float(np.polyfit(zz, y[m] * 100, 1)[0]))
    out["FM（每標準差、下5日%）"] = {k: (round(float(np.mean(v)), 3), round(nw_t(pd.Series(v), 4), 2)) for k, v in fm.items()}; print(out["FM（每標準差、下5日%）"])
    json.dump(out, open(f"{REPO}/research/results/weekly_pick.json", "w"), ensure_ascii=False, indent=1, default=str)
