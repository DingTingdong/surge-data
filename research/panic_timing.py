"""恐慌中的進場時機：撿跌深前 3 固定，換不同觸發點（第 45 輪；2026/11/24 10:35 事先登錄，看結果前寫下，看到結果後不修改）

起因：外部覆核建議固定前 3 選股規則，只比較進場時機；正好對應 11/24（恐慌中大盤單日大漲）該不該隔天追。
名單：撿跌深（W2，依自己漲勢）在觸發日收盤重算，取前 3（開盤一價鎖漲停買不到→往下一名遞補）；另列前 20、全部可買股票、0050。
觸發（每一波 t0＝上市空頭家數第一次到門檻、前 20 日都在門檻下；觀察窗 t0～t0+40）：
  E0 t0（第一次到門檻）
  E1 窗內第一次 ≥ 800
  E2 窗內第二次 ≥ 800（第一次後先連續 ≥3 天 < 800，再回到 ≥ 800）
  E3 窗內第一次 0050 單日漲 3%～5%
  E4 窗內第一次 0050 單日漲 ≥ 5%
進場：觸發日隔天開盤；另報「晚一天」（隔兩天開盤，名單不變）。
看：持有 1／3／5／10／20 個交易日報酬（扣 0.6%），前 3 減全部股票，持有 20 日內組合收盤最大回落（從持有期間最高點算，含買進價）。
門檻：750（主要）與 700 分開報告。2006–2019 用恐慌旗標的波段（同 panic_pick_old.py）只測 E0、E3、E4，描述。
判斷（每個觸發各自，750）：波數 ≥ 5，且前 3 的 20 日減全部股票平均 > 0、≥ 60% 的波 > 0、晚一天版本平均也 > 0 → 「支持在這個時點後追前 3」；
  波數 < 5 → 「樣本不足，不下結論」；其他 → 「不支持」。700 與 2006–2019 只當旁證。
結果（11/24 10:45）：750 只有 E0 達到判斷條件（支持）；E1（4 波）、E2（1 波）、E3（4 波）、E4（3 波）樣本不足，但方向都正。
  750 前3減全部 20 日：E0 +20.75（7/7，晚一天 +20.42）；E1 +34.11（4/4）；E2 +18.24（1/1）；E3 +17.32（3/4，晚一天 +13.63）；E4 +16.00（3/3，晚一天 +10.85）。
  700 旁證：E0 +9.42（10/12，支持）；E3 +12.20（6/9，晚一天 +10.58，支持）；E4 +22.99（4/4）。
  2006–2019：E3 +2.86（3 波，2/3）、+11.33（5 波，3/5）；E4 2006–09 2 波 +1.73（1/2），2010–19 沒有。
  報酬多在 3～10 天出現（750 E0：1 日 +0.9、5 日 +8.4、10 日 +15.7）。前 3 持有 20 日內的真正最大回落（從高點、收盤）平均 −11%，最差 −24% ～ −25%。
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
exec(open("research/panic_winners.py", encoding="utf-8").read().split("out = {}")[0])
Hh, Ll = P["H"], P["L"]; COST = 0.6; HS = (1, 3, 5, 10, 20)
et = pd.read_csv("data/etf.csv"); et["date"] = pd.to_datetime(et.date)
def adj_series(code):
    x = et[et.code == code].set_index("date").reindex(days); x[["open", "close"]] = x[["open", "close"]].ffill(limit=10)
    fct = pd.Series(1.0, index=days); a = pd.read_csv("data/adjust.csv", dtype={"code": str}); a = a[a.code == code]
    for _, z in a.iterrows(): fct[days < pd.Timestamp(z.date)] *= z.ratio
    r = (x.close * fct).pct_change()
    for d in r.index[r.abs() > 0.3]:
        ratio = (x.close.shift(1)[d] * fct.shift(1)[d]) / (x.close[d] * fct[d]); fct[days < d] /= ratio
    return x.open * fct, x.close * fct
o50, c50 = adj_series("0050"); r50 = c50.pct_change().values

def evaluate(i, delay=0):
    """i＝觸發日；買進日 e＝i+1+delay 開盤。回傳各持有天數報酬與回落。"""
    e = i + 1 + delay
    if e + 19 >= N: return None
    m = inb.iloc[i] & C0.iloc[i].notna(); o = O.iloc[e]
    lk = (O.iloc[e] == Hh.iloc[e]) & (O.iloc[e] == Ll.iloc[e]) & (O.iloc[e] / C0.iloc[e - 1] - 1 >= 0.094)
    buy = o.notna() & (o > 0) & ~lk
    hi20 = C0.iloc[i - 19:i + 1].max(); run = (hi20 / C0.iloc[i - 100:i - 19].min() - 1).where(m)
    W2 = (run.rank(pct=True) > 0.7) & lim.iloc[i - 59:i + 1].any() & (C0.iloc[i] / hi20 - 1 <= -0.2) & m
    ranked = list(run[W2].sort_values(ascending=False).index)
    t3 = [c for c in ranked if buy[c]][:3]; t20 = [c for c in ranked[:20] if buy[c]]
    if len(t3) < 3: return None
    r = {"d": str(days[i].date())}
    for h in HS:
        x = e + h - 1; R = (C0.iloc[x] / o - 1) * 100 - COST
        r[f"t3_{h}"] = R[t3].mean(); r[f"t20_{h}"] = R[t20].mean(); r[f"u_{h}"] = R[m & buy].mean()
        r[f"b_{h}"] = (c50.iloc[x] / o50.iloc[e] - 1) * 100 - COST
    path = np.r_[1.0, (C0.iloc[e:e + 20][t3].div(o[t3])).mean(axis=1).values]
    r["dd"] = float((path / np.maximum.accumulate(path) - 1).min() * 100)
    return r

def triggers(th):
    out = []
    for t0 in signals(th):
        w = range(t0, min(t0 + 41, N)); tr = {"E0": t0}
        e1 = next((t for t in w if bear.iloc[t] >= 800), None); tr["E1"] = e1
        if e1 is not None:
            bl = 0
            for t in range(e1 + 1, min(t0 + 41, N)):
                if bear.iloc[t] < 800: bl += 1
                else:
                    if bl >= 3: tr["E2"] = t; break
                    bl = 0
        tr["E3"] = next((t for t in w if 0.03 <= r50[t] < 0.05), None)
        tr["E4"] = next((t for t in w if r50[t] >= 0.05), None)
        out.append((str(days[t0].date()), tr))
    return out

def summarize(rows, rows_d):
    if not rows: return {"波數": 0}
    T = pd.DataFrame(rows); s = {"波數": len(T), "日期": T.d.tolist()}
    for h in HS:
        s[f"{h}日 前3"] = round(T[f"t3_{h}"].mean(), 2); s[f"{h}日 前3減全部"] = round((T[f"t3_{h}"] - T[f"u_{h}"]).mean(), 2)
    s["20日 前20"] = round(T.t20_20.mean(), 2); s["20日 全部"] = round(T.u_20.mean(), 2); s["20日 0050"] = round(T.b_20.mean(), 2)
    ex = T.t3_20 - T.u_20; s["20日 前3減全部>0"] = f"{(ex > 0).sum()}/{len(T)}"; s["前3最大回落平均"] = round(T.dd.mean(), 2); s["前3最大回落最差"] = round(T.dd.min(), 2)
    if rows_d:
        Td = pd.DataFrame(rows_d); s["晚一天 20日 前3減全部"] = round((Td.t3_20 - Td.u_20).mean(), 2); s["晚一天 20日 前3"] = round(Td.t3_20.mean(), 2)
    if len(T) < 5: s["判斷"] = "樣本不足"
    else: s["判斷"] = "支持" if (ex.mean() > 0 and (ex > 0).mean() >= 0.6 and s.get("晚一天 20日 前3減全部", -1) > 0) else "不支持"
    return s

res = {}
for th in (750, 700):
    TR = triggers(th); res[th] = {}
    for k in ("E0", "E1", "E2", "E3", "E4"):
        rows, rows_d = [], []
        for d0, tr in TR:
            t = tr.get(k)
            if t is None: continue
            a, b = evaluate(t), evaluate(t, 1)
            if a: a["wave"] = d0; rows.append(a)
            if b: rows_d.append(b)
        res[th][k] = summarize(rows, rows_d); print(th, k, json.dumps(res[th][k], ensure_ascii=False))

def run_old(path, start, end):
    X = pd.read_pickle(path); dd = X["days"]; NN = len(dd)
    C, Oo, prevc, pn, ib, Hx, Lx = X["C"], X["O"], X["prevc"], X["panic"].astype(bool), X["inb"], X["H"], X["L"]
    bx = np.asarray(X["bidx"], float); rb = np.r_[np.nan, bx[1:] / bx[:-1] - 1]
    Cff = pd.DataFrame(C).ffill(limit=60).values; ch = (C / prevc - 1) * 100
    lm = np.where(dd < pd.Timestamp("2015-06-01"), 6.5, 9.5)[:, None]; islu = ch >= lm
    pidx = [t for t in range(NN) if pn[t] and pd.Timestamp(start) <= dd[t] < pd.Timestamp(end) and t >= 100 and t + 45 < NN]
    W, cur, last = [], [], -99
    for t in pidx:
        if t - last > 20 and cur: W.append(cur); cur = []
        cur.append(t); last = t
    if cur: W.append(cur)
    def ev(i, delay=0):
        e = i + 1 + delay; o = Oo[e]; lkup = lm[e, 0] - 0.1
        lk = (o == Hx[e]) & (o == Lx[e]) & ((o / prevc[e] - 1) * 100 >= lkup)
        okb = (o > 0) & ~lk
        hi20 = np.nanmax(Cff[i - 19:i + 1], axis=0); lo = np.nanmin(Cff[i - 99:i - 19], axis=0); rv = hi20 / lo - 1
        b = ib[i] & np.isfinite(rv); rr = pd.Series(rv[b]).rank(pct=True).values; rk = np.full(len(rv), np.nan); rk[b] = rr
        c = np.where((rk > 0.7) & islu[i - 59:i + 1].any(axis=0) & (Cff[i] / hi20 - 1 <= -0.2) & b)[0]
        c = [j for j in c[np.argsort(-rv[c])] if okb[j] and np.isfinite(Cff[e + 19, j])]
        if len(c) < 3: return None
        R = (Cff[e + 19] / o - 1) * 100 - COST; u = np.nanmean(np.where(ib[i] & okb, R, np.nan))
        return {"d": str(dd[i].date()), "t3_20": float(np.mean(R[c[:3]])), "u_20": float(u)}
    out = {}
    for k in ("E0", "E3", "E4"):
        rows, rows_d = [], []
        for w in W:
            t0 = w[0]; win = range(t0, min(t0 + 41, NN - 21))
            t = t0 if k == "E0" else next((x for x in win if (0.03 <= rb[x] < 0.05 if k == "E3" else rb[x] >= 0.05)), None)
            if t is None: continue
            a, b = ev(t), ev(t, 1)
            if a: rows.append(a)
            if b: rows_d.append(b)
        if rows:
            T = pd.DataFrame(rows); ex = T.t3_20 - T.u_20
            out[k] = {"波數": len(T), "日期": T.d.tolist(), "20日 前3": round(T.t3_20.mean(), 2), "20日 前3減全部": round(ex.mean(), 2), ">0": f"{(ex > 0).sum()}/{len(T)}",
                      "晚一天 前3減全部": round(float(np.mean([r["t3_20"] - r["u_20"] for r in rows_d])), 2) if rows_d else None}
        else: out[k] = {"波數": 0}
        print(start[:4], k, json.dumps(out[k], ensure_ascii=False))
    return out
res["2006–2009"] = run_old("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01")
res["2010–2019"] = run_old("/tmp/model_v2/vs0050_extra.pkl", "2010-01-01", "2020-01-01")
json.dump(res, open("research/results/panic_timing.json", "w"), ensure_ascii=False, indent=1, default=str)
