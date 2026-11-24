"""恐慌後段才選，撿跌深前 3／前 20 還有沒有效（第 44 輪；2026/11/24 10:05 事先登錄，看結果前寫下，看到結果後不修改）

起因：第 43 輪是在恐慌第一天選。網站名單每天重算，使用者常在恐慌後段（例如 11/23 第二次碰到、11/24 大漲開高）才看到。
名單：網站撿跌深（W2，依自己漲勢排序），在選股日當天重算；隔天開盤買（開盤無價不算），抱 20 個交易日，等權。
選股日（每一波＝上市空頭家數第一次到門檻 t0，且前 20 日都在門檻下）：
  S1  t0（第一天，對照）
  S5  t0 後第 4 個交易日（恐慌第 5 天）
  S10 t0 後第 9 個交易日
  S2  第二次碰到：t0 之後 40 個交易日內，先連續 ≥3 天低於門檻、再回到門檻以上的第一天（沒有就不算）
  SA  t0 起 30 個交易日內每個在門檻以上的日子都選，波內先平均
看：前 3、前 20 的 20 日報酬減「全部股票同期平均」（超額），以及減 0050；贏的波數。
資料：2020–2026（門檻 700／750）；另用 2006–2019（恐慌旗標，同 panic_pick_old.py；S2＝波內連續 ≥3 天非恐慌後再恐慌）看 S1／S5／S2／SA。
判斷（每個後段選法各自）：2020–2026 的 700 與 750 都「前 3 超額平均 > 0、且 ≥ 60% 的波 > 0」→ 網站寫「恐慌後段選仍有效」；
  否則寫「只在恐慌剛開始選有效，後段名單只供觀察」。2006–2019 結果只描述、不影響判斷。
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
exec(open("research/panic_winners.py", encoding="utf-8").read().split("out = {}")[0])
et = pd.read_csv("data/etf.csv"); et["date"] = pd.to_datetime(et.date)
def adj_series(code):
    x = et[et.code == code].set_index("date").reindex(days); x[["open", "close"]] = x[["open", "close"]].ffill(limit=10)
    fct = pd.Series(1.0, index=days); a = pd.read_csv("data/adjust.csv", dtype={"code": str}); a = a[a.code == code]
    for _, z in a.iterrows(): fct[days < pd.Timestamp(z.date)] *= z.ratio
    r = (x.close * fct).pct_change()
    for d in r.index[r.abs() > 0.3]:
        ratio = (x.close.shift(1)[d] * fct.shift(1)[d]) / (x.close[d] * fct[d]); fct[days < d] /= ratio
    return x.open * fct, x.close * fct
o50, c50 = adj_series("0050"); H = 20

def pick_eval(i):
    if i + H >= N: return None
    m = inb.iloc[i] & C0.iloc[i].notna(); o = O.iloc[i + 1]; okb = o.notna()
    hi20 = C0.iloc[i - 19:i + 1].max(); run = (hi20 / C0.iloc[i - 100:i - 19].min() - 1).where(m)
    W2 = (run.rank(pct=True) > 0.7) & lim.iloc[i - 59:i + 1].any() & (C0.iloc[i] / hi20 - 1 <= -0.2) & m & okb
    top = list(run[W2].sort_values(ascending=False).index[:20])
    R = ((C0.iloc[i + H] / o - 1) * 100); u = R[m & okb].mean(); b = (c50.iloc[i + H] / o50.iloc[i + 1] - 1) * 100
    if len(top) < 3: return None
    return {"d": str(days[i].date()), "top3": R[top[:3]].mean() - u, "top20": R[top].mean() - u, "top3_b": R[top[:3]].mean() - b, "top20_b": R[top].mean() - b}

def waves_new(th):
    out = []
    for t0 in signals(th):
        ds = {"S1": [t0], "S5": [t0 + 4], "S10": [t0 + 9], "SA": [t for t in range(t0, min(t0 + 30, N)) if bear.iloc[t] >= th]}
        below, s2 = 0, None
        for t in range(t0 + 1, min(t0 + 41, N)):
            if bear.iloc[t] < th: below += 1
            else:
                if below >= 3: s2 = t; break
                below = 0
        ds["S2"] = [s2] if s2 is not None else []
        out.append((str(days[t0].date()), ds))
    return out

res = {}
for th in (700, 750):
    rows = []
    for d0, ds in waves_new(th):
        r = {"wave": d0}
        for k, ts in ds.items():
            ev = [e for e in (pick_eval(t) for t in ts) if e]
            if ev:
                for c in ("top3", "top20", "top3_b", "top20_b"): r[f"{k}_{c}"] = float(np.mean([e[c] for e in ev]))
                if k == "S2": r["S2_d"] = ev[0]["d"]
        rows.append(r)
    T = pd.DataFrame(rows); s = {}
    for k in ("S1", "S5", "S10", "S2", "SA"):
        col = f"{k}_top3"
        if col not in T: continue
        g = T.dropna(subset=[col])
        s[k] = {"波數": len(g), "前3超額": round(g[col].mean(), 2), "前3超額>0": f"{(g[col] > 0).sum()}/{len(g)}", "前20超額": round(g[f"{k}_top20"].mean(), 2),
                "前20超額>0": f"{(g[f'{k}_top20'] > 0).sum()}/{len(g)}", "前3減0050": round(g[f"{k}_top3_b"].mean(), 2), "前20減0050": round(g[f"{k}_top20_b"].mean(), 2)}
        s[k]["成立"] = bool(len(g) and g[col].mean() > 0 and (g[col] > 0).mean() >= 0.6)
    res[th] = {"摘要": s, "各波": T.round(2).to_dict("records")}
    print(f"== {th}"); [print(" ", k, v) for k, v in s.items()]
    print(T.round(1).to_string(index=False))

# 2006–2019（描述）
def run_old(path, start, end):
    X = pd.read_pickle(path); dd = X["days"]; NN = len(dd)
    C, Oo, prevc, pn, ib, lk = X["C"], X["O"], X["prevc"], X["panic"].astype(bool), X["inb"], X["lockn"]
    Cff = pd.DataFrame(C).ffill(limit=60).values; ch = (C / prevc - 1) * 100
    lm = np.where(dd < pd.Timestamp("2015-06-01"), 6.5, 9.5)[:, None]; islu = ch >= lm
    On = np.vstack([Oo[1:], np.full((1, C.shape[1]), np.nan)]); fut = np.vstack([Cff[H:], np.full((H, C.shape[1]), np.nan)])
    y = (fut / On - 1) * 100; ok = (On > 0) & ~lk; u = np.nanmean(np.where(ib & ok, y, np.nan), axis=1)
    pidx = [t for t in range(NN) if pn[t] and pd.Timestamp(start) <= dd[t] < pd.Timestamp(end) and t >= 100 and t + H + 10 < NN]
    W, cur, last = [], [], -99
    for t in pidx:
        if t - last > 20 and cur: W.append(cur); cur = []
        cur.append(t); last = t
    if cur: W.append(cur)
    def ev(t):
        hi20 = np.nanmax(Cff[t - 19:t + 1], axis=0); lo = np.nanmin(Cff[t - 99:t - 19], axis=0); rv = hi20 / lo - 1
        b = ib[t] & np.isfinite(rv); rr = pd.Series(rv[b]).rank(pct=True).values; rk = np.full(len(rv), np.nan); rk[b] = rr
        c = np.where((rk > 0.7) & islu[t - 59:t + 1].any(axis=0) & (Cff[t] / hi20 - 1 <= -0.2) & b)[0]
        c = c[np.argsort(-rv[c])][:20]; c = [j for j in c if ok[t, j] and np.isfinite(y[t, j])]
        return (np.mean(y[t, c[:3]]) - u[t], np.mean(y[t, c]) - u[t]) if len(c) >= 3 else None
    rows = []
    for w in W:
        t0 = w[0]; ws = set(w); ds = {"S1": [t0], "S5": [t0 + 4], "SA": [t for t in w if t < t0 + 30]}
        s2, gap = None, 0
        for t in range(t0 + 1, min(t0 + 41, NN)):
            if t in ws:
                if gap >= 3: s2 = t; break
                gap = 0
            else: gap += 1
        ds["S2"] = [s2] if s2 else []
        r = {"wave": str(dd[t0].date())}
        for k, ts in ds.items():
            e = [x for x in (ev(t) for t in ts if t + H < NN) if x]
            if e: r[f"{k}_top3"] = float(np.mean([x[0] for x in e])); r[f"{k}_top20"] = float(np.mean([x[1] for x in e]))
        rows.append(r)
    T = pd.DataFrame(rows); s = {}
    for k in ("S1", "S5", "S2", "SA"):
        col = f"{k}_top3"
        if col not in T: continue
        g = T.dropna(subset=[col])
        s[k] = {"波數": len(g), "前3超額": round(g[col].mean(), 2), "前3超額>0": f"{(g[col] > 0).sum()}/{len(g)}", "前20超額": round(g[f"{k}_top20"].mean(), 2), "前20超額>0": f"{(g[f'{k}_top20'] > 0).sum()}/{len(g)}"}
    return s
for k, (p, a, b) in {"2006–2009": ("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01"), "2010–2019": ("/tmp/model_v2/vs0050_extra.pkl", "2010-01-01", "2020-01-01")}.items():
    res[k] = run_old(p, a, b); print("==", k); [print(" ", kk, v) for kk, v in res[k].items()]
judge = {k: all(res[th]["摘要"].get(k, {}).get("成立", False) for th in (700, 750)) for k in ("S5", "S10", "S2", "SA")}
res["判斷"] = judge; print("後段選仍有效：", judge)
json.dump(res, open("research/results/panic_late.json", "w"), ensure_ascii=False, indent=1, default=str)
