"""撿跌深前 3 名在 2006–2019 是否也成立（第 43b 輪；2026/11/24 09:50 事先登錄，看結果前寫下，看到結果後不修改）

起因：第 43 輪 P2（撿跌深依自己漲勢前 3 名）在 2020–2026 通過，但「依自己漲勢排序」是 11/20 看過這些恐慌後選的，需要沒看過的期間驗證。
名單、恐慌波段、進場、抱 20 日、扣 1.1%：同 research/panic_pick_old.py（第 15 輪）。
比較：前 20（P0）、前 3（P2）、前 20 裡波動最低 3 檔（P1）、前 20 隨機 3 檔 1,000 次。
判斷：2006–2009、2010–2019 兩段各自 P2 各波平均 ≥ P0 − 1 且 P2 平均在隨機 3 檔的第 50 百分位以上 → 網站標「2006–2019 也成立」；
  只有一段成立或都不成立 → 網站寫「只在 2020 年後明顯」，提醒不要重壓。
結果（11/24 09:55）：2010–2019 成立（13 波 P2 +4.55、P0 0.00、P1 +0.05，P2 贏 P0 9/13，隨機百分位 100）；
  2006–2009 不成立（6 波 P2 +2.90、P0 +3.39、P1 +0.72，P2 中位數 6.13 較高但平均在隨機第 43 百分位）。
  → 不算「2006–2019 也成立」；網站寫出分段結果（2010–2019 較好、2006–2009 沒有比較好），提醒不要重壓。挑低波動（P1）在每段都較差。
"""
import os, sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(20261124)

def run(path, start, end, H=20):
    X = pd.read_pickle(path); days = X["days"]; N = len(days)
    C, O, prevc, pn, inb, lk = X["C"], X["O"], X["prevc"], X["panic"].astype(bool), X["inb"], X["lockn"]
    Cff = pd.DataFrame(C).ffill(limit=60).values; ch = (C / prevc - 1) * 100
    lim = np.where(days < pd.Timestamp("2015-06-01"), 6.5, 9.5)[:, None]; islu = ch >= lim
    Onext = np.vstack([O[1:], np.full((1, C.shape[1]), np.nan)]); fut = np.vstack([Cff[H:], np.full((H, C.shape[1]), np.nan)])
    y = (fut / Onext - 1) * 100; ok = (Onext > 0) & ~lk
    pidx = [t for t in range(N) if pn[t] and pd.Timestamp(start) <= days[t] < pd.Timestamp(end) and t >= 100 and t + H < N]
    waves, cur, last = [], [], -99
    for t in pidx:
        if t - last > 20 and cur: waves.append(cur); cur = []
        cur.append(t); last = t
    if cur: waves.append(cur)
    rows, R = [], []
    for w in waves:
        t = w[0]
        hi20 = np.nanmax(Cff[t - 19:t + 1], axis=0); lo = np.nanmin(Cff[t - 99:t - 19], axis=0); runv = hi20 / lo - 1
        b = inb[t] & np.isfinite(runv); rr = pd.Series(runv[b]).rank(pct=True).values; runr = np.full(len(runv), np.nan); runr[b] = rr
        cand = np.where((runr > 0.7) & islu[t - 59:t + 1].any(axis=0) & (Cff[t] / hi20 - 1 <= -0.20) & b)[0]
        cand = cand[np.argsort(-runv[cand])][:20]; c = [j for j in cand if ok[t, j] and np.isfinite(y[t, j])]
        if len(c) < 3: continue
        vol = np.nanstd(np.diff(Cff[t - 20:t + 1][:, c], axis=0) / Cff[t - 20:t][:, c], axis=0, ddof=1)
        p1 = [c[k] for k in np.argsort(vol)[:3]]
        rows.append({"起點": str(days[t].date()), "檔數": len(c), "P0": round(float(np.mean(y[t, c]) - 1.1), 2),
                     "P2": round(float(np.mean(y[t, c[:3]]) - 1.1), 2), "P1": round(float(np.mean(y[t, p1]) - 1.1), 2)})
        R.append([np.mean(y[t, rng.choice(c, 3, replace=False)]) - 1.1 for _ in range(1000)])
    return rows, np.array(R)

if __name__ == "__main__":
    out = {}
    for k, (p, s, e) in {"2006–2009": ("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01"),
                         "2010–2019": ("/tmp/model_v2/vs0050_extra.pkl", "2010-01-01", "2020-01-01")}.items():
        rows, R = run(p, s, e); g = pd.DataFrame(rows); rm = R.mean(0)
        o = {"波數": len(g), **{f"{c}平均": round(g[c].mean(), 2) for c in ("P0", "P2", "P1")}, **{f"{c}中位數": round(g[c].median(), 2) for c in ("P0", "P2", "P1")},
             "P2贏P0波數": f"{(g.P2 > g.P0).sum()}/{len(g)}", "P2賺錢波數": f"{(g.P2 > 0).sum()}/{len(g)}", "P2最差": round(g.P2.min(), 2), "P0最差": round(g.P0.min(), 2),
             "隨機3檔平均中位數": round(float(np.median(rm)), 2), "P2在隨機的百分位": round(float((rm < g.P2.mean()).mean() * 100)),
             "P1在隨機的百分位": round(float((rm < g.P1.mean()).mean() * 100))}
        o["成立"] = bool(o["P2平均"] >= o["P0平均"] - 1 and o["P2在隨機的百分位"] >= 50)
        out[k] = {"摘要": o, "各波": rows}; print("==", k, json.dumps(o, ensure_ascii=False)); print(g.to_string(index=False))
    out["2006–2019 也成立"] = all(out[k]["摘要"]["成立"] for k in ("2006–2009", "2010–2019")); print("2006–2019 也成立：", out["2006–2019 也成立"])
    json.dump(out, open(f"{REPO}/research/results/panic_pick3_old.json", "w"), ensure_ascii=False, indent=1)
