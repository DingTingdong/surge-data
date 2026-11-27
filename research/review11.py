"""第九份覆核要求的補充數字（第 68 輪；2026/11/27 11:20 登錄，看結果前寫下，全部描述性，不改規則）

起因：第九份覆核（固定 99c40f2）指出 ① 第 66 輪 B 的「最差單月 −28.7%」其實是「最差單月減 0050」，要另報月選本身最差單月，
  並補上每日市值回落的隨機比較；② 要固定報「排除最好一期」的績效；③ 第 66 輪 C2 用了舊的 500 元上限、樣本 176 期與 C1 的 190 期不同。
做法（規則同第 66 輪 review9.sim，11/27 現行月選）：
  B' 低波動 vs 300 組隨機（種子 0～299，同第 66 輪）：每月減 0050、最差單月（絕對）、最差單月減 0050、月底回落、每日市值回落；報低波動在隨機裡的百分位。
  E  排除最好 1 期、最好 3 期後的每月減 0050 與 2024–26（現行規則）。
  C2' k=10 分數前 5（不跳過高價股、排除 TDR）在 d0+11 開盤 → d0+15 收盤減分析範圍等權，只用 C1 和 C2 都有的月份；同時列 C1 k10 段在同一批月份的數字。
"""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import review7 as R7
import mp_exec as M
from review7 import rescore
from review9 import sim, mdd, daily_dd, NOTDR
from mp_exec import prep
from price_cap import addraw
from vs0050 import nw_t
import robust_select as RS

if __name__ == "__main__":
    res = {}
    PK = "/tmp/model_v2/vs0050_dl_extra.pkl"; D = prep(PK, "2011-01-01", "2027-01-01"); D["cols"] = list(pd.read_pickle(PK)["cols"]); D = addraw(D, PK); D["F0"] = D["F"].copy()
    rescore(D, keep=NOTDR); D["FDS"] = sorted(D["F"].date.unique())
    def met(T):
        x = T.r - T.b; return {"每月減0050": float(x.mean()), "最差單月": float(T.r.min()), "最差單月減0050": float(x.min()), "月底回落": mdd(T.r), "每日市值回落": daily_dd(D, T)}
    T0 = sim(D); base = met(T0); rnd = [met(sim(D, order="rand", seed=k)) for k in range(300)]
    res["B' 低波動 vs 隨機"] = {"低波動": {k: round(v, 2) for k, v in base.items()},
        "隨機中位數": {k: round(float(np.median([r[k] for r in rnd])), 2) for k in base},
        "隨機 5%～95%": {k: [round(float(np.percentile([r[k] for r in rnd], q)), 2) for q in (5, 95)] for k in base},
        "低波動的百分位（值越大＝數字越大；回落與虧損是負數，越大＝越淺）": {k: round(float(np.mean([r[k] <= base[k] for r in rnd]) * 100)) for k in base}}
    print("B'", json.dumps(res["B' 低波動 vs 隨機"], ensure_ascii=False), flush=True)
    x = (T0.r - T0.b); e = {}
    for k in (0, 1, 3):
        keep = x.drop(x.nlargest(k).index) if k else x
        e[f"排除最好{k}期"] = {"每月減0050": round(float(keep.mean()), 2), "t": round(nw_t(keep, 3), 2), "2024–26": round(float(keep[keep.index >= "2024-01-01"].mean()), 2),
                           "排除的期": [d.strftime("%Y-%m") for d in x.nlargest(k).index]}
    res["E 排除最好的期"] = e; print("E", json.dumps(e, ensure_ascii=False), flush=True)
    # C2'
    X = pd.read_pickle("/tmp/model_v2/vs0050_fdx_mid_extra.pkl"); days = pd.DatetimeIndex(X["days"]); C = np.asarray(X["C"], float); Ov = np.asarray(X["O"], float)
    inb = np.asarray(X["inb"], bool); b = np.asarray(X["bidx"], float); e5 = M.E50.reindex(days); bO = b * (e5.open / e5.close).where((e5.open > 0) & (e5.close > 0)).fillna(1.0).values
    Cf = pd.DataFrame(C).ffill(limit=10).values; pc = np.asarray(X["prevc"], float)
    def ew(t0, t1):
        p0 = Ov[t0]; m = inb[t0 - 1] & (p0 > 0) & np.isfinite(Cf[t1]); return float(np.nanmean(Cf[t1, m] / p0[m] - 1) * 100)
    Dm = RS.load("/tmp/model_v2/vs0050_fdx_mid_extra.pkl", "2011-01-01", "2027-01-01"); G = {d: g for d, g in Dm["F"].groupby("date")}; CJ = Dm["CJ"]
    ds = pd.Series(days); d0 = ds[ds.dt.day >= 11].groupby(ds.dt.to_period("M")).first(); pos = {d: i for i, d in enumerate(days)}; rows = []
    for d in d0:
        i = pos[d]
        if d < pd.Timestamp("2011-01-01") or i + 15 >= len(days): continue
        ie = i + 11; uw = ew(ie, i + 15); bb = (b[i + 15] / bO[ie] - 1) * 100
        r = {"date": d, "C1 k10段 等權減0050": uw - bb}
        if days[i + 10] in G:
            g = G[days[i + 10]]; g = g[~g.code.str.match(r"^91\d\d$")].sort_values("cs", ascending=False).head(20); js = []
            thr = 0.095 if days[ie] >= pd.Timestamp("2015-06-01") else 0.065
            for c in g.code:
                j = CJ[c]
                if not (Ov[ie, j] > 0) or (pc[ie, j] > 0 and Ov[ie, j] / pc[ie, j] - 1 >= thr - 1e-9): continue
                js.append(j)
                if len(js) >= 5: break
            if js: top = float(np.nanmean(Cf[i + 15, js] / Ov[ie, js] - 1) * 100) - 0.6; r["C2' 前5減等權"] = top - uw; r["C2' 前5減0050"] = top - (bb - 0.18)
        rows.append(r)
    W = pd.DataFrame(rows).set_index("date"); W = W.dropna(); sg = RS.seg(W.index)
    res["C' 同一批月份"] = {k: {"平均": round(float(W[k].mean()), 2), "t": round(nw_t(W[k].reset_index(drop=True), 3), 2), "四段": [round(float(W[k][sg == q].mean()), 2) for q in R7.SEG4], "次數": int(len(W))} for k in W.columns}
    print("C'", json.dumps(res["C' 同一批月份"], ensure_ascii=False), flush=True)
    json.dump(res, open(f"{REPO}/research/results/review11.json", "w"), ensure_ascii=False, indent=1, default=str)
