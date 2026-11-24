"""每檔各買 1 張（金額跟股價走）時，哪種月選挑法最好（第 48 輪；2026/11/24 23:30 事先登錄，看結果前寫下，看到結果後不修改）

起因：使用者指出零股價差大、流動性差、撮合慢，實際上常是每檔買 1 張；這時權重跟股價成正比，穩健版最大回落從 −23% 變 −34%，
  以前沒採用的挑法在這種買法下可能反而比較好。
權重：每期依買進價（期初開盤價）占比分配，等同每檔 1 張；另列等金額（原本的算法）對照。持有規則、成本同 calm3.py（0.6%）。
比較的挑法（都在每月換股日、前 20 名內續抱）：
  S1 穩健版：前 20 名挑 20 日波動最低 3 檔（現行）
  S2 分數前 3：直接買分數最高 3 檔（第 24 輪的 V0）
  S3 前 10 名挑波動最低 3 檔（第 24 輪的 V2）
  S4 前 20 名挑 60 日波動最低 3 檔（第 35 輪參數格點之一）
  S5 前 20 名挑「股價 × 波動」最低 3 檔（1 張時的實際風險，金額小、波動小的優先）
  R  前 20 名隨機 3 檔 200 次（1 張權重），看分布
期間：2006–09＋2011–2026 五段；另看三個換股日（第 1／3／5 個交易日）。
判斷（1 張權重下）：某挑法要取代穩健版，須 ① 2011–2026 每月比 0050 ≥ 穩健版（1 張）且 t ≥ 2.9；② 五段都贏 0050；
  ③ 最大回落比穩健版（1 張）淺 3 個百分點以上，或報酬多 0.3 以上且回落不更深；④ 三個換股日平均也符合 ③ 的方向。
  有一個以上通過 → 網站加「每檔 1 張的建議挑法」並說明；都沒過 → 照實報告，網站加說明兩種買法的差別。
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
import robust_select as RS
from wr_confirm import stats, nth_set

def prep(D):
    Cff = D["Cff"]; r = np.vstack([np.full((1, Cff.shape[1]), np.nan), Cff[1:] / Cff[:-1] - 1])
    D["_v60"] = pd.DataFrame(r).rolling(60, min_periods=45).std().values; return D

def sim(D, method, weight="px", n=3, keep_rank=20, cost=0.6, rng=None):
    O, Cff, lk, bidx, POS, CJ, ND = D["O"], D["Cff"], D["lockn"], np.asarray(D["bidx"], float), D["POS"], D["CJ"], D["ND"]
    G = {d: g for d, g in D["F"].groupby("date")}; rows, held = [], []
    for a, b_ in zip(D["FDS"][:-1], D["FDS"][1:]):
        ie, ix = POS[a] + 1, POS[b_] + 1
        if ix >= ND: break
        g = G[a].sort_values("cs", ascending=False); rank = {c: i + 1 for i, c in enumerate(g.code)}; t = ie - 1
        keep = [c for c in held if rank.get(c, 1e9) <= keep_rank][:n]; new = []
        if method == "S2": cand = g.code.tolist()
        elif method == "R":
            top = g.head(20).code.tolist(); cand = list(rng.permutation(top)) + g.code.iloc[20:].tolist()
        else:
            K = 10 if method == "S3" else 20; top = g.head(K).copy()
            if method == "S4": top["v"] = [D["_v60"][t, CJ[c]] for c in top.code]
            elif method == "S5": top["v"] = [top.vol20.iloc[i] * (O[ie, CJ[c]] if O[ie, CJ[c]] > 0 else Cff[t, CJ[c]]) for i, c in enumerate(top.code)]
            else: top["v"] = top.vol20
            cand = top.assign(v=top.v.fillna(np.inf)).sort_values(["v", "cs"], ascending=[True, False]).code.tolist() + g.code.iloc[K:].tolist()
        for c in cand:
            if len(keep) + len(new) >= n: break
            if c in keep: continue
            j = CJ[c]
            if O[ie, j] > 0 and not lk[ie - 1, j]: new.append(c)
        to = 1.0 if not rows else len([c for c in held if c not in keep]) / n; rs, ws, hn = [], [], []
        for c in keep + new:
            j = CJ[c]; o = O[ie, j] if O[ie, j] > 0 else Cff[ie - 1, j]; ex = O[ix, j] if O[ix, j] > 0 else Cff[ix, j]
            if not np.isfinite(ex): ex = Cff[ix - 1, j]
            if np.isfinite(ex) and np.isfinite(o) and o > 0: rs.append((ex / o - 1) * 100); ws.append(o if weight == "px" else 1.0); hn.append(c)
        held = hn; w = np.array(ws) / np.sum(ws) if ws else np.array([])
        rows.append({"date": a, "r": (float(np.dot(w, rs)) if rs else 0) - cost * to, "b": (bidx[ix - 1] / bidx[ie - 1] - 1) * 100, "wmax": float(w.max()) if len(w) else np.nan})
    T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index); return T

if __name__ == "__main__":
    Dn = prep(RS.load("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01")); Do = prep(RS.load("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01"))
    Dx = prep(RS.load("/tmp/model_v2/vs0050_fdx_extra.pkl", "2011-01-01", "2027-01-01")); Ex = [prep(nth_set(Dx, k)) for k in range(3)]
    res = {}
    for m in ("S1", "S2", "S3", "S4", "S5"):
        for wt in ("px", "eq"):
            T = pd.concat([sim(Do, m, wt), sim(Dn, m, wt)]); s = stats(T[["r", "b"]])
            tt = [stats(sim(E, m, wt)[["r", "b"]]) for E in Ex]
            res[f"{m} {wt}"] = {**s, "單檔最大占比中位數": round(float(T.wmax.median() * 100)), "三日": [(t["2011–2026"], t["2024-26"], t["最大回落"]) for t in tt],
                                "三日平均": (round(float(np.mean([t["2011–2026"][0] for t in tt])), 2), round(float(np.mean([t["最大回落"] for t in tt])), 1))}
            print(m, wt, json.dumps(res[f"{m} {wt}"], ensure_ascii=False, default=str), flush=True)
    rng = np.random.default_rng(20261124); RR = []
    for k in range(200):
        s = stats(pd.concat([sim(Do, "R", "px", rng=rng), sim(Dn, "R", "px", rng=rng)])[["r", "b"]]); RR.append((s["2011–2026"][0], s["最大回落"]))
    RR = np.array(RR); res["隨機3檔 1張"] = {"每月中位數": round(float(np.median(RR[:, 0])), 2), "回落中位數": round(float(np.median(RR[:, 1])), 1), "回落5%分位": round(float(np.percentile(RR[:, 1], 5)), 1)}
    print("R", res["隨機3檔 1張"])
    b = res["S1 px"]; chk = {}
    for m in ("S2", "S3", "S4", "S5"):
        s = res[f"{m} px"]
        c = {"①": bool(s["2011–2026"][0] >= b["2011–2026"][0] and s["2011–2026"][1] >= 2.9), "②": bool(all(s[k] > 0 for k in RS.SEGS)),
             "③": bool(s["最大回落"] >= b["最大回落"] + 3 or (s["2011–2026"][0] >= b["2011–2026"][0] + 0.3 and s["最大回落"] >= b["最大回落"])),
             "④": bool(s["三日平均"][1] >= b["三日平均"][1] + 3 or (s["三日平均"][0] >= b["三日平均"][0] + 0.3 and s["三日平均"][1] >= b["三日平均"][1]))}
        chk[m] = {"各項": c, "通過": all(c.values())}
    res["判斷"] = chk; print(json.dumps(chk, ensure_ascii=False))
    json.dump(res, open(f"{REPO}/research/results/lot_weight.json", "w"), ensure_ascii=False, indent=1, default=str)
