"""分批買 3 檔：降低換股日運氣（第 21 輪；2026/11/23 12:25 登錄，看結果前寫下，看到結果後不修改）

起因：第 20 輪發現月選 3 檔很看換股日：11 日後第 1／3／5 個交易日換股，2024–26 每月減 0050 +1.14／−1.50／−3.59，最大回落 −49／−45／−67%。
  使用者只能買幾檔，所以不能靠多買來分散；改成把資金分 3 份、在不同日子各買 1 檔，看能不能降低運氣成分。
作法（月選分數、2011–2026、成本每次賣出 0.6%）：
  S 分批：資金分 3 份，第 1 份在每月 11 日後第 1 個交易日、第 2 份第 3 個、第 3 份第 5 個交易日換股；每份只持有 1 檔：
    當天收盤後，手上那檔還在前 20 名就續抱，否則隔天開盤賣掉，換成分數最高、其他兩份沒持有、開盤買得到的那檔。三份各自複利，不互相調整。
  對照：3 檔一次買（同第 20 輪的規則）在三個換股日的結果平均（代表「不知道哪天換比較好」時的期望）。
資料：/tmp/model_v2/vs0050_fdx_extra.pkl（FD_OFFSETS=0,2,4）。
採用條件（S 全部符合才把網站改成分批）：
  ① 2011–2026 每月減 0050 平均 ≥ 對照平均，且 Newey-West t（落後 1）≥ 2.9；② 四段（2011–15、2016–19、2020–23、2024–26）都 > 0；
  ③ 最大回落比對照三個的平均淺；④ 成本 1.1% 時每月減 0050 仍 > 0、t ≥ 2。
另報：每份單獨的結果、每份每年換股次數。
結果（11/23 12:35）：不通過（② 2024–26 −0.17、③ 回落 −58% 比對照平均 −54% 深）→ 網站維持一次買 3 檔。
  分批 +3.42（t 3.23）四段 2.94／2.50／7.50／−0.17；對照（3 檔一次買，三個換股日平均）+2.77，2024–26 −1.32，回落平均 −54%。
  每份單獨：第 1 個交易日那份 +5.79（四段 5.94／5.72／8.67／1.57），第 3 個 +2.65，第 5 個 +1.83（2024–26 −2.99）
  → 優勢集中在月營收公布後馬上換股，晚幾天就少很多；最近兩年 3 檔版不論哪天換，平均大約和 0050 打平或略輸。
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import robust_select as RS
from vs0050 import nw_t, per, PER
from wr_confirm import sim2, nth_set

def stagger(D, score="cs", cost=0.6, keep_rank=20):
    O, Cff, lk, bidx, POS, CJ, ND, F = D["O"], D["Cff"], D["lockn"], D["bidx"], D["POS"], D["CJ"], D["ND"], D["F"]
    d = pd.Series(sorted(F.date.unique())); grp = d.groupby(d.dt.to_period("M"))
    ev = sorted([(pd.Timestamp(x), k) for k in range(3) for x in grp.nth(k)]); G = {x: g.sort_values(score, ascending=False) for x, g in F.groupby("date")}
    hold = [None, None, None]; recs = {0: [], 1: [], 2: []}; nsw = [0, 0, 0]
    for dt, k in ev:
        ie = POS[dt] + 1
        if ie >= ND: break
        g = G[dt]; top = list(g.code[:keep_rank]); h = hold[k]
        if h is not None:
            j = CJ[h["c"]]; ex = O[ie, j] if O[ie, j] > 0 else Cff[ie, j]
            if not np.isfinite(ex): ex = Cff[ie - 1, j]
            keep = h["c"] in top and np.isfinite(Cff[ie, j])
            r = (ex / h["px"] - 1) * 100; b = (bidx[ie - 1] / bidx[h["i"] - 1] - 1) * 100
            if keep:  # 續抱：這一段先結算（不扣成本），從今天開盤重新起算
                recs[k].append({"date": h["d"], "r": r - h["cost"], "b": b}); hold[k] = {"c": h["c"], "px": ex, "i": ie, "d": dt, "cost": 0.0}; continue
            recs[k].append({"date": h["d"], "r": r - h["cost"] - cost, "b": b}); hold[k] = None; nsw[k] += 1
        others = {x["c"] for i, x in enumerate(hold) if i != k and x is not None}
        for c in g.code:
            j = CJ[c]
            if c in others or not (O[ie, j] > 0) or lk[ie - 1, j]: continue
            hold[k] = {"c": c, "px": O[ie, j], "i": ie, "d": dt, "cost": 0.0}; break
    T = {k: pd.DataFrame(v).set_index("date") for k, v in recs.items()}
    for k in T: T[k].index = pd.DatetimeIndex(T[k].index); T[k]["m"] = T[k].index.to_period("M")
    # 三份各自複利；每月的組合報酬＝三份當月報酬平均（各份權重 1/3，按月近似）
    M = pd.concat([T[k].set_index("m")[["r", "b"]].add_suffix(str(k)) for k in T], axis=1).dropna()
    M["r"] = M[["r0", "r1", "r2"]].mean(axis=1); M["b"] = M[["b0", "b1", "b2"]].mean(axis=1); M.index = M.index.to_timestamp()
    yrs = (F.date.max() - F.date.min()).days / 365.25
    return M, {f"第{k + 1}份每年換股": round(nsw[k] / yrs, 1) for k in range(3)}

def stats(M):
    x = M.r - M.b; pr = per(x.index); w = (1 + M.r / 100).cumprod()
    return {"2011–2026": (round(float(x.mean()), 2), round(nw_t(x, 1), 2)), **{p: round(float(x[pr == p].mean()), 2) for p in PER},
            "最大回落": round(float((w / w.cummax() - 1).min() * 100)), "年化": round(float(w.iloc[-1] ** (12 / len(w)) - 1) * 100)}

if __name__ == "__main__":
    Dx = RS.load("/tmp/model_v2/vs0050_fdx_extra.pkl", "2011-01-01", "2027-01-01"); out = {}
    ref = [stats(sim2(nth_set(Dx, k), "cs", 3)) for k in range(3)]
    avg = {"2011–2026": round(float(np.mean([r["2011–2026"][0] for r in ref])), 2), **{p: round(float(np.mean([r[p] for r in ref])), 2) for p in PER},
           "最大回落": round(float(np.mean([r["最大回落"] for r in ref])))}
    out["對照 3 檔一次買（第1/3/5個交易日）"] = ref; out["對照平均"] = avg; print("對照", ref, "\n平均", avg, flush=True)
    M, info = stagger(Dx); s = stats(M); M11, _ = stagger(Dx, cost=1.1); s11 = stats(M11)
    each = {k: stats(M[[f"r{k}", f"b{k}"]].rename(columns={f"r{k}": "r", f"b{k}": "b"})) for k in range(3)}
    out["S 分批"] = {**s, **info}; out["S 每份單獨"] = each; out["S 成本1.1%"] = s11; print("S 分批", s, info, "\n每份", each, "\n1.1%", s11, flush=True)
    c = {"①": s["2011–2026"][0] >= avg["2011–2026"] and s["2011–2026"][1] >= 2.9, "②": all(s[p] > 0 for p in PER),
         "③": s["最大回落"] > avg["最大回落"], "④": s11["2011–2026"][0] > 0 and s11["2011–2026"][1] >= 2}
    out["各項"] = c; out["通過"] = all(c.values()); print("各項", c, "通過" if all(c.values()) else "不通過")
    json.dump(out, open(f"{REPO}/research/results/stagger.json", "w"), ensure_ascii=False, indent=1, default=str)
