"""3 檔不同產業（第 22 輪；2026/11/23 12:45 登錄，看結果前寫下，看到結果後不修改）

起因：網站月選每期只買 3 檔，最大回落 −49%（換股日晚幾天可到 −67%）。3 檔常集中在同一族群，族群一起跌時傷害最大。
作法：同月選 3 檔規則（前 3 名、開盤漲停買不到往下遞補、還在前 20 名續抱、成本 0.6%），但新買進時跳過和手上（含本期已選）同產業的股票，
  一路往下找到不同產業的為止（產業用 data/industry.csv 的現行分類）。續抱的不受限制。
資料：2006–2009＋2011–2026（五段）；換股時點用 vs0050_fdx_extra.pkl（11 日後第 1／3／5 個交易日）。
採用條件（全部符合才把網站改成「不同產業」）：
  ① 2011–2026 每月減 0050 平均不比原規則少 0.3 以上，t ≥ 2.9；② 五段都 > 0；③ 最大回落比原規則淺 5 個百分點以上；
  ④ 三個換股日的最大回落平均比原規則淺、且每月減 0050 平均不少 0.3 以上。
結果（11/23 12:50）：不通過 → 網站維持原規則。不同產業 +2.64（t 2.88）vs 原 +3.23；五段都 > 0 但都較低；最大回落 −50 vs −49；
  三個換股日回落平均 −57 vs −54 → 分散產業沒有降低回落，反而少賺（大跌時是全部一起跌，不是單一族群）。
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import robust_select as RS
from vs0050 import nw_t
from wr_confirm import sim2, stats, nth_set

IND = pd.read_csv(f"{REPO}/data/industry.csv", dtype={"code": str}).set_index("code")["ind_name"].to_dict()

def sim_div(D, score="cs", n=3, keep_rank=20, cost=0.6):
    O, Cff, lk, bidx, POS, CJ, ND = D["O"], D["Cff"], D["lockn"], D["bidx"], D["POS"], D["CJ"], D["ND"]
    G = {d: g for d, g in D["F"].groupby("date")}; rows, held = [], []
    for a, b_ in zip(D["FDS"][:-1], D["FDS"][1:]):
        ie, ix = POS[a] + 1, POS[b_] + 1
        if ix >= ND: break
        g = G[a].sort_values(score, ascending=False); rank = {c: i + 1 for i, c in enumerate(g.code)}
        keep = [c for c in held if rank.get(c, 1e9) <= keep_rank][:n]; new = []; used = {IND.get(c, "?" + c) for c in keep}
        for c in g.code:
            if len(keep) + len(new) >= n: break
            if c in keep: continue
            j = CJ[c]; ind = IND.get(c, "?" + c)
            if ind in used or not (O[ie, j] > 0) or lk[ie - 1, j]: continue
            new.append(c); used.add(ind)
        to = 1.0 if not rows else len([c for c in held if c not in keep]) / n; rs, hn = [], []
        for c in keep + new:
            j = CJ[c]; o = O[ie, j] if O[ie, j] > 0 else Cff[ie - 1, j]; ex = O[ix, j] if O[ix, j] > 0 else Cff[ix, j]
            if not np.isfinite(ex): ex = Cff[ix - 1, j]
            if np.isfinite(ex) and np.isfinite(o) and o > 0: rs.append((ex / o - 1) * 100); hn.append(c)
        held = hn; rows.append({"date": a, "r": (np.mean(rs) if rs else 0) - cost * to, "b": (bidx[ix - 1] / bidx[ie - 1] - 1) * 100})
    T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index); return T

if __name__ == "__main__":
    Dn = RS.load("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01"); Do = RS.load("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01")
    A = stats(pd.concat([sim2(Do, "cs", 3), sim2(Dn, "cs", 3)])); B = stats(pd.concat([sim_div(Do), sim_div(Dn)])); out = {"原規則": A, "不同產業": B}; print("原規則", A, "\n不同產業", B, flush=True)
    Dx = RS.load("/tmp/model_v2/vs0050_fdx_extra.pkl", "2011-01-01", "2027-01-01"); tim = []
    for k in range(3):
        E = nth_set(Dx, k); a, b = stats(sim2(E, "cs", 3)), stats(sim_div(E)); tim.append({"原": a, "不同產業": b}); print(f"第{2 * k + 1}個交易日", "原", a, "不同產業", b, flush=True)
    out["換股時點"] = tim
    ma = np.mean([t["原"]["最大回落"] for t in tim]); mb = np.mean([t["不同產業"]["最大回落"] for t in tim]); xa = np.mean([t["原"]["2011–2026"][0] for t in tim]); xb = np.mean([t["不同產業"]["2011–2026"][0] for t in tim])
    c = {"①": B["2011–2026"][0] >= A["2011–2026"][0] - 0.3 and B["2011–2026"][1] >= 2.9, "②": all(B[s] > 0 for s in RS.SEGS),
         "③": B["最大回落"] >= A["最大回落"] + 5, "④": mb > ma and xb >= xa - 0.3}
    out["三日平均 回落 原/不同"] = (round(ma), round(mb)); out["三日平均 減0050 原/不同"] = (round(xa, 2), round(xb, 2)); out["各項"] = c; out["通過"] = all(c.values())
    print(out["三日平均 回落 原/不同"], out["三日平均 減0050 原/不同"], "各項", c, "通過" if all(c.values()) else "不通過")
    json.dump(out, open(f"{REPO}/research/results/div3.json", "w"), ensure_ascii=False, indent=1, default=str)
