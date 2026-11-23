"""前段班裡挑波動低的 3 檔（第 24 輪；2026/11/23 12:15 登錄，看結果前寫下，看到結果後不修改）

起因：網站月選只買 3 檔，最大回落 −49%（換股日晚幾天可到 −67%），單檔暴跌的傷害很大。第 10–13 輪的特性檢查裡，
  「20 日波動低」「20 日最大單日漲幅低」在五段都和下個月報酬正相關（t ≥ 3），但放進綜合分數沒有幫助；這輪改成只用來在前段班裡挑。
作法（月選分數、續抱＝還在前 20 名、成本每次賣出 0.6%、開盤漲停買不到就換下一檔）：
  V0 現行：分數前 3 名。
  V1 分數前 20 名裡，挑 20 日波動（日報酬標準差）最低的 3 檔。
  V2 分數前 10 名裡，挑 20 日波動最低的 3 檔。
資料：2006–2009＋2011–2026（五段）；換股時點用 vs0050_fdx_extra.pkl（第 1／3／5 個交易日）。
採用條件（V1 或 V2 全部符合才換；兩個都過選回落較淺的）：
  ① 2011–2026 每月減 0050 不比 V0 少 0.3 以上，t ≥ 2.9；② 五段都 > 0；③ 最大回落比 V0 淺 5 個百分點以上；
  ④ 三個換股日的最大回落平均比 V0 淺 5 個百分點以上，且每月減 0050 平均不比 V0 少 0.3 以上。
結果（11/23 12:20）：兩個都不通過（① 平均少超過 0.3、④ 三日平均少超過 0.3）→ 依規則網站不換；但呈現明顯的「少賺、少跌」取捨：
  V1 前 20 挑低波動：+2.35（t 4.40，比 V0 的 3.45 高），五段 2.42／2.50／1.96／2.57／2.32，最大回落 −23%（V0 −49%、0050 −30%），年化 49%；
     三個換股日 +2.35／+1.05／+1.71，2024–26 +2.32／+0.76／+0.27，回落 −23／−30／−31（V0 −49／−45／−67）。
  V2 前 10 挑低波動：+1.99（t 2.94），回落 −33%；三日 2024–26 +1.99／+0.08／0.00。
  → 是否改用 V1 屬於風險偏好的選擇（每月少約 1%，回落少一半、2024–26 三個換股日都贏），交給使用者決定。
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import robust_select as RS
from vs0050 import nw_t
from wr_confirm import sim2, stats, nth_set

def sim_calm(D, K, n=3, keep_rank=20, cost=0.6):
    O, Cff, lk, bidx, POS, CJ, ND = D["O"], D["Cff"], D["lockn"], D["bidx"], D["POS"], D["CJ"], D["ND"]
    G = {d: g for d, g in D["F"].groupby("date")}; rows, held = [], []
    for a, b_ in zip(D["FDS"][:-1], D["FDS"][1:]):
        ie, ix = POS[a] + 1, POS[b_] + 1
        if ix >= ND: break
        g = G[a].sort_values("cs", ascending=False); rank = {c: i + 1 for i, c in enumerate(g.code)}
        keep = [c for c in held if rank.get(c, 1e9) <= keep_rank][:n]; new = []
        cand = g.head(K).assign(v=lambda z: z.vol20.fillna(np.inf)).sort_values(["v", "cs"], ascending=[True, False]).code.tolist() + g.code.iloc[K:].tolist()
        for c in cand:
            if len(keep) + len(new) >= n: break
            if c in keep: continue
            j = CJ[c]
            if O[ie, j] > 0 and not lk[ie - 1, j]: new.append(c)
        to = 1.0 if not rows else len([c for c in held if c not in keep]) / n; rs, hn = [], []
        for c in keep + new:
            j = CJ[c]; o = O[ie, j] if O[ie, j] > 0 else Cff[ie - 1, j]; ex = O[ix, j] if O[ix, j] > 0 else Cff[ix, j]
            if not np.isfinite(ex): ex = Cff[ix - 1, j]
            if np.isfinite(ex) and np.isfinite(o) and o > 0: rs.append((ex / o - 1) * 100); hn.append(c)
        held = hn; rows.append({"date": a, "r": (np.mean(rs) if rs else 0) - cost * to, "b": (bidx[ix - 1] / bidx[ie - 1] - 1) * 100})
    T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index); return T

if __name__ == "__main__":
    Dn = RS.load("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01"); Do = RS.load("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01")
    Dx = RS.load("/tmp/model_v2/vs0050_fdx_extra.pkl", "2011-01-01", "2027-01-01"); Ex = [nth_set(Dx, k) for k in range(3)]
    V0 = stats(pd.concat([sim2(Do, "cs", 3), sim2(Dn, "cs", 3)])); t0 = [stats(sim2(E, "cs", 3)) for E in Ex]; out = {"V0": V0, "V0 三個換股日": t0}; print("V0", V0, flush=True)
    m0, x0 = np.mean([t["最大回落"] for t in t0]), np.mean([t["2011–2026"][0] for t in t0]); best = None
    for lab, K in (("V1 前20挑低波動", 20), ("V2 前10挑低波動", 10)):
        s = stats(pd.concat([sim_calm(Do, K), sim_calm(Dn, K)])); tt = [stats(sim_calm(E, K)) for E in Ex]
        m1, x1 = np.mean([t["最大回落"] for t in tt]), np.mean([t["2011–2026"][0] for t in tt])
        c = {"①": s["2011–2026"][0] >= V0["2011–2026"][0] - 0.3 and s["2011–2026"][1] >= 2.9, "②": all(s[k] > 0 for k in RS.SEGS),
             "③": s["最大回落"] >= V0["最大回落"] + 5, "④": m1 >= m0 + 5 and x1 >= x0 - 0.3}
        out[lab] = {**s, "三個換股日": tt, "三日平均 回落/減0050": (round(m1), round(x1, 2)), "V0 三日平均": (round(m0), round(x0, 2)), "各項": c, "通過": all(c.values())}
        print(lab, s, "\n  三日", [(t["2011–2026"], t["2024-26"], t["最大回落"]) for t in tt], "平均", (round(m1), round(x1, 2)), "V0", (round(m0), round(x0, 2)), c, flush=True)
        if all(c.values()) and (best is None or s["最大回落"] > out[best]["最大回落"]): best = lab
    out["採用"] = best; print("採用", best)
    json.dump(out, open(f"{REPO}/research/results/calm3.json", "w"), ensure_ascii=False, indent=1, default=str)
