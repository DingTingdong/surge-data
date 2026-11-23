"""穩健版降低運氣成分（第 39、40 輪；2026/11/23 23:50 登錄，看結果前寫下，看到結果後不修改）

起因：第 35 輪發現穩健版參數敏感（前 10／15／20 名都好，其他變差），第 20 輪發現換股日差兩天結果差很多。使用者只能買 3 檔，
  所以不能靠多買來分散，改用：
  第 39 輪 E 合併設定：前 20 名裡每檔算「在前 10、前 15、前 20 三個候選池中，是不是池內近 20 日波動最低的 3 檔」的票數，
    依票數多到少、再依波動低到高挑 3 檔（續抱＝還在前 20 名）。
  第 40 輪 T 分兩天進出：挑法和穩健版完全一樣（同 3 檔），但每檔資金一半在換股日隔天開盤買、一半再晚 2 個交易日開盤買；
    賣出也一樣分兩天（下次換股日隔天、再晚 2 天）。
資料：2006–2009＋2011–2026；成本每次賣出 0.6%。
採用條件（各自，全部符合才改網站）：
  ① 2011–2026 每月減 0050 不比穩健版少 0.3 以上，t ≥ 2.9；② 五段都 > 0；
  ③ E：換股日改第 3、第 5 個交易日時（vs0050_fdx_extra.pkl）三個換股日的平均每月減 0050 比穩健版高、最大回落平均不比穩健版深；
     T：最大回落比穩健版淺，且 2024–26 不比穩健版差 0.5 以上。
結果（11/23 23:55）：兩個都不通過 → 網站維持穩健版。
  E 合併設定：+2.18（t 3.49），2016–19 只 +0.83、2024–26 +1.48，回落 −21%；三個換股日平均 +1.32、回落 −33%（穩健版 +1.70、−28%）→ 更差。
  T 分兩天進出：+2.13（t 4.17），五段都正，但回落 −25%（穩健版 −23%）、2024–26 +2.00 → 沒有比較穩。
  另：穩健版三個換股日平均每月 +1.70%（第 1 天 +2.35、第 3 天 +1.05、第 5 天 +1.71）→ 較保守的期望值約 +1.7%／月。
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import robust_select as RS
from calm3 import sim_calm
from wr_confirm import stats, nth_set

def picks_calm(g, held, O, lk, CJ, ie, mode):
    top = g.head(20); keep = [c for c in held if c in set(top.code)][:3]
    if mode == "E":
        v = top.set_index("code").vol20.fillna(np.inf); votes = pd.Series(0, index=v.index)
        for k in (10, 15, 20): votes[v.loc[list(top.code[:k])].nsmallest(3).index] += 1
        order = sorted(v.index, key=lambda c: (-votes[c], v[c]))
    else:
        order = top.assign(v=lambda z: z.vol20.fillna(np.inf)).sort_values(["v", "cs"], ascending=[True, False]).code.tolist()
    new = []
    for c in order + g.code.iloc[20:].tolist():
        if len(keep) + len(new) >= 3: break
        j = CJ[c]
        if c not in keep and O[ie, j] > 0 and not lk[ie - 1, j]: new.append(c)
    return keep, new

def sim(D, mode="calm", split=0, cost=0.6):
    O, Cff, lk, bidx, POS, CJ, ND = D["O"], D["Cff"], D["lockn"], D["bidx"], D["POS"], D["CJ"], D["ND"]
    G = {d: g.sort_values("cs", ascending=False) for d, g in D["F"].groupby("date")}; rows, held = [], []
    for a, b_ in zip(D["FDS"][:-1], D["FDS"][1:]):
        ie, ix = POS[a] + 1, POS[b_] + 1
        if ix + split >= ND: break
        keep, new = picks_calm(G[a], held, O, lk, CJ, ie, "E" if mode == "E" else "calm")
        to = 1.0 if not rows else len([c for c in held if c not in keep]) / 3; rs, hn = [], []
        for c in keep + new:
            j = CJ[c]; sub = []
            for off in ((0, split) if split else (0,)):
                o = O[ie + off, j] if O[ie + off, j] > 0 else Cff[ie + off - 1, j]; ex = O[ix + off, j] if O[ix + off, j] > 0 else Cff[ix + off, j]
                if not np.isfinite(ex): ex = Cff[ix + off - 1, j]
                if np.isfinite(ex) and np.isfinite(o) and o > 0: sub.append((ex / o - 1) * 100)
            if sub: rs.append(np.mean(sub)); hn.append(c)
        b = np.mean([(bidx[ix + off - 1] / bidx[ie + off - 1] - 1) * 100 for off in ((0, split) if split else (0,))])
        held = hn; rows.append({"date": a, "r": (np.mean(rs) if rs else 0) - cost * to, "b": b})
    T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index); return T

if __name__ == "__main__":
    Dn = RS.load("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01"); Do = RS.load("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01")
    B = stats(pd.concat([sim(Do), sim(Dn)])); E = stats(pd.concat([sim(Do, "E"), sim(Dn, "E")])); T = stats(pd.concat([sim(Do, split=2), sim(Dn, split=2)]))
    out = {"穩健版": B, "E 合併設定": E, "T 分兩天進出": T}
    Dx = RS.load("/tmp/model_v2/vs0050_fdx_extra.pkl", "2011-01-01", "2027-01-01"); tb, te = [], []
    for k in range(3): Ek = nth_set(Dx, k); tb.append(stats(sim(Ek))); te.append(stats(sim(Ek, "E")))
    out["換股日第1/3/5天 穩健版"] = tb; out["換股日第1/3/5天 E"] = te
    mb = (np.mean([x["2011–2026"][0] for x in tb]), np.mean([x["最大回落"] for x in tb])); me = (np.mean([x["2011–2026"][0] for x in te]), np.mean([x["最大回落"] for x in te]))
    ok = lambda s: s["2011–2026"][0] >= B["2011–2026"][0] - 0.3 and s["2011–2026"][1] >= 2.9 and all(s[k] > 0 for k in RS.SEGS)
    out["E 通過"] = bool(ok(E) and me[0] > mb[0] and me[1] >= mb[1]); out["T 通過"] = bool(ok(T) and T["最大回落"] > B["最大回落"] and T["2024-26"] >= B["2024-26"] - 0.5)
    out["三日平均 穩健版/E（減0050, 回落）"] = (tuple(round(v, 2) for v in mb), tuple(round(v, 2) for v in me))
    for k, v in out.items(): print(k, json.dumps(v, ensure_ascii=False, default=str), flush=True)
    json.dump(out, open(f"{REPO}/research/results/calm_ensemble.json", "w"), ensure_ascii=False, indent=1, default=str)
