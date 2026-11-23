"""穩健版 3 檔：大跌中要不要先換 0050（第 26 輪；2026/11/23 12:45 登錄，看結果前寫下，看到結果後不修改）

起因：網站寫「大跌中照常抱到換股日」，依據是 20 檔版的檢驗（第 9 輪）；網站已改成穩健版 3 檔，要重新確認。
作法（穩健版規則不變：前 20 名裡近 20 日波動最低 3 檔、還在前 20 名續抱、成本每次賣股 0.6%）：
  H 照常抱到換股日（現行）。
  P 持有期間第一次出現恐慌日（網站的恐慌定義，vs0050 快照的 panic）→ 隔天開盤全部賣掉換 0050，抱到下次換股日（0050 進出各扣 0.1%）；
    下次換股日照常重新選股（上期賣掉的不算續抱，要重新買進）。換股日當天已在恐慌中 → 照常買（同網站）。
資料：2006–2009＋2011–2026（五段）。
採用條件（P 全部符合才把網站說明改成「大跌時換 0050」）：
  ① 2011–2026 每月減 0050，P 減 H 平均 > 0 且 Newey-West t ≥ 2；② 五段中至少四段 P ≥ H；③ P 的最大回落比 H 淺。
另報：持有期間有恐慌的月份數、這些月份 H 與 P 的平均。
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import robust_select as RS
from vs0050 import nw_t
from wr_confirm import stats

def sim(D, panic, sell_on_panic, n=3, keep_rank=20, cost=0.6):
    O, Cff, lk, bidx, POS, CJ, ND = D["O"], D["Cff"], D["lockn"], D["bidx"], D["POS"], D["CJ"], D["ND"]
    G = {d: g for d, g in D["F"].groupby("date")}; rows, held = [], []
    for a, b_ in zip(D["FDS"][:-1], D["FDS"][1:]):
        ie, ix = POS[a] + 1, POS[b_] + 1
        if ix >= ND: break
        g = G[a].sort_values("cs", ascending=False); rank = {c: i + 1 for i, c in enumerate(g.code)}
        keep = [c for c in held if rank.get(c, 1e9) <= keep_rank][:n]; new = []
        for c in g.head(keep_rank).assign(v=lambda z: z.vol20.fillna(np.inf)).sort_values(["v", "cs"], ascending=[True, False]).code.tolist() + g.code.iloc[keep_rank:].tolist():
            if len(keep) + len(new) >= n: break
            j = CJ[c]
            if c not in keep and O[ie, j] > 0 and not lk[ie - 1, j]: new.append(c)
        to = 1.0 if not rows else len([c for c in held if c not in keep]) / n
        pd_ = [p for p in range(ie, ix - 1) if panic[p]] if sell_on_panic else []; p0 = pd_[0] if pd_ else None
        rs, hn = [], []
        for c in keep + new:
            j = CJ[c]; o = O[ie, j] if O[ie, j] > 0 else Cff[ie - 1, j]
            if p0 is not None:
                ex = O[p0 + 1, j] if O[p0 + 1, j] > 0 else Cff[p0 + 1, j]
                if not np.isfinite(ex): ex = Cff[p0, j]
                if np.isfinite(ex) and o > 0: rs.append(((ex / o) * (1 - cost / 100 - 0.001) * (bidx[ix - 1] / bidx[p0]) * (1 - 0.001) - 1) * 100)
            else:
                ex = O[ix, j] if O[ix, j] > 0 else Cff[ix, j]
                if not np.isfinite(ex): ex = Cff[ix - 1, j]
                if np.isfinite(ex) and o > 0: rs.append((ex / o - 1) * 100); hn.append(c)
        held = hn if p0 is None else []
        # 恐慌賣出的成本已在上面扣；下一期因 held=[]，不會再對這些股票扣一次
        rows.append({"date": a, "r": (np.mean(rs) if rs else 0) - cost * to, "b": (bidx[ix - 1] / bidx[ie - 1] - 1) * 100, "pan": p0 is not None or any(panic[p] for p in range(ie, ix - 1))})
    T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index); return T

if __name__ == "__main__":
    out = {}; TH, TP = [], []
    for path, s, e in (("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01"), ("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01")):
        D = RS.load(path, s, e); pan = np.asarray(pd.read_pickle(path)["panic"]).astype(bool)
        TH.append(sim(D, pan, False)); TP.append(sim(D, pan, True))
    H, P = pd.concat(TH), pd.concat(TP); sH, sP = stats(H), stats(P); d = (P.r - H.r); sg = RS.seg(d.index); m = d.index >= "2011-01-01"
    pm = H.pan.values
    out = {"H 照常抱": sH, "P 恐慌換0050": sP, "P減H 2011–26": (round(float(d[m].mean()), 2), round(nw_t(d[m], 1), 2)), "P減H 五段": {s: round(float(d[sg == s].mean()), 2) for s in RS.SEGS},
           "有恐慌的月份": int(pm.sum()), "這些月份 H/P 減0050": (round(float((H.r - H.b)[pm].mean()), 2), round(float((P.r - P.b)[pm].mean()), 2))}
    c = {"①": out["P減H 2011–26"][0] > 0 and out["P減H 2011–26"][1] >= 2, "②": sum(v >= 0 for v in out["P減H 五段"].values()) >= 4, "③": sP["最大回落"] > sH["最大回落"]}
    out["各項"] = c; out["通過"] = all(c.values()); print(json.dumps(out, ensure_ascii=False, default=str))
    json.dump(out, open(f"{REPO}/research/results/calm_panic.json", "w"), ensure_ascii=False, indent=1, default=str)
