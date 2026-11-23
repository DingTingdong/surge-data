"""月選只買前 N 檔（2026/11/23，探索：使用者不可能買 20 檔，要縮減名單）。看過結果才決定網站用 3 檔。"""
import sys, numpy as np, pandas as pd; sys.path.insert(0,"/home/claude/repo/research"); sys.argv=["x"]
import robust_select as RS
from vs0050 import nw_t
def sim2(D, score, n, keep_rank, cost=0.6):
    O, Cff, lk, bidx, POS, CJ, ND = D["O"], D["Cff"], D["lockn"], D["bidx"], D["POS"], D["CJ"], D["ND"]
    G = {d: g for d, g in D["F"].groupby("date")}; rows, held = [], []
    for a, b_ in zip(D["FDS"][:-1], D["FDS"][1:]):
        ie, ix = POS[a] + 1, POS[b_] + 1
        if ix >= ND: break
        g = G[a].sort_values(score, ascending=False); rank = {c: i + 1 for i, c in enumerate(g.code)}
        keep = [c for c in held if rank.get(c, 1e9) <= keep_rank][:n]; rs, hn = [], []
        cand = [c for c in g.code if c not in keep]; new = []
        for c in cand:  # 買不到（開盤鎖漲停）就往下一名遞補
            if len(keep) + len(new) >= n: break
            j = CJ[c]
            if O[ie, j] > 0 and not lk[ie - 1, j]: new.append(c)
        sells = len([c for c in held if c not in keep]); to = 1.0 if not rows else sells / n
        for c in keep + new:
            j = CJ[c]; o = O[ie, j] if O[ie, j] > 0 else Cff[ie - 1, j]; ex = O[ix, j] if O[ix, j] > 0 else Cff[ix, j]
            if not np.isfinite(ex): ex = Cff[ix - 1, j]
            if np.isfinite(ex) and np.isfinite(o) and o > 0: rs.append((ex / o - 1) * 100); hn.append(c)
        held = hn; rows.append({"date": a, "r": (np.mean(rs) if rs else 0) - cost * to, "b": (bidx[ix - 1] / bidx[ie - 1] - 1) * 100, "to": to})
    T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index); return T
Dn = RS.load("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01"); Do = RS.load("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01")
rows = []
for n in (1, 3, 5, 10, 20):
    for kr in sorted({n, 20}):
        T = pd.concat([sim2(Do, "cs", n, kr), sim2(Dn, "cs", n, kr)]); x = T.r - T.b; sg = RS.seg(T.index); m = T.index >= "2011-01-01"
        w = (1 + T.r[m] / 100).cumprod(); wb = (1 + T.b[m] / 100).cumprod(); yrs = m.sum() / 12
        yr = pd.DataFrame({"s": (1 + T.r[m] / 100).groupby(T.index[m].year).prod(), "b": (1 + T.b[m] / 100).groupby(T.index[m].year).prod()})
        rows.append({"檔數": n, "續抱": f"前{kr}內", "11–26每月減0050": round(x[m].mean(), 2), "t": round(nw_t(x[m], 1), 2),
                     **{s: round(x[sg == s].mean(), 2) for s in RS.SEGS}, "每月贏0050%": round((x[m] > 0).mean() * 100), "贏的年數": f"{int((yr.s > yr.b).sum())}/{len(yr)}",
                     "年化": round((w.iloc[-1] ** (1 / yrs) - 1) * 100), "最大回落": round(((w / w.cummax() - 1).min()) * 100), "最差一個月減0050": round(x[m].min()), "每月換手%": round(T.to.mean() * 100)})
print(pd.DataFrame(rows).to_string(index=False))
print("0050 年化", round((wb.iloc[-1] ** (1 / yrs) - 1) * 100), "最大回落", round(((wb / wb.cummax() - 1).min()) * 100))
