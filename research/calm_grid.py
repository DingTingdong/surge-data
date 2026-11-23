"""穩健版的參數穩定度檢查（第 35 輪；2026/11/23 17:45 登錄，只檢查、不換參數；看結果前寫下）

目的：網站月選（穩健版）用「前 20 名裡近 20 日波動最低 3 檔」。要知道 20 名、20 日是不是剛好選到好的，換成鄰近的設定結果會不會差很多。
  另外依使用者要求，單獨看「一般盤勢」（換股日不在恐慌、持有期間也沒碰到恐慌）的表現。
網格：候選池 前 10／15／20／30／40 名 × 波動期間 20／60 日，其他規則不變（續抱＝還在候選池內；成本 0.6%）。資料 2006–2009＋2011–2026。
判斷（不影響網站，只記錄）：若網格中「每月減 0050」大多數 > +1.5%、且五段都 > 0 的設定佔一半以上 → 穩健版不是巧合；
  若只有 20／20 附近特別好 → 標示「參數敏感」，之後以前瞻紀錄為準。
另報：一般盤勢月份（持有期間沒有恐慌日）與恐慌月份的每月減 0050。
結果（11/23 17:55）：好設定只佔 3／10 → 標示「參數敏感」。
  20 日波動＋前 10／15／20 名：+1.87／+2.24／+2.35，五段都正，回落 −33／−22／−23%（這一區都好）；
  改 60 日波動：+1.01／+1.21／+1.97，2024–26 都 ≤ 0；前 30／40 名：+1.68／+0.56，2024–26 −1.68／−3.59。
  → 要同時「分數夠前面（前 20 內）」＋「用近 20 日波動」才好；以前瞻紀錄為準。
  一般盤勢（持有期間沒碰到恐慌）185 期：每月減 0050 +2.44，63% 的月份贏；恐慌 52 期 +2.08。
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import robust_select as RS
from wr_confirm import stats
from calm_panic import sim

def with_vol60(D, path):
    X = pd.read_pickle(path); C = pd.DataFrame(X["C"], index=X["days"], columns=X["cols"]); v = C.pct_change(fill_method=None).rolling(60, min_periods=45).std()
    pos = {d: i for i, d in enumerate(X["days"])}; cj = {c: j for j, c in enumerate(X["cols"])}; F = D["F"]
    F["vol60"] = v.values[F.date.map(pos).values, F.code.map(cj).values]; return D

def sim_pool(D, pool, vcol, cost=0.6, n=3):
    O, Cff, lk, bidx, POS, CJ, ND = D["O"], D["Cff"], D["lockn"], D["bidx"], D["POS"], D["CJ"], D["ND"]
    G = {d: g for d, g in D["F"].groupby("date")}; rows, held = [], []
    for a, b_ in zip(D["FDS"][:-1], D["FDS"][1:]):
        ie, ix = POS[a] + 1, POS[b_] + 1
        if ix >= ND: break
        g = G[a].sort_values("cs", ascending=False); top = g.head(pool); keep = [c for c in held if c in set(top.code)][:n]; new = []
        for c in top.assign(v=lambda z: z[vcol].fillna(np.inf)).sort_values(["v", "cs"], ascending=[True, False]).code.tolist() + g.code.iloc[pool:].tolist():
            if len(keep) + len(new) >= n: break
            j = CJ[c]
            if c not in keep and O[ie, j] > 0 and not lk[ie - 1, j]: new.append(c)
        to = 1.0 if not rows else len([c for c in held if c not in keep]) / n; rs, hn = [], []
        for c in keep + new:
            j = CJ[c]; o = O[ie, j] if O[ie, j] > 0 else Cff[ie - 1, j]; ex = O[ix, j] if O[ix, j] > 0 else Cff[ix, j]
            if not np.isfinite(ex): ex = Cff[ix - 1, j]
            if np.isfinite(ex) and np.isfinite(o) and o > 0: rs.append((ex / o - 1) * 100); hn.append(c)
        held = hn; rows.append({"date": a, "r": (np.mean(rs) if rs else 0) - cost * to, "b": (bidx[ix - 1] / bidx[ie - 1] - 1) * 100})
    T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index); return T

if __name__ == "__main__":
    P = {"n": "/tmp/model_v2/vs0050_extra.pkl", "o": "/tmp/model_v2/vs0050_oldestonly_extra.pkl"}
    Dn = with_vol60(RS.load(P["n"], "2011-01-01", "2027-01-01"), P["n"]); Do = with_vol60(RS.load(P["o"], "2006-01-01", "2010-01-01"), P["o"])
    out = {}; rows = []
    for pool in (10, 15, 20, 30, 40):
        for vcol in ("vol20", "vol60"):
            s = stats(pd.concat([sim_pool(Do, pool, vcol), sim_pool(Dn, pool, vcol)])); k = f"前{pool}名 {vcol}"; out[k] = s
            rows.append({"設定": k, "每月減0050": s["2011–2026"][0], "t": s["2011–2026"][1], **{x: s[x] for x in RS.SEGS}, "回落": s["最大回落"]})
    G = pd.DataFrame(rows); print(G.to_string(index=False))
    good = ((G["每月減0050"] > 1.5) & (G[RS.SEGS] > 0).all(axis=1)).mean(); out["好設定比例"] = round(float(good), 2); print("每月 > 1.5 且五段都正的設定比例", round(float(good), 2))
    # 一般盤勢 vs 恐慌（現行穩健版）
    TH = []
    for path, s_, e_ in ((P["o"], "2006-01-01", "2010-01-01"), (P["n"], "2011-01-01", "2027-01-01")):
        D = RS.load(path, s_, e_); pan = np.asarray(pd.read_pickle(path)["panic"]).astype(bool); TH.append(sim(D, pan, False))
    H = pd.concat(TH); x = H.r - H.b
    out["一般盤勢 vs 恐慌"] = {"一般盤勢月份": int((~H.pan).sum()), "一般盤勢 每月減0050": round(float(x[~H.pan].mean()), 2), "一般盤勢 贏0050月份%": round(float((x[~H.pan] > 0).mean() * 100)),
                          "恐慌月份": int(H.pan.sum()), "恐慌 每月減0050": round(float(x[H.pan].mean()), 2)}
    print(out["一般盤勢 vs 恐慌"])
    json.dump(out, open(f"{REPO}/research/results/calm_grid.json", "w"), ensure_ascii=False, indent=1, default=str)
