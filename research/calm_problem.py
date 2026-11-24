"""穩健版會不會挑到「跌少漲少、其實有問題」的股票（第 46 輪；2026/11/24 13:00 事先登錄，看結果前寫下，看到結果後不修改）

起因：使用者指出低波動可能是「大盤參與度低、緩跌」的問題股，緩跌有時比大跌可怕。
A 描述（不影響判斷）：
  A1 前 20 名每檔下個月報酬，依 60 日 beta（對 0050 指數）分三等分、依 20 日波動分三等分，看平均、中位數、跌 >10% 的比例。
  A2 穩健版挑到的股票，「緩跌」（當月跌 >10% 且持有期間日波動仍在前 20 名的低三分之一）有多常見、和隨機挑 3 檔比。
B 改良（每個各自和穩健版比）：
  B1 參與度：前 20 名先剔除 60 日 beta 最低的三分之一，再挑波動最低 3 檔。
  B2 趨勢：前 20 名先剔除收盤在 20 日均線下的，再挑波動最低 3 檔（不足就照原排序補）。
  B3 月中出場：持有中收盤跌破 60 日均線，隔天開盤賣出，剩下時間改抱 0050 到換股日。
  B4 月中出場：持有中收盤比買進價跌 15%，隔天開盤賣出改抱 0050。
回測：同 calm3.py（2006–09＋2011–2026，成本 0.6%，月中出場多扣 0.3% 單邊），並看三個換股日（第 1／3／5 個交易日）。
採用條件（各自）：① 2011–2026 每月比 0050 不少於穩健版 −0.3 且 t ≥ 2.9；② 五段都贏 0050；③ 最大回落不比穩健版差；
  ④ 每月多 0.3 以上，或最大回落淺 3 個百分點以上；⑤ 三個換股日平均也符合 ④ 的方向（不比穩健版差）。全過才改網站，多個過選最大回落最淺的。
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
import robust_select as RS
from wr_confirm import stats, nth_set
from calm3 import sim_calm

def prep(D):
    Cff, bidx = D["Cff"], np.asarray(D["bidx"], float)
    r = np.vstack([np.full((1, Cff.shape[1]), np.nan), Cff[1:] / Cff[:-1] - 1]); rb = np.r_[np.nan, bidx[1:] / bidx[:-1] - 1]
    D["_r"], D["_rb"] = r, rb
    D["_ma20"] = pd.DataFrame(Cff).rolling(20, min_periods=15).mean().values
    D["_ma60"] = pd.DataFrame(Cff).rolling(60, min_periods=45).mean().values
    return D

def beta(D, t, j, w=60):
    x, y = D["_rb"][t - w + 1:t + 1], D["_r"][t - w + 1:t + 1, j]; ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 40: return np.nan
    x, y = x[ok], y[ok]; return float(np.cov(x, y)[0, 1] / np.var(x, ddof=1))

def sim(D, mode, n=3, K=20, keep_rank=20, cost=0.6, cost_exit=0.3):
    O, Cff, lk, bidx, POS, CJ, ND = D["O"], D["Cff"], D["lockn"], np.asarray(D["bidx"], float), D["POS"], D["CJ"], D["ND"]
    G = {d: g for d, g in D["F"].groupby("date")}; rows, held, recs = [], [], []
    for a, b_ in zip(D["FDS"][:-1], D["FDS"][1:]):
        ie, ix = POS[a] + 1, POS[b_] + 1
        if ix >= ND: break
        g = G[a].sort_values("cs", ascending=False); rank = {c: i + 1 for i, c in enumerate(g.code)}; t = ie - 1
        top = g.head(K).copy()
        if mode == "B1":
            top["beta"] = [beta(D, t, CJ[c]) for c in top.code]; cut = top.beta.quantile(1 / 3)
            top = pd.concat([top[top.beta > cut], top[~(top.beta > cut)]]) ; elig = set(top.code[top.beta > cut])
        elif mode == "B2":
            elig = {c for c in top.code if Cff[t, CJ[c]] >= D["_ma20"][t, CJ[c]]}
        else: elig = set(top.code)
        keep = [c for c in held if rank.get(c, 1e9) <= keep_rank][:n]; new = []
        tv = top.assign(v=lambda z: z.vol20.fillna(np.inf))
        cand = tv[tv.code.isin(elig)].sort_values(["v", "cs"], ascending=[True, False]).code.tolist() + tv[~tv.code.isin(elig)].sort_values(["v", "cs"], ascending=[True, False]).code.tolist() + g.code.iloc[K:].tolist()
        for c in cand:
            if len(keep) + len(new) >= n: break
            if c in keep: continue
            j = CJ[c]
            if O[ie, j] > 0 and not lk[ie - 1, j]: new.append(c)
        to = 1.0 if not rows else len([c for c in held if c not in keep]) / n; rs, hn, extra = [], [], 0.0
        for c in keep + new:
            j = CJ[c]; o = O[ie, j] if O[ie, j] > 0 else Cff[ie - 1, j]
            ex = O[ix, j] if O[ix, j] > 0 else Cff[ix, j]
            if not np.isfinite(ex): ex = Cff[ix - 1, j]
            if not (np.isfinite(ex) and np.isfinite(o) and o > 0): continue
            r = (ex / o - 1) * 100; stop = None
            if mode in ("B3", "B4"):
                for tt in range(ie, ix - 1):
                    px = Cff[tt, j]
                    if not np.isfinite(px): continue
                    if (mode == "B3" and px < D["_ma60"][tt, j]) or (mode == "B4" and px < o * 0.85): stop = tt; break
            if stop is not None:
                so = O[stop + 1, j] if O[stop + 1, j] > 0 else Cff[stop + 1, j]
                r = ((so / o) * (bidx[ix - 1] / bidx[stop]) - 1) * 100 - cost_exit; extra += 1
            rs.append(r); hn.append(c)
            recs.append({"date": a, "code": c, "r": r, "vol": float(np.nanstd(D["_r"][ie:ix, j])) * 100})
        held = hn; rows.append({"date": a, "r": (np.mean(rs) if rs else 0) - cost * to, "b": (bidx[ix - 1] / bidx[ie - 1] - 1) * 100, "stops": extra})
    T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index); return T, pd.DataFrame(recs)

def describe(D):
    O, Cff, POS, CJ, ND = D["O"], D["Cff"], D["POS"], D["CJ"], D["ND"]; G = {d: g for d, g in D["F"].groupby("date")}; R = []
    for a, b_ in zip(D["FDS"][:-1], D["FDS"][1:]):
        ie, ix = POS[a] + 1, POS[b_] + 1
        if ix >= ND: break
        top = G[a].sort_values("cs", ascending=False).head(20); bm = (D["bidx"][ix - 1] / D["bidx"][ie - 1] - 1) * 100
        for c, v in zip(top.code, top.vol20):
            j = CJ[c]; o = O[ie, j]; ex = O[ix, j] if O[ix, j] > 0 else Cff[ix - 1, j]
            if o > 0 and np.isfinite(ex): R.append({"date": a, "code": c, "beta": beta(D, ie - 1, j), "vol": v, "r": (ex / o - 1) * 100, "ex": (ex / o - 1) * 100 - bm})
    R = pd.DataFrame(R); out = {}
    for k in ("beta", "vol"):
        R[k + "_q"] = R.groupby("date")[k].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["低", "中", "高"]))
        out[k] = {q: {"平均減0050": round(g.ex.mean(), 2), "中位數減0050": round(g.ex.median(), 2), "跌>10%比例": round((g.r < -10).mean() * 100, 1), "n": len(g)} for q, g in R.groupby(k + "_q")}
    lowboth = R[(R.beta_q == "低") & (R.vol_q == "低")]
    out["低beta且低波動"] = {"平均減0050": round(lowboth.ex.mean(), 2), "中位數減0050": round(lowboth.ex.median(), 2), "跌>10%比例": round((lowboth.r < -10).mean() * 100, 1), "n": len(lowboth)}
    return out

if __name__ == "__main__":
    Dn = prep(RS.load("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01")); Do = prep(RS.load("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01"))
    Dx = prep(RS.load("/tmp/model_v2/vs0050_fdx_extra.pkl", "2011-01-01", "2027-01-01")); Ex = [prep(nth_set(Dx, k)) for k in range(3)]
    out = {"A1 2011–2026": describe(Dn), "A1 2006–2009": describe(Do)}; print(json.dumps(out, ensure_ascii=False), flush=True)
    res = {}
    for mode in ("base", "B1", "B2", "B3", "B4"):
        To, ro = sim(Do, mode); Tn, rn = sim(Dn, mode); s = stats(pd.concat([To[["r", "b"]], Tn[["r", "b"]]]))
        tt = [stats(sim(E, mode)[0][["r", "b"]]) for E in Ex]
        rr = pd.concat([ro, rn]); slow = ((rr.r < -10)).mean() * 100
        res[mode] = {**s, "三日": [(t["2011–2026"], t["2024-26"], t["最大回落"]) for t in tt], "三日平均": (round(np.mean([t["2011–2026"][0] for t in tt]), 2), round(np.mean([t["最大回落"] for t in tt]), 1)),
                     "單檔當月跌>10%比例": round(slow, 1), "月中出場次數": int(pd.concat([To, Tn]).stops.sum())}
        print(mode, json.dumps(res[mode], ensure_ascii=False, default=str), flush=True)
    b = res["base"]; chk = {}
    for m in ("B1", "B2", "B3", "B4"):
        s = res[m]; c = {"①": s["2011–2026"][0] >= b["2011–2026"][0] - 0.3 and s["2011–2026"][1] >= 2.9, "②": all(s[k] > 0 for k in RS.SEGS), "③": s["最大回落"] >= b["最大回落"],
                          "④": s["2011–2026"][0] >= b["2011–2026"][0] + 0.3 or s["最大回落"] >= b["最大回落"] + 3,
                          "⑤": s["三日平均"][0] >= b["三日平均"][0] - 0.3 and s["三日平均"][1] >= b["三日平均"][1]}
        chk[m] = {"各項": c, "通過": all(c.values())}
    passed = [m for m in chk if chk[m]["通過"]]; best = max(passed, key=lambda m: res[m]["最大回落"]) if passed else None
    out.update({"B": res, "判斷": chk, "採用": best}); print(json.dumps(chk, ensure_ascii=False), "採用", best)
    json.dump(out, open(f"{REPO}/research/results/calm_problem.json", "w"), ensure_ascii=False, indent=1, default=str)
