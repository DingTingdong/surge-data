"""每檔 1 張時剔除高價股（第 49 輪；2026/11/24 23:50 事先登錄，看結果前寫下，看到結果後不修改）

起因：第 48 輪每檔 1 張時，高價股常占一半以上，最大回落 −34%；使用者提議把太高價的剔除。
做法：穩健版（前 20 名挑 20 日波動最低 3 檔），但換股日先剔除不符合條件的再挑；續抱的不因價格賣出（只影響新買）。每檔 1 張權重、成本 0.6%。
  C1 股價 > 100 元剔除；C2 > 150 元；C3 > 200 元；
  C4 相對上限：3 檔中最高價不超過最低價的 2 倍（依波動由低到高逐一加入，會讓最高價超過已選最低價 2 倍、或最低價低於已選最高價一半的跳過）。
  另報：穩健版每檔 1 張（基準）與等金額；每月報酬分布（最好／最差月、賠錢月比例、逐年）。
判斷：某條件要採用，須 ① 每月比 0050 不少於基準 −0.3 且 t ≥ 2.9；② 五段都贏 0050；③ 最大回落比基準淺 3 個百分點以上；④ 三個換股日平均回落也淺 3 以上。
  多個通過選回落最淺的；通過 → 網站月選加「每檔 1 張」版本的名單說明。
注意：絕對價格上限在 2006 與 2026 意義不同（股價水準變了），所以另列 C4 相對版本。
股價：權重與上限用當時實際股價（未還原的原始開盤價）。第 48 輪的「1 張」權重用的是還原價，這輪的基準會重算。
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
import robust_select as RS
from wr_confirm import stats, nth_set

def sim(D, cap=None, rel=None, weight="px", n=3, K=20, keep_rank=20, cost=0.6):
    O, Cff, lk, bidx, POS, CJ, ND = D["O"], D["Cff"], D["lockn"], np.asarray(D["bidx"], float), D["POS"], D["CJ"], D["ND"]
    RAW = D.get("rawO"); G = {d: g for d, g in D["F"].groupby("date")}; rows, held = [], []
    for a, b_ in zip(D["FDS"][:-1], D["FDS"][1:]):
        ie, ix = POS[a] + 1, POS[b_] + 1
        if ix >= ND: break
        g = G[a].sort_values("cs", ascending=False); rank = {c: i + 1 for i, c in enumerate(g.code)}
        keep = [c for c in held if rank.get(c, 1e9) <= keep_rank][:n]; new = []
        cand = g.head(K).assign(v=lambda z: z.vol20.fillna(np.inf)).sort_values(["v", "cs"], ascending=[True, False]).code.tolist() + g.code.iloc[K:].tolist()
        px = lambda c: (RAW[ie, CJ[c]] if RAW is not None and np.isfinite(RAW[ie, CJ[c]]) else O[ie, CJ[c]])
        for c in cand:
            if len(keep) + len(new) >= n: break
            if c in keep: continue
            j = CJ[c]
            if not (O[ie, j] > 0 and not lk[ie - 1, j]): continue
            p = px(c)
            if cap is not None and p > cap: continue
            if rel is not None:
                ps = [px(x) for x in keep + new]
                if ps and (p > rel * min(ps) or p < max(ps) / rel): continue
            new.append(c)
        to = 1.0 if not rows else len([c for c in held if c not in keep]) / n; rs, ws, hn = [], [], []
        for c in keep + new:
            j = CJ[c]; o = O[ie, j] if O[ie, j] > 0 else Cff[ie - 1, j]; ex = O[ix, j] if O[ix, j] > 0 else Cff[ix, j]
            if not np.isfinite(ex): ex = Cff[ix - 1, j]
            if np.isfinite(ex) and np.isfinite(o) and o > 0: rs.append((ex / o - 1) * 100); ws.append(px(c) if weight == "px" else 1.0); hn.append(c)
        held = hn; w = np.array(ws, float) / np.sum(ws) if ws else np.array([])
        rows.append({"date": a, "r": (float(np.dot(w, rs)) if rs else 0) - cost * to, "b": (bidx[ix - 1] / bidx[ie - 1] - 1) * 100, "wmax": float(w.max()) if len(w) else np.nan})
    T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index); return T

_RAW = None
def addraw(D, path=None):
    """1 張的權重與價格上限要用「當時的實際股價」：data/oldest|older|old/history 與 data/history 的原始開盤價（未還原）。
    （11/24 跑之前發現：pickle 的 C、O、prevc 都是還原價，例如台積電 2016-01-04 顯示 105，實際約 143。）"""
    global _RAW
    if _RAW is None:
        import glob
        fs = sorted(glob.glob(f"{REPO}/data/oldest/history/*.csv.gz") + glob.glob(f"{REPO}/data/older/history/*.csv.gz") + glob.glob(f"{REPO}/data/old/history/*.csv.gz") + glob.glob(f"{REPO}/data/history/*.csv.gz"))
        parts = [pd.read_csv(f, usecols=["date", "code", "open", "close"], dtype={"code": str}) for f in fs]
        pr = f"{REPO}/data/prices.csv"
        if os.path.exists(pr): parts.append(pd.read_csv(pr, usecols=["date", "code", "open", "close"], dtype={"code": str}))
        R = pd.concat(parts, ignore_index=True); R["date"] = pd.to_datetime(R.date); R = R.drop_duplicates(["date", "code"], keep="last")
        R["px"] = R.open.where(R.open > 0, R.close); _RAW = R.set_index(["date", "code"]).px
    days, cols = D["days"] if "days" in D else None, D["cols"] if "cols" in D else None
    if days is None:
        X = pd.read_pickle(path); days, cols = X["days"], X["cols"]
    need = sorted({D["POS"][a] + 1 for a in D["FDS"] if D["POS"][a] + 1 < len(days)})
    arr = np.full((len(days), len(cols)), np.nan)
    sub = _RAW[_RAW.index.get_level_values(0).isin(pd.DatetimeIndex([days[i] for i in need]))].unstack()
    sub = sub.reindex(index=pd.DatetimeIndex([days[i] for i in need]), columns=list(cols))
    arr[need] = sub.values; D["rawO"] = arr; return D

def dist(T):
    x = T.r; y = T.groupby(T.index.year).apply(lambda z: ((1 + z.r / 100).prod() - 1) * 100)
    yb = T.groupby(T.index.year).apply(lambda z: ((1 + z.b / 100).prod() - 1) * 100)
    return {"每月平均": round(x.mean(), 2), "最好月": round(x.max(), 1), "最差月": round(x.min(), 1), "賠錢月%": round((x < 0).mean() * 100), "輸0050月%": round((x < T.b).mean() * 100),
            "最好年": round(y.max()), "最差年": round(y.min()), "賠錢年": f"{int((y < 0).sum())}/{len(y)}", "輸0050年": f"{int((y < yb).sum())}/{len(y)}", "逐年": {int(k): round(v) for k, v in y.items()}}

if __name__ == "__main__":
    Dn = addraw(RS.load("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01"), "/tmp/model_v2/vs0050_extra.pkl")
    Do = addraw(RS.load("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01"), "/tmp/model_v2/vs0050_oldestonly_extra.pkl")
    Dx = RS.load("/tmp/model_v2/vs0050_fdx_extra.pkl", "2011-01-01", "2027-01-01"); Ex = [addraw(nth_set(Dx, k), "/tmp/model_v2/vs0050_fdx_extra.pkl") for k in range(3)]
    res = {}
    for lab, kw in (("基準 1張", {}), ("等金額", {"weight": "eq"}), ("C1 ≤100元", {"cap": 100}), ("C2 ≤150元", {"cap": 150}), ("C3 ≤200元", {"cap": 200}), ("C4 最高≤最低2倍", {"rel": 2.0})):
        T = pd.concat([sim(Do, **kw), sim(Dn, **kw)]); s = stats(T[["r", "b"]]); tt = [stats(sim(E, **kw)[["r", "b"]]) for E in Ex]
        res[lab] = {**s, "單檔最大占比中位數": round(float(T.wmax.median() * 100)), "三日": [(t["2011–2026"], t["2024-26"], t["最大回落"]) for t in tt],
                    "三日平均": (round(float(np.mean([t["2011–2026"][0] for t in tt])), 2), round(float(np.mean([t["最大回落"] for t in tt])), 1)), "分布": dist(T)}
        print(lab, json.dumps(res[lab], ensure_ascii=False, default=str), flush=True)
    b = res["基準 1張"]; chk = {}
    for lab in ("C1 ≤100元", "C2 ≤150元", "C3 ≤200元", "C4 最高≤最低2倍"):
        s = res[lab]; c = {"①": bool(s["2011–2026"][0] >= b["2011–2026"][0] - 0.3 and s["2011–2026"][1] >= 2.9), "②": bool(all(s[k] > 0 for k in RS.SEGS)),
                           "③": bool(s["最大回落"] >= b["最大回落"] + 3), "④": bool(s["三日平均"][1] >= b["三日平均"][1] + 3)}
        chk[lab] = {"各項": c, "通過": all(c.values())}
    passed = [k for k in chk if chk[k]["通過"]]; res["判斷"] = chk; res["採用"] = max(passed, key=lambda k: res[k]["最大回落"]) if passed else None
    print(json.dumps(chk, ensure_ascii=False), "採用", res["採用"])
    json.dump(res, open(f"{REPO}/research/results/price_cap.json", "w"), ensure_ascii=False, indent=1, default=str)
