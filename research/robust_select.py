"""更合理、更穩的選股：流動性門檻與「每段期間都有效」的特性（第 10 輪；2026/11/23 08:45 登錄，看結果前寫下，看到結果後不修改）

起因：月選平均贏 0050，但最差的期間（2024–26 每月 +0.42%）幾乎打平；11/11 這期 19 檔全是小型股，大跌時一起重挫。使用者要「合理的選股」。
資料：2011–2026（vs0050_extra.pkl）與 2006–2009（vs0050_oldestonly_extra.pkl，上櫃 2007/7 起）。持有方式同月選（換股日隔天開盤買、還在前 20 的續抱），成本 0.6%。
五段期間：2006–09、2011–15、2016–19、2020–23、2024–26。
候選：
  L1 月選，但只在 20 日均額 ≥ 5,000 萬的股票裡選前 20（分數仍在原分析範圍內算）。
  L2 同上，門檻 1 億。
  R 「每段期間都有效」的綜合分數：候選特性＝月選 12 個 ＋ r60（12 週漲幅）、r120（＋），收縮比 vcr、20 日波動、20 日最大單日漲幅、5 日漲幅（－，依文獻方向）。
    每個特性每月算「特性百分位 vs 之後 20 日報酬」的等級相關（分析範圍內），五段的平均都 > 0 且全部月份 Newey-West t ≥ 3 才入選；
    入選特性（依方向）百分位等權平均 → 前 20。
判斷（全部符合才採用）：① 2011–2026 每月減 0050 的 NW t ≥ 2.9；② 五段每月減 0050 都 > 0；③ 五段中最差那段比月選的最差段（目前 +0.42）高；
  ④ 2011–2026 最大回落不比月選深 3 個百分點以上。多個符合 → 用最差段最高的。採用 → 網站「月選」換成新規則；否則維持現狀。
累計檢驗 23 個規則。另報：使用者原本的「12 週漲幅前 20」（r60 最高的 20 檔，同持有方式）在五段的表現；R 入選了哪些特性；各候選的年化與逐期。
結果（11/23 08:55）：都不採用。月選 最差段 +0.42（2024–26）；L1 +1.36（t 2.58）最差 +0.25、回落 −35.9；L2 +1.16（t 2.18）最差 −0.41；
  R（入選 d_hi250、yoy1、yoy3、rev_acc、rev_hi12、vol20、max20）2011–2026 0.00，2024–26 −2.61，回落 −21.6；12 週漲幅前 20 +0.58（t 0.91），回落 −57.5。
  特性 IC（五段都正、t ≥ 3）：營收 4 個（t 4.8–8.7）、d_hi250（4.05）、低波動 vol20（5.18）、低 max20（6.28）。r20（−2.56）、rs_g20（−3.05）、mg20（−2.84）五段多為負。

第 11 輪（11/23 09:00 登錄；看了第 10 輪的特性分析後才提出，所以照樣事先登錄）：
  RV 「營收＋高點」：yoy1、yoy3、rev_acc、rev_hi12、d_hi250 五個（五段都有效、而且符合「強勢股」方向的特性）百分位等權 → 前 20。判斷同第 10 輪 ①–④。
  另報（不判定）：只用價格特性（r20、mom6_1、d_hi250、m60、am、rs_g20）、只用營收特性（yoy1、yoy3、rev_acc、rev_hi12、mom1）的前 20。
"""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
from vs0050 import COMP, nw_t, per, PER
EXTRA = {"r60": +1, "r120": +1, "vcr": -1, "vol20": -1, "max20": -1, "r5": -1}
CANDS = {**{c: +1 for c in COMP}, **EXTRA}

def load(path, start, end):
    X = pd.read_pickle(path); days, cols = X["days"], X["cols"]
    D = {"O": X["O"], "Cff": pd.DataFrame(X["C"]).ffill(limit=60).values, "lockn": X["lockn"], "bidx": X["bidx"],
         "POS": {d: i for i, d in enumerate(days)}, "CJ": {c: j for j, c in enumerate(cols)}, "ND": len(days)}
    C = pd.DataFrame(X["C"], index=days, columns=cols); r1 = C.pct_change(fill_method=None)
    P = {"r60": C / C.shift(60) - 1, "vcr": r1.rolling(10, min_periods=8).std() / r1.rolling(60, min_periods=45).std(),
         "vol20": r1.rolling(20, min_periods=15).std(), "max20": r1.rolling(20, min_periods=15).max(), "r5": C / C.shift(5) - 1}
    F = X["fd"].copy(); F = F[(F.date >= start) & (F.date < end)].copy()
    ti = F.date.map(D["POS"]).values; ji = F.code.map(D["CJ"]).values
    for k, v in P.items(): F[k] = v.values[ti, ji]
    F = F.replace([np.inf, -np.inf], np.nan)
    for c, s in CANDS.items(): F["p_" + c] = F.groupby("date")[c].rank(pct=True, ascending=(s > 0)).fillna(0.5)
    F["cs"] = F[["p_" + c for c in COMP]].mean(axis=1); F["tgt"] = F.groupby("date").y20.rank(pct=True)
    D["F"] = F; D["FDS"] = sorted(F.date.unique()); return D

def sim(D, score, n=20, minamt=None, cost=0.6):
    O, Cff, lk, bidx, POS, CJ, ND = D["O"], D["Cff"], D["lockn"], D["bidx"], D["POS"], D["CJ"], D["ND"]
    G = {d: g for d, g in D["F"].groupby("date")}; rows, held = [], []
    for a, b_ in zip(D["FDS"][:-1], D["FDS"][1:]):
        ta, tb = POS[a], POS[b_]; ie, ix = ta + 1, tb + 1
        if ix >= ND: break
        g = G[a]
        if minamt: g = g[g.a20 >= minamt]
        g = g.sort_values(score, ascending=False); rank = {c: i + 1 for i, c in enumerate(g.code)}
        keep = [c for c in held if rank.get(c, 1e9) <= n]; new = [c for c in g.code if c not in keep][:n - len(keep)]
        sells = len([c for c in held if c not in keep]); to = 1.0 if not rows else sells / n; rs, hn = [], []
        for c in keep + new:
            j = CJ[c]
            if c in new and (not (O[ie, j] > 0) or lk[ie - 1, j]): continue
            o = O[ie, j] if O[ie, j] > 0 else Cff[ie - 1, j]; ex = O[ix, j] if O[ix, j] > 0 else Cff[ix, j]
            if not np.isfinite(ex): ex = Cff[ix - 1, j]
            if np.isfinite(ex) and np.isfinite(o) and o > 0: rs.append((ex / o - 1) * 100); hn.append(c)
        held = hn; rows.append({"date": a, "r": (np.mean(rs) if rs else 0) - cost * to, "b": (bidx[ix - 1] / bidx[ie - 1] - 1) * 100})
    T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index); return T

def seg(d): return np.where(d < pd.Timestamp("2011-01-01"), "2006-09", per(d))
SEGS = ["2006-09", "2011-15", "2016-19", "2020-23", "2024-26"]
def mdd(r):
    w = (1 + r / 100).cumprod(); return round(float((w / w.cummax() - 1).min() * 100), 1)

def evaluate(TA, TB):
    """TA：2011–2026，TB：2006–2009（只取 2006-01～2009-12）。"""
    TB = TB[(TB.index >= "2006-01-01") & (TB.index < "2010-01-01")]; T = pd.concat([TB, TA]); x = T.r - T.b; s = seg(T.index)
    xa = TA.r - TA.b; w = (1 + TA.r / 100).cumprod(); wb = (1 + TA.b / 100).cumprod(); yrs = len(TA) / 12
    ps = {k: round(float(x[s == k].mean()), 2) for k in SEGS}
    return {"2011–2026 減0050": (round(float(xa.mean()), 2), round(nw_t(xa, 1), 2)), "五段": ps, "最差段": min(ps.values()),
            "2011–2026 年化 策略/0050": (round(float(w.iloc[-1] ** (1 / yrs) - 1) * 100, 1), round(float(wb.iloc[-1] ** (1 / yrs) - 1) * 100, 1)),
            "2011–2026 最大回落": mdd(TA.r), "每月贏0050%": round(float((x > 0).mean() * 100))}

if __name__ == "__main__":
    A = load("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01"); B = load("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2005-07-01", "2010-03-01")
    # R：五段都有效的特性
    FF = pd.concat([B["F"][(B["F"].date >= "2006-01-01") & (B["F"].date < "2010-01-01")], A["F"]]); ic = {}
    for c in CANDS:
        m = FF.groupby("date").apply(lambda g: g["p_" + c].corr(g.tgt, method="spearman") if g.tgt.notna().sum() > 50 else np.nan).dropna()
        s = seg(pd.DatetimeIndex(m.index)); ic[c] = {"五段": {k: round(float(m[s == k].mean()), 4) for k in SEGS}, "t": round(nw_t(m, 1), 2)}
    keepf = [c for c, v in ic.items() if all(x > 0 for x in v["五段"].values()) and v["t"] >= 3]
    print("特性 IC：", json.dumps(ic, ensure_ascii=False)); print("R 入選：", keepf, flush=True)
    for D in (A, B): D["F"]["rs"] = D["F"][["p_" + c for c in keepf]].mean(axis=1)
    out = {"特性IC": ic, "R入選": keepf}; base = None; res = {}
    for lab, kw in (("月選（原本）", {"score": "cs"}), ("L1 均額≥5,000萬", {"score": "cs", "minamt": 5e7}), ("L2 均額≥1億", {"score": "cs", "minamt": 1e8}),
                    ("R 五段都有效", {"score": "rs"}), ("參考 12週漲幅前20", {"score": "r60"})):
        e = evaluate(sim(A, **kw), sim(B, **kw)); res[lab] = e; print("==", lab, json.dumps(e, ensure_ascii=False), flush=True)
    b0 = res["月選（原本）"]; adopt = None
    for lab in ("L1 均額≥5,000萬", "L2 均額≥1億", "R 五段都有效"):
        e = res[lab]; ok = e["2011–2026 減0050"][1] >= 2.9 and all(v > 0 for v in e["五段"].values()) and e["最差段"] > b0["最差段"] and e["2011–2026 最大回落"] >= b0["2011–2026 最大回落"] - 3
        res[lab]["採用條件"] = bool(ok)
        if ok and (adopt is None or e["最差段"] > res[adopt]["最差段"]): adopt = lab
    out["結果"] = res; out["採用"] = adopt; print("採用：", adopt)
    json.dump(out, open(f"{REPO}/research/results/robust_select.json", "w"), ensure_ascii=False, indent=1, default=str)

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "r11":
    pass
