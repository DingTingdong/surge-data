"""8 特性分數 vs 月選，3 檔版（第 20 輪；2026/11/23 12:05 登錄，看結果前寫下，看到結果後不修改）

起因：第 19 輪的「勝率分」組合（20 檔）在三個沒用來挑特性的期間都比月選好，但那輪不是驗這個，只能當線索。
  網站已改成每期只買 3 檔（還在前 20 名續抱），所以這輪直接驗 3 檔版。
分數：
  A 月選分數（12 特性等權百分位）。
  B 8 特性分數（特性與方向沿用第 19 輪、不再調整）：半年動能、距 250 日高點、營收年增、三個月營收年增、營收加速、營收創 12 個月新高、
    120 日漲幅，以及近五日漲幅高（robust_select 的 p_r5 是反向排名，所以用 1 − p_r5），等權平均。
持有：每期買分數前 3 名（開盤漲停買不到往下一名遞補），下次換股日還在前 20 名的續抱；成本每次賣出 0.6%。另報 20 檔版。
資料：2011–2026（vs0050_extra.pkl）、2006–2009（vs0050_oldestonly_extra.pkl）；換股時點檢查用 FD_OFFSETS=0,2,4 重建的 vs0050_fdx_extra.pkl
  （每月 11 日後第 1、3、5 個交易日）。
採用條件（B 全部符合才把網站換成 B）：
  ① 2011–2026 每月減 0050 平均 > A，且 Newey-West t（落後 1）≥ 2.9；② 五段（2006–09、2011–15、2016–19、2020–23、2024–26）都 > 0；
  ③ 每月「B 減 A」在五段中至少四段 > 0，且 2024–26 > 0；④ 換股日改成第 3、第 5 個交易日時，B 都 > A，且 B 的 2024–26 減 0050 都 > 0；
  ⑤ 成本 1.1% 時 B 每月減 0050 > 0、t ≥ 2；⑥ 最大回落不比 A 深 10 個百分點以上。
  累計檢驗約 45 個規則，門檻維持 t ≥ 2.9。
結果（11/23 12:15）：B 不通過（①②③④ 沒過）→ 網站維持月選。
  3 檔：A 月選 +3.23（t 3.45）五段 2.91／3.19／2.47／5.53／1.14、回落 −49%；B +2.66（t 2.62）五段 1.17／3.67／0.96／5.06／−0.09、回落 −53%。
  20 檔：A +1.92、B +2.18（B 減 A t 1.09，不顯著；2024–26 +1.20 vs +0.42）。
  換股時點（3 檔）：A 第1／3／5 個交易日 +3.23／+2.91／+2.17，2024–26 +1.14／−1.50／−3.59，回落 −49／−45／−67%；
    B +2.66／+3.58／+1.38，2024–26 −0.09／+3.23／−2.72。
  重要：3 檔版長期在三個換股日都贏 0050，但 2024–26 好壞取決於哪一天換股（運氣），回落最深 −67%。
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import robust_select as RS
from vs0050 import nw_t

WR = ["p_mom6_1", "p_d_hi250", "p_yoy1", "p_yoy3", "p_rev_acc", "p_rev_hi12", "p_r120"]

def add_wr(D):
    F = D["F"]; F["wr"] = (F[WR].sum(axis=1) + (1 - F["p_r5"])) / 8; return D

def sim2(D, score, n, keep_rank=20, cost=0.6):
    O, Cff, lk, bidx, POS, CJ, ND = D["O"], D["Cff"], D["lockn"], D["bidx"], D["POS"], D["CJ"], D["ND"]
    G = {d: g for d, g in D["F"].groupby("date")}; rows, held = [], []
    for a, b_ in zip(D["FDS"][:-1], D["FDS"][1:]):
        ie, ix = POS[a] + 1, POS[b_] + 1
        if ix >= ND: break
        g = G[a].sort_values(score, ascending=False); rank = {c: i + 1 for i, c in enumerate(g.code)}
        keep = [c for c in held if rank.get(c, 1e9) <= keep_rank][:n]; new = []
        for c in g.code:
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

def run(Ds, score, n, cost=0.6):
    return pd.concat([sim2(D, score, n, cost=cost) for D in Ds]).sort_index()

def stats(T):
    x = T.r - T.b; sg = RS.seg(T.index); m = T.index >= "2011-01-01"; w = (1 + T.r[m] / 100).cumprod()
    return {"2011–2026": (round(float(x[m].mean()), 2), round(nw_t(x[m], 1), 2)), **{s: round(float(x[sg == s].mean()), 2) for s in RS.SEGS if (sg == s).any()},
            "最大回落": round(float((w / w.cummax() - 1).min() * 100)), "年化": round(float(w.iloc[-1] ** (12 / m.sum()) - 1) * 100)}

def nth_set(D, k):
    F = D["F"]; d = pd.Series(sorted(F.date.unique())); fk = set(d.groupby(d.dt.to_period("M")).nth(k))
    E = dict(D); E["F"] = F[F.date.isin(fk)].copy(); E["FDS"] = sorted(fk); return E

if __name__ == "__main__":
    Dn = add_wr(RS.load("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01")); Do = add_wr(RS.load("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01"))
    out = {}
    for n in (3, 20):
        TA, TB = run([Do, Dn], "cs", n), run([Do, Dn], "wr", n); sA, sB = stats(TA), stats(TB); dx = (TB.r - TA.r); sg = RS.seg(dx.index)
        out[f"{n} 檔"] = {"A 月選": sA, "B 8特性": sB, "B減A": {s: round(float(dx[sg == s].mean()), 2) for s in RS.SEGS}, "B減A t（2011–26）": round(nw_t(dx[dx.index >= "2011-01-01"], 1), 2)}
        print(n, "檔", json.dumps(out[f"{n} 檔"], ensure_ascii=False), flush=True)
    Dx = add_wr(RS.load("/tmp/model_v2/vs0050_fdx_extra.pkl", "2011-01-01", "2027-01-01")); tim = {}
    for k, lab in ((0, "第1個交易日（核對）"), (1, "第3個交易日"), (2, "第5個交易日")):
        E = nth_set(Dx, k); a, b = stats(sim2(E, "cs", 3)), stats(sim2(E, "wr", 3)); tim[lab] = {"A": a, "B": b}; print(lab, "A", a, "B", b, flush=True)
    out["換股時點（3 檔）"] = tim
    s11 = stats(run([Do, Dn], "wr", 3, cost=1.1)); out["B 成本1.1%（3 檔）"] = s11
    A3, B3, d3 = out["3 檔"]["A 月選"], out["3 檔"]["B 8特性"], out["3 檔"]["B減A"]
    c = {"①": B3["2011–2026"][0] > A3["2011–2026"][0] and B3["2011–2026"][1] >= 2.9, "②": all(B3[s] > 0 for s in RS.SEGS),
         "③": sum(d3[s] > 0 for s in RS.SEGS) >= 4 and d3["2024-26"] > 0,
         "④": all(tim[l]["B"]["2011–2026"][0] > tim[l]["A"]["2011–2026"][0] and tim[l]["B"]["2024-26"] > 0 for l in ("第3個交易日", "第5個交易日")),
         "⑤": s11["2011–2026"][0] > 0 and s11["2011–2026"][1] >= 2, "⑥": B3["最大回落"] >= A3["最大回落"] - 10}
    out["各項"] = c; out["通過"] = all(c.values()); print("各項", c, "通過" if all(c.values()) else "不通過")
    json.dump(out, open(f"{REPO}/research/results/wr_confirm.json", "w"), ensure_ascii=False, indent=1, default=str)
