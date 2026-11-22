"""「每月前20」達標計畫（2026/11/22 起；每一輪都在看結果之前寫下並存檔，看到結果後不修改）

目標（使用者 11/22 要求設定一個目標一直做）：每月換股的強勢股名單，2011–2026、扣 1.1% 來回成本後，每月報酬減 0050（含息）：
  ① 全期平均 > 0 且 Newey-West t ≥ 2.9（落後 1）；② 2011–15、2016–19、2020–23、2024–26 四段平均都 > 0；
  ③ 發現期（2020–23，綜合分數的 12 個特性當初是用這段挑的）以外的月份，平均 > 0 且 t ≥ 2。三項都符合＝達標。
報酬算法同 research/top20_exact.py（換股日隔天開盤賣舊買新，每個交易日只算一次；0050 同段收盤到收盤）。
基準（top20_exact.py，成本 1.1%）：+1.43（t 2.73）；四段 +1.15、+0.91、+3.35、−0.08；發現期外 +0.77（t 1.39）→ ①②③ 都不到。
多重檢定：本計畫每多一個規則就記一次；門檻 t ≥ max(2.9, 該輪 Bonferroni)。

第 1 輪（11/22 20:00 登錄）：降低換股成本。依據：Novy-Marx & Velikov (2016) 換股緩衝（只在股票掉出較寬的名次才賣）大幅降低交易成本、保留大部分報酬。
  B1：已持有的股票只要還在當月分數前 40 名就續抱；空出的名額依分數由高到低補滿 20 檔。
  B2：同上，續抱門檻改成前 60 名。
  成本 1.1% × 換股比例（賣出檔數 ÷ 20）。新買的股票開盤一價漲停買不到，該月就少一檔（續抱的不受影響）；分數名次在當天分析範圍（20 日均額 ≥ 1,000 萬）內排，
  掉出分析範圍＝賣出。網站「每月前20」分頁採用達標的規則（兩個都達標用全期 t 較高的）；都不達標就用原本每月全換的前20，並標明「尚未達標」。
  另報：成本 0.6%、平均換股比例、年化、最大回落、逐年。
  第 1 輪結果（11/22）：B1 +1.31（t 2.48）、B2 +1.31（t 2.57），都比基準 +1.45（t 2.76）差，2024–26 仍為負 → 不達標。

第 2 輪（11/22 20:15 登錄；累計檢驗 5 個規則，門檻仍為 t ≥ 2.9）：卡住的是 2024–26（台積電帶動 0050）與發現期外的 t。
  V1 波動控管（Barroso & Santa-Clara 2015；Daniel & Moskowitz 2016）：σ＝基準策略「減 0050」在前 2～7 個月（6 個月，跳過還沒結束的最近一期）的標準差；
     目標＝到當月為止所有 σ 的中位數（只用過去）；選股比重 w＝min(1, 目標 ÷ σ)，其餘放 0050。每月減 0050＝w ×（選股 − 成本×換股比例 − 0050）− 0.5×成本×|w − 上月 w|。
  V2 依成交值加權：前 20 檔依 20 日均額加權，單檔上限 15%（超過的部分依比例分給其他檔），讓大型、成交活絡的股票比重較高。
  V3 逐檔和 0050 比強弱：前 20 檔中，過去 120 個交易日漲幅不到 0050 同期漲幅的，那一格改放 0050（1/20）。
  成本、報酬算法、判斷（①②③）同第 1 輪；達標的規則用於網站分頁（多個達標用全期 t 最高的）。
  第 2 輪結果（11/22，成本 1.1%）：V1 +1.29（t 2.95）四段都正、發現期外 +0.73（t 1.48）→ 差 ③；V2 +1.46（t 2.33），最大回落 −50%；
    V3 +1.37（t 2.78），2024–26 −0.34 → 都不達標。成本 0.6% 時基準、V1、V3 都達標（另報，不改判斷）。

第 3 輪（11/22 20:30 登錄；累計 7 個規則，門檻 t ≥ 2.9）：持股數。依據：Grinold 主動管理基本法則（資訊比率 ≈ 預測力 × √持股廣度），
  分數在前 20 名以外仍有預測力（offense.py H4、deep_normal 五等分差），擴大持股可降低和 0050 的差距波動。
  W30：分數前 30 檔等權（還在前 30 的不賣）；W40：前 40 檔等權（還在前 40 的不賣）。其餘同第 1 輪。
  第 3 輪結果（11/22，成本 1.1%）：W30 +1.16（t 2.44）、W40 +0.96（t 2.14），都比前 20 差 → 不達標。
  依第 1 輪事先定的規則：都不達標 → 網站分頁（11/22 上線，名稱「月選」）用基準前 20，並標明以 1.1% 成本計尚未達標。
"""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
from vs0050 import COMP, nw_t, per, PER
X = pd.read_pickle("/tmp/model_v2/vs0050_extra.pkl"); days = X["days"]; cols = X["cols"]; Ov, Cv, lockn, bidx = X["O"], X["C"], X["lockn"], X["bidx"]
Cff = pd.DataFrame(Cv).ffill(limit=60).values; POS = {d: i for i, d in enumerate(days)}; CJ = {c: j for j, c in enumerate(cols)}; ND = len(days)
F = X["fd"].copy(); F = F[F.date >= "2011-01-01"]
F["cs"] = np.mean([F.groupby("date")[c].rank(pct=True).fillna(0.5).values for c in COMP], axis=0)
FDS = sorted(F.date.unique()); G = {d: g.sort_values("cs", ascending=False).reset_index(drop=True) for d, g in F.groupby("date")}

def px_exit(tb, j):
    ex = Ov[tb + 1, j] if tb + 1 < ND and Ov[tb + 1, j] > 0 else Cff[min(tb + 1, ND - 1), j]
    return ex if np.isfinite(ex) else Cff[tb, j]

def simulate(keep_rank=None, n=20, score=None):
    """keep_rank=None：每月全換成前 n；否則已持有且名次 ≤ keep_rank 的續抱。score：自訂排序欄（預設 cs）。"""
    rows, held = [], []
    for a, b in zip(FDS[:-1], FDS[1:]):
        ta, tb = POS[a], POS[b]; g = G[a] if score is None else G[a].sort_values(score, ascending=False).reset_index(drop=True)
        rank = {c: i + 1 for i, c in enumerate(g.code)}
        keep = [c for c in held if keep_rank is not None and rank.get(c, 1e9) <= keep_rank]
        new = [c for c in g.code if c not in keep][:n - len(keep)]
        sells = len([c for c in held if c not in keep]); to = 1.0 if not held else sells / n
        rs, hold_next = [], []
        for c in keep + new:
            j = CJ[c]; is_new = c in new
            if is_new and (not (Ov[ta + 1, j] > 0) or lockn[ta, j]): continue
            # 續抱的股票從上一段的賣出價（= 這段的買進價）接著算
            o = Ov[ta + 1, j] if Ov[ta + 1, j] > 0 else Cff[ta, j]
            ex = px_exit(tb, j)
            if np.isfinite(ex) and np.isfinite(o) and o > 0: rs.append((ex / o - 1) * 100); hold_next.append(c)
        held = hold_next
        rows.append({"date": a, "g": np.mean(rs) if rs else 0.0, "to": to, "b": (bidx[tb] / bidx[ta] - 1) * 100, "n": len(rs)})
    T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index); return T

def evaluate(T, cost):
    r = T.g - cost * T.to; x = r - T.b; pr = per(T.index)
    w, wb = (1 + r / 100).cumprod(), (1 + T.b / 100).cumprod(); yrs = (days[POS[FDS[-1]]] - FDS[0]).days / 365.25
    a_ = (round(float(x.mean()), 2), round(nw_t(x, 1), 2)); ps = {p: (round(float(x[pr == p].mean()), 2), round(nw_t(x[pr == p], 1), 2)) for p in PER}
    oo = x[pr != "2020-23"]; oos = (round(float(oo.mean()), 2), round(nw_t(oo, 1), 2))
    ok = a_[0] > 0 and a_[1] >= 2.9 and all(ps[p][0] > 0 for p in PER) and oos[0] > 0 and oos[1] >= 2
    yr = pd.DataFrame({"策略%": (1 + r / 100).groupby(T.index.year).prod() * 100 - 100, "0050%": (1 + T.b / 100).groupby(T.index.year).prod() * 100 - 100}).round(1)
    return {"減0050 全期": a_, "四段": ps, "發現期外": oos, "達標": bool(ok), "平均換股比例": round(float(T.to.mean()), 2),
            "年化 策略/0050": (round((w.iloc[-1] ** (1 / yrs) - 1) * 100, 1), round((wb.iloc[-1] ** (1 / yrs) - 1) * 100, 1)),
            "最大回落": (round(float((w / w.cummax() - 1).min() * 100), 1), round(float((wb / wb.cummax() - 1).min() * 100), 1)),
            "贏0050年數": f"{int((yr['策略%'] > yr['0050%']).sum())}/{len(yr)}", "逐年": {int(k): (v["策略%"], v["0050%"]) for k, v in yr.iterrows()}}

if __name__ == "__main__" and (len(sys.argv) < 2 or sys.argv[1] == "r1"):
    out = {}
    for lab, kr in (("基準 每月全換（還在前20的不賣）", 20), ("B1 前40續抱", 40), ("B2 前60續抱", 60)):
        T = simulate(kr); out[lab] = {"成本1.1%": evaluate(T, 1.1), "成本0.6%": evaluate(T, 0.6)}
        o = out[lab]["成本1.1%"]; print(lab, "1.1%:", {k: v for k, v in o.items() if k != "逐年"}, "| 0.6%:", out[lab]["成本0.6%"]["減0050 全期"], out[lab]["成本0.6%"]["年化 策略/0050"], flush=True)
    json.dump(out, open(f"{REPO}/research/results/top20_goal_r1.json", "w"), ensure_ascii=False, indent=1)

def evaluate_x(x, r, b):
    """x：每月減0050；r：策略報酬；b：0050（都是 %，同一組日期）。判斷同 evaluate。"""
    pr = per(x.index); w, wb = (1 + r / 100).cumprod(), (1 + b / 100).cumprod(); yrs = (days[POS[FDS[-1]]] - FDS[0]).days / 365.25
    a_ = (round(float(x.mean()), 2), round(nw_t(x, 1), 2)); ps = {p: (round(float(x[pr == p].mean()), 2), round(nw_t(x[pr == p], 1), 2)) for p in PER}
    oo = x[pr != "2020-23"]; oos = (round(float(oo.mean()), 2), round(nw_t(oo, 1), 2))
    ok = a_[0] > 0 and a_[1] >= 2.9 and all(ps[p][0] > 0 for p in PER) and oos[0] > 0 and oos[1] >= 2
    yr = pd.DataFrame({"策略%": (1 + r / 100).groupby(x.index.year).prod() * 100 - 100, "0050%": (1 + b / 100).groupby(x.index.year).prod() * 100 - 100}).round(1)
    return {"減0050 全期": a_, "四段": ps, "發現期外": oos, "達標": bool(ok),
            "年化 策略/0050": (round(float((w.iloc[-1] ** (1 / yrs) - 1) * 100), 1), round(float((wb.iloc[-1] ** (1 / yrs) - 1) * 100), 1)),
            "最大回落": (round(float((w / w.cummax() - 1).min() * 100), 1), round(float((wb / wb.cummax() - 1).min() * 100), 1)),
            "贏0050年數": f"{int((yr['策略%'] > yr['0050%']).sum())}/{len(yr)}", "逐年": {int(k): (v["策略%"], v["0050%"]) for k, v in yr.iterrows()}}

def simulate2(mode, n=20, keep_rank=20, cost=1.1):
    """mode='a20'：依 20 日均額加權（單檔上限 15%）；mode='vs0050'：r120 不到 0050 的格子改放 0050。回傳每月 r（已扣成本）、b。"""
    rows, held = [], []
    for a, b_ in zip(FDS[:-1], FDS[1:]):
        ta, tb = POS[a], POS[b_]; g = G[a]; rank = {c: i + 1 for i, c in enumerate(g.code)}
        keep = [c for c in held if rank.get(c, 1e9) <= keep_rank]; new = [c for c in g.code if c not in keep][:n - len(keep)]
        sells = len([c for c in held if c not in keep]); to = 1.0 if not held else sells / n
        b = (bidx[tb] / bidx[ta] - 1) * 100; b120 = bidx[ta] / bidx[ta - 120] - 1 if ta >= 120 else np.nan
        info = g.set_index("code"); picks, hold_next = [], []
        for c in keep + new:
            j = CJ[c]
            if c in new and (not (Ov[ta + 1, j] > 0) or lockn[ta, j]): continue
            o = Ov[ta + 1, j] if Ov[ta + 1, j] > 0 else Cff[ta, j]; ex = px_exit(tb, j)
            if np.isfinite(ex) and np.isfinite(o) and o > 0: picks.append((c, (ex / o - 1) * 100, float(info.loc[c, "a20"]), float(info.loc[c, "r120"]))); hold_next.append(c)
        held = hold_next
        if not picks: rows.append({"date": a, "r": b, "b": b}); continue
        P_ = pd.DataFrame(picks, columns=["code", "y", "a20", "r120"])
        if mode == "a20":
            w = P_.a20.fillna(P_.a20.median()).clip(lower=1).values.astype(float); w = w / w.sum()
            for _ in range(20):
                over = w > 0.15
                if not over.any(): break
                ex_ = (w[over] - 0.15).sum(); w[over] = 0.15; w[~over] += ex_ * w[~over] / w[~over].sum()
            r = float((P_.y.values * w).sum()) - cost * to
        else:
            weak = P_.r120.values < b120 if np.isfinite(b120) else np.zeros(len(P_), bool)
            k = len(P_); r = (P_.y.values[~weak].sum() + weak.sum() * b) / k - cost * to * (1 - weak.mean())
        rows.append({"date": a, "r": r, "b": b})
    T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index); return T

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "r2":
    out = {}
    for cost in (1.1, 0.6):
        base = simulate(20); x0 = base.g - cost * base.to - base.b
        sig = pd.Series([x0.iloc[max(0, i - 7):max(0, i - 1)].std(ddof=1) if i >= 7 else np.nan for i in range(len(x0))], index=x0.index)
        tgt = sig.expanding(min_periods=6).median(); w = (tgt / sig).clip(upper=1.0).fillna(1.0)
        x1 = w * x0 - 0.5 * cost * w.diff().abs().fillna(0)
        out[f"V1 波動控管 成本{cost}%"] = evaluate_x(x1, base.b + x1, base.b); out[f"V1 平均選股比重 成本{cost}%"] = round(float(w.mean()), 2)
        T2 = simulate2("a20", cost=cost); out[f"V2 成交值加權 成本{cost}%"] = evaluate_x(T2.r - T2.b, T2.r, T2.b)
        T3 = simulate2("vs0050", cost=cost); out[f"V3 逐檔和0050比強弱 成本{cost}%"] = evaluate_x(T3.r - T3.b, T3.r, T3.b)
    for k, v in out.items(): print(k, {kk: vv for kk, vv in v.items() if kk != "逐年"} if isinstance(v, dict) else v, flush=True)
    json.dump(out, open(f"{REPO}/research/results/top20_goal_r2.json", "w"), ensure_ascii=False, indent=1)

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "r3":
    out = {}
    for lab, n in (("W30 前30", 30), ("W40 前40", 40)):
        T = simulate(n, n=n); out[lab] = {"成本1.1%": evaluate(T, 1.1), "成本0.6%": evaluate(T, 0.6)}
        print(lab, "1.1%:", {k: v for k, v in out[lab]["成本1.1%"].items() if k != "逐年"}, "| 0.6%:", out[lab]["成本0.6%"]["減0050 全期"], out[lab]["成本0.6%"]["發現期外"], out[lab]["成本0.6%"]["達標"], flush=True)
    json.dump(out, open(f"{REPO}/research/results/top20_goal_r3.json", "w"), ensure_ascii=False, indent=1)
