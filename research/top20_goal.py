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

第 4 輪（11/22 22:15 登錄）：使用者實際手續費 2.8 折（來回約 0.4% ＋滑價），之後成本以 0.6% 為主（1.1% 另報）。使用者要「績效正常」＝穩定、不靠某段期間或成本假設。
  依第 2 輪「多個達標用全期 t 最高的」：0.6% 時基準（t 3.58）、V1（t 3.81）、V3（t 3.62）三項都達標 → V1 為候選。
  這輪不加新規則，只檢查 V1 穩不穩，以下變化都在看結果前定好：
   a. 波動回看 3 個月、12 個月（原 6）；目標改用過去 σ 的平均（原中位數）
   b. 換股日改成 11 日後第 3、第 5 個交易日（原第 1 個）
   c. 晚一天進場與換股（換股日後第 2 個交易日開盤）
   d. 每月扣掉報酬最好的 1 檔
   e. 成本 1.1%
   f. 從分數前 40 名隨機抽 20 檔，重複 200 次
  判斷「穩」：a–e 每個變化都要「全期每月減 0050 > 0 且 t ≥ 2、2024–26 > 0」；f 的 200 次中至少 90% 全期 > 0。
  穩 → 網站「月選」改用 V1（顯示本期選股比重，其餘放 0050）；不穩 → 維持基準前 20。
  另報穩定度：16 年中贏 0050 的年數、有沒有連兩年輸、滾動 12 個月贏 0050 的比例、最差的滾動 12 個月、最大回落。
  第 4 輪結果（11/22）：不穩 → 網站維持基準前 20。
    通過：a 回看 3／12 個月、目標用平均；c 晚一天進場 +1.42（t 3.32）；e 成本 1.1% +1.29（t 2.95）；f 隨機抽 200 次 100% 全期 > 0（中位數 +1.12）。
    沒過：b 換股日改第 3 個交易日 +1.19（t 2.74）但 2024–26 −0.44、發現期外 t 1.33；第 5 個交易日 +1.09，2024–26 −0.40；
          d 每月扣掉最好 1 檔 −0.13。（另報：隨機 20 檔一般股票扣掉最好 1 檔也從 −1.13 掉到 −2.52；前 20 扣掉後仍比它高 2.27，挑股能力還在，
          但「贏 0050」靠每月抓到大漲股。前 20 有 58% 的月份贏 0050，每月中位數 +1.33。）
    解讀：優勢集中在月營收公布後的頭幾天，要在換股日當天選、隔天開盤買；越晚越弱。
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

# ---------------- 第 4 輪：V1 穩健度檢查 ----------------
def v1_apply(T, cost, look=6, tgt="median"):
    x0 = T.g - cost * T.to - T.b
    sig = pd.Series([x0.iloc[max(0, i - look - 1):max(0, i - 1)].std(ddof=1) if i >= look + 1 else np.nan for i in range(len(x0))], index=x0.index)
    t_ = sig.expanding(min_periods=6).median() if tgt == "median" else sig.expanding(min_periods=6).mean()
    w = (t_ / sig).clip(upper=1.0).fillna(1.0)
    x1 = w * x0 - 0.5 * cost * w.diff().abs().fillna(0)
    return x1, T.b + x1, w

def stability(x, r, b):
    pr = per(x.index); yr = pd.DataFrame({"s": (1 + r / 100).groupby(x.index.year).prod(), "b": (1 + b / 100).groupby(x.index.year).prod()})
    lose = (yr.s <= yr.b).values; consec = any(lose[i] and lose[i + 1] for i in range(len(lose) - 1))
    rs = (1 + r / 100).rolling(12).apply(np.prod, raw=True); rb = (1 + b / 100).rolling(12).apply(np.prod, raw=True); rel = (rs / rb - 1).dropna() * 100
    w, wb = (1 + r / 100).cumprod(), (1 + b / 100).cumprod(); yrs = (x.index[-1] - x.index[0]).days / 365.25 + 1 / 12
    return {"全期": (round(float(x.mean()), 2), round(nw_t(x, 1), 2)), "2024-26": round(float(x[pr == "2024-26"].mean()), 2),
            "四段": {p: round(float(x[pr == p].mean()), 2) for p in PER}, "發現期外": (round(float(x[pr != "2020-23"].mean()), 2), round(nw_t(x[pr != "2020-23"], 1), 2)),
            "贏0050年數": f"{int((~lose).sum())}/{len(lose)}", "連兩年輸": bool(consec), "滾動12月贏0050%": round(float((rel > 0).mean() * 100), 0),
            "最差滾動12月%": round(float(rel.min()), 1), "最大回落 策略/0050": (round(float((w / w.cummax() - 1).min() * 100), 1), round(float((wb / wb.cummax() - 1).min() * 100), 1)),
            "年化 策略/0050": (round(float(w.iloc[-1] ** (1 / yrs) - 1) * 100, 1), round(float(wb.iloc[-1] ** (1 / yrs) - 1) * 100, 1))}

def sim_general(fds, G, n=20, keep_rank=20, off=1, drop_best=False, rnd=None):
    rows, held = [], []
    for a, b_ in zip(fds[:-1], fds[1:]):
        ta, tb = POS[a], POS[b_]
        if tb + off >= ND: break
        g = G[a]
        if rnd is not None:
            pool = list(g.code[:40]); chosen = list(rnd.choice(pool, size=n, replace=False)); keep = [c for c in held if c in chosen]; new = [c for c in chosen if c not in keep]
        else:
            rank = {c: i + 1 for i, c in enumerate(g.code)}; keep = [c for c in held if rank.get(c, 1e9) <= keep_rank]; new = [c for c in g.code if c not in keep][:n - len(keep)]
        sells = len([c for c in held if c not in keep]); to = 1.0 if not held else sells / n
        rs, hn = [], []
        for c in keep + new:
            j = CJ[c]; ie = ta + off; ix = tb + off
            if c in new and (not (Ov[ie, j] > 0) or lockn[ie - 1, j]): continue
            o = Ov[ie, j] if Ov[ie, j] > 0 else Cff[ie - 1, j]
            ex = Ov[ix, j] if ix < ND and Ov[ix, j] > 0 else Cff[min(ix, ND - 1), j]
            if not np.isfinite(ex): ex = Cff[ix - 1, j]
            if np.isfinite(ex) and np.isfinite(o) and o > 0: rs.append((ex / o - 1) * 100); hn.append(c)
        held = hn
        if drop_best and len(rs) > 1: rs = sorted(rs)[:-1]
        rows.append({"date": a, "g": np.mean(rs) if rs else 0.0, "to": to, "b": (bidx[tb + off - 1] / bidx[ta + off - 1] - 1) * 100})
    T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index); return T

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "r4":
    XX = pd.read_pickle("/tmp/model_v2/vs0050_fdx_extra.pkl"); FF = XX["fd"].copy(); FF = FF[FF.date >= "2011-01-01"]
    FF["cs"] = np.mean([FF.groupby("date")[c].rank(pct=True).fillna(0.5).values for c in COMP], axis=0)
    dser = pd.Series(sorted(FF.date.unique())); gm = dser.groupby(dser.dt.to_period("M"))
    GG = {d: g.sort_values("cs", ascending=False).reset_index(drop=True) for d, g in FF.groupby("date")}
    out, judge = {}, {}
    def run(lab, T, cost=0.6, **kw):
        x, r, w = v1_apply(T, cost, **kw); s = stability(x, r, T.b); out[lab] = s
        ok = s["全期"][0] > 0 and s["全期"][1] >= 2 and s["2024-26"] > 0; judge[lab] = ok
        print(lab, "穩" if ok else "不穩", {k: v for k, v in s.items() if k != "四段"}, flush=True); return x
    T0 = sim_general(FDS, G)
    # 基準與 V1 本身（另報穩定度）
    s0 = stability(T0.g - 0.6 * T0.to - T0.b, T0.g - 0.6 * T0.to, T0.b); out["參考 基準前20（不控管）"] = s0; print("參考 基準前20", {k: v for k, v in s0.items() if k != "四段"})
    run("V1 原設定（回看6、中位數）", T0)
    run("a 回看3個月", T0, look=3); run("a 回看12個月", T0, look=12); run("a 目標用平均", T0, tgt="mean")
    for k, nth in ((2, 1), (4, 2)):  # 每月只存了第 1、3、5 個交易日 → 存檔序號 0、1、2
        fk = sorted(gm.nth(nth).values); fk = [pd.Timestamp(d) for d in fk]; Gk = {d: GG[d] for d in fk}
        run(f"b 換股日改第{k + 1}個交易日", sim_general(fk, Gk))
    run("c 晚一天進場", sim_general(FDS, G, off=2))
    run("d 每月扣掉最好的1檔", sim_general(FDS, G, drop_best=True))
    run("e 成本1.1%", T0, cost=1.1)
    rng = np.random.default_rng(20261122); pos_ = []
    for i in range(200):
        Tr = sim_general(FDS, G, rnd=rng); x, r, w = v1_apply(Tr, 0.6); pos_.append((float(x.mean()), float(x[per(x.index) == "2024-26"].mean())))
    pos_ = np.array(pos_); out["f 前40隨機抽20（200次）"] = {"全期>0比例%": round(float((pos_[:, 0] > 0).mean() * 100), 1), "全期平均 中位數/5%/95%": [round(float(np.percentile(pos_[:, 0], q)), 2) for q in (50, 5, 95)],
                                                  "2024-26>0比例%": round(float((pos_[:, 1] > 0).mean() * 100), 1)}
    judge["f"] = out["f 前40隨機抽20（200次）"]["全期>0比例%"] >= 90; print("f", out["f 前40隨機抽20（200次）"], flush=True)
    w_now = v1_apply(T0, 0.6)[2]; out["V1 最近的選股比重"] = {str(k.date()): round(float(v), 2) for k, v in w_now.tail(8).items()}
    out["判斷"] = {"各項": judge, "穩": bool(all(judge[k] for k in judge if not k.startswith("V1 原")))}
    print("最近的選股比重", out["V1 最近的選股比重"]); print("判斷", out["判斷"])
    json.dump(out, open(f"{REPO}/research/results/top20_goal_r4.json", "w"), ensure_ascii=False, indent=1)
