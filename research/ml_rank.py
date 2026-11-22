"""機器學習綜合排序：能不能比「12 個特性等權平均」挑得更好？（第 7 輪；2026/11/23 00:30 登錄，看結果前寫下，看到結果後不修改）

依據：Gu, Kelly & Xiu (2020, RFS) 美股、Leippold, Wang & Zhou (2022, JFE) 中國股市：梯度提升樹等非線性模型的橫斷面選股，樣本外優於線性綜合分數。
換股日與分析範圍同月選（每月 11 日後第一個交易日收盤後，20 日均額 ≥ 1,000 萬）。
特性（換股日當天就知道的）：月選的 12 個（r20、mom6_1、d_hi250、m60、am、yoy1、yoy3、rev_acc、rev_hi12、mom1、mg20、rs_g20）
  ＋ r120、20 日均額取 log、近 5 日漲幅、20 日波動、20 日最大單日漲幅、距 20 日高點、大盤（0050）近 20 日漲幅與 20 日波動（讓模型依大盤狀況調整）。
目標：之後 20 個交易日報酬（隔天開盤起算）在當月的百分位。
模型（參數事先固定，不調）：scikit-learn HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
  min_samples_leaf=200, l2_regularization=1.0, random_state=0)。
逐年滾動：預測 Y 年時，只用 2011 年起到 Y−1 年 11 月的換股日訓練（排除 Y−1 年 12 月，因為它的報酬期間會跨進 Y 年）；預測 2014～2026 年。
  2011–2013 只當訓練資料，這段的名單沿用綜合分數。
比較：ML 分數前 20 vs 綜合分數前 20，持有方式完全同月選（還在前 20 的續抱、真實換股方式），成本 0.6%，期間 2014-01～2026-11。
判斷（兩項都要）：
  M1 每月「ML 前 20 − 綜合前 20」平均 > 0 且 Newey-West t（落後 1）≥ 2；
  M2 ML 前 20 減 0050：全期 t ≥ 2.9，四段（2014–15、2016–19、2020–23、2024–26）都 > 0，發現期（2020–23）以外 t ≥ 2。
  通過 → 再用 2011–2026 全部資料訓練最終模型，套到 2006–2009 年（data/oldest，完全沒用過），ML 前 20 每月減 0050 > 0 才換上網站；否則網站維持現狀。
累計檢驗 15 個規則。另報：每月贏 0050 的比例、每檔入選股贏 0050 的比例、逐年、最大回落、特性重要度（排列重要度）。
結果（11/23 00:40）：不通過 → 網站維持現狀。M1 ML−綜合 每月 −1.30（t −2.40）；ML 減0050 +0.70（t 1.41），四段 +2.39、+0.51、+1.63、−1.52；
  綜合同期 +2.00（t 3.29）。每月贏0050 ML 57%、綜合 58%；入選股 20 日贏 0050 比例 ML 48.8%、綜合 49.5%、全部 41.1%。特性重要度都接近 0（雜訊大）。
"""
import os, sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
import top20_goal as TG
from vs0050 import COMP, nw_t, per, PER
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

def add_features(F, X):
    days, cols = X["days"], X["cols"]; C = pd.DataFrame(X["C"], index=days, columns=cols); b = pd.Series(X["bidx"], index=days)
    r1 = C.pct_change(fill_method=None)
    P = {"vol20": r1.rolling(20, min_periods=15).std(), "max20": r1.rolling(20, min_periods=15).max(), "r5": C / C.shift(5) - 1, "d_hi20": C / C.rolling(20, min_periods=15).max() - 1}
    pos = {d: i for i, d in enumerate(days)}; cj = {c: j for j, c in enumerate(cols)}; ti = F.date.map(pos).values; ji = F.code.map(cj).values
    F = F.copy()
    for k, v in P.items(): F[k] = v.values[ti, ji]
    br = b.pct_change(); F["mkt_r20"] = (b / b.shift(20) - 1).values[ti]; F["mkt_vol20"] = br.rolling(20).std().values[ti]
    F["la20"] = np.log(F.a20.clip(lower=1)); return F.replace([np.inf, -np.inf], np.nan)

FEATS = COMP + ["r120", "la20", "r5", "vol20", "max20", "d_hi20", "mkt_r20", "mkt_vol20"]
PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, random_state=0)

if __name__ == "__main__":
    F = add_features(TG.F, TG.X); F["tgt"] = F.groupby("date").y20.rank(pct=True)
    F["ml"] = F.cs.values
    last_model, imp_set = None, None
    for Y in range(2014, 2027):
        tr = F[(F.date.dt.year < Y) & ~((F.date.dt.year == Y - 1) & (F.date.dt.month == 12)) & F.tgt.notna()]
        te = F.date.dt.year == Y
        m = HistGradientBoostingRegressor(**PARAMS).fit(tr[FEATS], tr.tgt)
        F.loc[te, "ml"] = m.predict(F.loc[te, FEATS]); last_model, imp_set = m, F[te & F.tgt.notna()]
        print(Y, "訓練筆數", len(tr), flush=True)
    TG.G = {d: g.sort_values("cs", ascending=False).reset_index(drop=True) for d, g in F.groupby("date")}
    out = {}
    for cost in (0.6, 1.1):
        Tm = TG.simulate(20, score="ml"); Tc = TG.simulate(20)
        Tm, Tc = Tm[Tm.index >= "2014-01-01"], Tc[Tc.index >= "2014-01-01"]
        d = (Tm.g - cost * Tm.to) - (Tc.g - cost * Tc.to)
        em, ec = TG.evaluate(Tm, cost), TG.evaluate(Tc, cost)
        xm = Tm.g - cost * Tm.to - Tm.b; xc = Tc.g - cost * Tc.to - Tc.b
        out[f"成本{cost}%"] = {"M1 ML−綜合 每月 平均/t": (round(float(d.mean()), 2), round(nw_t(d, 1), 2)), "ML 減0050": {k: em[k] for k in ("減0050 全期", "四段", "發現期外", "年化 策略/0050", "最大回落", "贏0050年數", "平均換股比例")},
                               "綜合 減0050": {k: ec[k] for k in ("減0050 全期", "四段", "發現期外", "年化 策略/0050", "最大回落", "贏0050年數", "平均換股比例")},
                               "每月贏0050% ML/綜合": (round(float((xm > 0).mean() * 100)), round(float((xc > 0).mean() * 100))), "逐年 ML": em["逐年"], "逐年 綜合": ec["逐年"]}
        if cost == 0.6:
            a, ps, oo = em["減0050 全期"], em["四段"], em["發現期外"]
            m1 = d.mean() > 0 and nw_t(d, 1) >= 2; m2 = a[0] > 0 and a[1] >= 2.9 and all(v[0] > 0 for v in ps.values()) and oo[0] > 0 and oo[1] >= 2
            out["M1"], out["M2"], out["通過"] = bool(m1), bool(m2), bool(m1 and m2)
        print(f"== 成本 {cost}%", json.dumps({k: v for k, v in out[f"成本{cost}%"].items() if not k.startswith("逐年")}, ensure_ascii=False), flush=True)
    # 入選股層級的勝率（20 日報酬贏 0050 的比例；描述用）
    E = F[F.date >= "2014-01-01"]; topm = E.sort_values("ml", ascending=False).groupby("date").head(20); topc = E.sort_values("cs", ascending=False).groupby("date").head(20)
    hr = lambda z: round(float(((z.y20 - z.b20) > 0).mean() * 100), 1)
    out["入選股20日贏0050比例% ML/綜合/全部"] = (hr(topm), hr(topc), hr(E))
    pi = permutation_importance(last_model, imp_set[FEATS], imp_set.tgt, n_repeats=5, random_state=0)
    out["特性重要度（2026 年）"] = dict(sorted({f: round(float(v), 4) for f, v in zip(FEATS, pi.importances_mean)}.items(), key=lambda kv: -kv[1]))
    print("入選股勝率", out["入選股20日贏0050比例% ML/綜合/全部"]); print("重要度", out["特性重要度（2026 年）"]); print("M1", out["M1"], "M2", out["M2"], "通過", out["通過"])
    json.dump(out, open(f"{REPO}/research/results/ml_rank.json", "w"), ensure_ascii=False, indent=1, default=str)
    F[["date", "code", "ml", "cs"]].to_pickle("/tmp/model_v2/ml_scores.pkl")
