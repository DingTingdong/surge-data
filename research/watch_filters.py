"""觀察名單加條件：今日發動、新高突破（第 32 輪；2026/11/23 17:05 登錄，看結果前寫下，看到結果後不修改）

起因：使用者要讓觀察名單變好。今日發動平常時期每筆比 0050 少 1.6～4.0%；新高突破賺錢、贏一般股票，但不穩定贏 0050。
  這輪用前幾輪找到的三個方向當條件：
  F1 低波動：訊號日前 20 個交易日（不含當天）日報酬標準差，在當天分析範圍內最低的三分之一（第 24 輪：前段班挑低波動回落少一半）；
  F2 營收催化：訊號日已公布的最近一月營收年增 ≥ 20% 且創 12 個月新高（第 30 輪：低基期突破加這個後五段都贏一般股票）；
  F3 分數高：月選綜合分數（最近一次每月快照的百分位）≥ 0.8（之前描述性結果：分數 ≥ 0.8 的今日發動約和 0050 打平）。
  另測組合 F1+F2、F1+F3、F2+F3、F1+F2+F3。
樣本：vs0050 快照的事件表（2005–2010 與 2010–2026），只用分析範圍內、非恐慌期間；2010 年不算（同其他輪）。
  今日發動：隔天開盤買、抱 20 個交易日（y20）；新高突破：隔天開盤買、抱 60 個交易日（y60）。成本 0.6%；減 0050（含息）同期。
判斷（每個名單、每個條件各自判斷；全部符合才在網站該分頁加標記、排前面）：
  ① 五段（2006–09、2011–15、2016–19、2020–23、2024–26）每筆減 0050 平均都 > 0，且每段至少 20 筆；
  ② 全部平均 > 0，以進場月份分群的 Newey-West t ≥ 2.9（共 14 個條件×名單，門檻維持 2.9）。
另報：贏 0050 比例、中位數、每年筆數、減同日全部股票平均。
結果（11/23 17:15）：全部不通過（t 都沒到 2.9）→ 網站不改成正式訊號。
  今日發動（平常）全部 −1.45；F3 分數 ≥ 0.8：+1.46（t 2.23，五段 2.02／1.28／0.19／2.37／1.16 都正）；
  F2＋F3（營收強＋分數高）：+1.65（t 2.54，五段 2.65／1.20／0.98／2.27／1.26 都正，一年約 90 筆，贏 0050 46%）；
  F1＋F2＋F3：+3.55（贏 0050 56%）但 2016–19 −1.92（21 筆）；F1 低波動單獨沒用（−1.44）。
  新高突破全部 +0.44；F3 +2.71、F2 +2.43，但 2016–19 或 2024–26 為負。
  → 最接近的是今日發動 F2＋F3（五段都正、t 2.5）：網站只加標記當參考，用之後的新資料再驗。
"""
import glob, json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
from vs0050 import nw_t
from base_breakout_rev import revenue
SEGS = ["2006-09", "2011-15", "2016-19", "2020-23", "2024-26"]
def seg(d): return np.where(d < pd.Timestamp("2010-01-01"), "2006-09", np.where(d < pd.Timestamp("2011-01-01"), "2010", np.where(d < pd.Timestamp("2016-01-01"), "2011-15", np.where(d < pd.Timestamp("2020-01-01"), "2016-19", np.where(d < pd.Timestamp("2024-01-01"), "2020-23", "2024-26")))))

def volpct(extra, E):
    X = pd.read_pickle(extra); days, cols = X["days"], X["cols"]; C = pd.DataFrame(X["C"], index=days, columns=cols)
    v = C.pct_change(fill_method=None).shift(1).rolling(20, min_periods=15).std()
    v = v.where(np.asarray(X["inb"]).astype(bool)).rank(axis=1, pct=True)
    pos = {d: i for i, d in enumerate(days)}; cj = {c: j for j, c in enumerate(cols)}; V = v.values
    ti = E.date.map(pos); ji = E.code.map(cj); ok = ti.notna() & ji.notna()
    out = np.full(len(E), np.nan); out[ok.values] = V[ti[ok].astype(int).values, ji[ok].astype(int).values]; return out

if __name__ == "__main__":
    parts = []
    for ev, extra in (("/tmp/model_v2/vs0050_oldestonly.pkl", "/tmp/model_v2/vs0050_oldestonly_extra.pkl"), ("/tmp/model_v2/vs0050.pkl", "/tmp/model_v2/vs0050_extra.pkl")):
        E = pd.read_pickle(ev); E = E[E.base & ~E.panic.astype(bool)].copy(); E["vp"] = volpct(extra, E); parts.append(E)
    E = pd.concat(parts, ignore_index=True); E["seg"] = seg(E.date); E = E[E.seg != "2010"]
    E = E.drop_duplicates(["date", "code", "kind"])
    R = revenue(); E = pd.merge_asof(E.sort_values("date"), R.sort_values("pub"), left_on="date", right_on="pub", by="code", direction="backward")
    E["F1"] = E.vp <= 1 / 3; E["F2"] = (E.r1.fillna(False) & E.r2.fillna(False)).astype(bool); E["F3"] = E.cs >= 0.8
    E = E.reset_index(drop=True)
    conds = {"全部": pd.Series(True, index=E.index), "F1": E.F1, "F2": E.F2, "F3": E.F3, "F1+F2": E.F1 & E.F2, "F1+F3": E.F1 & E.F3, "F2+F3": E.F2 & E.F3, "F1+F2+F3": E.F1 & E.F2 & E.F3}
    out = {}; passed = []
    for kind, lab, y, b, u in (("pk", "今日發動", "y20", "b20", "u20"), ("nh", "新高突破", "y60", "b60", "u60")):
        K = E[E.kind == kind]
        for cn, m in conds.items():
            z = K[m.loc[K.index].astype(bool)].dropna(subset=[y, b]).copy()
            z["x"] = z[y] - 0.6 - z[b]; z["xu"] = z[y] - z[u]
            sg = {s: (int((z.seg == s).sum()), round(float(z[z.seg == s].x.mean()), 2) if (z.seg == s).any() else None) for s in SEGS}
            mm = z.groupby(z.date.dt.to_period("M")).x.mean()
            res = {"筆數": int(len(z)), "平均": round(float(z.x.mean()), 2), "中位": round(float(z.x.median()), 2), "贏0050%": round(float((z.x > 0).mean() * 100)),
                   "t": round(nw_t(mm.reset_index(drop=True), 3), 2) if len(mm) > 10 else None, "五段（筆數, 平均）": sg,
                   "減全部股票": round(float(z.xu.mean()), 2), "每年筆數": round(len(z) / 19.0, 1)}
            ok = all(v[0] >= 20 and v[1] is not None and v[1] > 0 for v in sg.values()) and res["平均"] > 0 and (res["t"] or 0) >= 2.9
            res["通過"] = ok; out[f"{lab} {cn}"] = res
            if ok and cn != "全部": passed.append(f"{lab} {cn}")
            print(lab, cn, json.dumps(res, ensure_ascii=False), flush=True)
    out["通過的"] = passed; print("通過：", passed)
    json.dump(out, open(f"{REPO}/research/results/watch_filters.json", "w"), ensure_ascii=False, indent=1, default=str)
