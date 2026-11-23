"""觀察名單再加「借券放空不多」（第 34 輪；2026/11/23 17:50 登錄，看結果前寫下，看到結果後不修改）

起因：第 32 輪今日發動「營收強＋分數高」五段都贏 0050（+1.65、t 2.54）但沒達標；第 33 輪另報借券比前 20% 的今日發動明顯較差。
  這輪把兩者合起來：排除借券比（借券賣出餘額 ÷ 20 日平均成交股數，當天全部股票百分位）前 20% 的事件；沒有借券資料的保留。
  注意：兩個條件都是在同一份資料上看到的，這輪不是全新期間，門檻維持 t ≥ 2.9，之後要用前瞻資料確認。
規則（平常時期、分析範圍內；今日發動抱 20 天、新高突破抱 60 天；成本 0.6%；減 0050）：
  W1 今日發動＋營收強＋分數高＋借券不多；W2 今日發動＋分數高＋借券不多；W3 新高突破＋分數高＋借券不多。
判斷（各自）：① 五段每筆減 0050 平均都 > 0，每段至少 20 筆；② 全部平均 > 0，進場月份分群 Newey-West t ≥ 2.9。通過 → 網站該分頁標「回測達標」。
結果（11/23 17:55）：三個都不通過。W1 +1.69（t 2.51，五段都正）、W2 +1.44（t 1.94）、W3 +2.68（t 1.85，2016–19 −0.40）；
  分數高的事件很少同時借券多，排除後幾乎沒變 → 今日發動「營收強＋分數高」仍維持參考標記。
"""
import json, os, sys, warnings
import numpy as np, pandas as pd, glob
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
from vs0050 import nw_t
from base_breakout_rev import revenue
from watch_filters import volpct, seg, SEGS
from sbl_filter import sbl_feats
if __name__ == "__main__":
    S = pd.concat([pd.read_csv(f, dtype={"code": str}) for f in sorted(glob.glob(f"{REPO}/data/sbl/*.csv.gz"))], ignore_index=True); S["date"] = pd.to_datetime(S.date); S = S.drop_duplicates(["date", "code"], keep="last")
    parts = []
    for ev, extra in (("/tmp/model_v2/vs0050_oldestonly.pkl", "/tmp/model_v2/vs0050_oldestonly_extra.pkl"), ("/tmp/model_v2/vs0050.pkl", "/tmp/model_v2/vs0050_extra.pkl")):
        E = pd.read_pickle(ev); E = E[E.base & ~E.panic.astype(bool)].copy()
        X, fl, ra = sbl_feats(extra, S); PR = ra.rank(axis=1, pct=True); pos = {d: i for i, d in enumerate(X["days"])}; cj = {c: j for j, c in enumerate(X["cols"])}
        ti = E.date.map(pos); ji = E.code.map(cj); ok = (ti.notna() & ji.notna()).values; v = np.full(len(E), np.nan); v[ok] = PR.values[ti[ok].astype(int), ji[ok].astype(int)]; E["sp"] = v
        parts.append(E); del X, fl, ra, PR
    E = pd.concat(parts, ignore_index=True); E["seg"] = seg(E.date); E = E[E.seg != "2010"].drop_duplicates(["date", "code", "kind"])
    R = revenue(); E = pd.merge_asof(E.sort_values("date"), R.sort_values("pub"), left_on="date", right_on="pub", by="code", direction="backward").reset_index(drop=True)
    E["F2"] = (E.r1.fillna(False) & E.r2.fillna(False)).astype(bool); E["F3"] = E.cs >= 0.8; E["NS"] = ~(E.sp > 0.8)
    rules = {"W1 今日發動＋營收強＋分數高＋借券不多": ("pk", "y20", "b20", E.F2 & E.F3 & E.NS), "W2 今日發動＋分數高＋借券不多": ("pk", "y20", "b20", E.F3 & E.NS),
             "W3 新高突破＋分數高＋借券不多": ("nh", "y60", "b60", E.F3 & E.NS)}
    out = {}
    for lab, (kind, y, b, m) in rules.items():
        z = E[(E.kind == kind) & m].dropna(subset=[y, b]).copy(); z["x"] = z[y] - 0.6 - z[b]
        sg = {s: (int((z.seg == s).sum()), round(float(z[z.seg == s].x.mean()), 2) if (z.seg == s).any() else None) for s in SEGS}
        mm = z.groupby(z.date.dt.to_period("M")).x.mean(); t = round(nw_t(mm.reset_index(drop=True), 3), 2)
        ok = all(v[0] >= 20 and v[1] is not None and v[1] > 0 for v in sg.values()) and z.x.mean() > 0 and t >= 2.9
        out[lab] = {"筆數": int(len(z)), "平均": round(float(z.x.mean()), 2), "中位": round(float(z.x.median()), 2), "贏0050%": round(float((z.x > 0).mean() * 100)), "t": t, "五段": sg, "通過": ok}
        print(lab, json.dumps(out[lab], ensure_ascii=False), flush=True)
    json.dump(out, open(f"{REPO}/research/results/watch_sbl.json", "w"), ensure_ascii=False, indent=1, default=str)
