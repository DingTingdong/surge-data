"""平常時期的結論，用外部檢查的教訓重查（2026/11/22；探索，不改任何事先登錄的判斷）。
① 加權：逐筆／每月內逐筆再每月等權／每天先平均、月內日期等權、每月等權；② 去重：同一檔 20 個交易日內只算第一次（新高突破本來就要求前60日沒創過）；
③ 持有期重疊：60 日報酬放在每月序列裡會重疊約 3 個月，t 值改用 Newey-West（落後 = 持有日數 ÷ 20）；④ 研究範圍：成交量小（20日均額 < 1,000 萬）的今日發動另外看。
資料：2011-01～2019-12（data/older＋data/old）與 2020-01～2026-11（data/history），定義同 research/older_era.py（7% 時代漲停 ≥ 6.5%）。
"""
import os, sys, json, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import older_era as oe
CACHE = "/tmp/model_v2"
def get(kind):
    fn = f"{CACHE}/recheck_{kind}.pkl"
    if os.path.exists(fn): return pd.read_pickle(fn)
    R, E = oe.build(kind); pd.to_pickle((R, E), fn); return R, E
def nw_t(s, L):
    s = pd.Series(s).dropna().values; n = len(s)
    if n < 5: return np.nan
    x = s - s.mean(); v = (x @ x) / n
    for k in range(1, L + 1): v += 2 * (1 - k / (L + 1)) * (x[k:] @ x[:-k]) / n
    return float(s.mean() / np.sqrt(v / n))
def per(d): return np.where(d < pd.Timestamp("2016-01-01"), "2011-15", np.where(d < pd.Timestamp("2020-01-01"), "2016-19", np.where(d < pd.Timestamp("2024-01-01"), "2020-23", "2024-26")))
def score_pk(R):
    U = R[R.fd & R.base & R.y20.notna()].copy(); U["m"] = U.date.dt.to_period("M")
    K = R[R.pk & R.y20.notna()].copy(); K["m"] = K.date.dt.to_period("M"); sc = np.zeros(len(K))
    for fc in oe.COMP:
        pr = np.full(len(K), 0.5)
        for m, g in K.groupby("m"):
            ref = np.sort(U.loc[U.m == m, fc].dropna().values)
            if len(ref) < 50: continue
            v = g[fc].values; pr[K.index.get_indexer(g.index)] = np.where(np.isnan(v), 0.5, np.searchsorted(ref, v, side="right") / len(ref))
        sc += pr
    K["cs"] = sc / len(oe.COMP); return K
def dedup20(z, days):
    pos = {d: i for i, d in enumerate(days)}; z = z.sort_values("date"); keep, last = [], {}
    for i, (d, c) in zip(z.index, zip(z.date, z.code)):
        t = pos[d]
        if c in last and t - last[c] < 20: continue
        last[c] = t; keep.append(i)
    return z.loc[keep]
def four(z, col, L=1):
    m = z.date.dt.to_period("M")
    a = z[col].mean(); b = z.groupby(m)[col].mean(); c = z.groupby([m, "date"])[col].mean().groupby(level=0).mean()
    return {"逐筆": round(float(a), 2), "每月等權": (round(float(b.mean()), 2), round(nw_t(b, L), 2)), "每天先平均·每月等權": (round(float(c.mean()), 2), round(nw_t(c, L), 2)), "月數": int(len(b)), "筆數": int(len(z))}
if __name__ == "__main__":
    out = {}
    for kind, a, b in (("older", "2011-01-01", "2019-12-31"), ("new", "2020-01-01", "2026-12-31")):
        R, E = get(kind); days = sorted(R.date.unique())
        K = score_pk(R); K = K[(K.date >= a) & (K.date <= b) & ~K.panic]; K["yn"] = K.y20 - 1.1; K["xb"] = K.yn - K.b20; K["xu"] = K.yn - K.u20; K["per"] = per(K.date)
        Ee = E[(E.date >= a) & (E.date <= b)].copy(); Ee["xu"] = Ee.y - 0.6 - Ee.u; Ee["per"] = per(Ee.date)
        for p in sorted(K.per.unique()):
            k = K[K.per == p]; kb = k[k.base]; e = Ee[Ee.per == p]
            o = {"弱(<0.5)減0050": four(kb[kb.cs < 0.5], "xb"), "弱(<0.5)減0050 去重": four(dedup20(kb[kb.cs < 0.5], days), "xb"),
                 "強(≥0.8)減0050": four(kb[kb.cs >= 0.8], "xb"), "強(≥0.8)減等權": four(kb[kb.cs >= 0.8], "xu"),
                 "全部今日發動減0050": four(kb, "xb"), "量小的今日發動 弱(<0.5)減0050": four(k[~k.base & (k.cs < 0.5)], "xb"),
                 "量小的今日發動 全部減0050": four(k[~k.base], "xb"), "新高突破60日減等權": four(e, "xu", L=3)}
            out[p] = o; print(f"\n== {p}"); [print("  ", kk, vv) for kk, vv in o.items()]
    json.dump(out, open(os.path.join(oe.REPO, "research/results/normal_recheck.json"), "w"), ensure_ascii=False, indent=1)
