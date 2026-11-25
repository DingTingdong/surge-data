"""可轉債代收後 10 天：照第三份外部覆核修正細節後重算（第 57b 輪；2026/11/25 19:10 登錄。這是穩健性檢查，不改第 57 輪「不通過」的判定，也不改主要持有天數）

覆核指出、這輪修正：
  F1 公告日不是交易日（週末、假日）時，原本是「之後第一個交易日」的再隔天才買，晚一天 → 改成之後第一個交易日開盤買。
  F2 原本要求公告後還有 60 天資料，近期能算 10 天的被排除 → 只要求 10 天（5／20 天各自要求）。
  F3 平均與 t 口徑：每月平均（月內先平均、月份等權），t 用同一組月平均；另報「沒訊號的月份算 0 筆、用日曆月」的區塊重抽 95% 區間（月份區塊，2,000 次）。
  F4 勝率只算有報酬的筆數。
  F5 統計 0050 開盤價缺值、改用收盤價代替的筆數。
  F6 出場日一價跌停（開＝高＝低＝收、跌幅 ≥ 9%（2015/6 前 6.5%））賣不掉 → 順延到第一個沒鎖跌停的交易日開盤賣；出場日沒成交（停牌、下市）→ 順延到下一個有成交的開盤，找不到記為 −100%（下市）。報影響筆數。
  F7 去重規則完整寫出：抓得到「第 N 次」照公司＋次數；抓不到的同公司 90 天內一次；兩者合併後同公司 30 天內只留一筆。
其餘規則同第 56 輪 B／第 57 輪。兩段：2006–2010（第 57 輪考題）、2011–2026（第 56 輪）。
"""
import json, os, sys, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
from vs0050 import nw_t
from cb_holdout import events
COST, BCOST = 0.6, 0.18
fs = sorted(glob.glob(f"{REPO}/data/*/history/*.csv.gz") + glob.glob(f"{REPO}/data/history/*.csv.gz"))
E50 = pd.concat([pd.read_csv(f, dtype={"code": str}, usecols=["date", "code", "open", "close"]).query("code in ['50','0050']") for f in fs]).drop_duplicates("date")
E50["date"] = pd.to_datetime(E50.date); E50 = E50.set_index("date")

def run(path, a, b_, EV, fixed):
    X = pd.read_pickle(path); days = pd.DatetimeIndex(X["days"]); N = len(days); cj = {c: j for j, c in enumerate(X["cols"])}
    Craw = np.asarray(X["C"], float); Cff = pd.DataFrame(Craw).ffill(limit=5).values; O = np.asarray(X["O"], float); Hh = np.asarray(X["H"], float); Ll = np.asarray(X["L"], float)
    inb = np.asarray(X["inb"]).astype(bool); lk = np.asarray(X["lockn"]).astype(bool); b = np.asarray(X["bidx"], float)
    e5 = E50.reindex(days); okb = ((e5.open > 0) & (e5.close > 0)).values; bO = b * np.where(okb, (e5.open / e5.close).values, 1.0)
    RU = {}
    for H in (5, 10, 20):
        R = np.full(N, np.nan)
        for ie in range(1, N - H + 1):
            rr = Cff[ie + H - 1] / np.where(O[ie] > 0, O[ie], np.nan) - 1; m = inb[ie - 1] & np.isfinite(rr) & ~lk[ie - 1]
            if m.sum() > 50: R[ie] = np.nanmean(rr[m]) * 100 - COST
        RU[H] = R
    rows, cnt = [], {"週末假日公告": 0, "0050開盤缺值": 0, "出場鎖跌停": 0, "出場沒成交": 0, "下市": 0}
    for d, c in zip(EV.d, EV.code):
        if not (pd.Timestamp(a) <= d < pd.Timestamp(b_)): continue
        j = cj.get(c); i = int(np.searchsorted(days.values, np.datetime64(d)))
        if j is None or i >= N: continue
        nt = days[i] != d
        ie = i if (fixed and nt) else i + 1; isig = ie - 1
        if ie >= N or not (O[ie, j] > 0) or lk[ie - 1, j] or not inb[isig, j]: continue
        need = 10 if fixed else 60
        if ie + need >= N: continue
        cnt["週末假日公告"] += int(nt); cnt["0050開盤缺值"] += int(not okb[ie])
        r = {"date": days[isig], "code": c}
        for H in (5, 10, 20):
            ix = ie + H - 1
            if ix >= N: r[f"x{H}"] = r[f"u{H}"] = np.nan; continue
            px, bx_ = Cff[ix, j], b[ix]
            if fixed:
                pv = Craw[ix - 1, j] if Craw[ix - 1, j] > 0 else Cff[ix - 1, j]; lim = 0.91 if days[ix] >= pd.Timestamp("2015-06-01") else 0.935
                ld = Craw[ix, j] > 0 and O[ix, j] == Hh[ix, j] == Ll[ix, j] == Craw[ix, j] and pv > 0 and Craw[ix, j] / pv <= lim
                nd = not (Craw[ix, j] > 0)
                if ld or nd:
                    if H == 10: cnt["出場鎖跌停" if ld else "出場沒成交"] += 1
                    q = ix + 1
                    while q < N and not (O[q, j] > 0 and not (O[q, j] == Hh[q, j] == Ll[q, j] and Craw[q - 1, j] > 0 and O[q, j] / Craw[q - 1, j] <= lim)): q += 1
                    if q < N and q - ix <= 60: px, bx_ = O[q, j], bO[q]
                    elif q >= N: r[f"x{H}"] = r[f"u{H}"] = np.nan; continue
                    else:
                        px = 0.0; bx_ = b[min(ix, N - 1)]
                        if H == 10: cnt["下市"] += 1
            rH = (px / O[ie, j] - 1) * 100 - COST; r[f"x{H}"] = rH - ((bx_ / bO[ie] - 1) * 100 - BCOST); r[f"u{H}"] = rH - RU[H][ie]
        rows.append(r)
    return pd.DataFrame(rows), cnt

def stat(V, col, a, b_):
    Z = V.dropna(subset=[col])
    if not len(Z): return {}
    m = Z.groupby(Z.date.dt.to_period("M"))[col].mean()
    cal = m.reindex(pd.period_range(pd.Timestamp(a), pd.Timestamp(b_) - pd.Timedelta(days=1), freq="M"))   # 日曆月（沒訊號＝缺）
    rng = np.random.default_rng(0); vals = cal.values; L = len(vals); bs = []
    for _ in range(2000):
        st = rng.integers(0, L, size=L // 6 + 1); idx = np.concatenate([np.arange(s, s + 6) % L for s in st])[:L]
        v = vals[idx]; v = v[np.isfinite(v)]
        if len(v): bs.append(v.mean())
    return {"筆數": int(len(Z)), "每筆平均": round(float(Z[col].mean()), 2), "每月平均": round(float(m.mean()), 2), "t(月平均)": round(nw_t(m.reset_index(drop=True), 3), 2),
            "中位": round(float(Z[col].median()), 2), "贏%": round(float((Z[col] > 0).mean() * 100)), "6個月區塊重抽95%": [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]}

if __name__ == "__main__":
    EV = events(); res = {}
    for lab, path, a, b_ in (("2006–2010", "/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2011-01-01"), ("2011–2026", "/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01")):
        for fx in (False, True):
            V, cnt = run(path, a, b_, EV, fx); k = f"{lab}｜{'修正後' if fx else '原算法'}"
            res[k] = {"影響筆數": cnt, **{f"{H}日減0050": stat(V, f"x{H}", a, b_) for H in (5, 10, 20)}, "10日減分析範圍": stat(V, "u10", a, b_).get("每月平均")}
            print(k, json.dumps(res[k], ensure_ascii=False), flush=True)
    json.dump(res, open(f"{REPO}/research/results/cb_robust.json", "w"), ensure_ascii=False, indent=1)
