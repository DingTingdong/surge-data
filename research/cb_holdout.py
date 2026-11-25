"""可轉債代收公告後 10 天：用沒看過的 2006–2010 驗證（第 57 輪；2026/11/25 18:30 事先登錄，看結果前寫下，看到結果後不修改）

起因：第 56 輪 B 在 2011–2026 看到「代收公告隔天開盤買、抱 10 天」比 0050 多 1.79%（t 4.49），但 10 天是看過 5 個天數後最好的，不算驗證。
  2006–2010 的公告這幾輪都沒算過 10 天（第 50 輪只報過 2006–09 的 60 日減全部 +2.34，那個數字已看過，這輪不用它判斷），拿來當考題。
規則完全照第 56 輪 B，不改：主旨含「轉換公司債」且含「代收」，排除私募／海外／外國／子公司；抓得到「第 N 次」照次數去重，抓不到的同公司 90 天內算一次；
  同公司 30 天內只留一筆。公告日（或之後第一個交易日）的隔天開盤買，開盤一價漲停買不到就放棄；需在分析範圍內；抱 10 個交易日收盤賣，扣 0.6%。
  0050 同一天開盤買、扣 0.18%（含息指數用 0050 當天開盤／收盤比例換算）。資料：vs0050_oldestonly_extra.pkl，公告日 2006-01-01～2010-12-31。
判斷（事先寫下）：10 日減 0050 每筆平均 > 0，且以公告月份分群 NW t（落後 3）≥ 2.0 → 通過。
  另報（不影響判斷）：中位數、贏 0050 比例、2006–08 與 2009–10 兩段、減分析範圍、5／20 日。
通過 → 網站卡片「剛發可轉債」改標前 10 個交易日、開始 Telegram 提醒（小部位），並照常往後記錄；不通過 → 只往後記錄，不提醒。
"""
import json, os, sys, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
from vs0050 import nw_t
COST, BCOST = 0.6, 0.18

def events():
    N_ = pd.concat([pd.read_csv(f, dtype={"code": str, "date": str, "time": str}) for f in sorted(glob.glob(f"{REPO}/data/mopsnews/*.csv.gz"))]).drop_duplicates(["date", "time", "code", "subject"]).reset_index(drop=True)
    s = N_.subject.fillna("").str.replace(r"\s+", "", regex=True)
    k = N_[s.str.contains("轉換公司債") & s.str.contains("代收") & ~s.str.contains("私募|海外|外國|子公司")].copy()
    k["n"] = s[k.index].str.extract(r"第([一二三四五六七八九十\d]+)次")[0]; k["d"] = pd.to_datetime(k.date, format="%Y%m%d", errors="coerce")
    k = k.dropna(subset=["d"]).sort_values("d"); a = k[k.n.notna()].drop_duplicates(["code", "n"]); rows, last = [], {}
    for d, c in zip(k[k.n.isna()].d, k[k.n.isna()].code):
        if c in last and (d - last[c]).days < 90: continue
        last[c] = d; rows.append((d, c))
    iss = pd.concat([a[["d", "code"]], pd.DataFrame(rows, columns=["d", "code"])]).sort_values("d"); keep, lc = [], {}
    for d, c in zip(iss.d, iss.code):
        if c in lc and (d - lc[c]).days < 30: continue
        lc[c] = d; keep.append((d, c))
    return pd.DataFrame(keep, columns=["d", "code"])

if __name__ == "__main__":
    X = pd.read_pickle("/tmp/model_v2/vs0050_oldestonly_extra.pkl"); days = pd.DatetimeIndex(X["days"]); N = len(days); cj = {c: j for j, c in enumerate(X["cols"])}
    Cff = pd.DataFrame(X["C"]).ffill(limit=5).values; O = np.asarray(X["O"], float); inb = np.asarray(X["inb"]).astype(bool); lk = np.asarray(X["lockn"]).astype(bool); b = np.asarray(X["bidx"], float)
    fs = sorted(glob.glob(f"{REPO}/data/*/history/*.csv.gz") + glob.glob(f"{REPO}/data/history/*.csv.gz"))
    E5 = pd.concat([pd.read_csv(f, dtype={"code": str}, usecols=["date", "code", "open", "close"]).query("code in ['50','0050']") for f in fs]).drop_duplicates("date")
    E5["date"] = pd.to_datetime(E5.date); E5 = E5.set_index("date").reindex(days)
    bO = b * (E5.open / E5.close).where((E5.open > 0) & (E5.close > 0)).fillna(1.0).values
    RU = {}
    for H in (5, 10, 20):
        R = np.full(N, np.nan)
        for ie in range(1, N - H + 1):
            rr = Cff[ie + H - 1] / np.where(O[ie] > 0, O[ie], np.nan) - 1; m = inb[ie - 1] & np.isfinite(rr) & ~lk[ie - 1]
            if m.sum() > 50: R[ie] = np.nanmean(rr[m]) * 100 - COST
        RU[H] = R
    EV = events(); EV = EV[(EV.d >= "2006-01-01") & (EV.d < "2011-01-01")]; print("2006–2010 事件（去重後）", len(EV), flush=True)
    rows, skip = [], {"不在資料": 0, "期間外或不足": 0, "買不到或不在範圍": 0}
    for d, c in zip(EV.d, EV.code):
        j = cj.get(c)
        if j is None: skip["不在資料"] += 1; continue
        i = int(np.searchsorted(days.values, np.datetime64(d))); ie = i + 1
        if i >= N or ie + 9 >= N: skip["期間外或不足"] += 1; continue
        if not (O[ie, j] > 0) or lk[ie - 1, j] or not inb[i, j]: skip["買不到或不在範圍"] += 1; continue
        r = {"date": days[i], "code": c}
        for H in (5, 10, 20):
            ix = ie + H - 1
            if ix >= N: r[f"x{H}"] = r[f"u{H}"] = np.nan; continue
            rH = (Cff[ix, j] / O[ie, j] - 1) * 100 - COST; r[f"x{H}"] = rH - ((b[ix] / bO[ie] - 1) * 100 - BCOST); r[f"u{H}"] = rH - RU[H][ie]
        rows.append(r)
    V = pd.DataFrame(rows); print("可用筆數", len(V), skip, flush=True)
    m = V.groupby(V.date.dt.to_period("M")).x10.mean(); t = nw_t(m.reset_index(drop=True), 3)
    res = {"事件數": int(len(EV)), "可用筆數": int(len(V)), "排除": skip, "公告月份數": int(len(m)),
           "10日減0050 每筆": round(float(V.x10.mean()), 2), "t": round(float(t), 2), "中位": round(float(V.x10.median()), 2), "贏0050%": round(float((V.x10 > 0).mean() * 100)),
           "2006–08": [int((V.date.dt.year <= 2008).sum()), round(float(V[V.date.dt.year <= 2008].x10.mean()), 2)], "2009–10": [int((V.date.dt.year >= 2009).sum()), round(float(V[V.date.dt.year >= 2009].x10.mean()), 2)],
           "10日減分析範圍": round(float(V.u10.mean()), 2), "5日減0050": round(float(V.x5.mean()), 2), "20日減0050": round(float(V.x20.mean()), 2)}
    res["通過"] = bool(res["10日減0050 每筆"] > 0 and res["t"] >= 2.0)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    json.dump(res, open(f"{REPO}/research/results/cb_holdout.json", "w"), ensure_ascii=False, indent=1)
