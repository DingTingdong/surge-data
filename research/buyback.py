"""庫藏股買回公告之後的股價（第 17 輪；2026/11/23 10:05 登錄，資料抓取之前寫下，看到結果後不修改）

依據：公司宣布買回自家股票，之後常有正的異常報酬（Ikenberry, Lakonishok & Vermaelen 1995；管理層認為股價被低估的訊號）。
資料：公開資訊觀測站 ajax_t35sc09（tools/fetch_buyback.py → data/buyback/buyback.csv.gz），2006–2026，上市＋上櫃。
事件：董事會決議日期；決議日之後的第一個交易日開盤買（公告常在收盤後），整天一價鎖死買不到不計；抱 20 個交易日，扣 1.1%；
  只算分析範圍內（20 日均額 ≥ 1,000 萬）的股票；減等權分析範圍同期報酬。同一家公司 60 天內的第二次決議不重複算。
彙總：同一個月的事件先平均，每月一筆；Newey-West t（落後 1）。五段：2006–09、2011–15、2016–19、2020–23、2024–26（2010 併入 2011–15）。
判斷：
  B1 全部事件：每月平均減等權 > 0、t ≥ 2.9、五段都 > 0 → 網站卡片標「庫藏股」（決議後 20 個交易日內）。
  B2 恐慌期間的事件（決議日在恐慌期間）：平均減等權 > 0，而且 2006–2019、2020–2026 兩段都 > 0 → 撿跌深名單裡有庫藏股的排前面。
另報：依買回目的（轉讓員工／維護股東權益／註銷）、決議前 20 日跌幅分組；5 日、60 日的結果。
"""
import json, os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
from vs0050 import nw_t

def load_bb():
    B = pd.read_csv(f"{REPO}/data/buyback/buyback.csv.gz", dtype=str)
    cc = [c for c in B.columns if "代號" in c][0]; dc = [c for c in B.columns if "決議日期" in c][0]; pc = [c for c in B.columns if "目的" in c][0]
    def roc(s):
        try: y, m, d = re.split(r"[/.-]", str(s).strip())[:3]; return pd.Timestamp(int(y) + 1911, int(m), int(d))
        except Exception: return pd.NaT
    import re
    E = pd.DataFrame({"code": B[cc].str.strip(), "date": B[dc].map(roc), "purpose": B[pc].astype(str).str.strip()}).dropna(subset=["date"]).drop_duplicates(["code", "date"]).sort_values(["code", "date"])
    keep, last = [], {}
    for i, c, d in zip(E.index, E.code, E.date):
        if c in last and (d - last[c]).days < 60: continue
        last[c] = d; keep.append(i)
    return E.loc[keep]

def returns(path, E, start, end):
    X = pd.read_pickle(path); days, cols = X["days"], X["cols"]; N = len(days); J = len(cols)
    Cff = pd.DataFrame(X["C"]).ffill(limit=60).values; O = X["O"]; lk = X["lockn"]; inb = X["inb"]; pn = X["panic"].astype(bool)
    Onext = np.vstack([O[1:], np.full((1, J), np.nan)]); ok = (Onext > 0) & ~lk; out = []
    U = {}
    for H in (5, 20, 60):
        fut = np.vstack([Cff[H:], np.full((H, J), np.nan)]); y = (fut / Onext - 1) * 100; U[H] = (y, np.nanmean(np.where(inb & ok, y, np.nan), axis=1))
    cj = {c: j for j, c in enumerate(cols)}; r20 = np.vstack([np.full((20, J), np.nan), Cff[20:] / Cff[:-20] - 1]) * 100
    for c, d, p in zip(E.code, E.date, E.purpose):
        if c not in cj or d < pd.Timestamp(start) or d >= pd.Timestamp(end): continue
        s = int(np.searchsorted(days.values, np.datetime64(d), side="right")) - 1
        if s < 20 or s + 60 >= N: continue
        j = cj[c]
        if not inb[s, j] or not ok[s, j]: continue
        row = {"date": days[s], "code": c, "purpose": p, "panic": bool(pn[s]), "pre20": r20[s, j]}
        for H in (5, 20, 60): row[f"x{H}"] = U[H][0][s, j] - 1.1 - U[H][1][s]
        out.append(row)
    return pd.DataFrame(out)

def seg(d): return np.where(d < pd.Timestamp("2010-01-01"), "2006-09", np.where(d < pd.Timestamp("2016-01-01"), "2011-15", np.where(d < pd.Timestamp("2020-01-01"), "2016-19", np.where(d < pd.Timestamp("2024-01-01"), "2020-23", "2024-26"))))
def cal(z, col="x20"):
    m = z.groupby(z.date.dt.to_period("M"))[col].mean(); return round(float(m.mean()), 2), round(nw_t(m, 1), 2), int(len(z)), int(len(m))

if __name__ == "__main__":
    E = load_bb(); print("決議（60 天內去重）", len(E), "年份", E.date.dt.year.value_counts().sort_index().to_dict(), flush=True)
    R = pd.concat([returns("/tmp/model_v2/vs0050_oldestonly_extra.pkl", E, "2006-01-01", "2010-01-01"), returns("/tmp/model_v2/vs0050_extra.pkl", E, "2010-01-01", "2027-01-01")], ignore_index=True)
    R["seg"] = seg(R.date); out = {}
    a = cal(R); segs = {k: cal(R[R.seg == k]) for k in ("2006-09", "2011-15", "2016-19", "2020-23", "2024-26")}
    out["B1 全部 20日減等權（平均, t, 筆數, 月數）"] = a; out["B1 五段"] = segs
    out["B1 成立"] = bool(a[0] > 0 and a[1] >= 2.9 and all(v[0] > 0 for v in segs.values()))
    P = R[R.panic]; p1, p2 = P[P.date < "2020-01-01"], P[P.date >= "2020-01-01"]
    out["B2 恐慌期間 全部／2006–2019／2020–2026"] = (cal(P), cal(p1), cal(p2))
    out["B2 成立"] = bool(len(P) and P.x20.mean() > 0 and len(p1) and p1.x20.mean() > 0 and len(p2) and p2.x20.mean() > 0)
    out["另報 5日／60日"] = (cal(R, "x5"), cal(R, "x60"))
    out["另報 依目的"] = {k: cal(z) for k, z in R.groupby(R.purpose.str.slice(0, 10)) if len(z) >= 30}
    R["q"] = pd.qcut(R.pre20, 3, labels=["決議前漲最多", "中間", "決議前跌最多"][::-1])
    out["另報 依決議前20日漲跌"] = {str(k): cal(z) for k, z in R.groupby("q")}
    for k, v in out.items(): print("==", k, json.dumps(v, ensure_ascii=False, default=str))
    json.dump(out, open(f"{REPO}/research/results/buyback.json", "w"), ensure_ascii=False, indent=1, default=str)
