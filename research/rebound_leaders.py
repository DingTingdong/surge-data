"""大跌後反彈最強的股票，之後 60 天好不好（第 61 輪；2026/11/26 09:05 登錄，看結果前寫下，看到結果後不修改）

起因：使用者問「每次大跌之後 10 日內反彈最好的股票，到 60 日的股價通常好不好」（這波 11/20 低點後，AI 供應鏈反彈最多）。
大跌事件（兩種，各自報）：
  E1 上市空頭家數第一次 ≥ 750（前 20 個交易日都 < 750；當天市場別、排除 TDR，資料 research/audit/out/B_bear_all_pit_notdr.csv）→ 2014 起。
  E2 0050 含息指數收盤比近 60 日最高跌 ≥ 10%，第一次（前 20 個交易日都沒有）→ 2006 起，次數較多。
  t0＝事件日。排名日 t＝t0 後第 K 個交易日（K＝10 主要；K＝5 另報）。
反彈＝t 日收盤 ÷ 該股 t0～t 之間最低收盤 − 1（排名日當時就知道）。範圍＝t 日的雷達分析範圍。
買：t 的隔天開盤；開盤比前收漲 ≥ 9.5%（2015/6 前 6.5%）或沒開盤 → 不算這檔（只用開盤價判斷）。抱 60 個交易日收盤，扣 0.6%。
比較：同日開盤買的分析範圍等權（同樣排除）、0050（同日開盤、含息、扣 0.18%）。另報反彈最弱 20 檔、反彈依十分位分組。也報 20 日。
判斷（描述性，事先寫下）：主要＝E1、K＝10、反彈前 20 檔、60 日。
  「通常好」＝事件平均減分析範圍 > 0 且 ≥ 2/3 事件 > 0，且事件平均減 0050 > 0；E2 方向相同才算穩定。不改網站，只回答問題。
"""
import json, os, sys, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
COST, BCOST = 0.6, 0.18
fs = sorted(glob.glob(f"{REPO}/data/*/history/*.csv.gz") + glob.glob(f"{REPO}/data/history/*.csv.gz"))
E50 = pd.concat([pd.read_csv(f, dtype={"code": str}, usecols=["date", "code", "open", "close"]).query("code in ['50','0050']") for f in fs]).drop_duplicates("date")
E50["date"] = pd.to_datetime(E50.date); E50 = E50.set_index("date")[["open", "close"]]
_e = pd.read_csv(f"{REPO}/data/etf.csv", dtype={"code": str}).query("code == '0050'"); _e["date"] = pd.to_datetime(_e.date); E50 = E50.combine_first(_e.set_index("date")[["open", "close"]])

def load(path):
    X = pd.read_pickle(path); days = pd.DatetimeIndex(X["days"]); b = np.asarray(X["bidx"], float)
    e5 = E50.reindex(days); r = (e5.open / e5.close).where((e5.open > 0) & (e5.close > 0)).fillna(1.0).values
    return dict(days=days, cols=list(X["cols"]), C=pd.DataFrame(X["C"]).ffill(limit=5).values, O=np.asarray(X["O"], float), pc=np.asarray(X["prevc"], float),
                inb=np.asarray(X["inb"]).astype(bool), b=b, bO=b * r)

def events():
    B = pd.read_csv(f"{REPO}/research/audit/out/B_bear_all_pit_notdr.csv", index_col=0, parse_dates=True).iloc[:, 0]
    v = B.values; e1 = [B.index[t] for t in range(20, len(v)) if v[t] >= 750 and (v[t - 20:t] < 750).all()]
    out = {"E1 空頭家數第一次750": e1, "E2 0050跌10%": []}
    for path in ("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "/tmp/model_v2/vs0050_extra.pkl"):
        X = pd.read_pickle(path); d = pd.DatetimeIndex(X["days"]); b = pd.Series(np.asarray(X["bidx"], float), index=d)
        dd = (b / b.rolling(60, min_periods=40).max() - 1) <= -0.10
        for t in range(20, len(d)):
            if dd.iloc[t] and not dd.iloc[t - 20:t].any() and d[t] >= pd.Timestamp("2006-01-01"): out["E2 0050跌10%"].append(d[t])
    out["E2 0050跌10%"] = sorted(set(out["E2 0050跌10%"])); return out

def one(D, t0, K, N=20):
    days = D["days"]; i0 = int(np.searchsorted(days.values, np.datetime64(t0)))
    if i0 >= len(days) or days[i0] != t0: return None
    t = i0 + K; ie = t + 1
    if ie + 59 >= len(days): return None
    C, O, pc, inb = D["C"], D["O"], D["pc"], D["inb"]
    lo = np.nanmin(C[i0:t + 1], axis=0); reb = C[t] / lo - 1
    thr = 0.095 if days[ie] >= pd.Timestamp("2015-06-01") else 0.065
    ok = inb[t] & np.isfinite(reb) & (O[ie] > 0) & ~((pc[ie] > 0) & (O[ie] / np.where(pc[ie] > 0, pc[ie], np.nan) - 1 >= thr - 1e-9))
    out = {"t0": str(t0.date()), "rank": str(days[t].date())}
    for H in (20, 60):
        ix = ie + H - 1; r = (C[ix] / np.where(O[ie] > 0, O[ie], np.nan) - 1) * 100 - COST; okH = ok & np.isfinite(r)
        u = float(np.nanmean(r[okH])); bb = (D["b"][ix] / D["bO"][ie] - 1) * 100 - BCOST
        idx = np.where(okH)[0]; order = idx[np.argsort(-reb[idx])]
        top, bot = order[:N], order[-N:]
        dec = np.array_split(order, 10)
        out[f"top{H}"] = float(np.mean(r[top])); out[f"bot{H}"] = float(np.mean(r[bot])); out[f"u{H}"] = u; out[f"b{H}"] = bb
        out[f"dec{H}"] = [round(float(np.mean(r[g])) - u, 2) for g in dec]
        if H == 60: out["top_names"] = [D["cols"][j] for j in top[:5]]; out["top_reb"] = round(float(np.mean(reb[top]) * 100), 1)
    return out

if __name__ == "__main__":
    EV = events(); print({k: len(v) for k, v in EV.items()}, flush=True)
    Ds = [(pd.Timestamp("2005-01-01"), pd.Timestamp("2010-12-31"), load("/tmp/model_v2/vs0050_oldestonly_extra.pkl")), (pd.Timestamp("2011-01-01"), pd.Timestamp("2027-01-01"), load("/tmp/model_v2/vs0050_extra.pkl"))]
    res = {}
    for ek, ts in EV.items():
        for K in (10, 5):
            rows = []
            for t0 in ts:
                D = next((d for a, b_, d in Ds if a <= t0 <= b_), None)
                if D is None: continue
                r = one(D, t0, K)
                if r: rows.append(r)
            if not rows: continue
            R = pd.DataFrame(rows); s = {"事件數": len(R)}
            for H in (60, 20):
                xu, xb, bu = R[f"top{H}"] - R[f"u{H}"], R[f"top{H}"] - R[f"b{H}"], R[f"bot{H}"] - R[f"u{H}"]
                s[f"{H}日"] = {"反彈前20 平均": round(R[f"top{H}"].mean(), 2), "分析範圍": round(R[f"u{H}"].mean(), 2), "0050": round(R[f"b{H}"].mean(), 2),
                              "前20減範圍": round(xu.mean(), 2), "贏範圍次數": f"{int((xu > 0).sum())}/{len(R)}", "前20減0050": round(xb.mean(), 2), "贏0050次數": f"{int((xb > 0).sum())}/{len(R)}",
                              "反彈最弱20減範圍": round(bu.mean(), 2), "十分位減範圍(反彈強→弱)": [round(float(np.mean([d[i] for d in R[f'dec{H}']])), 2) for i in range(10)]}
            if K == 10 and ek.startswith("E1"):
                s["逐次"] = R[["t0", "rank", "top_reb", "top60", "u60", "b60", "top_names"]].round(1).to_dict("records")
            key = f"{ek}｜K={K}"; res[key] = s; print(key, json.dumps(s, ensure_ascii=False)[:1500], flush=True)
    m = res.get("E1 空頭家數第一次750｜K=10", {}).get("60日", {}); m2 = res.get("E2 0050跌10%｜K=10", {}).get("60日", {})
    def good(m):
        if not m: return False
        a, n = m["贏範圍次數"].split("/"); return m["前20減範圍"] > 0 and int(a) >= 2 / 3 * int(n) and m["前20減0050"] > 0
    res["結論"] = {"E1通常好": good(m), "E2通常好": good(m2), "穩定": good(m) and good(m2)}; print(res["結論"])
    json.dump(res, open(f"{REPO}/research/results/rebound_leaders.json", "w"), ensure_ascii=False, indent=1, default=str)
