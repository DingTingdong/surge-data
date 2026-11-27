"""今日發動的當天成交值門檻（第 69 輪；2026/11/27 12:05 登錄，看結果前寫下，看到結果後不修改）

起因：使用者 11/27 問「今日發動要求當天成交值 ≥ 2,000 萬會不會太高？很多漲停一早就鎖住，成交量本來就不大」（宣德 11/12 漲停只有 1,640 萬）。
事件：2011–2026，上市＋上櫃普通股（排除 0 開頭、91xx TDR），當天漲 ≥ 9%（2015/6 前 ≥ 6.5% 改用 ≥ 6%）、前 10 個交易日沒有同樣的漲幅、前 20 日平均成交值 ≥ 300 萬（同現行）。
依當天成交值分組：< 500 萬、500～1,000 萬、1,000～2,000 萬、≥ 2,000 萬（現行）。另報「一價到底（開＝高＝低＝收）」的比例。
進出：隔天開盤買（開盤比前收漲 ≥ 9.5%／2015/6 前 6.5% → 買不到，不計入）；抱 20 個交易日收盤賣，扣 0.6%；減同期 0050（隔天開盤 → 第 20 天收盤，含息、扣 0.18%）。
報：每組筆數、買不到的比例、每筆平均減 0050、中位數、贏 0050 比例、月份 NW t（落後 3）、四段。
判斷（事先寫下）：若 1,000～2,000 萬這組「每筆平均減 0050 ≥ 現行組、中位數 ≥ 現行組、四段至少 3 段不低於現行組」，網站今日發動門檻降到 1,000 萬；
  否則不改。今日發動本身仍只是觀察用（平常時期平均輸 0050）。
"""
import glob, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.path.insert(0, REPO); sys.argv = sys.argv[:1]
from mp_exec import E50
from vs0050 import nw_t
import robust_select as RS
SEG4 = ["2011-15", "2016-19", "2020-23", "2024-26"]

if __name__ == "__main__":
    os.chdir(REPO)
    fs = sorted(glob.glob("data/older/history/*.csv.gz") + glob.glob("data/old/history/*.csv.gz") + glob.glob("data/history/*.csv.gz"))
    p = pd.concat([pd.read_csv(f, dtype={"code": str}, usecols=["date", "code", "open", "high", "low", "close", "amount", "market"]) for f in fs] +
                  [pd.read_csv("data/prices.csv", dtype={"code": str}, usecols=["date", "code", "open", "high", "low", "close", "amount", "market"])], ignore_index=True)
    p = p[~p.code.str.startswith("0") & ~p.code.str.match(r"^91\d\d$")].drop_duplicates(["date", "code"], keep="last"); p["date"] = pd.to_datetime(p.date)
    tw = set(p.loc[p.market == "TWSE", "date"]); p = p[(p.market == "TWSE") | p.date.isin(tw)]
    pv = lambda c: p.pivot(index="date", columns="code", values=c).sort_index()
    C, O, H, L, A = pv("close"), pv("open"), pv("high"), pv("low"), pv("amount")
    adj = pd.read_csv("data/adjust.csv", dtype={"code": str}); adj["date"] = pd.to_datetime(adj.date)
    adj = adj[adj.code.isin(C.columns) & adj.date.isin(C.index) & (adj.ratio > 0.05) & (adj.ratio < 20)]
    fac = adj.groupby(["date", "code"]).ratio.prod().unstack().reindex(index=C.index, columns=C.columns).fillna(1.0)
    Mx = fac.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0); Ca, Oa = C * Mx, O * Mx
    days = C.index; thr = pd.Series(np.where(days >= pd.Timestamp("2015-06-01"), 0.09, 0.06), index=days)
    r1 = Ca.pct_change(fill_method=None); up = r1.ge(thr, axis=0) & r1.notna()
    prior = up.shift(1).rolling(10, min_periods=1).max().fillna(0).astype(bool)
    a20p = A.shift(1).rolling(20, min_periods=20).mean()
    ev = up & ~prior & (a20p >= 3e6)
    e5 = E50.reindex(days); bidx = (e5.close / e5.close.shift(1)).fillna(1.0)
    # 0050 含息：用 adjust.csv 的 0050 比例
    a50 = pd.read_csv("data/adjust.csv", dtype={"code": str}); a50 = a50[a50.code == "0050"]; a50["date"] = pd.to_datetime(a50.date)
    f50 = a50.groupby("date").ratio.prod().reindex(days).fillna(1.0); f50 = f50.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)
    c5, o5 = e5.close * f50, e5.open * f50
    Cv, Ov, Ocv = Ca.values, Oa.values, O.values; pcv = C.shift(1).values; N = len(days); rows = []
    lock = ((O == H) & (H == L) & (L == C)).values
    for t, j in zip(*np.nonzero(ev.values)):
        if days[t] < pd.Timestamp("2011-01-01") or t + 20 >= N: continue
        ie = t + 1; o = Ov[ie, j]; craw = C.values[t, j]
        gap = 0.095 if days[ie] >= pd.Timestamp("2015-06-01") else 0.065
        miss = not (Ocv[ie, j] > 0) or (craw > 0 and Ocv[ie, j] / craw - 1 >= gap - 1e-9)
        x = np.nan
        if not miss:
            cl = pd.Series(Cv[ie:ie + 20, j]).ffill().iloc[-1]
            if cl == cl and o > 0: x = (cl / o - 1) * 100 - 0.6 - ((c5.iloc[ie + 19] / o5.iloc[ie] - 1) * 100 - 0.18)
        rows.append({"date": days[t], "code": C.columns[j], "amt": A.values[t, j], "lock": bool(lock[t, j]), "miss": miss, "x": x})
    E = pd.DataFrame(rows); E["組"] = pd.cut(E.amt, [0, 5e6, 1e7, 2e7, np.inf], right=False, labels=["<500萬", "500～1,000萬", "1,000～2,000萬", "≥2,000萬（現行）"])
    res = {}
    for g, z in E.groupby("組", observed=True):
        y = z.dropna(subset=["x"]); m = y.groupby(y.date.dt.to_period("M")).x.mean(); m.index = m.index.to_timestamp(); sg = RS.seg(y.date)
        res[str(g)] = {"筆數": len(z), "一價到底%": round(float(z.lock.mean() * 100)), "隔天買不到%": round(float(z.miss.mean() * 100)), "可買筆數": len(y),
                       "每筆平均減0050": round(float(y.x.mean()), 2), "中位數": round(float(y.x.median()), 2), "贏0050%": round(float((y.x > 0).mean() * 100)),
                       "月份t": round(nw_t(m.reset_index(drop=True), 3), 2), "四段": [round(float(y.x[sg == q].mean()), 2) for q in SEG4]}
        print(g, json.dumps(res[str(g)], ensure_ascii=False), flush=True)
    a, b = res["1,000～2,000萬"], res["≥2,000萬（現行）"]
    ok = [a["每筆平均減0050"] >= b["每筆平均減0050"], a["中位數"] >= b["中位數"], sum(x >= y for x, y in zip(a["四段"], b["四段"])) >= 3]
    res["判斷"] = {"各條件": ok, "門檻降到1,000萬": all(ok)}; print(res["判斷"])
    json.dump(res, open("research/results/pk_amount.json", "w"), ensure_ascii=False, indent=1)
