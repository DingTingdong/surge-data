"""大盤判斷：Put/Call 比、課程「賭跳水」時機（第 53 輪；2026/11/25 12:45 事先登錄，看結果前寫下，看到結果後不修改）

起因：使用者只想判斷大盤，偶爾小額買選擇權當樂透。第 52 輪三大法人分計沒用；這輪測 2007 年起就有的 Put/Call 比，和課程說的跳水時機。
資料：期交所臺指選擇權 Put/Call 比（data/taifex/pcratio.csv，2007–2026）；大盤用 0050 還原含息日報酬。
A Put/Call 比（每天收盤後）：
  A1 未平倉量比（賣權未平倉 ÷ 買權未平倉），A2 成交量比；各自換成「過去 250 個交易日的百分位」，分五組，看之後 5、20 個交易日 0050 報酬。
  判斷（A1、A2 各自）：最高一組減最低一組的 20 日報酬差，用不重疊的 20 日樣本算 t，|t| ≥ 2.9，且 2007–2015、2016–2026 兩段同方向 → 雷達加「Put/Call 比」大盤訊號（標明方向）；否則只顯示數值。
B 跳水時機（課程：「賭明天破 20 日線，或一天跌 150～200 點」；當時指數約 10,800，換成百分比 ≈ 1.5%）：
  B1 收盤由 20 日線上方跌破到下方；B2 單日跌 ≥ 1.5%；B3 夜盤（台指期）跌 ≥ 1.5%（data/night.csv，資料太少只報筆數）。
  看隔天：跌 ≥ 1.5% 的機率、漲 ≥ 1.5% 的機率；之後 5 天內最低點曾跌 ≥ 3% 的機率；隔天、5 天平均報酬。和全部日子比。
  判斷（B1、B2 各自）：隔天跌 ≥ 1.5% 的機率 ≥ 全部日子的 1.5 倍，且二項檢定 p < 0.01，兩段都成立 → 雷達加「跳水警示」；
  同時報告漲 ≥ 1.5% 的機率（大跌後常大彈，買賣權都是樂透）。
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
from math import comb
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
from opt_inst_week import p0050

def binom_p(k, n, p):  # 單尾 P(X ≥ k)
    from scipy.stats import binom
    return float(binom.sf(k - 1, n, p))

if __name__ == "__main__":
    px = p0050().dropna(); r = px.pct_change() * 100
    P = pd.read_csv(f"{REPO}/data/taifex/pcratio.csv", dtype=str); P.columns = [c.strip() for c in P.columns]; P["date"] = pd.to_datetime(P["日期"].str.strip())
    P = P.set_index("date").sort_index(); P = P[~P.index.duplicated()]
    num = lambda s: pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")
    A = pd.DataFrame({"A1 未平倉量比": num(P["買賣權未平倉量比率%"]), "A2 成交量比": num(P["買賣權成交量比率%"])}).reindex(px.index).ffill(limit=3)
    f5 = (px.shift(-5) / px - 1) * 100; f20 = (px.shift(-20) / px - 1) * 100; res = {}
    for k in A.columns:
        pct = A[k].rolling(250, min_periods=200).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
        q = pd.cut(pct, [-0.01, 0.2, 0.4, 0.6, 0.8, 1.01], labels=[1, 2, 3, 4, 5])
        tab = {int(g): {"5日": round(float(f5[q == g].mean()), 2), "20日": round(float(f20[q == g].mean()), 2), "天數": int((q == g).sum())} for g in [1, 2, 3, 4, 5]}
        out = {"五組（低→高）": tab}
        for lab, m in (("全期", slice(None)), ("2007–2015", slice("2007", "2015")), ("2016–2026", slice("2016", "2026"))):
            qq, ff = q[m], f20[m]; idx = np.arange(len(qq))[::20]; qq, ff = qq.iloc[idx], ff.iloc[idx]
            a, b = ff[qq == 5].dropna(), ff[qq == 1].dropna(); d = a.mean() - b.mean(); se = np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 2 and len(b) > 2 else np.nan
            out[lab] = {"高減低 20日": round(float(d), 2), "t（不重疊）": round(float(d / se), 2) if se == se and se > 0 else None, "樣本": (len(a), len(b))}
        out["通過"] = bool(abs(out["全期"]["t（不重疊）"] or 0) >= 2.9 and np.sign(out["2007–2015"]["高減低 20日"]) == np.sign(out["2016–2026"]["高減低 20日"]))
        res[k] = out; print(k, json.dumps(out, ensure_ascii=False), flush=True)
    ma20 = px.rolling(20).mean(); nxt = r.shift(-1); low5 = (px.rolling(5).min().shift(-5) / px - 1) * 100; f5r = f5
    base = {"天數": int(nxt.notna().sum()), "隔天跌≥1.5%": round(float((nxt <= -1.5).mean() * 100), 2), "隔天漲≥1.5%": round(float((nxt >= 1.5).mean() * 100), 2),
            "5天內曾跌≥3%": round(float((low5 <= -3).mean() * 100), 2), "隔天平均": round(float(nxt.mean()), 3), "5天平均": round(float(f5r.mean()), 2)}
    res["全部日子"] = base; print("全部日子", base)
    conds = {"B1 跌破20日線": (px < ma20) & (px.shift(1) >= ma20.shift(1)), "B2 單日跌≥1.5%": r <= -1.5}
    for k, c in conds.items():
        o = {}
        for lab, m in (("全期", slice(None)), ("2007–2015", slice("2007", "2015")), ("2016–2026", slice("2016", "2026"))):
            cc = c[m] & nxt[m].notna(); n = int(cc.sum()); kdn = int((nxt[m][cc] <= -1.5).sum()); p0 = float((nxt[m] <= -1.5).mean())
            o[lab] = {"次數": n, "隔天跌≥1.5%": round(kdn / n * 100, 2) if n else None, "倍數": round(kdn / n / p0, 2) if n and p0 else None, "p": round(binom_p(kdn, n, p0), 4) if n else None,
                      "隔天漲≥1.5%": round(float((nxt[m][cc] >= 1.5).mean() * 100), 2), "5天內曾跌≥3%": round(float((low5[m][cc] <= -3).mean() * 100), 2),
                      "隔天平均": round(float(nxt[m][cc].mean()), 3), "5天平均": round(float(f5r[m][cc].mean()), 2)}
        o["通過"] = bool(all(o[s]["倍數"] and o[s]["倍數"] >= 1.5 and o[s]["p"] < 0.01 for s in ("全期", "2007–2015", "2016–2026")))
        res[k] = o; print(k, json.dumps(o, ensure_ascii=False), flush=True)
    if os.path.exists(f"{REPO}/data/night.csv"):
        N = pd.read_csv(f"{REPO}/data/night.csv"); res["B3 夜盤跌≥1.5%"] = {"資料筆數": int(len(N)), "說明": "夜盤資料 11/25 才開始存，筆數太少，之後累積再看"}
    json.dump(res, open(f"{REPO}/research/results/market_timing_opt.json", "w"), ensure_ascii=False, indent=1, default=str)
