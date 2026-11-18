"""空頭家數到 800 家是不是底部（使用者老師的理論），以及那時候該抓哪種反彈股。

全部在看任何結果之前寫下（2026/11/18 09:15），看到結果後不修改。（寫之前沒看過空頭家數的任何數字。）

空頭家數：當天有收盤的所有上市櫃四碼股票中，符合「空頭排列」的家數。
  主要定義 D1：收盤 < 5日均線 < 10日均線 < 20日均線 < 60日均線
  敏感度  D2：收盤 < 5日均線 < 10日均線 < 20日均線（不看60日線）
  （坊間定義不一，兩種都報。）
訊號：空頭家數 ≥ 800，且前20個交易日都 < 800 的第一天（同一波只算一次）。另報門檻 600／700／900／1000，以及「占全部家數 ≥ 40%」版本（股票總數逐年增加）。
大盤：分析範圍（20日均額 ≥ 1,000 萬）等權指數（同 research/regime.py）。
問題1「800 是不是底部」，看訊號日之後：
  - 指數 5／10／20 日報酬（訊號日收盤起算），和所有交易日的平均比
  - 之後20日內指數還會再跌多少（最低收盤 ÷ 訊號日收盤 − 1）
  - 訊號日離這波真正低點多近：之後60日內的最低點是不是就在訊號日前後3天內
  判斷「是底部訊號」：D1@800 的訊號，10日指數報酬平均 > 所有交易日平均 + 2 個百分點，且至少 2/3 的訊號之後10日是正報酬。
問題2「抓反彈」：訊號日收盤，把分析範圍內股票按近20日漲跌幅分10組，隔天開盤買、持有10日，看跌最深的一組比全部平均多賺多少（超額報酬）。
  判斷有用：平均超額 ≥ 2 個百分點，且至少 2/3 的訊號為正。另報跌最淺一組、近20日最大跌幅組合中「仍在動」的股票。
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radar_core import load_prices, panels, features, tiers

p = load_prices(); P = panels(p)
C, O = P["C"], P["O"]
f = features(P); S = tiers(f, P["MK"])
days = C.index; N = len(days)
R = lambda n: C.rolling(n, min_periods=n).mean()
m5, m10, m20, m60 = R(5), R(10), R(20), R(60)
have = C.notna()
D1 = ((C < m5) & (m5 < m10) & (m10 < m20) & (m20 < m60)).sum(axis=1)
D2 = ((C < m5) & (m5 < m10) & (m10 < m20)).sum(axis=1)
tot = (have & m60.notna()).sum(axis=1)
inb = S[S.base].assign(v=True).pivot(index="date", columns="code", values="v").reindex(index=days, columns=C.columns).fillna(False).astype(bool)
ew = (1 + C.pct_change(fill_method=None).where(inb).mean(axis=1).fillna(0)).cumprod()
start = int(np.argmax(tot.values >= 1000))  # 均線資料足夠之後才開始

def signals(x, th):
    s = []
    for i in range(max(start, 20), N):
        if x.iloc[i] >= th and (x.iloc[i - 20:i] < th).all(): s.append(i)
    return s

def fwd_idx(i, h): return (ew.iloc[i + h] / ew.iloc[i] - 1) * 100 if i + h < N else np.nan
uncond = {h: float(np.nanmean([fwd_idx(i, h) for i in range(start, N)])) for h in (5, 10, 20)}

def evals(sig):
    rows = []
    for i in sig:
        lo20 = (ew.iloc[i + 1:i + 21].min() / ew.iloc[i] - 1) * 100 if i + 1 < N else np.nan
        w = ew.iloc[max(0, i - 60):i + 61]
        tr = days.get_loc(w.idxmin())
        rows.append({"date": days[i].date().isoformat(), "r5": fwd_idx(i, 5), "r10": fwd_idx(i, 10), "r20": fwd_idx(i, 20), "dd20": lo20,
                     "trough": days[tr].date().isoformat(), "td": tr - i, "near": abs(tr - i) <= 3 and i + 60 < N})
    return pd.DataFrame(rows)

def summ(T):
    if T.empty: return {"n": 0}
    return {"n": int(len(T)), "r5": round(T.r5.mean(), 2), "r10": round(T.r10.mean(), 2), "r20": round(T.r20.mean(), 2),
            "pos10": round((T.r10.dropna() > 0).mean() * 100), "dd20": round(T.dd20.mean(), 2), "dd20_worst": round(T.dd20.min(), 2),
            "near_trough": int(T.near.sum()), "median_days_to_trough": float(T.td.median())}

res = {"uncond": {k: round(v, 2) for k, v in uncond.items()}, "variants": {}}
for nm, x in (("D1", D1), ("D2", D2)):
    for th in (600, 700, 800, 900, 1000):
        T = evals(signals(x, th)); res["variants"][f"{nm}@{th}"] = {"sum": summ(T), "events": T.round(2).to_dict("records")}
    pct = (x / tot * 100)
    T = evals(signals(pct, 40)); res["variants"][f"{nm}@40%"] = {"sum": summ(T), "events": T.round(2).to_dict("records")}
main = res["variants"]["D1@800"]["sum"]
res["is_bottom_signal"] = bool(main.get("n", 0) > 0 and main["r10"] > uncond[10] + 2 and main["pos10"] >= 67)

# 問題2：抓反彈
rows = []
for i in signals(D1, 800):
    if i + 10 >= N: continue
    m = inb.iloc[i] & O.iloc[i + 1].notna()
    r20 = ((C.iloc[i] / C.iloc[i - 20] - 1) * 100).where(m)
    ret = ((C.iloc[i + 10] / O.iloc[i + 1] - 1) * 100).where(m)
    ex = ret - ret.mean()
    q = r20.rank(pct=True)
    hot = pd.Series(S[(S.date == days[i]) & S.hot].code.values)
    rows.append({"date": days[i].date().isoformat(), "deep": round(ex[q <= 0.1].mean(), 2), "shallow": round(ex[q > 0.9].mean(), 2),
                 "deep_hot": round(ex[(q <= 0.1) & ex.index.isin(hot)].mean(), 2), "n_deep_hot": int(((q <= 0.1) & ex.index.isin(hot)).sum()),
                 "avg_all": round(ret.mean(), 2)})
Q2 = pd.DataFrame(rows)
res["rebound"] = {"events": Q2.to_dict("records"),
                  "deep_avg": round(Q2.deep.mean(), 2) if len(Q2) else None, "deep_pos": round((Q2.deep > 0).mean() * 100) if len(Q2) else None,
                  "shallow_avg": round(Q2.shallow.mean(), 2) if len(Q2) else None}
res["rebound"]["useful"] = bool(len(Q2) and Q2.deep.mean() >= 2 and (Q2.deep > 0).mean() >= 2 / 3)
res["daily"] = {"date": [d.date().isoformat() for d in days[start:]], "D1": D1.iloc[start:].astype(int).tolist(), "D2": D2.iloc[start:].astype(int).tolist(), "tot": tot.iloc[start:].astype(int).tolist()}
json.dump(res, open("research/results/bearcount.json", "w"), ensure_ascii=False, indent=1, default=str)
print("uncond", res["uncond"])
for k, v in res["variants"].items(): print(k, v["sum"])
print(pd.DataFrame(res["variants"]["D1@800"]["events"]).to_string())
print("是底部訊號:", res["is_bottom_signal"]); print(Q2.to_string()); print({k: v for k, v in res["rebound"].items() if k != "events"})
print("最近：", list(zip(res["daily"]["date"][-12:], res["daily"]["D1"][-12:], res["daily"]["D2"][-12:], res["daily"]["tot"][-12:])))
print("D1 歷年最大：", pd.Series(D1.iloc[start:].values, index=days[start:]).groupby(days[start:].year).max().to_dict())
