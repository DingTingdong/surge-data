"""空頭家數 ≥ 800 時，撿「跌深的好公司」：哪一種好公司反彈最好。

全部在看結果之前寫下（2026/11/18 11:40），看到結果後不修改。
（已知：bearcount.py 的「近20日跌最深一成」超額 +2.65、主要靠 2026/7。）

時點：
  T1 第一次衝上 800（前20天都 < 800）＝ bearcount.py 的訊號（2020 年以來 5 次已完成）
  T2 敏感度：門檻 700 的第一次（10 次），檢查結果是不是只靠少數幾次
股票：訊號日在分析範圍（20日均額 ≥ 1,000 萬）。
「跌深」＝ 收盤比自己近20日最高收盤低 ≥ 20%。
「好公司」用手上有的資料定義（沒有財報），各自和「跌深」合併：
  G1 大跌前是多頭：訊號日前20日那天，收盤在自己的120日均線之上
  G2 大跌前強：訊號日前 80→20 日的漲幅前 30%
  G3 大公司：股本 ≥ 30 億（現在的股本）
  G4 法人沒跑：近20日三大法人合計買超 > 0
  G5 G1 且 G3（多頭的大公司）
對照：全部股票平均、只看跌深。
結果：隔天開盤買，10 日與 20 日收盤報酬；主要看 20 日「超額報酬」（減去當次全部股票平均），以及 20 日內最低點（買進後還會再跌多少）。
判斷：T1 中，某組 20 日超額平均比「只看跌深」高 ≥ 2 個百分點，且 5 次中至少 4 次贏過「只看跌深」，才算比較好；T2 要方向相同。
"""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radar_core import load_prices, panels, features, tiers

p = load_prices(); P = panels(p)
C, O = P["C"], P["O"]
f = features(P); S = tiers(f, P["MK"])
days = C.index; N = len(days)
name = p.sort_values("date").drop_duplicates("code", keep="last").set_index("code").name
R = lambda n: C.rolling(n, min_periods=n).mean()
m5, m10, m20, m60, m120 = R(5), R(10), R(20), R(60), R(120)
bear = ((C < m5) & (m5 < m10) & (m10 < m20) & (m20 < m60)).sum(axis=1)
inb = S[S.base].assign(v=True).pivot(index="date", columns="code", values="v").reindex(index=days, columns=C.columns).fillna(False).astype(bool)
ind = pd.read_csv("data/industry.csv", dtype={"code": str}).set_index("code")
cap = pd.Series(ind["cap"]).reindex(C.columns)
g = pd.Series(ind["ind_name"]).reindex(C.columns)
ch = pd.concat([pd.read_csv(fn, dtype={"code": str}, usecols=["date", "code", "inst"]) for fn in sorted(glob.glob("data/chips/*.csv*"), key=lambda x: (os.path.basename(x)[:4], x.endswith(".csv")))], ignore_index=True)
ch["date"] = pd.to_datetime(ch.date); ch = ch.drop_duplicates(["date", "code"], keep="last")
inst = ch.pivot(index="date", columns="code", values="inst").reindex(index=days, columns=C.columns)
start = int(np.argmax((C.notna() & m60.notna()).sum(axis=1).values >= 1000))

def signals(th):
    return [i for i in range(max(start, 20), N) if bear.iloc[i] >= th and (bear.iloc[i - 20:i] < th).all()]

def groups(i):
    m = inb.iloc[i] & C.iloc[i].notna()
    deep = (C.iloc[i] / C.iloc[i - 19:i + 1].max() - 1 <= -0.20) & m
    g1 = C.iloc[i - 20] > m120.iloc[i - 20]
    pre = C.iloc[i - 20] / C.iloc[i - 80] - 1
    g2 = pre.where(m).rank(pct=True) > 0.7
    g3 = cap >= 30
    g4 = inst.iloc[i - 19:i + 1].sum(min_count=1) > 0
    return m, {"全部股票": m, "只看跌深": deep, "G1 跌深＋原本多頭": deep & g1, "G2 跌深＋大跌前強": deep & g2,
               "G3 跌深＋大公司": deep & g3, "G4 跌深＋法人沒跑": deep & g4, "G5 跌深＋多頭大公司": deep & g1 & g3}

def run(th):
    rows = []
    for i in signals(th):
        if i + 20 >= N: continue
        m, G = groups(i)
        o = O.iloc[i + 1]
        r10 = (C.iloc[i + 10] / o - 1) * 100; r20 = (C.iloc[i + 20] / o - 1) * 100
        lo = (C.iloc[i + 1:i + 21].min() / o - 1) * 100
        base10, base20 = r10[m].mean(), r20[m].mean()
        for k, sel in G.items():
            sel = sel & o.notna()
            rows.append({"date": days[i].date().isoformat(), "grp": k, "n": int(sel.sum()), "r10": r10[sel].mean(), "r20": r20[sel].mean(),
                         "ex10": r10[sel].mean() - base10, "ex20": r20[sel].mean() - base20, "low20": lo[sel].mean(),
                         "win20": (r20[sel] > 0).mean() * 100})
    return pd.DataFrame(rows)

out = {}
for th in (800, 700):
    T = run(th)
    piv = T.pivot(index="date", columns="grp", values="ex20")
    s = T.groupby("grp")[["n", "r10", "r20", "ex10", "ex20", "low20", "win20"]].mean().round(2)
    s["beat_deep"] = [int((piv[k] > piv["只看跌深"]).sum()) for k in s.index]
    s["events"] = int(piv.shape[0])
    out[str(th)] = {"summary": s.reset_index().to_dict("records"), "by_event": T.round(2).to_dict("records")}
    print("== 門檻", th); print(s.to_string()); print(T.pivot(index="date", columns="grp", values="ex20").round(1).to_string())
s8 = pd.DataFrame(out["800"]["summary"]).set_index("grp"); s7 = pd.DataFrame(out["700"]["summary"]).set_index("grp")
better = [k for k in s8.index if k.startswith("G") and s8.at[k, "ex20"] - s8.at["只看跌深", "ex20"] >= 2 and s8.at[k, "beat_deep"] >= 4
          and (s7.at[k, "ex20"] - s7.at["只看跌深", "ex20"]) > 0]
out["better_than_deep"] = better

# 這一波：11/04 第一次衝上800；11/17 第二次 ≥800。列出 11/17 符合各組的股票
i = len(days) - 1
m, G = groups(i)
dd = (C.iloc[i] / C.iloc[i - 19:i + 1].max() - 1) * 100
now = {}
for k in ("只看跌深", "G1 跌深＋原本多頭", "G3 跌深＋大公司", "G5 跌深＋多頭大公司"):
    sel = G[k]
    now[k] = [[c, name.get(c, ""), g.get(c) if isinstance(g.get(c), str) else "", round(dd[c], 1), None if cap.get(c) != cap.get(c) else round(cap[c], 1)]
              for c in dd[sel].sort_values().index]
out["now"] = {"date": days[i].date().isoformat(), "bear": int(bear.iloc[i]), "lists": now}
json.dump(out, open("research/results/panic_pick.json", "w"), ensure_ascii=False, indent=1, default=str)
print("比跌深更好的組:", better or "無")
print("11/17 空頭家數", int(bear.iloc[i]), {k: len(v) for k, v in now.items()})
for x in now["G5 跌深＋多頭大公司"][:40]: print(x)
