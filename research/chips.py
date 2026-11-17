"""籌碼資料能不能讓「今日發動」更準。

全部在資料到齊、跑之前寫下（2026/11/17 18:15），看到結果後不修改。

訊號：今日發動（v2）。籌碼用訊號當天的資料（法人買賣超約下午公布、融資融券約晚上公布，都在隔天開盤前）。
候選過濾條件（缺資料 → 視為不符合）：
  C1 投信當天買超 > 0
  C2 外資（含外資自營商）當天買超 > 0
  C3 三大法人合計當天買超 > 0
  C4 融資餘額比5個交易日前沒有增加（≤ 0），散戶還沒追
  C5 排除融資餘額比5個交易日前增加超過 10% 的（散戶已大量追價）
結果：隔天開盤買。主要看「移動停損」出場（research/exits.py 的 X6）的每筆平均報酬；另報中位數、
20個交易日內最高收盤 ≥ +20% 的比例、10個交易日內最低收盤 ≤ −10% 的比例。扣 1.1%。
期間：2020–2023 挑選，2024–2026/11 驗證。
挑選規則：在 2020–2023，某條件「保留至少 30% 的訊號」且「X6 平均比全部今日發動高 ≥ 1.0 個百分點」才算有用；
有用的當中選提升最多的。沒有任何條件有用 → 結論是籌碼沒有幫助。驗證期照實報告全部條件。
"""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radar_core import load_prices, panels, features, tiers

COST, CAP, LOCK = 1.1, 20, 0.095
TRAIN_END = pd.Timestamp("2023-12-31")
p = load_prices(); P = panels(p)
C, O, H, L = P["C"], P["O"], P["H"], P["L"]
f = features(P); S = tiers(f, P["MK"])
days = C.index; N = len(days)
Cv, Ov, Hv, Lv = C.values, O.values, H.values, L.values
pos = {d: i for i, d in enumerate(days)}; col = {c: j for j, c in enumerate(C.columns)}

ch = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in sorted(glob.glob("data/chips/*.csv*"), key=lambda x: (os.path.basename(x)[:4], x.endswith(".csv")))], ignore_index=True)
ch["date"] = pd.to_datetime(ch.date)
ch = ch.drop_duplicates(["date", "code"], keep="last")
piv = lambda c: ch.pivot(index="date", columns="code", values=c).reindex(index=days, columns=C.columns)
trust, foreign, inst, margin = piv("trust"), piv("foreign"), piv("inst"), piv("margin")
m5 = margin - margin.shift(5)
m5p = m5 / margin.shift(5)

def locked(i, j, up):
    o, h, l = Ov[i, j], Hv[i, j], Lv[i, j]
    k = i - 1
    while k >= 0 and not Cv[k, j] == Cv[k, j]: k -= 1
    if k < 0 or not (o == o and h == h and l == l): return False
    r = o / Cv[k, j] - 1
    return o == h == l and (r >= LOCK if up else r <= -LOCK)

def x6(t, j):
    e = t + 1; end = e + CAP - 1
    if end >= N: return None
    o = Ov[e, j]
    if not o == o or o <= 0 or locked(e, j, True): return None
    peak = o
    for k in range(e, end + 1):
        c = Cv[k, j]
        if not c == c: continue
        peak = max(peak, c)
        if c <= peak * 0.9 and k < end:
            for s in range(k + 1, min(k + 6, N)):
                if Ov[s, j] == Ov[s, j] and not locked(s, j, False): return (Ov[s, j] / o - 1) * 100 - COST
            return None
    k = end
    while k > e and not Cv[k, j] == Cv[k, j]: k -= 1
    return (Cv[k, j] / o - 1) * 100 - COST

rows = []
for d, c in zip(S[S.pk].date, S[S.pk].code):
    t, j = pos[d], col[c]
    r = x6(t, j)
    if r is None: continue
    o = Ov[t + 1, j]
    rows.append({"date": d, "train": d <= TRAIN_END, "x6": r,
                 "hit20": np.nanmax(Cv[t + 1:t + 21, j]) / o - 1 >= 0.2, "drop10": np.nanmin(Cv[t + 1:t + 11, j]) / o - 1 <= -0.1,
                 "trust": trust.iat[t, j], "foreign": foreign.iat[t, j], "inst": inst.iat[t, j], "m5": m5.iat[t, j], "m5p": m5p.iat[t, j]})
T = pd.DataFrame(rows)
cond = {"全部今日發動": pd.Series(True, index=T.index), "C1 投信買超": T.trust > 0, "C2 外資買超": T.foreign > 0,
        "C3 三大法人買超": T.inst > 0, "C4 融資5日沒增加": T.m5 <= 0, "C5 排除融資5日增>10%": T.m5p.notna() & (T.m5p <= 0.10)}
def stats(g, base_n):
    return {"n": int(len(g)), "keep": round(len(g) / max(1, base_n) * 100, 1), "x6_avg": round(g.x6.mean(), 2), "x6_med": round(g.x6.median(), 2),
            "hit20": round(g.hit20.mean() * 100, 1), "drop10": round(g.drop10.mean() * 100, 1)}
res = {}
for tr, per in ((True, "2020-2023"), (False, "2024-2026")):
    g0 = T[T.train == tr]
    res[per] = {k: stats(g0[m[g0.index]], len(g0)) for k, m in cond.items()}
cov = {k: round(float(T[k].notna().mean() * 100), 1) for k in ("trust", "foreign", "inst", "m5")}
tr = res["2020-2023"]; base = tr["全部今日發動"]["x6_avg"]
useful = [k for k in cond if k != "全部今日發動" and tr[k]["keep"] >= 30 and tr[k]["x6_avg"] - base >= 1.0]
pick = max(useful, key=lambda k: tr[k]["x6_avg"]) if useful else None
out = {"pick": pick, "useful": useful, "coverage_pct": cov, "results": res}
json.dump(out, open("research/results/chips.json", "w"), ensure_ascii=False, indent=1)
for per in res: print("==", per); print(pd.DataFrame(res[per]).T.to_string())
print("資料覆蓋率%:", cov); print("有用的條件:", useful or "無", "→ 採用:", pick)
