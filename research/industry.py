"""產業集中（族群性）與股本大小，能不能讓「今日發動」更準。

全部在跑之前寫下（2026/11/18 01:50），看到結果後不修改。

訊號：今日發動（v2）。隔天開盤買，移動停損出場（research/exits.py 的 X6），扣 1.1%。
產業與股本用 data/industry.csv（目前的官方分類與實收資本額，不是當時的；已下市公司沒有資料 → 不計入，另報覆蓋率）。
「活躍」= 當天今日發動或仍在動。
候選條件：
  G1 同產業當天另外有 ≥ 2 檔活躍
  G2 同產業當天另外有 ≥ 1 檔今日發動
  G3 同產業當天活躍比例是全體的 ≥ 1.5 倍，且同產業活躍 ≥ 3 檔（網站上「今天集中的產業」的規則）
  G4 同產業當天沒有其他活躍股（孤軍）
  K1 股本 ≤ 10 億
  K2 股本 ≤ 30 億
  K3 股本 > 30 億
期間：2020–2023 挑選，2024–2026/11 驗證。
挑選規則（同 research/chips.py）：在 2020–2023，某條件「保留至少 30% 的訊號」且「X6 平均比全部今日發動（有產業資料者）高 ≥ 1.0 個百分點」才算有用；
有用的當中選提升最多的。沒有 → 結論是沒有幫助。驗證期照實報告全部條件。
"""
import json, os, sys
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
ind = pd.read_csv("data/industry.csv", dtype={"code": str, "ind": str}).set_index("code")

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

B = S[S.base][["date", "code", "pk", "hot"]].copy()
B["act"] = B.pk | B.hot
B["g"] = B.code.map(ind.ind_name)
B["cap"] = B.code.map(ind["cap"])
day_rate = B.groupby("date").act.mean()
gstat = B.dropna(subset=["g"]).groupby(["date", "g"]).agg(n=("act", "size"), na=("act", "sum"), npk=("pk", "sum")).reset_index()
B = B.merge(gstat, on=["date", "g"], how="left")
sig = B[B.pk].copy()
sig["oth_act"] = sig.na - 1; sig["oth_pk"] = sig.npk - 1
sig["lift"] = (sig.na / sig.n) / sig.date.map(day_rate)
rows = []
for r in sig.itertuples():
    t, j = pos[r.date], col[r.code]
    x = x6(t, j)
    if x is None: continue
    rows.append({"date": r.date, "train": r.date <= TRAIN_END, "x6": x, "has": r.g == r.g, "oth_act": r.oth_act, "oth_pk": r.oth_pk,
                 "lift": r.lift, "na": r.na, "cap": r.cap,
                 "hit20": np.nanmax(Cv[t + 1:t + 21, j]) / Ov[t + 1, j] - 1 >= 0.2, "drop10": np.nanmin(Cv[t + 1:t + 11, j]) / Ov[t + 1, j] - 1 <= -0.1})
T = pd.DataFrame(rows)
cov = round(float(T.has.mean() * 100), 1)
T = T[T.has]
cond = {"全部今日發動（有產業資料）": pd.Series(True, index=T.index), "G1 同產業另有≥2檔活躍": T.oth_act >= 2, "G2 同產業另有今日發動": T.oth_pk >= 1,
        "G3 產業集中（網站規則）": (T.lift >= 1.5) & (T.na >= 3), "G4 孤軍（同產業沒有其他活躍）": T.oth_act == 0,
        "K1 股本≤10億": T.cap <= 10, "K2 股本≤30億": T.cap <= 30, "K3 股本>30億": T.cap > 30}
def stats(g, n0):
    return {"n": int(len(g)), "keep": round(len(g) / max(1, n0) * 100, 1), "x6_avg": round(g.x6.mean(), 2), "x6_med": round(g.x6.median(), 2),
            "win": round((g.x6 > 0).mean() * 100, 1), "hit20": round(g.hit20.mean() * 100, 1), "drop10": round(g.drop10.mean() * 100, 1)}
res = {}
for tr, per in ((True, "2020-2023"), (False, "2024-2026")):
    g0 = T[T.train == tr]; res[per] = {k: stats(g0[m[g0.index]], len(g0)) for k, m in cond.items()}
yearly = {str(y): {k: round(g[m[g.index]].x6.mean(), 2) for k, m in cond.items()} for y, g in T.groupby(T.date.dt.year)}
tr = res["2020-2023"]; base = tr["全部今日發動（有產業資料）"]["x6_avg"]
useful = [k for k in cond if not k.startswith("全部") and tr[k]["keep"] >= 30 and tr[k]["x6_avg"] - base >= 1.0]
pick = max(useful, key=lambda k: tr[k]["x6_avg"]) if useful else None
out = {"pick": pick, "useful": useful, "coverage_pct": cov, "results": res, "yearly": yearly}
json.dump(out, open("research/results/industry.json", "w"), ensure_ascii=False, indent=1)
for per in res: print("==", per); print(pd.DataFrame(res[per]).T.to_string())
print(pd.DataFrame(yearly).to_string())
print("產業資料覆蓋率%:", cov, "有用的條件:", useful or "無", "→ 採用:", pick)
