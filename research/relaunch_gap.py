# 探索：今日發動＋再發動，隔天開盤跳空幅度與移動停損結果（恐慌期間 vs 平常）
import os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
src = open("research/relaunch.py", encoding="utf-8").read()
src = src.replace('json.dump({"results"', 'pass  # json.dump({"results"').replace('for per in res: print', 'pass  # for per in res: print').replace('print(yearly.to_string()); print("加進雷達：", add)', '')
src = src.split("# 穩健性")[0].replace('print("恐慌期間（True）vs 平常（False）："); print(q.to_string())', '')
exec(src)
# 隔天開盤跳空幅度
gap = []
for r in T.itertuples():
    t = pos[r.date]
    gap.append(np.nan)
T2 = []
for r in B.itertuples():
    t, j = pos[r.date], col[r.code]
    if t + 1 >= N: continue
    o, c0 = Ov[t + 1, j], Cv[t, j]
    T2.append({"date": r.date, "grp": r.grp, "gap": (o / c0 - 1) * 100 if o == o and c0 == c0 else np.nan})
G = pd.DataFrame(T2)
T = T.merge(G, on=["date", "grp"], how="left") if False else T
# 直接重算：把 gap 對上每筆
rows = []
for r in B.itertuples():
    t, j = pos[r.date], col[r.code]
    x = x6(t, j)
    if x is None: continue
    o, c0 = Ov[t + 1, j], Cv[t, j]
    rows.append({"date": r.date, "grp": r.grp, "blocked": x == "blocked", "x6": np.nan if x == "blocked" else x, "gap": (o / c0 - 1) * 100})
Z = pd.DataFrame(rows); Z["panic"] = Z.date.map(panic)
Z["跳空"] = pd.cut(Z.gap, [-100, 0, 3, 6, 9.4, 100], labels=["開平或開低", "開高0-3%", "開高3-6%", "開高6-9%", "開漲停"])
for pn, lab in ((True, "恐慌期間"), (False, "平常")):
    z = Z[(Z.panic == pn) & Z.grp.isin(["F 今日發動", "R 再發動"])]
    s = z.groupby("跳空", observed=True).agg(筆數=("gap", "size"), 買不到=("blocked", "mean"), X6平均=("x6", "mean"), X6中位數=("x6", "median"))
    s["占比%"] = (s.筆數 / s.筆數.sum() * 100).round(0); s["買不到"] = (s.買不到 * 100).round(0)
    print("==", lab, "（今日發動＋再發動，隔天開盤買、移動停損）"); print(s.round(2).to_string())
