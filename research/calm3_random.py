"""穩健版檢查：前 20 名「隨機挑 3 檔」200 次，和「挑波動最低 3 檔」比（2026/11/23，使用者選定穩健版後的檢查；描述性）。"""
import sys, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import robust_select as RS
from calm3 import sim_calm
from wr_confirm import stats
Dn = RS.load("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01"); rng = np.random.default_rng(20261123); F0 = Dn["F"].copy(); res = []
for i in range(200):
    F = F0.copy(); F["vol20"] = rng.random(len(F)); Dn["F"] = F  # 用隨機數取代波動 → 前 20 名裡隨機挑
    s = stats(sim_calm(Dn, 20)); res.append((s["2011–2026"][0], s["2024-26"], s["最大回落"]))
Dn["F"] = F0; v1 = stats(sim_calm(Dn, 20)); R = np.array(res)
print("挑低波動", v1["2011–2026"], v1["2024-26"], v1["最大回落"])
for k, lab in enumerate(["每月減0050", "2024–26", "最大回落"]):
    print(f"隨機 {lab}：中位數 {np.median(R[:, k]):.2f}，5%～95% {np.percentile(R[:, k], 5):.2f}～{np.percentile(R[:, k], 95):.2f}，低波動版的排名百分位 {(R[:, k] < [v1['2011–2026'][0], v1['2024-26'], v1['最大回落']][k]).mean() * 100:.0f}%")
