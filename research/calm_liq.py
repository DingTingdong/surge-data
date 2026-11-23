"""穩健版的流動性與成本（第 28 輪；2026/11/23 14:00 登錄，看結果前寫下，看到結果後不修改）

起因：穩健版入選股的 20 日平均成交值中位數只有 0.35～0.52 億，55% 低於 5 千萬；低波動有一部分可能只是成交清淡。
  使用者資金不大，買得到，但滑價可能比大型股高。
檢驗：
  (1) 成本敏感度（另報、不影響判斷）：每次賣出成本 0.6%（現行）、1.1%、1.6%。
  (2) L 穩健版加流動性門檻：前 20 名裡只考慮 20 日平均成交值 ≥ 5 千萬的，再挑波動最低 3 檔（續抱規則不變）。
採用條件（L 全部符合才把網站改成加門檻）：① 2011–2026 每月減 0050 不比穩健版少 0.3 以上、t ≥ 2.9；② 五段都 > 0；③ 最大回落不比穩健版深 5 個百分點以上。
結果（11/23 14:05）：L 不通過（③ 回落 −33% 比穩健版 −23% 深 10 點）→ 網站維持不設門檻。
  成本 0.6／1.1／1.6%：每月減 0050 +2.35／+1.92／+1.49，五段都正，回落 −23／−26／−30% → 多 1% 滑價仍明顯贏。
  L 加門檻：+2.52（t 4.03），五段 0.57／1.95／1.32／3.17／4.30（2024–26 較好、2006–09 較差），回落 −33%。
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import robust_select as RS
from calm3 import sim_calm
from wr_confirm import stats
Dn = RS.load("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01"); Do = RS.load("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01"); out = {}
for c in (0.6, 1.1, 1.6):
    out[f"成本{c}%"] = stats(pd.concat([sim_calm(Do, 20, cost=c), sim_calm(Dn, 20, cost=c)])); print(f"成本{c}%", out[f"成本{c}%"], flush=True)
def liq(D, floor=5e7):
    E = dict(D); F = D["F"].copy(); F["rk"] = F.groupby("date").cs.rank(ascending=False); F.loc[(F.rk <= 20) & ~(F.a20 >= floor), "vol20"] = np.inf; E["F"] = F; return E
L = stats(pd.concat([sim_calm(liq(Do), 20), sim_calm(liq(Dn), 20)])); B = out["成本0.6%"]; out["L 加門檻5千萬"] = L; print("L", L)
c = {"①": L["2011–2026"][0] >= B["2011–2026"][0] - 0.3 and L["2011–2026"][1] >= 2.9, "②": all(L[s] > 0 for s in RS.SEGS), "③": L["最大回落"] >= B["最大回落"] - 5}
out["各項"] = c; out["通過"] = all(c.values()); print(c)
json.dump(out, open(f"{REPO}/research/results/calm_liq.json", "w"), ensure_ascii=False, indent=1, default=str)
