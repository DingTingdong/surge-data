"""月選每期落後 0050 的歸因（第 71 輪；2026/11/27 13:20 登錄，描述性）：同一期（隔天開盤 → 下次換股隔天開盤）
風格＝分析範圍等權 − 0050，選股＝月選 − 分析範圍等權（月選已扣成本，等權不扣）。報 2011–2026 與 2024–26 的平均，以及輸 0050 的期裡兩者各占多少。
結果（11/27 13:25）：2011–2026 風格 −0.50（t −1.87）、選股 +2.38（t 5.24）；2024–26 風格 −2.44（t −4.97）、選股 +3.20（t 3.49）。
  輸 0050 的期（2024–26 共 19 期）：風格平均 −3.37、選股平均 −2.30；84% 的期風格為負、68% 選股為負 → 兩者都有，風格略大。"""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
from review9 import sim, NOTDR
from review7 import rescore
from mp_exec import prep
from price_cap import addraw
from vs0050 import nw_t
if __name__ == "__main__":
    PK = "/tmp/model_v2/vs0050_dl_extra.pkl"; D = prep(PK, "2011-01-01", "2027-01-01"); D["cols"] = list(pd.read_pickle(PK)["cols"]); D = addraw(D, PK); D["F0"] = D["F"].copy()
    rescore(D, keep=NOTDR); D["FDS"] = sorted(D["F"].date.unique()); T = sim(D)
    X = pd.read_pickle(PK); inb = np.asarray(X["inb"], bool); O = D["O"]; Cf = D["Cff"]
    ew = []
    for a, r in T.iterrows():
        ie, ix = int(r.ie), int(r.ix); m = inb[ie - 1] & (O[ie] > 0); ex = np.where(O[ix] > 0, O[ix], Cf[ix]); ew.append(float(np.nanmean(ex[m] / O[ie, m] - 1) * 100))
    T["ew"] = ew; T["風格"] = T.ew - T.b; T["選股"] = T.r - T.ew; res = {}
    for lab, z in (("2011–2026", T), ("2024–26", T[T.index >= "2024-01-01"])):
        lo = z[z.r < z.b]
        res[lab] = {"期數": len(z), "風格平均": round(float(z.風格.mean()), 2), "風格t": round(nw_t(z.風格, 3), 2), "選股平均": round(float(z.選股.mean()), 2), "選股t": round(nw_t(z.選股, 3), 2),
                    "輸0050期數": len(lo), "輸的期：風格平均": round(float(lo.風格.mean()), 2), "輸的期：選股平均": round(float(lo.選股.mean()), 2),
                    "輸的期裡風格為負的比例%": round(float((lo.風格 < 0).mean() * 100)), "輸的期裡選股為負的比例%": round(float((lo.選股 < 0).mean() * 100))}
    res["2024–26 逐期"] = {a.strftime("%Y-%m"): [round(r.r, 1), round(r.ew, 1), round(r.b, 1)] for a, r in T[T.index >= "2024-01-01"].iterrows()}
    print(json.dumps({k: v for k, v in res.items() if k != "2024–26 逐期"}, ensure_ascii=False))
    json.dump(res, open(f"{REPO}/research/results/attrib.json", "w"), ensure_ascii=False, indent=1)
