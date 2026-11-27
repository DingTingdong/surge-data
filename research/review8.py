"""換股日分批、分析範圍成交值門檻、營收的重要性（第 65 輪；2026/11/27 09:48 登錄〔原寫 10:05 是筆誤，以 commit 965f561 時間為準〕，看結果前寫下，看到結果後不修改）

起因：使用者 11/27 問 ①「成交值 1,000 萬的限制會不會讓績效變差」②「營收真的那麼重要嗎」，並同意跑第七份覆核建議的「分 2 批換股」。
基準（11/27 現行規則）：第 58 輪換股日、第 59 輪成交修正、5 檔、不跳過高價股、排除 TDR、單檔權重上限 30%（減到 25%），成本 0.6%，2011–2026。
資料：
  dl     ＝ vs0050_dl_extra.pkl（現有）
  dloff  ＝ vs0050_dloff_dl_extra.pkl：vs0050.py 以 FD_RULE=deadline FD_DLOFF=1,2,3,4 FD_TAG=dloff 重建：換股日之後第 1～4 個交易日也有快照
  amt0   ＝ vs0050_amt0_dl_extra.pkl：vs0050.py 以 FD_RULE=deadline RADAR_MIN_AMT=0 FD_TAG=amt0 重建：分析範圍不設成交值門檻（其他條件不變）
第一部分：換股日（dloff）
  S1～S5：只在換股日（S1，等於現行）或之後第 1～4 個交易日（S2～S5）換股，其他規則相同。
  B2：一半資金 S1、一半 S3（覆核建議：期限日後第 1 與第 3 個交易日各換一半）；B5：S1～S5 各 1/5。
    合併方式：同一個月份的各帳戶報酬平均、0050 同樣平均（近似，各帳戶之間不再平衡）。
  判斷 B2（降低換股日運氣，不要求變好）：每月減 0050 ≥ S1 − 0.2、t（落後3）≥ 2.9、2024–26 ≥ S1 − 0.2、最大回落不比 S1 深 3 點以上 → 全部符合才採用。
  另報：S1～S5 每月超額兩兩相關、各自 2024–26、B2 與 S1 的每月超額標準差。
第二部分：成交值門檻（amt0；在 amt0 裡取子集合後重新算百分位）
  U0 不設門檻、U3 ≥ 300 萬、U5 ≥ 500 萬、U10 ≥ 1,000 萬（現行；應接近 dl 的結果）、U20 ≥ 2,000 萬、U30 ≥ 3,000 萬。
  注意：rs_g20 的產業報酬是依 amt0 的分析範圍算，和現行略有不同；所以 U10 與 dl 基準的差距就是這個誤差。
  判斷（比 U10）：低於現行的門檻（U0、U3、U5）要全部符合才採用：每月減 0050 比 U10 高、t ≥ 2.9、2024–26 比 U10 高 ≥ 0.3、
    四段至少 3 段不低於 U10、回落不比 U10 深 3 點以上，而且「新買的股票 20 日均額 < 1,000 萬者額外扣 1% 滑價」後每月仍比 U10 高。
  U20、U30 只描述（第 64 輪已判斷 3,000 萬不採用）。另報：各門檻每期新買股票的 20 日均額中位數。
第三部分：營收的重要性（dl；描述性，不改規則）
  價格 7 項（r20、mom6_1、d_hi250、m60、am、mg20、rs_g20）、營收 5 項（yoy1、yoy3、rev_acc、rev_hi12、mom1）、現行 12 項，以及 12 項各拿掉 1 個營收特性。
結果（11/27 09:53）：
  第一部分（每月減 0050｜2024–26｜回落）：S1 現行 +1.89｜+0.76｜−21；S2 +1.43｜−0.01｜−31；S3 +1.03｜−0.69｜−31；S4 +1.04｜−1.37｜−33；S5 +1.30｜−0.86｜−29。
    四段 S1 都高於 S3（2.33／1.31／2.70／0.76 對 1.49／0.72／2.00／−0.69）。S1 與 S2～S5 每月超額相關 0.70／0.65／0.55／0.51。
    B2 +1.46（t 3.38）、2024–26 +0.04、回落 −24 → 不採用（低於 S1 − 0.2，2024–26 也低）。B5 +1.34。每月超額標準差 S1 7.76、B2 6.83。
    → 晚換股不是「運氣不同」而已，是一致變差：營收期限後第一個交易日換最好，晚一天就少約 0.5%／月。實盤要準時在換股日隔天開盤買。
  第二部分（amt0 子集合）：U0 +1.80（2024–26 −0.74）、U3 +1.80（+0.12）、U5 +2.06（+0.21，扣低流動 1% 後 +1.92）、U10 +1.86（+0.97）、
    U20 +1.69（+1.59，回落 −28）、U30 +1.72（+1.80，回落 −33）→ 降低門檻都不採用（2024–26 比現行差、四段不一致）；1,000 萬維持。
  第三部分：現行 12 項 +1.89（回落 −21）；只用價格 7 項 +1.80（t 2.53，2024–26 −0.77，回落 −41）；只用營收 5 項 −0.58（回落 −52）；
    各拿掉一個營收特性 +1.37～+1.81（全部變差）。→ 營收單獨沒用，但和價格合用時把回落從 −41% 降到 −21%、2024–26 從 −0.77 變 +0.76。
"""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import review7 as R7
from review7 import rescore, sim_cap, summ, SEG4
from mp_exec import prep, stats
from price_cap import addraw
from vs0050 import COMP, nw_t
import robust_select as RS
KW = dict(L=True, W=True, Xs=True, B=True, n=5, cap=None)
NOTDR = lambda F: ~F.code.str.match(r"^91\d\d$")
PRICE = ["r20", "mom6_1", "d_hi250", "m60", "am", "mg20", "rs_g20"]; REV = ["yoy1", "yoy3", "rev_acc", "rev_hi12", "mom1"]

def load(pk):
    D = prep(pk, "2011-01-01", "2027-01-01"); D["cols"] = list(pd.read_pickle(pk)["cols"]); D = addraw(D, pk); D["F0"] = D["F"].copy(); return D

def run(D, feats=COMP, keep=NOTDR, dates=None, xcost=None):
    rescore(D, feats=feats, keep=keep)
    if dates is not None: D["F"] = D["F"][D["F"].date.isin(dates)]
    D["FDS"] = sorted(D["F"].date.unique())
    T, cnt = sim_cap(D, wcap=0.30, xcost=xcost, **KW); return T

def mix(Ts):
    k = []
    for T in Ts:
        x = T[["r", "b"]].copy(); x.index = x.index.to_period("M"); k.append(x)
    idx = k[0].index
    for x in k[1:]: idx = idx.intersection(x.index)
    T = sum(x.loc[idx] for x in k) / len(k); T.index = idx.to_timestamp(); return T

if __name__ == "__main__":
    res = {}
    # 第一部分
    pk = "/tmp/model_v2/vs0050_dloff_dl_extra.pkl"; D = load(pk); days = D["days"]
    ds = pd.Series(days); dl = ds[ds.dt.day >= 10].groupby(ds.dt.to_period("M")).first(); ix = np.searchsorted(ds.values, dl.values) + 1
    TS = {}
    for k in range(5):
        dd = set(ds.iloc[(ix + k)[(ix + k) < len(ds)]]); T = run(D, dates=dd); TS[f"S{k + 1}"] = T
        res[f"S{k + 1}"] = summ(T); print(f"S{k + 1}", json.dumps(res[f"S{k + 1}"], ensure_ascii=False), flush=True)
    for lab, ks in (("B2（S1＋S3 各半）", ["S1", "S3"]), ("B5（S1～S5 各 1/5）", ["S1", "S2", "S3", "S4", "S5"])):
        T = mix([TS[k] for k in ks]); s = summ(T); s["每月超額標準差"] = round(float((T.r - T.b).std()), 2); res[lab] = s; print(lab, json.dumps(s, ensure_ascii=False), flush=True)
    ex = pd.DataFrame({k: (T.r - T.b).groupby(T.index.to_period("M")).mean() for k, T in TS.items()}).dropna()
    res["S1～S5 每月超額相關"] = ex.corr().round(2).to_dict(); res["S1 每月超額標準差"] = round(float(ex.S1.std()), 2)
    s1, b2 = res["S1"], res["B2（S1＋S3 各半）"]
    ok = [b2["2011–2026"] >= s1["2011–2026"] - 0.2, b2["t(落後3)"] >= 2.9, b2["2024-26"] >= s1["2024-26"] - 0.2, b2["最大回落(月底)"] >= s1["最大回落(月底)"] - 3]
    res["B2 判斷"] = {"各條件": ok, "採用": all(ok)}
    del D
    # 第二部分
    pk = "/tmp/model_v2/vs0050_amt0_dl_extra.pkl"; D = load(pk); U = {}
    for lab, th in (("U0 不設", 0), ("U3 ≥300萬", 3e6), ("U5 ≥500萬", 5e6), ("U10 ≥1,000萬（現行）", 1e7), ("U20 ≥2,000萬", 2e7), ("U30 ≥3,000萬", 3e7)):
        kp = (lambda th: (lambda F: NOTDR(F) & (F.a20 >= th)))(th)
        T = run(D, keep=kp); s = summ(T)
        A = D["F"].set_index(["date", "code"]).a20
        s["新買均額中位(萬)"] = round(float(np.nanmedian([A.get((a, c), np.nan) for a, cs in zip(T.index, T.codes) for c in cs.split(",") if c])) / 1e4)
        if th < 1e7:
            xc = lambda a, new, w, A=A: 1.0 * sum(w.get(c, 0) for c in new if A.get((a, c), 1e12) < 1e7)
            T2 = run(D, keep=kp, xcost=xc); s["低流動額外扣1%後"] = round(float((T2.r - T2.b).mean()), 2)
        res[lab] = s; U[lab] = s; print(lab, json.dumps(s, ensure_ascii=False), flush=True)
    u10 = res["U10 ≥1,000萬（現行）"]
    for lab in ("U0 不設", "U3 ≥300萬", "U5 ≥500萬"):
        s = res[lab]; ok = [s["2011–2026"] > u10["2011–2026"], s["t(落後3)"] >= 2.9, s["2024-26"] - u10["2024-26"] >= 0.3,
                            sum(a >= b for a, b in zip(s["四段"], u10["四段"])) >= 3, s["最大回落(月底)"] >= u10["最大回落(月底)"] - 3, s["低流動額外扣1%後"] > u10["2011–2026"]]
        res[lab]["判斷"] = {"各條件": ok, "採用": all(ok)}
    del D
    # 第三部分
    D = load("/tmp/model_v2/vs0050_dl_extra.pkl")
    for lab, fs in (("現行 12 項", COMP), ("只用價格 7 項", PRICE), ("只用營收 5 項", REV), *[(f"拿掉 {r}", [c for c in COMP if c != r]) for r in REV]):
        res[lab] = summ(run(D, feats=fs)); print(lab, json.dumps(res[lab], ensure_ascii=False), flush=True)
    json.dump(res, open(f"{REPO}/research/results/review8.json", "w"), ensure_ascii=False, indent=1, default=str)
