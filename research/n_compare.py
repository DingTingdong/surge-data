"""月選買幾檔：3、5、7、9 檔並列（第 60 輪；2026/11/25 23:50 登錄，看結果前寫下。使用者要求列出差異，描述性，不設通過門檻）

規則：同現行月選（第 58 輪換股日、第 59 輪全部成交修正），只改檔數 n：前 20 名裡 20 日波動最低的 n 檔；新買的股票進場開盤原始價 ≥ 500 元跳過（同第 49 輪的價格依據）。
  另列：前 20 名全買（參考）。每檔等金額。資料 2011–2026（換股日第 1 天；換股日敏感度沒有重算，所以各檔數之間的差異也可能受換股日運氣影響）。
另外對每個 n 算「一半 0050」的三種做法：每月調回五五、每年 1 月調回五五、一開始五五之後完全不動。
報：每月減 0050 與 t、2024–26、年化、最大回落（月底）、輸 0050 的年數、2024／2025／2026 報酬、每期買 1 張各檔所需資金的中位數（原始價 × 1000 × 檔數）。
結果（11/26 00:10）：（跳過 500 元以上；2026＝1～10 月換股，算到 11/12）
  檔數   每月減0050(t)  2024–26  年化  回落  輸0050年  2024／2025／2026   每期 1 張資金中位（近一年）
  3      +1.89 (3.62)   +0.17    40%   −26%  2/16     28.0／71.9／26.2    17 萬（24 萬）
  5      +1.72 (3.38)   −0.01    38%   −21%  3/16     55.8／43.2／28.7    31 萬（66 萬）
  7      +1.88 (3.67)   −0.49    42%   −23%  3/16     52.8／47.5／14.1    48 萬（93 萬）
  9      +1.91 (3.68)   −0.30    42%   −19%  3/16     63.1／39.8／23.2    61 萬（112 萬）
  前20   +1.80 (3.46)   −0.41    40%   −31%  3/16     61.8／32.2／25.6   168 萬（1,207 萬，不跳過 500 元）
  0050                            16%   −30%          51.5／47.5／48.8
  → 檔數多寡平均報酬差不多，多買主要是降回落、降單年運氣。2024–26 各檔數都約打平 0050。
  一半 0050（3 檔）：每月調回 年化 29%、回落 −21%；每年 1 月調回 28.6%、−21%；放著不動 34.5%、−23%（月選部位越長越大）。
注意：不跳過 500 元的 3 檔，2025 +153.6%、2026 −13.5%；跳過後 2025 +71.9%、2026 +26.2%，差別只來自 3～4 次換掉的股票 → 3 檔的單年結果主要是運氣。

"""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
from mp_exec import prep, sim
from price_cap import addraw
from vs0050 import nw_t
def yearly(r): return r.groupby(r.index.year).apply(lambda z: ((1 + z / 100).prod() - 1) * 100)
def summ(r, b):
    x = r - b; c = (1 + r / 100).cumprod(); yr, yb = yearly(r), yearly(b)
    return {"每月減0050": round(float(x.mean()), 2), "t": round(nw_t(x, 3), 2), "2024–26每月減0050": round(float(x[x.index >= "2024-01-01"].mean()), 2),
            "年化%": round(float(c.iloc[-1] ** (12 / len(c)) - 1) * 100, 1), "最大回落%": round(float((c / c.cummax() - 1).min() * 100)),
            "輸0050年": f"{int((yr < yb).sum())}/{len(yr)}", "2024": round(float(yr.get(2024)), 1), "2025": round(float(yr.get(2025)), 1), "2026": round(float(yr.get(2026)), 1)}
def mix(T, mode):
    r, b = T.r / 100, T.b / 100
    if mode == "每月調回": return 50 * r + 50 * b
    vs, vb, out = 0.5, 0.5, []
    for d, ri, bi in zip(T.index, r, b):
        if mode == "每年1月調回" and d.month == 1 and out: tot = vs + vb; vs, vb = tot / 2, tot / 2
        v0 = vs + vb; vs *= 1 + ri; vb *= 1 + bi; out.append(((vs + vb) / v0 - 1) * 100)
    return pd.Series(out, index=T.index)
if __name__ == "__main__":
    PK = "/tmp/model_v2/vs0050_dl_extra.pkl"; D = prep(PK, "2011-01-01", "2027-01-01"); D["cols"] = list(pd.read_pickle(PK)["cols"]); D = addraw(D, PK); res = {}
    for n in (3, 5, 7, 9, 20):
        T, _ = sim(D, L=True, W=True, Xs=True, B=True, n=n, cap=500 if n < 20 else None)
        cap = []
        for a, cs in zip(T.index, T.codes):
            ie = D["POS"][a] + 1; cap.append(np.nansum([D["rawO"][ie, D["CJ"][c]] for c in cs.split(",") if c]) * 1000 / 1e4)
        k = "前20全買" if n == 20 else f"{n}檔"
        res[k] = {"全部月選": summ(T.r, T.b), "每期1張所需資金中位(萬)": round(float(np.median(cap))), "近一年資金中位(萬)": round(float(np.median(cap[-12:])))}
        for md in ("每月調回", "每年1月調回", "放著不動"): res[k][f"一半0050｜{md}"] = summ(mix(T, md), T.b)
        print(k, json.dumps(res[k], ensure_ascii=False), flush=True)
    res["0050"] = summ(T.b, T.b); json.dump(res, open(f"{REPO}/research/results/n_compare.json", "w"), ensure_ascii=False, indent=1)
