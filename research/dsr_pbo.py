"""DSR 與 PBO：73 輪研究的最後統計檢驗（第 74 輪；2026/11/27 17:55 登錄，看結果前寫下，看到結果後不修改；結果不拿來改任何規則）

起因：第十六份覆核要求在凍結規則後做 Deflated Sharpe Ratio（Bailey & López de Prado 2014）與 PBO／CSCV（Bailey, Borwein, López de Prado & Zhu 2017）。
月選與今日分數分開檢驗。績效＝每期（月選：每個換股期；今日分數：每個月）報酬減 0050，成本一律 0.6%。
一、月選的試驗集合（用現有快取能完整重建的版本，全部 2011–2026、同一套成交規則；基準＝11/27 現行：5 檔、低波動、不跳過高價股、排除 TDR、單檔上限 30%）：
  檔數 3、4、6、7、8、9、10、前 20 全買（8）；改用分數最高 5 檔（1）；跳過 ≥500 元、≥2,000 元（2）；不排除 TDR（1）；
  12 個特性各拿掉一個（12）；只用價格 7 項、只用營收 5 項（2）；分析範圍門檻 0／300 萬／500 萬／2,000 萬／3,000 萬（5，amt0 快取）；
  換股日晚 1～4 個交易日（4，dloff 快取）；停損 −10%／−15%／−20%（3）；加上現行（1）＝ 39 個。
  沒放進來的：期限日當天換股的舊規則（有偷看營收的前視偏差）、第 1～57 輪的早期規則（不同資料版本，序列無法完整重建）、隨機挑選（不是研究選項）。→ 揭露為限制。
二、今日分數的試驗集合：第 72 輪 T1～T4 × 抱 5／10／20 天（12）＋第 73 輪 T1-A、T1-B、T1-C（3）＝ 15 個；受檢對象＝T1-B。
DSR：受檢策略每期超額的 Sharpe（不年化）SR；SR0＝√V[SR_n] ×（(1−γ)Φ⁻¹(1−1/N) ＋ γΦ⁻¹(1−1/(N·e))），γ＝0.5772；
  DSR＝Φ((SR−SR0)·√(T−1) ／ √(1 − 偏態·SR ＋ (峰態−1)/4·SR²))。V[SR_n] 用試驗集合各版本的 SR 變異數。
  N：主要用「有效試驗數」＝試驗集合超額報酬相關矩陣特徵值的參與比（(Σλ)²/Σλ²）；另報 N＝試驗數、100、300、1,000 的敏感度
  （73 輪實際試過的規則總數無法精確計數，估計在數百個）。
PBO（CSCV）：取所有版本都有資料的期數，平均切 16 塊，所有 C(16,8)＝12,870 種「一半樣本內、一半樣本外」組合；
  樣本內選平均超額最高的版本，看它在樣本外的名次百分位 ω，PBO＝ω ≤ 0.5（樣本外不到中位數）的比例。另報樣本外平均超額、樣本外虧損機率。
判讀（覆核建議，事先寫下）：DSR ≥ 95% 且 PBO ≤ 20% 證據較強；DSR 80～95% 或 PBO 20～50% 證據有限；DSR < 80% 或 PBO > 50% 過度擬合疑慮較高。
  DSR 以主要 N（有效試驗數）判讀，N＝300、1,000 的結果一併揭露。
結果（11/27 18:10）：
  月選（39 版、190 期）：有效試驗數 1.9（各版本高度相關）；受檢 SR 0.243／期。DSR：主要 99.9%、N＝39 94.1%、N＝100 89.3%、N＝300 82.0%、N＝1,000 72.2%。
    PBO 81.0%（樣本內最佳版本在樣本外名次中位只在第 29 百分位），但樣本內最佳在樣本外平均仍 +1.39%／期、樣本外虧損機率 0.9%。→ 依事先標準：過度擬合疑慮較高。
    解讀：39 個版本之間「誰比較好」是雜訊（5 檔、低波動、門檻等細部選擇沒有穩定優勢），但整個家族在樣本外幾乎都贏 0050。
  今日分數（15 版、191 個月）：有效試驗數 2.1；T1-B SR 0.217／月。DSR：主要 99.2%、N＝15 90.2%、N＝100 72.0%、N＝300 59.6%、N＝1,000 46.3%。
    PBO 1.0% → 依事先標準：證據較強。但看過結果後發現：12,870 組裡 12,735 組樣本內最佳都是 T1-C（個股占 99%、其他版本約 20%），PBO 低反映的是「高曝險版本一直最好」，不是 T1-B 穩健。
    事後補充（未事先登錄）：拿掉 T1-C 後 PBO 19.0%。
"""
import itertools, json, math, os, sys
import numpy as np, pandas as pd
from scipy.stats import norm, skew, kurtosis
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import review7 as R7
from review7 import rescore, sim_cap
from review9 import sim as sim9, NOTDR
from review12 import sim_stop
from mp_exec import prep
from price_cap import addraw
from vs0050 import COMP
KW = dict(L=True, W=True, Xs=True, B=True, n=5, cap=None)
PRICE = ["r20", "mom6_1", "d_hi250", "m60", "am", "mg20", "rs_g20"]; REV = ["yoy1", "yoy3", "rev_acc", "rev_hi12", "mom1"]

def load(pk):
    D = prep(pk, "2011-01-01", "2027-01-01"); D["cols"] = list(pd.read_pickle(pk)["cols"]); D = addraw(D, pk); D["F0"] = D["F"].copy(); return D

def run(D, feats=COMP, keep=NOTDR, dates=None, **kw):
    rescore(D, feats=feats, keep=keep)
    if dates is not None: D["F"] = D["F"][D["F"].date.isin(dates)]
    D["FDS"] = sorted(D["F"].date.unique()); T, _ = sim_cap(D, wcap=0.30, **{**KW, **kw}); x = T.r - T.b; x.index = x.index.to_period("M"); return x

def dsr(x, srs, N):
    x = np.asarray(x, float); T = len(x); sr = x.mean() / x.std(ddof=1); g = 0.5772156649
    v = np.var(srs, ddof=1); sr0 = math.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / N) + g * norm.ppf(1 - 1 / (N * math.e)))
    sk, ku = skew(x), kurtosis(x, fisher=False)
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    return {"SR": round(sr, 3), "SR0": round(sr0, 3), "DSR%": round(float(norm.cdf(z)) * 100, 1)}

def pbo(M, S=16):
    M = M.dropna(); T, Nn = M.shape; blocks = np.array_split(np.arange(T), S); w, oos, loss = [], [], []
    for comb in itertools.combinations(range(S), S // 2):
        ins = np.concatenate([blocks[i] for i in comb]); out = np.concatenate([blocks[i] for i in range(S) if i not in comb])
        a = M.values[ins].mean(axis=0); b = M.values[out].mean(axis=0); k = int(np.argmax(a))
        r = (b < b[k]).sum() + 0.5 * ((b == b[k]).sum() - 1); w.append((r + 0.5) / Nn); oos.append(b[k]); loss.append(b[k] < 0)
    w = np.array(w); lam = np.log(w / (1 - w))
    return {"期數": T, "版本數": Nn, "PBO%": round(float((w <= 0.5).mean()) * 100, 1), "樣本外名次百分位中位": round(float(np.median(w)) * 100),
            "樣本內最佳在樣本外的平均超額": round(float(np.mean(oos)), 2), "樣本外虧損機率%": round(float(np.mean(loss)) * 100, 1)}

def neff(M):
    c = np.corrcoef(M.dropna().values.T); lam = np.linalg.eigvalsh(c); return float(lam.sum() ** 2 / (lam ** 2).sum())

def evaluate(M, target):
    srs = (M.mean() / M.std(ddof=1)).values; ne = neff(M); out = {"有效試驗數": round(ne, 1), "各版本SR範圍": [round(float(srs.min()), 3), round(float(srs.max()), 3)]}
    for lab, N in (("主要（有效試驗數）", max(ne, 2)), (f"N＝試驗數 {M.shape[1]}", M.shape[1]), ("N＝100", 100), ("N＝300", 300), ("N＝1000", 1000)):
        out[f"DSR {lab}"] = dsr(M[target].dropna(), srs, N)
    out["PBO"] = pbo(M); d, p = out["DSR 主要（有效試驗數）"]["DSR%"], out["PBO"]["PBO%"]
    out["判讀"] = "證據較強" if d >= 95 and p <= 20 else ("過度擬合疑慮較高" if d < 80 or p > 50 else "證據有限")
    return out

if __name__ == "__main__":
    res = {}; S = {}
    D = load("/tmp/model_v2/vs0050_dl_extra.pkl")
    S["現行"] = run(D)
    for n in (3, 4, 6, 7, 8, 9, 10, 20): S[f"{n}檔"] = run(D, n=n)
    rescore(D, keep=NOTDR); D["FDS"] = sorted(D["F"].date.unique()); T = sim9(D, order="score"); x = T.r - T.b; x.index = x.index.to_period("M"); S["分數最高5檔"] = x
    S["跳過≥500"] = run(D, cap=500); S["跳過≥2000"] = run(D, cap=2000); S["不排除TDR"] = run(D, keep=None)
    for c in COMP: S[f"拿掉{c}"] = run(D, feats=[k for k in COMP if k != c])
    S["只用價格"] = run(D, feats=PRICE); S["只用營收"] = run(D, feats=REV)
    rescore(D, keep=NOTDR); D["FDS"] = sorted(D["F"].date.unique())
    for st in (0.10, 0.15, 0.20):
        T, _ = sim_stop(D, st); x = T.r - T.b; x.index = x.index.to_period("M"); S[f"停損{int(st*100)}%"] = x
    del D
    D = load("/tmp/model_v2/vs0050_amt0_dl_extra.pkl")
    for lab, th in (("門檻0", 0), ("門檻300萬", 3e6), ("門檻500萬", 5e6), ("門檻2000萬", 2e7), ("門檻3000萬", 3e7)):
        S[lab] = run(D, keep=(lambda th: (lambda F: NOTDR(F) & (F.a20 >= th)))(th))
    del D
    D = load("/tmp/model_v2/vs0050_dloff_dl_extra.pkl"); ds = pd.Series(D["days"]); dl = ds[ds.dt.day >= 10].groupby(ds.dt.to_period("M")).first(); ix = np.searchsorted(ds.values, dl.values) + 1
    for k in range(1, 5): S[f"晚{k}天"] = run(D, dates=set(ds.iloc[(ix + k)[(ix + k) < len(ds)]]))
    del D
    M = pd.DataFrame(S); M = M[~M.index.duplicated()]; print("月選", M.shape, flush=True)
    res["月選"] = evaluate(M, "現行"); res["月選"]["版本"] = list(M.columns); print("月選", json.dumps({k: v for k, v in res["月選"].items() if k != "版本"}, ensure_ascii=False), flush=True)
    # 今日分數
    v2, v3 = pd.read_pickle("/tmp/ser_v2.pkl"), pd.read_pickle("/tmp/ser_v3.pkl"); Q = {}
    names = ["T1", "T2", "T3", "T4"]; cfg = ["H10 0.6", "H10 1.1", "H5 0.6", "H20 0.6"]
    for i, x in enumerate(v2):
        t, c = names[i // 4], cfg[i % 4]
        if "1.1" not in c: Q[f"{t} {c}"] = x
    Q["T1-A"], Q["T1-B"], Q["T1-C"] = v3[0], v3[2], v3[4]
    M2 = pd.DataFrame(Q); print("今日分數", M2.shape, flush=True)
    res["今日分數"] = evaluate(M2, "T1-B"); res["今日分數"]["版本"] = list(M2.columns); print("今日分數", json.dumps({k: v for k, v in res["今日分數"].items() if k != "版本"}, ensure_ascii=False), flush=True)
    json.dump(res, open(f"{REPO}/research/results/dsr_pbo.json", "w"), ensure_ascii=False, indent=1)
