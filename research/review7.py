"""第七份外部覆核的建議驗證（第 64 輪；2026/11/27 09:20 登錄，看結果前寫下，看到結果後不修改）

起因：使用者 11/27 貼來第七份覆核，主張 ①「≥ 500 元跳過」是 2024–26 打平 0050 的最大主因；② 應排除 TDR、把分析範圍門檻提高到 3,000 萬；
  ③ 單檔權重上限 30%（超過在換股日減到 25%）；④ mg20 做消融測試；⑤ rs_g20 有產業分類前視偏差。
資料：vs0050_dl_extra.pkl（第 58 輪換股日），2011–2026；成交照第 59 輪全部修正（L+W+X+B）、5 檔、成本 0.6%（同第 60 輪 n_compare）。
並列（每個只改一件事，最後一個是覆核建議的組合）：
  V0  現行（新買跳過進場開盤原始價 ≥ 500 元）——應重現第 60 輪 +1.72
  A1  不跳過高價股；A2 跳過 ≥ 2,000 元
  B   排除 91xx TDR（從分析範圍拿掉後重算百分位）
  C   分析範圍改成 20 日均額 ≥ 3,000 萬（重算百分位）
  ABC 覆核的組合：A1＋B＋C
  D1  拿掉 mg20（11 個特性）；D2 拿掉 rs_g20（11 個特性）
  E   單檔權重上限：換股日續抱股權重 > 30% 就減到 25%，多出的平均分給其他持股，減碼金額扣 0.6%
判斷（事先寫下）：
  追求報酬的改法（A1、A2、C、ABC、D1、D2）全部符合才採用：每月減 0050 比 V0 高、t ≥ 2.9、2024–26 比 V0 高 ≥ 0.3、
    四段（2011–15、2016–19、2020–23、2024–26）至少 3 段不低於 V0、最大回落（月底）不比 V0 深 3 個百分點以上。
  整理／風控的改法（B、E）：每月減 0050 不比 V0 低 0.2 以上、回落不比 V0 深 3 點以上就採用（理由事先就成立，不要求變好）。
  累計多檢驗 8 個規則，沒有做多重檢定校正，所以採用門檻比單一檢驗嚴。
另報（描述性）：
  ① V0 裡「低波動 5 檔本來會選、但因 ≥ 500 元跳過」的次數與年份；A1 實際選到 ≥ 500 元股票的期數；那些股票該期的報酬減 0050。
  ② 停用標準的機率：用 V0 每月減 0050 的標準差，在真實每月超額 = +0.5、+0.8、+1.72 時，
     「12 期累計落後 0050 超過 12%」與「18 期累計超額 < 0」的機率（常態近似，月份獨立）。
結果（11/27 09:45）：每月減 0050（t 落後3）｜2024–26｜四段｜回落
  V0  現行      +1.72（3.38）｜−0.01｜2.23／1.34／2.67／−0.01｜−21   （重現第 60 輪）
  A1  不跳過    +1.89（3.86）｜+0.76｜2.36／1.32／2.67／0.76 ｜−21   → 五個條件都過
  A2  ≥2,000    +1.81（3.71）｜+0.32｜2.36／1.32／2.67／0.32 ｜−21   → 都過（但每一項都不如 A1）
  B   排除 TDR  +1.70（3.34）｜−0.01｜回落 −21 → 採用（整理類）
  C   3,000 萬  +1.56（3.31）｜+1.70｜2016–19 −0.01｜回落 −36 → 不採用
  ABC 組合      +1.59（3.44）｜+1.82｜2016–19 −0.12｜回落 −36 → 不採用
  D1  拿掉 mg20 +1.35（2.39）｜−0.70｜回落 −30 → 不採用（mg20 有貢獻，不是可有可無）
  D2  拿掉 rs_g20 +1.21（2.46）｜−0.71 → 不採用（但 rs_g20 用現在產業分類，無法排除前視偏差的貢獻）
  E   權重上限  +1.72（3.38）｜回落 −21｜16 年只減碼 5 次 → 採用（整理類，幾乎沒影響）
  另報①：A1 和 V0 只有 22 期不同（17 期在 2024 後）。2024–26 的差異合計 +26.2 個百分點，其中 2026-07 一期（川湖 8,050 元）就 +22.7，
    去掉那一期只剩 +3.5（34 個月）→ 「500 元是 2024–26 打平主因」不成立；改善幾乎來自單一個月。
  另報②：每月超額標準差 8.05%。真實每月 +0.5／+0.8／+1.72 時，「12 期落後 > 12%」機率 26／22／12%，「18 期累計 < 0」40／34／18%。
  → A1 依事先規則通過，但 500 元是使用者自訂的買得起條件，改不改由使用者決定；B、E 網站 11/27 起採用。
  11/27 09:30 使用者決定採用 A1（取消 500 元上限），網站同日起改。
"""
import json, os, sys
import numpy as np, pandas as pd
from math import erf, sqrt
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import mp_exec as M
from mp_exec import prep, stats
from price_cap import addraw
from vs0050 import COMP, nw_t
import robust_select as RS
PK = "/tmp/model_v2/vs0050_dl_extra.pkl"
SEG4 = ["2011-15", "2016-19", "2020-23", "2024-26"]

def rescore(D, feats=COMP, keep=None):
    F = D["F0"] if keep is None else D["F0"][keep(D["F0"])]
    F = F.copy()
    for c in feats: F["p_" + c] = F.groupby("date")[c].rank(pct=True).fillna(0.5)
    F["cs"] = F[["p_" + c for c in feats]].mean(axis=1); D["F"] = F; return D

def sim_cap(D, wcap=None, **kw):
    """mp_exec.sim 的權重上限版本（只在 W=True 時有作用）：先照原規則換股，再把 > wcap 的持股減到 0.25。"""
    if wcap is None: return M.sim(D, **kw)
    orig = M.sim
    # 直接重寫一份：和 mp_exec.sim 相同，只在算完 w 之後加上限
    O, Cff, bO, POS, CJ, ND, pc, days = D["O"], D["Cff"], D["bO"], D["POS"], D["CJ"], D["ND"], D["prevc"], D["days"]
    n, cap = kw.get("n", 5), kw.get("cap")
    G = {d: g for d, g in D["F"].groupby("date")}; rows, held, val = [], [], {}; trims = 0
    FDS = D["FDS"]
    for a, b_ in zip(FDS[:-1], FDS[1:]):
        ie, ix = POS[a] + 1, POS[b_] + 1
        if ix >= ND: break
        g = G[a].sort_values("cs", ascending=False); rank = {c: i + 1 for i, c in enumerate(g.code)}
        keep = [c for c in held if rank.get(c, 1e9) <= 20][:n]
        stuck = [c for c in held if c not in keep and not (O[ie, CJ[c]] > 0)]
        new = []
        cand = g.head(20).assign(v=lambda z: z.vol20.fillna(np.inf)).sort_values(["v", "cs"], ascending=[True, False]).code.tolist() + g.code.iloc[20:].tolist()
        for c in cand:
            if len(keep) + len(stuck) + len(new) >= n: break
            if c in keep or c in stuck: continue
            j = CJ[c]
            if not (O[ie, j] > 0): continue
            thr = 0.095 if days[ie] >= pd.Timestamp("2015-06-01") else 0.065
            if pc[ie, j] > 0 and O[ie, j] / pc[ie, j] - 1 >= thr - 1e-9: continue
            if cap and np.isfinite(D["rawO"][ie, j]) and D["rawO"][ie, j] >= cap: continue
            new.append(c)
        sold = [c for c in held if c not in keep and c not in stuck]
        if not rows: w = {c: 1 / len(new) for c in new} if new else {}; to = 1.0
        else:
            tot = sum(val.values()) or 1.0; w0 = {c: v / tot for c, v in val.items()}
            sv = sum(w0.get(c, 0) for c in sold); w = {c: w0[c] for c in keep + stuck if c in w0}
            for c in new: w[c] = sv / len(new)
            to = sv
            if not new and sv > 0: w["_cash"] = sv
        over = [c for c in w if c != "_cash" and w[c] > wcap]
        if over:
            ex = sum(w[c] - 0.25 for c in over); trims += len(over)
            for c in over: w[c] = 0.25
            oth = [c for c in w if c != "_cash" and c not in over]
            if oth:
                for c in oth: w[c] += ex / len(oth)
            else: w["_cash"] = w.get("_cash", 0) + ex
            to += ex
        rr = {}
        for c in [k for k in w if k != "_cash"]:
            j = CJ[c]; o = O[ie, j] if O[ie, j] > 0 else Cff[ie - 1, j]; exn = O[ix, j] if O[ix, j] > 0 else np.nan
            if not np.isfinite(exn): exn = Cff[ix, j] if np.isfinite(Cff[ix, j]) else Cff[ix - 1, j]
            rr[c] = (exn / o - 1) if (np.isfinite(exn) and o > 0) else 0.0
        R = sum(w[c] * (rr.get(c, 0.0) if c != "_cash" else 0.0) for c in w) * 100 - M.COST * to
        val = {c: w[c] * (1 + rr.get(c, 0.0)) for c in w if c != "_cash"}
        held = [c for c in w if c != "_cash"]
        rows.append({"date": a, "r": R, "b": (bO[ix] / bO[ie] - 1) * 100, "codes": ",".join(new + keep + stuck)})
    T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index); return T, {"減碼次數": trims}

def summ(T):
    s = stats(T); x = T.r - T.b; sg = RS.seg(T.index)
    s["四段"] = [round(float(x[sg == q].mean()), 2) for q in SEG4]; s["期數"] = len(T); return s

def ncdf(z): return 0.5 * (1 + erf(z / sqrt(2)))

if __name__ == "__main__":
    D = prep(PK, "2011-01-01", "2027-01-01"); D["cols"] = list(pd.read_pickle(PK)["cols"]); D = addraw(D, PK); D["F0"] = D["F"].copy()
    KW = dict(L=True, W=True, Xs=True, B=True, n=5)
    runs = {
        "V0 現行（跳過 ≥500）": (dict(), dict(cap=500)),
        "A1 不跳過高價股": (dict(), dict(cap=None)),
        "A2 跳過 ≥2,000": (dict(), dict(cap=2000)),
        "B 排除 TDR": (dict(keep=lambda F: ~F.code.str.match(r"^91\d\d$")), dict(cap=500)),
        "C 均額 ≥3,000 萬": (dict(keep=lambda F: F.a20 >= 3e7), dict(cap=500)),
        "ABC 覆核組合": (dict(keep=lambda F: (~F.code.str.match(r"^91\d\d$")) & (F.a20 >= 3e7)), dict(cap=None)),
        "D1 拿掉 mg20": (dict(feats=[c for c in COMP if c != "mg20"]), dict(cap=500)),
        "D2 拿掉 rs_g20": (dict(feats=[c for c in COMP if c != "rs_g20"]), dict(cap=500)),
        "E 權重上限 30%": (dict(), dict(cap=500, wcap=0.30)),
    }
    res, TT = {}, {}
    for lab, (rk, sk) in runs.items():
        rescore(D, **rk); wc = sk.pop("wcap", None)
        T, cnt = sim_cap(D, wcap=wc, **{**KW, **sk}); TT[lab] = T
        s = summ(T); s["次數"] = cnt; res[lab] = s; print(lab, json.dumps(s, ensure_ascii=False), flush=True)
    # 判斷
    v0 = res["V0 現行（跳過 ≥500）"]; key = "2011–2026"
    for lab in ("A1 不跳過高價股", "A2 跳過 ≥2,000", "C 均額 ≥3,000 萬", "ABC 覆核組合", "D1 拿掉 mg20", "D2 拿掉 rs_g20"):
        s = res[lab]; ok = [s[key] > v0[key], s["t(落後3)"] >= 2.9, s["2024-26"] - v0["2024-26"] >= 0.3,
                            sum(a >= b for a, b in zip(s["四段"], v0["四段"])) >= 3, s["最大回落(月底)"] >= v0["最大回落(月底)"] - 3]
        res[lab]["判斷"] = {"各條件": ok, "採用": all(ok)}
    for lab in ("B 排除 TDR", "E 權重上限 30%"):
        s = res[lab]; ok = [s[key] >= v0[key] - 0.2, s["最大回落(月底)"] >= v0["最大回落(月底)"] - 3]
        res[lab]["判斷"] = {"各條件": ok, "採用": all(ok)}
    # 另報 ①：高價股
    rescore(D); T0, TA = TT["V0 現行（跳過 ≥500）"], TT["A1 不跳過高價股"]; hp = []
    for a in TA.index:
        ie = D["POS"][a] + 1; ca = [c for c in TA.codes[a].split(",") if c]; c0 = set(c for c in T0.codes[a].split(",") if c)
        for c in ca:
            if c not in c0 and D["rawO"][ie, D["CJ"][c]] >= 500: hp.append({"date": a.strftime("%Y-%m"), "code": c, "px": round(float(D["rawO"][ie, D["CJ"][c]]))})
    res["另報_高價股"] = {"A1 多選到 ≥500 元的筆數": len(hp), "逐筆": hp}
    # 另報 ②：停用標準
    x = (T0.r - T0.b); sd = float(x.std()); out = {"每月超額標準差": round(sd, 2)}
    for m in (0.5, 0.8, 1.72):
        out[f"真實 +{m}"] = {"12期落後>12%": round(ncdf((-12 - 12 * m) / (sd * sqrt(12))) * 100), "18期累計<0": round(ncdf((-18 * m) / (sd * sqrt(18))) * 100)}
    res["另報_停用標準機率%"] = out
    print(json.dumps({k: res[k] for k in ("另報_高價股", "另報_停用標準機率%")}, ensure_ascii=False))
    json.dump(res, open(f"{REPO}/research/results/review7.json", "w"), ensure_ascii=False, indent=1)
