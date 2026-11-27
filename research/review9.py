"""第八份覆核的三個檢驗（第 66 輪；2026/11/27 10:35 登錄，看結果前寫下，看到結果後不修改。全部描述性，不改規則）

起因：使用者 11/27 貼來第八份覆核。它指出 ① 只報每月平均超額會掩蓋複利差距、回落用月底算會低估；② 低波動 5 檔要和隨機 5 檔比；
  ③ 今日分數「k=5 抱 10 天」和「k=10 抱 5 天」的後半段是同一段日曆區間，超額可能是月中到下月初的時段效應，不是選股。
規則：11/27 現行月選（第 58 輪換股日、第 59 輪成交、5 檔、不跳過高價股、排除 TDR、單檔上限 30%），2011–2026，資料 vs0050_dl_extra.pkl。
A 現行規則的完整數字：每月減 0050、t、四段、月底回落（含起點）、每日市值回落（期中用收盤價逐日估值）、
  2024 年 1 月起複利累積報酬與 0050、相對財富（月選 ÷ 0050 − 1）、依換股日年份歸類的年度報酬（不是嚴格曆年，照實標示）。
B 低波動 vs 隨機：同一批前 20 名，把「依 20 日波動由低到高」換成隨機順序（300 組固定種子 0～299），其他規則全部相同；
  另列「分數最高 5 檔」。報低波動版在隨機組裡的百分位：每月減 0050、2024–26、月底回落、最差單月、每月超額標準差。
C 時段效應（今日分數快照 vs0050_fdx_mid_extra.pkl；d0＝每月 11 日起第一個交易日，與第 63 輪相同）：
  C1 分析範圍等權（不選股）減 0050 在三段：前段＝d0+6 開盤 → d0+10 收盤；後段＝d0+10 收盤 → d0+15 收盤；k10 段＝d0+11 開盤 → d0+15 收盤。
  C2 k=10 快照分數前 5 檔（第 63 輪 A，跳過開盤漲 ≥9.5%、≥500 元）在 k10 段減「分析範圍等權」——看選股本身有沒有額外貢獻。
  C3 月底前後：分析範圍等權減 0050，以「每月最後一個交易日」為 0，第 −5～+5 個交易日逐日的平均與 t（月份 NW 落後 3）。
  解讀規則（事先寫下）：C1 後段平均 > 0 且 t ≥ 2、C2 ≈ 0 → 判定第 63 輪短抱的超額主要是時段效應，今日分數本身沒有證據；
    C2 > 0 且 t ≥ 2 → 選股在該時段有額外貢獻，列入往後驗證。
結果（11/27 10:45）：
  A 現行規則：每月 +1.89（t 3.84），2024–26 +0.76；月底回落（含起點）−20.6%（0050 −30.1%），每日市值回落 −29.7%。
    2024/1 起 34 期複利：月選 +283%、0050 +233%，相對財富 +15%（但 2026-07 一期貢獻很大，見第 64 輪）；依換股日年份 2024／2025／2026：62.6／70.4／38.4 對 51.5／47.5／48.8。
  B 低波動 vs 300 組隨機：每月超額 +1.89 對隨機中位 +1.90（第 49 百分位，報酬沒有優勢）；月底回落 −20.6% 對隨機中位 −42.3%（比 300 組都淺）；
    超額標準差 7.76 對 9.83（最低）；但最差單月 −28.7 比 93% 的隨機組差。分數最高 5 檔 +2.14、回落 −40.7%、2024–26 −0.41。
    → 低波動的作用是降回落，不是檔數造成的（隨機 5 檔回落中位 −42%）。
  C1 分析範圍等權減 0050：前段 −0.27（t −1.8）、後段 +0.33（t 2.13）、k10 段 +0.11（t 0.68）。
  C2 k=10 分數前 5 在 k10 段：減等權 +0.75（t 2.44，四段 0.43／−0.15／1.69／1.26）、減 0050 +1.04。
  C3 月底效應：分析範圍等權減 0050，最後交易日 +0.13（t 1.79）、下月第 1 天 +0.20（t 3.06）、第 4～5 天 −0.19／−0.15（t −2.2）。
  → 依事先規則：C2 > 0 且 t ≥ 2，選股在該時段有額外貢獻 → 列入往後驗證（每日分數 11/27 起完整保存）；
    另外確實有月底到月初的中小型股效應，但 k10 段（從開盤買起）只占 +0.11。仍是探索結果，不當買進規則。
"""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import review7 as R7
import mp_exec as M
from review7 import rescore, summ
from mp_exec import prep
from price_cap import addraw
from vs0050 import nw_t
import robust_select as RS
NOTDR = lambda F: ~F.code.str.match(r"^91\d\d$")
KW = dict(L=True, W=True, Xs=True, B=True, n=5, cap=None)

def sim(D, order="vol", seed=None, wcap=0.30):
    """與 review7.sim_cap 相同，但可換候選順序，並回傳每期權重（算每日市值）。"""
    O, Cff, bO, POS, CJ, ND, pc, days = D["O"], D["Cff"], D["bO"], D["POS"], D["CJ"], D["ND"], D["prevc"], D["days"]
    rng = np.random.default_rng(seed) if seed is not None else None
    G = {d: g for d, g in D["F"].groupby("date")}; rows, held, val = [], [], {}; FDS = D["FDS"]
    for a, b_ in zip(FDS[:-1], FDS[1:]):
        ie, ix = POS[a] + 1, POS[b_] + 1
        if ix >= ND: break
        g = G[a].sort_values("cs", ascending=False); rank = {c: i + 1 for i, c in enumerate(g.code)}
        keep = [c for c in held if rank.get(c, 1e9) <= 20][:5]; stuck = [c for c in held if c not in keep and not (O[ie, CJ[c]] > 0)]
        top = g.head(20)
        if order == "vol": cand = top.assign(v=lambda z: z.vol20.fillna(np.inf)).sort_values(["v", "cs"], ascending=[True, False]).code.tolist()
        elif order == "score": cand = top.code.tolist()
        else: cand = list(rng.permutation(top.code.tolist()))
        cand += g.code.iloc[20:].tolist(); new = []
        for c in cand:
            if len(keep) + len(stuck) + len(new) >= 5: break
            if c in keep or c in stuck: continue
            j = CJ[c]
            if not (O[ie, j] > 0): continue
            thr = 0.095 if days[ie] >= pd.Timestamp("2015-06-01") else 0.065
            if pc[ie, j] > 0 and O[ie, j] / pc[ie, j] - 1 >= thr - 1e-9: continue
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
            ex = sum(w[c] - 0.25 for c in over)
            for c in over: w[c] = 0.25
            oth = [c for c in w if c != "_cash" and c not in over]
            for c in oth: w[c] += ex / len(oth)
            to += ex
        rr, ent = {}, {}
        for c in [k for k in w if k != "_cash"]:
            j = CJ[c]; o = O[ie, j] if O[ie, j] > 0 else Cff[ie - 1, j]; exn = O[ix, j] if O[ix, j] > 0 else np.nan
            if not np.isfinite(exn): exn = Cff[ix, j] if np.isfinite(Cff[ix, j]) else Cff[ix - 1, j]
            rr[c] = (exn / o - 1) if (np.isfinite(exn) and o > 0) else 0.0; ent[c] = o
        R = sum(w[c] * (rr.get(c, 0.0) if c != "_cash" else 0.0) for c in w) * 100 - M.COST * to
        val = {c: w[c] * (1 + rr.get(c, 0.0)) for c in w if c != "_cash"}; held = [c for c in w if c != "_cash"]
        rows.append({"date": a, "r": R, "b": (bO[ix] / bO[ie] - 1) * 100, "ie": ie, "ix": ix, "w": dict(w), "ent": ent, "to": to})
    T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index); return T

def daily_dd(D, T):
    Cff, CJ, O = D["Cff"], D["CJ"], D["O"]; nav, path = 1.0, [1.0]
    for _, r in T.iterrows():
        base = nav * (1 - M.COST * r.to / 100)
        for t in range(int(r.ie), int(r.ix)):
            v = sum(wt * (Cff[t, CJ[c]] / r.ent[c] if np.isfinite(Cff[t, CJ[c]]) else 1.0) for c, wt in r.w.items() if c != "_cash") + r.w.get("_cash", 0)
            path.append(base * v)
        nav *= 1 + r.r / 100; path.append(nav)
    p = np.array(path); return round(float((p / np.maximum.accumulate(p) - 1).min() * 100), 1)

def mdd(r):
    v = np.r_[1.0, (1 + r / 100).cumprod().values]; return round(float((v / np.maximum.accumulate(v) - 1).min() * 100), 1)

if __name__ == "__main__":
    res = {}
    PK = "/tmp/model_v2/vs0050_dl_extra.pkl"; D = prep(PK, "2011-01-01", "2027-01-01"); D["cols"] = list(pd.read_pickle(PK)["cols"]); D = addraw(D, PK); D["F0"] = D["F"].copy()
    rescore(D, keep=NOTDR); D["FDS"] = sorted(D["F"].date.unique())
    # A
    T = sim(D); s = summ(T); s["月底回落(含起點)"] = mdd(T.r); s["0050 月底回落"] = mdd(T.b); s["每日市值回落"] = daily_dd(D, T)
    y = T[T.index >= "2024-01-01"]; cr, cb = float((1 + y.r / 100).prod() - 1) * 100, float((1 + y.b / 100).prod() - 1) * 100
    s["2024 起複利"] = {"月選": round(cr, 1), "0050": round(cb, 1), "相對財富": round(((1 + cr / 100) / (1 + cb / 100) - 1) * 100, 1), "期數": len(y)}
    s["依換股日年份"] = {int(k): [round(float((1 + g.r / 100).prod() - 1) * 100, 1), round(float((1 + g.b / 100).prod() - 1) * 100, 1)] for k, g in T.groupby(T.index.year)}
    res["A 現行規則"] = s; print("A", json.dumps(s, ensure_ascii=False), flush=True)
    # B
    def met(T):
        x = T.r - T.b; return {"每月減0050": float(x.mean()), "2024–26": float(x[x.index >= "2024-01-01"].mean()), "月底回落": mdd(T.r), "最差單月": float(x.min()), "超額標準差": float(x.std())}
    base = met(T); sc = met(sim(D, order="score")); rnd = [met(sim(D, order="rand", seed=k)) for k in range(300)]
    pct = {k: round(float(np.mean([r[k] <= base[k] for r in rnd]) * 100)) for k in base}
    res["B 低波動 vs 隨機"] = {"低波動": {k: round(v, 2) for k, v in base.items()}, "分數最高5": {k: round(v, 2) for k, v in sc.items()},
                          "隨機中位數": {k: round(float(np.median([r[k] for r in rnd])), 2) for k in base}, "隨機 5%～95%": {k: [round(float(np.percentile([r[k] for r in rnd], q)), 2) for q in (5, 95)] for k in base},
                          "低波動在隨機裡的百分位（值越大＝數字越大）": pct}
    print("B", json.dumps(res["B 低波動 vs 隨機"], ensure_ascii=False), flush=True)
    # C
    X = pd.read_pickle("/tmp/model_v2/vs0050_fdx_mid_extra.pkl"); days = pd.DatetimeIndex(X["days"]); C = np.asarray(X["C"], float); Ov = np.asarray(X["O"], float)
    inb = np.asarray(X["inb"], bool); b = np.asarray(X["bidx"], float); e5 = M.E50.reindex(days); bO = b * (e5.open / e5.close).where((e5.open > 0) & (e5.close > 0)).fillna(1.0).values
    Cf = pd.DataFrame(C).ffill(limit=10).values
    def ew(t0, t1, op=False):
        p0 = (Ov[t0] if op else Cf[t0]); m = inb[t0 - (1 if op else 0)] & (p0 > 0) & np.isfinite(Cf[t1])
        return float(np.nanmean(Cf[t1, m] / p0[m] - 1) * 100)
    ds = pd.Series(days); d0 = ds[ds.dt.day >= 11].groupby(ds.dt.to_period("M")).first(); pos = {d: i for i, d in enumerate(days)}
    rows = []
    for d in d0:
        if d < pd.Timestamp("2011-01-01"): continue
        i = pos[d]
        if i + 15 >= len(days): continue
        f1 = ew(i + 6, i + 10, op=True) - ((b[i + 10] / bO[i + 6] - 1) * 100); f2 = ew(i + 10, i + 15) - ((b[i + 15] / b[i + 10] - 1) * 100)
        f3 = ew(i + 11, i + 15, op=True) - ((b[i + 15] / bO[i + 11] - 1) * 100); rows.append({"date": d, "前段": f1, "後段": f2, "k10段": f3})
    W = pd.DataFrame(rows).set_index("date"); sg = RS.seg(W.index); c1 = {}
    for k in W.columns:
        x = W[k]; c1[k] = {"平均": round(float(x.mean()), 2), "t": round(nw_t(x.reset_index(drop=True), 3), 2), "四段": [round(float(x[sg == q].mean()), 2) for q in R7.SEG4], "次數": int(len(x))}
    res["C1 分析範圍等權減0050"] = c1; print("C1", json.dumps(c1, ensure_ascii=False), flush=True)
    # C2：k=10 快照分數前 5（同第 63 輪 A）減分析範圍等權，k10 段
    Dm = RS.load("/tmp/model_v2/vs0050_fdx_mid_extra.pkl", "2011-01-01", "2027-01-01"); Dm["cols"] = list(X["cols"]); Dm = addraw(Dm, "/tmp/model_v2/vs0050_fdx_mid_extra.pkl")
    G = {d: g for d, g in Dm["F"].groupby("date")}; CJ = Dm["CJ"]; pc = np.asarray(X["prevc"], float); r2 = []
    for d in d0:
        i = pos[d]
        if i + 15 >= len(days) or days[i + 10] not in G or d < pd.Timestamp("2011-01-01"): continue
        g = G[days[i + 10]].sort_values("cs", ascending=False).head(20); ie = i + 11; js = []
        thr = 0.095 if days[ie] >= pd.Timestamp("2015-06-01") else 0.065
        for c in g.code:
            j = CJ[c]
            if not (Ov[ie, j] > 0) or (pc[ie, j] > 0 and Ov[ie, j] / pc[ie, j] - 1 >= thr - 1e-9) or Dm["rawO"][ie, j] >= 500: continue
            js.append(j)
            if len(js) >= 5: break
        if not js: continue
        top = float(np.nanmean(Cf[i + 15, js] / Ov[ie, js] - 1) * 100) - 0.6
        r2.append({"date": d, "前5減等權": top - ew(ie, i + 15, op=True), "前5減0050": top - ((b[i + 15] / bO[ie] - 1) * 100 - 0.18)})
    W2 = pd.DataFrame(r2).set_index("date"); sg2 = RS.seg(W2.index)
    res["C2 k=10 分數前5（k10 段）"] = {k: {"平均": round(float(W2[k].mean()), 2), "t": round(nw_t(W2[k].reset_index(drop=True), 3), 2), "四段": [round(float(W2[k][sg2 == q].mean()), 2) for q in R7.SEG4], "次數": int(len(W2))} for k in W2.columns}
    print("C2", json.dumps(res["C2 k=10 分數前5（k10 段）"], ensure_ascii=False), flush=True)
    # C3：月底前後逐日
    me = ds.groupby(ds.dt.to_period("M")).last(); c3 = {}
    for off in range(-5, 6):
        xs = []
        for d in me:
            i = pos[d] + off
            if i < 1 or i >= len(days) or d < pd.Timestamp("2011-01-01"): continue
            xs.append(ew(i - 1, i) - ((b[i] / b[i - 1] - 1) * 100))
        x = pd.Series(xs); c3[off] = {"平均": round(float(x.mean()), 3), "t": round(nw_t(x, 3), 2)}
    res["C3 月底前後（最後交易日＝0）分析範圍等權減0050 每日"] = c3; print("C3", json.dumps(c3, ensure_ascii=False), flush=True)
    json.dump(res, open(f"{REPO}/research/results/review9.json", "w"), ensure_ascii=False, indent=1, default=str)
