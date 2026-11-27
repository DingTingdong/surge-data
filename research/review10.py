"""月選的虧損月份、回落與停損（第 67 輪；2026/11/27 10:55 登錄，看結果前寫下，看到結果後不修改）

起因：使用者 11/27 問「回落那麼高的機率高嗎、歷史上幾個月賠錢、共同點是什麼、能不能提早預料而停損」。
規則：11/27 現行月選（同第 66 輪 review9.sim），2011–2026，vs0050_dl_extra.pkl，成本 0.6%。
A 描述：每期（約一個月）絕對報酬 < 0 的期數與比例、輸 0050 的期數、虧損期裡 0050 也下跌的比例；
  每日市值的回落事件（從高點跌 ≥ 10% 算一次，到創新高結束）：次數、最深、天數；以及任意時點往後一年內碰到 −20%／−30% 回落的比例。
B 共同點（描述性，看過結果的分組不能當規則）：最差 10% 的期 vs 其他期，換股日當天的
  0050 近 20 日漲幅、0050 相對 60 日均線、分析範圍站上 60 日均線的比例、5 檔平均 20 日漲幅、5 檔平均 20 日波動、5 檔平均分數；以及月份分布。
C 停損與避險（事先指定，全部和現行比）：
  C1 持有中組合市值（相對進場）收盤跌到 −10% → 隔天開盤全部賣、改抱 0050 到下次換股；下一期重新買 5 檔。
  C2 同上 −15%；C3 同上 −20%。
  C4 換股日 0050 收盤在 60 日均線之下 → 這期整期抱 0050（不買個股，已持有的全部賣）。
  採用條件（全部符合）：每月減 0050 不低於現行 0.2 以上、t ≥ 2.9、2024–26 不低於現行 0.2 以上、每日市值回落比現行淺 5 個百分點以上。
結果（11/27 11:05）：
  A 190 期中 69 期賠錢（36%），82 期輸 0050；賠錢期平均 −4.6%、賺錢期 +7.7%；賠錢期裡 59% 是 0050 也跌。
    最差 5 期：2026-01 −16.5（0050 +12.1）、2026-10 −13.2（−8.4）、2015-07 −11.6、2021-07 −11.5、2019-04 −9.6。
    每日市值從高點跌 ≥10% 共 31 次（約一年 2 次），其中 ≥20% 9 次，最深 −29.6%，最長 531 個交易日才回到高點。
    任一天起一年內碰到 −20% 回落的比例 45%、−30% 為 0%，中位數 −16.9%。
  B 最差 10% 的期和其他期，換股日的大盤條件幾乎一樣（0050 近 20 日 +1.5 對 +1.5、站上 60 日線比例 56% 對 55%）；
    只有 5 檔先前漲得較多（20 日 +18.2% 對 +14.6%）、波動略高（2.43 對 2.17）。月份分散在全年。→ 看不出可事先辨認的共同點。
  C 現行 +1.87、每日回落 −29.6%；停損 −10% +1.85、2024–26 +1.05、回落 −27.7%（只淺 1.9 點，13 次）；−15% 回落 −33.6%；−20% −33.2%；
    0050 在 60 日線下抱 0050：+1.13、回落 −32.6%。→ 全部不採用：停損沒有明顯降低回落，−15%、−20% 反而更深（賣在低點後錯過反彈）。
"""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import review7 as R7
import mp_exec as M
from review7 import rescore
from review9 import sim, mdd, NOTDR
from mp_exec import prep, stats
from price_cap import addraw
from vs0050 import nw_t
import robust_select as RS

def path_of(D, T, stop=None, mkt=None):
    """每期逐日市值；stop＝停損門檻（如 0.10）；mkt＝換股日要不要抱 0050 的布林序列。回傳（每期報酬表, 每日淨值）。
    簡化：停損或抱 0050 後，下一期視為空手重買（全部換手成本 0.6%）。"""
    Cff, CJ, O, bO, b = D["Cff"], D["CJ"], D["O"], D["bO"], D["bidx"]
    nav, path, rows, fresh, pmk = 1.0, [1.0], [], False, False
    for a, r in T.iterrows():
        ie, ix = int(r.ie), int(r.ix)
        to = 1.0 if fresh else r.to
        if mkt is not None and mkt.get(a, False):
            base = nav if pmk else nav * (1 - (0.0 if fresh else 0.3) / 100) * (1 - 0.09 / 100)   # 賣掉持股（單邊 0.3%）＋買 0050；連續抱 0050 不再算成本
            for t in range(ie, ix): path.append(base * b[t] / bO[ie])
            nav_new = base * (bO[ix] / bO[ie]); rows.append({"date": a, "r": (nav_new / nav - 1) * 100, "b": r.b, "stop": 0, "mk": 1}); nav = nav_new; path.append(nav); fresh = True; pmk = True; continue
        if pmk: nav *= 1 - 0.09 / 100   # 賣 0050
        pmk = False
        base = nav * (1 - M.COST * to / 100); stopped = None
        for t in range(ie, ix):
            v = sum(wt * (Cff[t, CJ[c]] / r.ent[c] if np.isfinite(Cff[t, CJ[c]]) else 1.0) for c, wt in r.w.items() if c != "_cash") + r.w.get("_cash", 0)
            path.append(base * v)
            if stop is not None and v <= 1 - stop and t + 1 < ix: stopped = t; break
        if stopped is None:
            nav_new = nav * (1 + (r.r + M.COST * r.to) / 100) * (1 - M.COST * to / 100); fresh = False
        else:
            t1 = stopped + 1
            v1 = sum(wt * ((O[t1, CJ[c]] if O[t1, CJ[c]] > 0 else Cff[t1, CJ[c]]) / r.ent[c]) for c, wt in r.w.items() if c != "_cash") + r.w.get("_cash", 0)
            mid = base * v1 * (1 - 0.3 / 100) * (1 - 0.09 / 100)
            for t in range(t1, ix): path.append(mid * b[t] / bO[t1])
            nav_new = mid * (bO[ix] / bO[t1]) * (1 - 0.09 / 100); fresh = True
        rows.append({"date": a, "r": (nav_new / nav - 1) * 100, "b": r.b, "stop": int(stopped is not None), "mk": 0}); nav = nav_new; path.append(nav)
    T2 = pd.DataFrame(rows).set_index("date"); T2.index = pd.DatetimeIndex(T2.index); return T2, np.array(path)

def dd_events(p, th=0.10):
    pk, ev, i0, low = p[0], [], 0, 0.0
    for i, v in enumerate(p):
        if v >= pk:
            if low <= -th: ev.append({"最深%": round(low * 100, 1), "交易日": i - i0})
            pk, i0, low = v, i, 0.0
        else: low = min(low, v / pk - 1)
    if low <= -th: ev.append({"最深%": round(low * 100, 1), "交易日": len(p) - i0, "尚未回到高點": True})
    return ev

if __name__ == "__main__":
    res = {}
    PK = "/tmp/model_v2/vs0050_dl_extra.pkl"; D = prep(PK, "2011-01-01", "2027-01-01"); D["cols"] = list(pd.read_pickle(PK)["cols"]); D = addraw(D, PK); D["F0"] = D["F"].copy()
    rescore(D, keep=NOTDR); D["FDS"] = sorted(D["F"].date.unique()); T = sim(D)
    T0, p0 = path_of(D, T)
    # A
    x = T0.r - T0.b; lose = T0.r < 0
    ev = dd_events(p0); fw = []
    for i in range(0, len(p0) - 250):
        seg = p0[i:i + 251]; fw.append((seg / np.maximum.accumulate(seg) - 1).min())
    fw = np.array(fw)
    res["A"] = {"期數": len(T0), "賠錢期": int(lose.sum()), "賠錢比例%": round(float(lose.mean() * 100)), "輸0050期": int((x < 0).sum()),
                "賠錢期裡0050也跌%": round(float((T0.b[lose] < 0).mean() * 100)), "賠錢期平均%": round(float(T0.r[lose].mean()), 1), "賺錢期平均%": round(float(T0.r[~lose].mean()), 1),
                "最差5期": {d.strftime("%Y-%m"): [round(float(v), 1), round(float(T0.b[d]), 1)] for d, v in T0.r.nsmallest(5).items()},
                "每日回落≥10%事件": ev, "事件數": len(ev),
                "任一天起一年內最大回落": {"≤−20%機率%": round(float((fw <= -0.20).mean() * 100)), "≤−30%機率%": round(float((fw <= -0.30).mean() * 100)), "中位數%": round(float(np.median(fw) * 100), 1)},
                "每日市值回落": round(float((p0 / np.maximum.accumulate(p0) - 1).min() * 100), 1)}
    print("A", json.dumps(res["A"], ensure_ascii=False), flush=True)
    # B
    b = D["bidx"]; POS = D["POS"]; days = D["days"]; ma60 = pd.Series(b).rolling(60).mean().values
    X = pd.read_pickle(PK); inb = np.asarray(X["inb"], bool); C = np.asarray(X["C"], float); Cma = pd.DataFrame(C).rolling(60, min_periods=55).mean().values
    G = {d: g.set_index("code") for d, g in D["F"].groupby("date")}; feats = []
    for a, r in T.iterrows():
        t = POS[a]; cs = [c for c in r.w if c != "_cash"]; g = G[a]
        feats.append({"date": a, "0050近20日%": (b[t] / b[t - 20] - 1) * 100, "0050相對60日線%": (b[t] / ma60[t] - 1) * 100,
                      "站上60日線比例%": float(np.nanmean((C[t] > Cma[t])[inb[t]]) * 100),
                      "5檔20日漲幅%": float(np.nanmean([g.r20.get(c, np.nan) for c in cs]) * 100), "5檔20日波動%": float(np.nanmean([g.vol20.get(c, np.nan) for c in cs]) * 100),
                      "5檔分數": float(np.nanmean([g.cs.get(c, np.nan) for c in cs])), "月": a.month})
    Fz = pd.DataFrame(feats).set_index("date"); worst = T0.r <= T0.r.quantile(0.10)
    res["B 最差10%期 vs 其他"] = {k: [round(float(Fz[k][worst].mean()), 2), round(float(Fz[k][~worst].mean()), 2)] for k in Fz.columns if k != "月"}
    res["B 最差10%期的月份"] = Fz["月"][worst].value_counts().sort_index().to_dict()
    print("B", json.dumps(res["B 最差10%期 vs 其他"], ensure_ascii=False), res["B 最差10%期的月份"], flush=True)
    # C
    base = None
    for lab, kw in (("現行", {}), ("C1 停損 −10%", {"stop": 0.10}), ("C2 停損 −15%", {"stop": 0.15}), ("C3 停損 −20%", {"stop": 0.20}),
                    ("C4 0050 在 60 日線下抱 0050", {"mkt": {a: bool(b[POS[a]] < ma60[POS[a]]) for a in T.index}})):
        T2, p2 = path_of(D, T, **kw); x2 = T2.r - T2.b; sg = RS.seg(T2.index)
        s = {"每月減0050": round(float(x2.mean()), 2), "t": round(nw_t(x2, 3), 2), "2024–26": round(float(x2[x2.index >= "2024-01-01"].mean()), 2),
             "四段": [round(float(x2[sg == q].mean()), 2) for q in R7.SEG4], "每日市值回落": round(float((p2 / np.maximum.accumulate(p2) - 1).min() * 100), 1),
             "月底回落": mdd(T2.r), "觸發期數": int(T2.stop.sum() + T2.mk.sum()), "年化%": round(float(p2[-1] ** (12 / len(T2)) - 1) * 100, 1)}
        if base is None: base = s
        else:
            ok = [s["每月減0050"] >= base["每月減0050"] - 0.2, s["t"] >= 2.9, s["2024–26"] >= base["2024–26"] - 0.2, s["每日市值回落"] >= base["每日市值回落"] + 5]
            s["判斷"] = {"各條件": ok, "採用": all(ok)}
        res[lab] = s; print(lab, json.dumps(s, ensure_ascii=False), flush=True)
    json.dump(res, open(f"{REPO}/research/results/review10.json", "w"), ensure_ascii=False, indent=1, default=str)
