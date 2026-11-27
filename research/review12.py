"""第十份覆核的更正與暫定名單配對測試（第 70 輪；2026/11/27 12:55 登錄，看結果前寫下，看到結果後不修改）

起因：第十份覆核（固定 9d1c289）指出：
  ① 第 67 輪停損版本停損後，下一期仍沿用「不停損版本」的持股與權重，沒有真的空手重選 → 停損結果失真。
  ② 每日市值序列把「每期開盤結算點」和每天收盤混在一起，滾動一年用 251 筆、回到高點的天數都不是嚴格的交易日。
  ③ 「8:40 正式名單沒出來就買暫定名單」沒有依據：第 65 輪測的是延後換股，不是暫定名單。
做法（11/27 現行月選，2011–2026，vs0050_dl_extra.pkl）：
  第一部分（更正第 67 輪 A、C1～C3）：選股與停損在同一個迴圈裡模擬；停損 → 隔天開盤全部賣、抱 0050 到下次換股；
    下一期真的空手，從前 20 名重選波動最低 5 檔。每日淨值每個交易日一筆（收盤），回落事件天數＝交易日數，滾動一年＝250 個交易日。
    採用條件同第 67 輪（每月減 0050 不低於現行 0.2 以上、t ≥ 2.9、2024–26 不低於現行 0.2 以上、每日回落淺 5 點以上）。
  第二部分（暫定名單）：每個換股日用三種融資版本各算一次前 20 名與低波動 5 檔：
    F 完整（現行）；P 用前一個交易日的融資算 mg20（網站暫定名單的做法）；N 融資一律 0.5。
    報：P、N 與 F 的前 20 名重疊、5 檔重疊（平均、最低、完全相同的比例）；
      同一期「5 檔等權、隔天開盤買到下次換股隔天開盤」報酬，P−F、N−F 的每期差（平均、t、四段）。
    判斷（事先寫下）：P 的 5 檔平均重疊 ≥ 4 檔、且 P−F 平均 ≥ −0.3 且 t > −2 → 正式名單來不及時可以買暫定名單；否則等正式名單。
"""
import glob, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import review7 as R7
import mp_exec as M
from review7 import rescore
from review9 import NOTDR
from mp_exec import prep
from price_cap import addraw
from vs0050 import nw_t, COMP
import robust_select as RS

def sim_stop(D, stop=None):
    O, Cff, bO, b, POS, CJ, ND, pc, days = D["O"], D["Cff"], D["bO"], D["bidx"], D["POS"], D["CJ"], D["ND"], D["prevc"], D["days"]
    G = {d: g for d, g in D["F"].groupby("date")}; FDS = D["FDS"]; rows, held, val = [], [], {}; nav = 1.0; daily = {}
    for a, b_ in zip(FDS[:-1], FDS[1:]):
        ie, ix = POS[a] + 1, POS[b_] + 1
        if ix >= ND: break
        g = G[a].sort_values("cs", ascending=False); rank = {c: i + 1 for i, c in enumerate(g.code)}
        keep = [c for c in held if rank.get(c, 1e9) <= 20][:5]; stuck = [c for c in held if c not in keep and not (O[ie, CJ[c]] > 0)]
        cand = g.head(20).assign(v=lambda z: z.vol20.fillna(np.inf)).sort_values(["v", "cs"], ascending=[True, False]).code.tolist() + g.code.iloc[20:].tolist(); new = []
        for c in cand:
            if len(keep) + len(stuck) + len(new) >= 5: break
            if c in keep or c in stuck: continue
            j = CJ[c]
            if not (O[ie, j] > 0): continue
            thr = 0.095 if days[ie] >= pd.Timestamp("2015-06-01") else 0.065
            if pc[ie, j] > 0 and O[ie, j] / pc[ie, j] - 1 >= thr - 1e-9: continue
            new.append(c)
        sold = [c for c in held if c not in keep and c not in stuck]
        if not held: w = {c: 1 / len(new) for c in new} if new else {}; to = 1.0
        else:
            tot = sum(val.values()) or 1.0; w0 = {c: v / tot for c, v in val.items()}
            sv = sum(w0.get(c, 0) for c in sold); w = {c: w0[c] for c in keep + stuck if c in w0}
            for c in new: w[c] = sv / len(new)
            to = sv
            if not new and sv > 0: w["_cash"] = sv
        over = [c for c in w if c != "_cash" and w[c] > 0.30]
        if over:
            ex = sum(w[c] - 0.25 for c in over)
            for c in over: w[c] = 0.25
            oth = [c for c in w if c != "_cash" and c not in over]
            for c in oth: w[c] += ex / len(oth)
            to += ex
        ent = {c: (O[ie, CJ[c]] if O[ie, CJ[c]] > 0 else Cff[ie - 1, CJ[c]]) for c in w if c != "_cash"}
        nav0 = nav; base = nav * (1 - M.COST * to / 100); stopped = None
        val_at = lambda t, px: sum(wt * (px(c, t) / ent[c]) for c, wt in w.items() if c != "_cash") + w.get("_cash", 0)
        pclose = lambda c, t: Cff[t, CJ[c]] if np.isfinite(Cff[t, CJ[c]]) else ent[c]
        for t in range(ie, ix):
            v = val_at(t, pclose); daily[t] = base * v
            if stop is not None and v <= 1 - stop and t + 1 < ix: stopped = t; break
        if stopped is None:
            popen = lambda c, t: O[t, CJ[c]] if O[t, CJ[c]] > 0 else pclose(c, t)
            nav = base * val_at(ix, popen); val = {c: w[c] * popen(c, ix) / ent[c] for c in w if c != "_cash"}; held = [c for c in w if c != "_cash"]
        else:
            t1 = stopped + 1; popen = lambda c, t: O[t, CJ[c]] if O[t, CJ[c]] > 0 else pclose(c, t)
            mid = base * val_at(t1, popen) * (1 - 0.3 / 100) * (1 - 0.09 / 100)
            for t in range(t1, ix): daily[t] = mid * b[t] / bO[t1]
            nav = mid * (bO[ix] / bO[t1]) * (1 - 0.09 / 100); held, val = [], {}
        rows.append({"date": a, "r": (nav / nav0 - 1) * 100, "b": (bO[ix] / bO[ie] - 1) * 100, "stop": int(stopped is not None)})
    T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index)
    s = pd.Series(daily).sort_index(); s.index = days[s.index]; return T, s

def dd_stats(s):
    v = s.values; pk = np.maximum.accumulate(np.r_[1.0, v])[1:]; dd = v / pk - 1
    ev, in_dd, low, i0 = [], False, 0.0, 0
    peak_i = 0
    for i, x in enumerate(v):
        if x >= pk[i] - 1e-12:
            if in_dd and low <= -0.10: ev.append({"最深%": round(low * 100, 1), "交易日": i - peak_i, "起": str(s.index[peak_i].date()), "回到高點": str(s.index[i].date())})
            in_dd, low, peak_i = False, 0.0, i
        else: in_dd = True; low = min(low, x / pk[i] - 1)
    if in_dd and low <= -0.10: ev.append({"最深%": round(low * 100, 1), "交易日": len(v) - 1 - peak_i, "起": str(s.index[peak_i].date()), "尚未回到高點": True})
    fw = np.array([(lambda z: (z / np.maximum.accumulate(z) - 1).min())(v[i:i + 251]) for i in range(0, len(v) - 250)])
    return {"每日回落": round(float(dd.min() * 100), 1), "≥10%事件數": len(ev), "≥20%事件數": sum(e["最深%"] <= -20 for e in ev), "最長交易日": max(e["交易日"] for e in ev) if ev else 0,
            "一年內碰到−20%（重疊視窗比例）%": round(float((fw <= -0.20).mean() * 100)), "一年內最大回落中位數%": round(float(np.median(fw) * 100), 1), "事件": ev}

if __name__ == "__main__":
    res = {}
    PK = "/tmp/model_v2/vs0050_dl_extra.pkl"; D = prep(PK, "2011-01-01", "2027-01-01"); D["cols"] = list(pd.read_pickle(PK)["cols"]); D = addraw(D, PK); D["F0"] = D["F"].copy()
    rescore(D, keep=NOTDR); D["FDS"] = sorted(D["F"].date.unique())
    base = None
    for lab, st in (("現行", None), ("C1 停損 −10%", 0.10), ("C2 停損 −15%", 0.15), ("C3 停損 −20%", 0.20)):
        T, s = sim_stop(D, st); x = T.r - T.b; sg = RS.seg(T.index); d = dd_stats(s)
        r = {"每月減0050": round(float(x.mean()), 2), "t": round(nw_t(x, 3), 2), "2024–26": round(float(x[x.index >= "2024-01-01"].mean()), 2), "四段": [round(float(x[sg == q].mean()), 2) for q in R7.SEG4],
             "賠錢期": int((T.r < 0).sum()), "期數": len(T), "停損次數": int(T.stop.sum()), **{k: v for k, v in d.items() if k != "事件"}}
        if base is None: base = r; r["回落事件"] = d["事件"]
        else:
            ok = [r["每月減0050"] >= base["每月減0050"] - 0.2, r["t"] >= 2.9, r["2024–26"] >= base["2024–26"] - 0.2, r["每日回落"] >= base["每日回落"] + 5]
            r["判斷"] = {"各條件": ok, "採用": all(ok)}
        res[lab] = r; print(lab, json.dumps({k: v for k, v in r.items() if k != "回落事件"}, ensure_ascii=False), flush=True)
    # 第二部分
    fs = sorted(glob.glob(f"{REPO}/data/older/chips/*.csv*") + glob.glob(f"{REPO}/data/old/chips/*.csv*") + glob.glob(f"{REPO}/data/chips/*.csv*"))
    ch = pd.concat([pd.read_csv(f, dtype={"code": str}, usecols=["date", "code", "margin"]) for f in fs], ignore_index=True)
    ch["date"] = pd.to_datetime(ch.date); ch = ch.dropna(subset=["margin"]).drop_duplicates(["date", "code"], keep="last")
    mg = ch.pivot(index="date", columns="code", values="margin").sort_index(); md = mg.index
    F0 = D["F0"]; F0 = F0[NOTDR(F0)].copy(); prevmg = {}
    for a in sorted(F0.date.unique()):
        k = md.searchsorted(a) - 1   # 前一個有融資的交易日
        if k < 20: continue
        prevmg[a] = (mg.iloc[k] / mg.iloc[k - 20] - 1).replace([np.inf, -np.inf], np.nan)
    F0["mg20_prev"] = [prevmg.get(d, pd.Series(dtype=float)).get(c, np.nan) for d, c in zip(F0.date, F0.code)]
    def ranks(F, col):
        F = F.copy(); F["mgx"] = F[col] if col else np.nan
        feats = [c if c != "mg20" else "mgx" for c in COMP]
        for c in feats: F["p_" + c] = F.groupby("date")[c].rank(pct=True).fillna(0.5)
        F["cs"] = F[["p_" + c for c in feats]].mean(axis=1); return F
    O, POS, CJ, days, pc = D["O"], D["POS"], D["CJ"], D["days"], D["prevc"]; bO = D["bO"]; Cff = D["Cff"]
    def picks(g, ie):
        top = g.sort_values("cs", ascending=False).head(20); cand = top.assign(v=lambda z: z.vol20.fillna(np.inf)).sort_values(["v", "cs"], ascending=[True, False]).code.tolist(); out = []
        for c in cand:
            j = CJ[c]; thr = 0.095 if days[ie] >= pd.Timestamp("2015-06-01") else 0.065
            if not (O[ie, j] > 0) or (pc[ie, j] > 0 and O[ie, j] / pc[ie, j] - 1 >= thr - 1e-9): continue
            out.append(c)
            if len(out) >= 5: break
        return set(top.code), out
    V = {"F": ranks(F0, "mg20"), "P": ranks(F0, "mg20_prev"), "N": ranks(F0, None)}
    FDS = sorted(F0.date.unique()); rows = []
    for a, b_ in zip(FDS[:-1], FDS[1:]):
        if a not in prevmg: continue
        ie, ix = POS[a] + 1, POS[b_] + 1
        if ix >= D["ND"]: break
        r = {"date": a}; base_t, base_p = None, None
        for k in "FPN":
            t20, p5 = picks(V[k][V[k].date == a], ie)
            ret = np.mean([((O[ix, CJ[c]] if O[ix, CJ[c]] > 0 else Cff[ix, CJ[c]]) / O[ie, CJ[c]] - 1) * 100 for c in p5]) if p5 else np.nan
            if k == "F": base_t, base_p, base_r = t20, set(p5), ret
            else: r[f"{k}前20重疊"] = len(t20 & base_t); r[f"{k}5檔重疊"] = len(set(p5) & base_p); r[f"{k}−F"] = ret - base_r
        rows.append(r)
    W = pd.DataFrame(rows).set_index("date"); W.index = pd.DatetimeIndex(W.index); sg = RS.seg(W.index); out = {"期數": len(W)}
    for k in "PN":
        dx = W[f"{k}−F"].dropna()
        out[k] = {"前20重疊平均": round(float(W[f"{k}前20重疊"].mean()), 1), "5檔重疊平均": round(float(W[f"{k}5檔重疊"].mean()), 2), "5檔重疊最低": int(W[f"{k}5檔重疊"].min()),
                  "5檔完全相同%": round(float((W[f"{k}5檔重疊"] == 5).mean() * 100)), "每期報酬差平均": round(float(dx.mean()), 2), "t": round(nw_t(dx.reset_index(drop=True), 3), 2),
                  "四段": [round(float(dx[sg[W.index.isin(dx.index)] == q].mean()), 2) for q in R7.SEG4]}
    p = out["P"]; out["判斷"] = {"各條件": [p["5檔重疊平均"] >= 4, p["每期報酬差平均"] >= -0.3 and p["t"] > -2], "可以買暫定名單": p["5檔重疊平均"] >= 4 and p["每期報酬差平均"] >= -0.3 and p["t"] > -2}
    res["第二部分 暫定名單"] = out; print("第二部分", json.dumps(out, ensure_ascii=False), flush=True)
    json.dump(res, open(f"{REPO}/research/results/review12.json", "w"), ensure_ascii=False, indent=1, default=str)
