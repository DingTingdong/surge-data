"""勝率導向：專門拉高「每檔這個月贏 0050」的機率（第 19 輪；2026/11/23 11:20 登錄，看結果前寫下，看到結果後不修改）

起因：使用者的目標是每檔股票短期（一個月）贏 0050 就好。月選每檔一個月贏 0050 的機率只有 43～55%，比隨便挑高 6～9 個百分點；
  之前每一輪都以「平均多賺多少」為目標，沒有專門拉高勝率。
定義：「贏」= 從換股日隔天開盤買、抱到下一個換股日隔天開盤賣，報酬減 0050（含息）同期報酬 > 0.6%（扣掉使用者 2.8 折一次進出）。
  換股日同月選（每月第一個 ≥ 11 號的交易日）。資料：vs0050_extra.pkl（2011–2026）與 vs0050_oldestonly_extra.pkl（2006–2009），只在分析範圍內。
候選特性（18 個，全部在換股日已知）：月選 12 個（r20、mom6_1、d_hi250、m60、am、yoy1、yoy3、rev_acc、rev_hi12、mom1、mg20、rs_g20），
  另加 r60、r120、vcr（收縮比）、vol20（20 日波動）、max20（20 日最大單日漲幅）、r5，以及 a20（20 日成交值，規模／流動性）。
發現期 2011–2019（只用這段決定）：每個特性看同月百分位前 20%、後 20% 的勝率減同月全部股票勝率（每月一個值）。
  方向：2011–15 與 2016–19 兩段都「前 20% 減後 20%」同號才保留，取勝率高的一端；且該端每月勝率差的 Newey-West t（落後 1）≥ 2。
  勝率分 = 保留特性（依方向翻轉後）百分位等權平均，缺值補 0.5。
比較的選法（每月選 20 檔）：
  H0 月選（綜合分數前 20）；H1 勝率分前 20；H2 月選分數與勝率分百分位平均前 20；H3 月選前 40 裡取勝率分最高的 20。
檢驗期：2006–2009、2020–2023、2024–2026（都沒用來決定特性）。
採用條件（全部符合才換掉月選）：
  ① 三個檢驗段每檔勝率都比 H0 高 ≥ 3 個百分點；② 三個檢驗段組合（真實持有：續抱仍在前 20 的、成本 0.6%）每月減 0050 平均都 > 0；
  ③ 2011–2026 組合每月減 0050 Newey-West t ≥ 2.9，且平均不比 H0 少 0.3 以上。
另報：各段每檔平均、中位數、贏時平均、輸時平均、20 檔合起來贏 0050 的月份比例。
結果（11/23 11:30）：三個都不通過（① 勝率沒有提高 3 個百分點）→ 網站維持月選。
  發現期保留 8 個：半年動能、距 250 日高點、營收年增、三個月營收年增、營收加速、營收創 12 個月新高、120 日漲幅、五日漲幅低（短線回檔）；
    各自前 20% 的勝率也只比全部股票高 1.7～3.3 個百分點。
  每檔勝率（檢驗段 2006–09／2020–23／2024–26）：月選 49.2／49.8／41.9；勝率分 48.8／49.1／42.2；平均 48.3／49.9／43.7；前40取 47.7／50.3／44.5。
    全部股票 44.4／42.0／32.7 → 每檔一個月贏 0050 的機率，用這些特性最多只能做到比隨便挑高 5～12 個百分點，大約一半一半。
  意外發現（不是這輪要驗的，只能當線索）：組合每月減 0050，勝率分 +2.21（t 4.36）、平均 +2.23（t 4.19），月選 +1.88（t 3.58）；
    三個檢驗段勝率分 1.46／4.13／1.27 vs 月選 1.38／3.79／0.42，都比較高 → 下一輪事先登錄、用沒看過的方式（換股日往後挪、前瞻紀錄）再驗。
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import robust_select as RS
from vs0050 import nw_t

FEATS = ["r20", "mom6_1", "d_hi250", "m60", "am", "yoy1", "yoy3", "rev_acc", "rev_hi12", "mom1", "mg20", "rs_g20", "r60", "r120", "vcr", "vol20", "max20", "r5", "a20"]
COST = 0.6

def picks_frame(D):
    """每個換股日、每檔：隔天開盤到下個換股日隔天開盤的報酬減 0050。"""
    O, Cff, lk, bidx, POS, CJ, ND, F = D["O"], D["Cff"], D["lockn"], D["bidx"], D["POS"], D["CJ"], D["ND"], D["F"]
    out = []
    for a, b_ in zip(D["FDS"][:-1], D["FDS"][1:]):
        ie, ix = POS[a] + 1, POS[b_] + 1
        if ix >= ND: break
        g = F[F.date == a].copy(); j = g.code.map(CJ).values
        o = O[ie, j]; ex = np.where(O[ix, j] > 0, O[ix, j], Cff[ix, j]); ex = np.where(np.isfinite(ex), ex, Cff[ix - 1, j])
        ok = (o > 0) & ~lk[ie - 1, j] & np.isfinite(ex)
        g["ret"] = np.where(ok, (ex / np.where(o > 0, o, np.nan) - 1) * 100, np.nan); g["bm"] = (bidx[ix - 1] / bidx[ie - 1] - 1) * 100
        out.append(g)
    P = pd.concat(out, ignore_index=True); P["x"] = P.ret - P.bm; P["win"] = (P.x > COST).astype(float).where(P.x.notna())
    P["p_a20"] = P.groupby("date").a20.rank(pct=True); P["seg"] = RS.seg(P.date); return P

def add_score(P, keep):
    P["wr"] = np.mean([(P["p_" + f] if d > 0 else 1 - P["p_" + f]).fillna(0.5).values for f, d in keep.items()], axis=0)
    P["p_wr"] = P.groupby("date").wr.rank(pct=True); P["p_cs"] = P.groupby("date").cs.rank(pct=True); P["h2"] = (P.p_wr + P.p_cs) / 2
    r40 = P.groupby("date").cs.rank(ascending=False) <= 40; P["h3"] = np.where(r40, P.wr, -1.0); return P

def per_pick(P, col):
    t = P.sort_values(col, ascending=False).groupby("date").head(20); t = t[t.x.notna()]; res = {}
    for s in RS.SEGS:
        b = t[t.seg == s]; a = P[(P.seg == s) & P.x.notna()]
        if not len(b): continue
        pm = b.groupby("date").x.mean()
        res[s] = {"勝率": round(float(b.win.mean() * 100), 1), "全部": round(float(a.win.mean() * 100), 1), "平均": round(float(b.x.mean()), 2), "中位": round(float(b.x.median()), 2),
                  "贏時": round(float(b.x[b.x > COST].mean()), 1), "輸時": round(float(b.x[b.x <= COST].mean()), 1), "合起來贏的月份%": round(float((pm > COST).mean() * 100))}
    return res

if __name__ == "__main__":
    Dn = RS.load("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01"); Do = RS.load("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01")
    Pn, Po = picks_frame(Dn), picks_frame(Do); P = pd.concat([Po, Pn], ignore_index=True); out = {}
    disc = P[P.seg.isin(["2011-15", "2016-19"]) & P.win.notna()]; keep = {}; rep = {}
    for f in FEATS:
        q = disc.assign(hi=disc["p_" + f] >= 0.8, lo=disc["p_" + f] <= 0.2, base=disc.groupby("date").win.transform("mean"))
        m = q.groupby(["seg", "date"]).apply(lambda g: pd.Series({"hi": g.win[g.hi].mean() - g.base.iloc[0], "lo": g.win[g.lo].mean() - g.base.iloc[0]}))
        sg = m.groupby(level=0).mean(); d1 = np.sign(sg.hi - sg.lo)
        if d1.nunique() == 1 and d1.iloc[0] != 0:
            d = int(d1.iloc[0]); ser = (m.hi if d > 0 else m.lo).dropna(); t = nw_t(ser.reset_index(drop=True), 1)
            rep[f] = {"方向": "高" if d > 0 else "低", "勝率差（百分點）": round(float(ser.mean() * 100), 1), "t": round(t, 2)}
            if ser.mean() > 0 and t >= 2: keep[f] = d
        else: rep[f] = {"方向": "兩段不一致", "前20%": sg.hi.round(3).to_dict(), "後20%": sg.lo.round(3).to_dict()}
    out["發現期各特性"] = rep; out["保留"] = {f: ("高" if d > 0 else "低") for f, d in keep.items()}; print("發現期", json.dumps(rep, ensure_ascii=False), "\n保留", out["保留"], flush=True)
    P = add_score(P, keep); Pn = P[P.date >= "2011-01-01"]; Po = P[P.date < "2011-01-01"]
    for D_, Q in ((Dn, Pn), (Do, Po)):
        F = D_["F"].merge(Q[["date", "code", "wr", "h2", "h3"]], on=["date", "code"], how="left"); F[["wr", "h2", "h3"]] = F[["wr", "h2", "h3"]].fillna(-1); D_["F"] = F
    pp = {}; port = {}
    for lab, col in (("H0 月選", "cs"), ("H1 勝率分", "wr"), ("H2 平均", "h2"), ("H3 月選前40取勝率", "h3")):
        pp[lab] = per_pick(P, col)
        Tn = RS.sim(Dn, col, cost=COST); To = RS.sim(Do, col, cost=COST); T = pd.concat([To, Tn]); x = T.r - T.b; sgm = RS.seg(T.index)
        port[lab] = {"2011–2026": (round(float(x[T.index >= "2011-01-01"].mean()), 2), round(nw_t(x[T.index >= "2011-01-01"], 1), 2)), **{s: round(float(x[sgm == s].mean()), 2) for s in RS.SEGS}}
        print("==", lab, "\n  每檔", json.dumps(pp[lab], ensure_ascii=False), "\n  組合 每月減0050", port[lab], flush=True)
    TEST = ["2006-09", "2020-23", "2024-26"]
    for lab in ("H1 勝率分", "H2 平均", "H3 月選前40取勝率"):
        c = {"①": all(pp[lab][s]["勝率"] - pp["H0 月選"][s]["勝率"] >= 3 for s in TEST), "②": all(port[lab][s] > 0 for s in TEST),
             "③": port[lab]["2011–2026"][1] >= 2.9 and port[lab]["2011–2026"][0] >= port["H0 月選"]["2011–2026"][0] - 0.3}
        out[lab] = {"每檔": pp[lab], "組合": port[lab], "各項": c, "通過": all(c.values())}; print(lab, c, "通過" if all(c.values()) else "")
    out["H0 月選"] = {"每檔": pp["H0 月選"], "組合": port["H0 月選"]}
    json.dump(out, open(f"{REPO}/research/results/winrate.json", "w"), ensure_ascii=False, indent=1, default=str)
