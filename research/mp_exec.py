"""月選穩健版的成交與資金帳修正（第 59 輪；2026/11/25 21:30 登錄，看結果前寫下。修正錯誤，結果不論好壞都採用）

起因：第六份外部覆核（固定 82f86ff 版本重算）指出：
  ① 開盤鎖漲停用「當天開＝高＝低」判斷（整天的行情），再改用同一天開盤價買下一名 → 開盤當下還不知道會不會整天鎖死，含事後資訊。
  ② 每月把持股報酬取平均＝每月重新等權，但成本只算換掉的檔數；續抱不調股數時，實際權重會漂移。
  ③ 補價：資料補值到 60 天，換股日沒開盤價時用補值後收盤價出場，等於假設停牌股按舊價賣掉。
  ④ 主數字 t 用 Newey-West 落後 1，文件寫落後 3。
  ⑤ 0050 用換股日收盤到下次換股日收盤，個股是隔天開盤到開盤。
修正（換股日用第 58 輪規則，資料 vs0050_dl／vs0050_oldestonly_dl）：
  L 鎖漲停改成只看開盤價：開盤價比前一天收盤漲 ≥ 9.5%（2015/6 前 6.5%）就當作買不到，改買下一名（假設 08:59 試撮價≈開盤價，可以在開盤前得知）。
  W 權重漂移：續抱的股票保留原市值，不重新等權；賣掉的市值扣 0.6%（來回）後平均買進新股；第一期等金額。
  X 出場沒有開盤價（停牌）：該股繼續留在帳上（不算賣掉、不換股），用之後第一個有開盤價的日子才賣；統計次數。
  B 0050 改成同一段「隔天開盤 → 下次換股隔天開盤」（含息指數 × 當天開盤／收盤比例；2026 由 etf.csv 補）。
  另外報 NW 落後 1 與 3。
並列：V0 原算法（應重現 +1.80）、V0+L、V0+W、V0+B、全部修正（L+W+X+B）。主數字改用「全部修正」。
結果（11/25 21:40）：修正後幾乎不變，主數字改用「全部修正」。
  V0 原算法 +1.80（t 落後1 3.34／落後3 3.37），2024–26 −0.39，回落 −27%。
  +L 開盤價判斷鎖漲停：+1.79（跳過 3 次）；+W 權重漂移：+1.80（2024–26 −0.29）；+B 0050 開盤到開盤：+1.81。
  全部修正：+1.79（t 3.35／3.36），五段 3.04／2.32／1.71／2.69／−0.28，回落 −27%，年化 39%，58% 的月份贏 0050；停牌延後賣 1 次。
  → 覆核指出的成交與資金帳問題都屬實，但實際影響每月 < 0.1%；月選的風險在於 2024–26 打平、換股日敏感，不在模擬方式。

"""
import json, os, sys, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import robust_select as RS
from vs0050 import nw_t
COST = 0.6
fs = sorted(glob.glob(f"{REPO}/data/*/history/*.csv.gz") + glob.glob(f"{REPO}/data/history/*.csv.gz"))
E50 = pd.concat([pd.read_csv(f, dtype={"code": str}, usecols=["date", "code", "open", "close"]).query("code in ['50','0050']") for f in fs]).drop_duplicates("date")
E50["date"] = pd.to_datetime(E50.date); E50 = E50.set_index("date")[["open", "close"]]
_e = pd.read_csv(f"{REPO}/data/etf.csv", dtype={"code": str}).query("code == '0050'"); _e["date"] = pd.to_datetime(_e.date); E50 = E50.combine_first(_e.set_index("date")[["open", "close"]])

def prep(path, a, b):
    D = RS.load(path, a, b); X = pd.read_pickle(path); days = pd.DatetimeIndex(X["days"])
    D["prevc"] = np.asarray(X["prevc"], float); D["Craw"] = np.asarray(X["C"], float); D["days"] = days
    e5 = E50.reindex(days); r = (e5.open / e5.close).where((e5.open > 0) & (e5.close > 0)).fillna(1.0).values; D["bO"] = np.asarray(D["bidx"], float) * r
    return D

def sim(D, L=False, W=False, Xs=False, B=False, n=3, K=20, cap=None):   # cap：新買的股票進場開盤原始價 ≥ cap 就跳過（11/25 第 60 輪加；預設不變）
    O, Cff, lk, bidx, bO, POS, CJ, ND, pc, days = D["O"], D["Cff"], D["lockn"], D["bidx"], D["bO"], D["POS"], D["CJ"], D["ND"], D["prevc"], D["days"]
    G = {d: g for d, g in D["F"].groupby("date")}; rows, held, val = [], [], {}; cnt = {"停牌延後賣": 0, "開盤近漲停跳過": 0, "原鎖漲停跳過": 0}
    def locked(ie, j):
        if L:
            thr = 0.095 if days[ie] >= pd.Timestamp("2015-06-01") else 0.065
            return pc[ie, j] > 0 and O[ie, j] / pc[ie, j] - 1 >= thr - 1e-9
        return bool(lk[ie - 1, j])
    FDS = D["FDS"]
    for a, b_ in zip(FDS[:-1], FDS[1:]):
        ie, ix = POS[a] + 1, POS[b_] + 1
        if ix >= ND: break
        g = G[a].sort_values("cs", ascending=False); rank = {c: i + 1 for i, c in enumerate(g.code)}
        keep = [c for c in held if rank.get(c, 1e9) <= 20][:n]; stuck = []
        if Xs:   # 停牌（換股日隔天沒開盤價）的舊持股賣不掉，留著
            stuck = [c for c in held if c not in keep and not (O[ie, CJ[c]] > 0)]
        new = []
        cand = g.head(K).assign(v=lambda z: z.vol20.fillna(np.inf)).sort_values(["v", "cs"], ascending=[True, False]).code.tolist() + g.code.iloc[K:].tolist()
        for c in cand:
            if len(keep) + len(stuck) + len(new) >= n: break
            if c in keep or c in stuck: continue
            j = CJ[c]
            if not (O[ie, j] > 0): continue
            if locked(ie, j): cnt["開盤近漲停跳過" if L else "原鎖漲停跳過"] += 1; continue
            if cap and "rawO" in D and np.isfinite(D["rawO"][ie, j]) and D["rawO"][ie, j] >= cap: continue
            new.append(c)
        sold = [c for c in held if c not in keep and c not in stuck]
        if not W:
            to = 1.0 if not rows else len(sold) / n
            port = keep + stuck + new; w = {c: 1 / len(port) for c in port} if port else {}
        else:
            if not rows: w = {c: 1 / len(new) for c in new} if new else {}; to = 1.0
            else:
                tot = sum(val.values()) or 1.0; w0 = {c: v / tot for c, v in val.items()}
                sv = sum(w0.get(c, 0) for c in sold); w = {c: w0[c] for c in keep + stuck if c in w0}
                for c in new: w[c] = sv / len(new) if new else 0
                to = sv
                if not new and sv > 0: w["_cash"] = sv
        rr = {}
        for c in [k for k in w if k != "_cash"]:
            j = CJ[c]; o = O[ie, j] if O[ie, j] > 0 else Cff[ie - 1, j]; ex = O[ix, j] if O[ix, j] > 0 else np.nan
            if not np.isfinite(ex):
                if Xs: cnt["停牌延後賣"] += 1
                ex = Cff[ix, j] if np.isfinite(Cff[ix, j]) else Cff[ix - 1, j]
            rr[c] = (ex / o - 1) if (np.isfinite(ex) and o > 0) else 0.0
        R = sum(w[c] * (rr.get(c, 0.0) if c != "_cash" else 0.0) for c in w) * 100 - COST * to
        val = {c: w[c] * (1 + rr.get(c, 0.0)) for c in w if c != "_cash"}
        held = [c for c in w if c != "_cash"]
        bb = (bO[ix] / bO[ie] - 1) * 100 if B else (bidx[ix - 1] / bidx[ie - 1] - 1) * 100
        rows.append({"date": a, "r": R, "b": bb, "codes": ",".join(new + keep + stuck)})
    T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index); return T, cnt

def stats(T):
    x = T.r - T.b; sg = RS.seg(T.index); m = T.index >= "2011-01-01"; w = (1 + T.r[m] / 100).cumprod()
    return {"2011–2026": round(float(x[m].mean()), 2), "t(落後1)": round(nw_t(x[m], 1), 2), "t(落後3)": round(nw_t(x[m], 3), 2),
            **{s: round(float(x[sg == s].mean()), 2) for s in RS.SEGS if (sg == s).any()}, "最大回落(月底)": round(float((w / w.cummax() - 1).min() * 100)),
            "年化": round(float(w.iloc[-1] ** (12 / m.sum()) - 1) * 100), "贏0050月%": round(float((x[m] > 0).mean() * 100))}

if __name__ == "__main__":
    Do = prep("/tmp/model_v2/vs0050_oldestonly_dl_extra.pkl", "2006-01-01", "2010-01-01"); Dn = prep("/tmp/model_v2/vs0050_dl_extra.pkl", "2011-01-01", "2027-01-01")
    out = {}
    for lab, kw in (("V0 原算法", {}), ("V0+L 開盤價判斷鎖漲停", {"L": True}), ("V0+W 權重漂移", {"W": True}), ("V0+B 0050開盤到開盤", {"B": True}),
                    ("全部修正 L+W+X+B", {"L": True, "W": True, "Xs": True, "B": True})):
        (To, co), (Tn, cn) = sim(Do, **kw), sim(Dn, **kw); s = stats(pd.concat([To, Tn])); s["次數"] = {k: co[k] + cn[k] for k in co}
        out[lab] = s; print(lab, json.dumps(s, ensure_ascii=False), flush=True)
    json.dump(out, open(f"{REPO}/research/results/mp_exec.json", "w"), ensure_ascii=False, indent=1)
