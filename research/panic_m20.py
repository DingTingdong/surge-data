"""大跌（恐慌）期間，月選要不要停？（第 9 輪；2026/11/23 01:25 登錄，看結果前寫下，看到結果後不修改）

起因：11/11 這期碰上 11/16 起的急跌，19 檔全跌、平均 −23.5%（0050 −18.4%）。使用者：「大跌期間好像不能撿這種股票」。
恐慌定義同研究頁（vs0050.py）：上市空頭排列家數近 10 日最高，2020 前用比例 ≥ 0.7499、2020 起用家數 ≥ 750。
基準：月選（top20_goal.py 基準，換股日隔天開盤買、還在前 20 的續抱），成本 0.6%。
規則：
  P1 換股日在恐慌期間 → 這個月不選股，全部放 0050（已持有的賣掉）。
  P2 P1 ＋ 持有期間一進入恐慌（第一天觸發）→ 隔天開盤全部賣出，放 0050 到下次換股日。
  停損或換到 0050 都算一次換股（0.6% ÷ 20 每檔），0050 進出扣 0.2%。
判斷（三項都要）：① 2011–2026 每月「規則 − 基準」平均 > 0 且 Newey-West t ≥ 2；② 2006–2009（data/oldest，含 2008 年金融海嘯）平均 > 0；
  ③ 2011–2026 最大回落比基準淺。符合 → 網站「月選」在恐慌期間提示「暫停、改放 0050」並照規則記錄；否則維持現狀，並在說明寫出大跌時的歷史表現。
另報：恐慌月份與平常月份，基準每月減 0050 的平均與月數；P1、P2 減 0050 的全期、四段、年化、最大回落；每次恐慌波段基準的表現。
結果（11/23 01:35）：兩個都不成立 → 網站不暫停，說明寫出大跌時的歷史表現。
  2011–2026 基準每月減0050：換股日已在恐慌 +2.05（15 月）、持有期間才進恐慌 −0.50（23 月）、平常 +2.22（152 月）。
  P1 − 基準 −0.19（t −1.44），最大回落 −34.2（基準 −31.0）；P2 − 基準 −0.63（t −2.71），最大回落 −32.9。
  2006–2009：恐慌換股月 −1.14（8 月）；P1 − 基準 +0.15、回落 −51.6（基準 −55.3）；P2 −0.42。
  解讀：虧損多發生在恐慌訊號出現之前；訊號出現後才賣，等於賣在低點、錯過反彈。
"""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
from vs0050 import COMP, nw_t, per, PER

def load(path, start, end):
    X = pd.read_pickle(path); days, cols = X["days"], X["cols"]
    D = {"O": X["O"], "Cff": pd.DataFrame(X["C"]).ffill(limit=60).values, "lockn": X["lockn"], "bidx": X["bidx"], "panic": X["panic"].astype(bool),
         "POS": {d: i for i, d in enumerate(days)}, "CJ": {c: j for j, c in enumerate(cols)}, "ND": len(days)}
    F = X["fd"].copy(); F = F[(F.date >= start) & (F.date < end)]
    F["cs"] = np.mean([F.groupby("date")[c].rank(pct=True).fillna(0.5).values for c in COMP], axis=0)
    D["FDS"] = sorted(F.date.unique()); D["G"] = {d: g.sort_values("cs", ascending=False).reset_index(drop=True) for d, g in F.groupby("date")}
    return D

def sim(D, mode, cost=0.6, n=20):
    O, Cff, lk, bidx, pn, POS, CJ, ND = D["O"], D["Cff"], D["lockn"], D["bidx"], D["panic"], D["POS"], D["CJ"], D["ND"]
    rows, held, in0050 = [], [], False
    for a, b_ in zip(D["FDS"][:-1], D["FDS"][1:]):
        ta, tb = POS[a], POS[b_]; ie, ix = ta + 1, tb + 1
        if ix >= ND: break
        first = not rows; bper = (bidx[ix - 1] / bidx[ie - 1] - 1) * 100
        if mode in ("P1", "P2") and pn[ta]:
            sells = len(held); r = bper - cost * sells / n - (0.0 if in0050 else 0.2); held = []; in0050 = True
            rows.append({"date": a, "r": r, "b": bper, "pf": True, "trig": False}); continue
        g = D["G"][a]; rank = {c: i + 1 for i, c in enumerate(g.code)}
        keep = [c for c in held if rank.get(c, 1e9) <= n]; new = [c for c in g.code if c not in keep][:n - len(keep)]
        sells = len([c for c in held if c not in keep])
        ktr = None
        if mode == "P2":
            for k in range(ie, ix - 1):
                if pn[k]: ktr = k; break
        rs, hn = [], []
        for c in keep + new:
            j = CJ[c]
            if c in new and (not (O[ie, j] > 0) or lk[ie - 1, j]): continue
            o = O[ie, j] if O[ie, j] > 0 else Cff[ie - 1, j]
            if not (np.isfinite(o) and o > 0): continue
            if ktr is not None:
                ep = O[ktr + 1, j]; ep = ep if ep > 0 else Cff[ktr + 1, j]
                if np.isfinite(ep): rs.append(((ep / o) * (bidx[ix - 1] / bidx[ktr]) - 1) * 100 - 0.2)
                continue
            ex = O[ix, j] if O[ix, j] > 0 else Cff[ix, j]
            if not np.isfinite(ex): ex = Cff[ix - 1, j]
            if np.isfinite(ex): rs.append((ex / o - 1) * 100); hn.append(c)
        nexit = len(rs) if ktr is not None else 0
        to = 1.0 if first else (sells + nexit) / n
        r = (np.mean(rs) if rs else bper) - cost * to - (0.1 if in0050 else 0.0)
        held = hn; in0050 = ktr is not None
        rows.append({"date": a, "r": r, "b": bper, "pf": bool(pn[ta]), "trig": ktr is not None or bool(pn[ie:ix].any())})
    T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index); return T

def mdd(r):
    w = (1 + r / 100).cumprod(); return round(float((w / w.cummax() - 1).min() * 100), 1)
def summ(T):
    x = T.r - T.b; pr = per(x.index); w, wb = (1 + T.r / 100).cumprod(), (1 + T.b / 100).cumprod(); yrs = len(x) / 12
    return {"減0050": (round(float(x.mean()), 2), round(nw_t(x, 1), 2)), "四段": {p: round(float(x[pr == p].mean()), 2) for p in PER if (pr == p).any()},
            "年化 策略/0050": (round(float(w.iloc[-1] ** (1 / yrs) - 1) * 100, 1), round(float(wb.iloc[-1] ** (1 / yrs) - 1) * 100, 1)), "最大回落 策略/0050": (mdd(T.r), mdd(T.b))}

if __name__ == "__main__":
    out = {}
    A = load("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01")
    B = load("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2005-07-01", "2010-03-01")
    for lab, D, s0, s1 in (("2011–2026", A, "2011-01-01", "2027-01-01"), ("2006–2009", B, "2006-01-01", "2010-01-01")):
        T = {m: sim(D, m) for m in ("基準", "P1", "P2")}
        T = {m: t[(t.index >= s0) & (t.index < s1)] for m, t in T.items()}
        base = T["基準"]; xb = base.r - base.b
        o = {"基準": summ(base), "恐慌換股月 基準減0050": (round(float(xb[base.pf].mean()), 2), int(base.pf.sum())),
             "持有期間碰到恐慌的月 基準減0050": (round(float(xb[base.trig & ~base.pf].mean()), 2), int((base.trig & ~base.pf).sum())),
             "平常月 基準減0050": (round(float(xb[~base.trig].mean()), 2), int((~base.trig).sum())),
             "碰到恐慌的月（逐月 基準/0050）": {str(k.date()): (round(float(v.r), 1), round(float(v.b), 1)) for k, v in base[base.trig].iterrows()}}
        for m in ("P1", "P2"):
            d = T[m].r - base.r
            o[m] = {**summ(T[m]), "規則−基準 每月": (round(float(d.mean()), 3), round(nw_t(d, 1), 2))}
        out[lab] = o
        print("==", lab, json.dumps({k: v for k, v in o.items() if not k.startswith("碰到")}, ensure_ascii=False), flush=True)
    j = {}
    for m in ("P1", "P2"):
        a_, b_ = out["2011–2026"][m], out["2006–2009"][m]
        j[m] = bool(a_["規則−基準 每月"][0] > 0 and a_["規則−基準 每月"][1] >= 2 and b_["規則−基準 每月"][0] > 0 and a_["最大回落 策略/0050"][0] > out["2011–2026"]["基準"]["最大回落 策略/0050"][0])
    out["判斷"] = j; print("判斷", j)
    json.dump(out, open(f"{REPO}/research/results/panic_m20.json", "w"), ensure_ascii=False, indent=1, default=str)
