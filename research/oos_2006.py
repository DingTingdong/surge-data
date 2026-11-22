"""月選在 2006–2009 年（完全沒用過、含 2008 金融海嘯）還成不成立？

全部在資料到之前寫下（2026/11/22 22:10；oldest-era 工作流程剛開始抓），看到結果後不修改。
資料：data/oldest（2005-01～2009-12；上櫃股價約 2007 年中才有，之前只有上市股；上市融資 2005 起、上櫃 2007 起；營收 2004 起）
  接上 data/older、data/old、data/history。用環境變數 INCLUDE_OLDEST=1 讓 research/vs0050.py 多讀 data/oldest、另存快取，不影響原本結果。
規則完全照網站「月選」（top20_goal.py 基準）：每月 11 日後第一個交易日收盤後，分析範圍（20 日均額 ≥ 1,000 萬）內 12 個「趨勢＋營收」特性
  當天百分位等權平均最高的 20 檔，隔天開盤買；下次換股日還在前 20 的續抱；開盤一價漲停（2015/6 前 ≥ 6.5%）買不到不計。
  報酬算法同 top20_exact.py（換股日隔天開盤賣舊買新，每天只算一次）；0050 含息、同段收盤到收盤。成本 0.6%（另報 1.1%）。
檢驗期：2006-01～2009-12 的換股（約 48 個月）。
主要判斷：每月減 0050 平均 > 0 且 Newey-West t（落後 1）≥ 2.0 → 「新期間也成立」。
  （這是事先固定的規則在新期間重複驗證，只有一個檢驗，所以用 2.0，不用找新訊號時的 2.9。）
另報（不判定）：逐年（2006、2007、2008、2009）策略與 0050、最大回落（2008 年那段）、減等權分析範圍、V1 波動控管版本、
  只用上市股（整段都一致）的結果、每月贏 0050 的比例。
補充（2026/11/23 00:05 登錄，2006–2009 資料仍未到）：10% 停損版（top20_exits.py 的 X1）也在這段驗證，並決定網站要不要加停損。
  理由：第 5、6 輪顯示「換股日晚兩天時 2024–26 也要贏」這一項，所有規則都只在雜訊範圍內（34 個月，每月標準誤約 1.3%），幾乎不可能通過；
  改用一段完全沒用過、而且有 2008 年大跌的期間，來檢驗停損是否真的有幫助。
  X1 規則照 top20_exits.py（收盤比這期起點價低 10% → 隔天開盤賣，資金放 0050 到下次換股日；停損賣出算換股成本，0050 進出扣 0.2%）。
  網站加停損的條件（三項都要）：2006–2009（成本 0.6%）X1 每月減 0050 平均 > 0；X1 的最大回落比基準淺；X1 每月減 0050 不比基準少 0.3 個百分點以上。
  符合 → 「月選」分頁顯示每檔的停損價（這期起點價 × 0.9），說明改成含停損的版本；不符合 → 維持現狀。
結果（11/23 00:15）：
  資料：上市 2005–2009 每日都有，上櫃 2007/7/2 起；營收 2004–2009 每月 1,170–1,310 家；無抓取失敗。0050 逐年 2006 +21.1、2007 +10.3、2008 −43.3、2009 +75.0。
  月選（成本 0.6%）：每月減 0050 +1.23（t 1.48）→ 未達 t ≥ 2.0，「新期間也成立」＝否。四年都贏 0050（30.4/15.1、22.4/6.3、−39.9/−41.3、130.3/84.6），
    每月贏 0050 58%；減等權 +0.56（t 0.99）→ 這段挑股優勢弱；最大回落 −55.3（0050 −51.2）。成本 1.1% +0.82（t 0.99）。只用上市股 +1.14（t 1.35）。V1 +1.04（t 1.31）。
  10% 停損：+0.27（t 0.36），最大回落 −65.0（基準 −55.3）→ 網站加停損＝否（大跌後急彈時被停損出場）。
"""
import glob, json, os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
from vs0050 import COMP, nw_t
# 11/23：快照只建 2005–2010（整段 2005–2026 會超過記憶體），用 OLDEST_ONLY=1 建
X = pd.read_pickle("/tmp/model_v2/vs0050_oldestonly_extra.pkl"); days, cols = X["days"], X["cols"]; Ov, Cv, lockn, bidx = X["O"], X["C"], X["lockn"], X["bidx"]
Cff = pd.DataFrame(Cv).ffill(limit=60).values; POS = {d: i for i, d in enumerate(days)}; CJ = {c: j for j, c in enumerate(cols)}; ND = len(days)
F = X["fd"].copy(); F = F[(F.date >= "2005-07-01") & (F.date < "2010-03-01")]
F["cs"] = np.mean([F.groupby("date")[c].rank(pct=True).fillna(0.5).values for c in COMP], axis=0)
mk = {}
for fn in sorted(glob.glob(f"{REPO}/data/oldest/history/*.csv.gz")):
    h = pd.read_csv(fn, usecols=["code", "market"], dtype={"code": str}); mk.update(dict(zip(h.code, h.market)))
FDS = sorted(F.date.unique())

def run(Fsub, n=20):
    G = {d: g.sort_values("cs", ascending=False).reset_index(drop=True) for d, g in Fsub.groupby("date")}
    rows, held = [], []
    for a, b_ in zip(FDS[:-1], FDS[1:]):
        ta, tb = POS[a], POS[b_]; g = G[a]; rank = {c: i + 1 for i, c in enumerate(g.code)}
        keep = [c for c in held if rank.get(c, 1e9) <= n]; new = [c for c in g.code if c not in keep][:n - len(keep)]
        sells = len([c for c in held if c not in keep]); to = 1.0 if not held else sells / n
        def ret(c):
            j = CJ[c]; o = Ov[ta + 1, j] if Ov[ta + 1, j] > 0 else Cff[ta, j]
            ex = Ov[tb + 1, j] if Ov[tb + 1, j] > 0 else Cff[tb + 1, j]
            if not np.isfinite(ex): ex = Cff[tb, j]
            return (ex / o - 1) * 100 if np.isfinite(ex) and np.isfinite(o) and o > 0 else None
        rs, hn = [], []
        for c in keep + new:
            j = CJ[c]
            if c in new and (not (Ov[ta + 1, j] > 0) or lockn[ta, j]): continue
            v = ret(c)
            if v is not None: rs.append(v); hn.append(c)
        held = hn
        us = [v for v in (ret(c) for c in g.code) if v is not None]
        rows.append({"date": a, "g": np.mean(rs), "to": to, "b": (bidx[tb] / bidx[ta] - 1) * 100, "u": np.mean(us), "n": len(rs), "N": len(g)})
    T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index); return T

def run_x1(Fsub, n=20, cost=0.6):
    """同 run，但加 10% 停損（top20_exits.py X1）。回傳每月 r（已扣成本）、b。"""
    G = {d: g.sort_values("cs", ascending=False).reset_index(drop=True) for d, g in Fsub.groupby("date")}
    rows, held = [], []
    for a, b_ in zip(FDS[:-1], FDS[1:]):
        ta, tb = POS[a], POS[b_]; ie, ix = ta + 1, tb + 1; first = not rows; g = G[a]; rank = {c: i + 1 for i, c in enumerate(g.code)}
        bper = (bidx[ix - 1] / bidx[ie - 1] - 1) * 100
        keep = [c for c in held if rank.get(c, 1e9) <= n]; new = [c for c in g.code if c not in keep][:n - len(keep)]
        sells = len([c for c in held if c not in keep]); rs, hn, nstop = [], [], 0
        for c in keep + new:
            j = CJ[c]
            if c in new and (not (Ov[ie, j] > 0) or lockn[ie - 1, j]): continue
            o = Ov[ie, j] if Ov[ie, j] > 0 else Cff[ie - 1, j]
            if not (np.isfinite(o) and o > 0): continue
            k_stop = None
            for k in range(ie, ix - 1):
                ck = Cff[k, j]
                if not np.isfinite(ck): break
                if ck <= o * 0.9: k_stop = k; break
            if k_stop is not None:
                ep = Ov[k_stop + 1, j]; ep = ep if ep > 0 else Cff[k_stop + 1, j]
                if np.isfinite(ep): rs.append(((ep / o) * (bidx[ix - 1] / bidx[k_stop]) - 1) * 100 - 0.2); nstop += 1; continue
            ex = Ov[ix, j] if ix < ND and Ov[ix, j] > 0 else Cff[min(ix, ND - 1), j]
            if not np.isfinite(ex): ex = Cff[ix - 1, j]
            if np.isfinite(ex): rs.append((ex / o - 1) * 100); hn.append(c)
        held = hn; to = 1.0 if first else (sells + nstop) / n
        rows.append({"date": a, "r": (np.mean(rs) if rs else bper) - cost * to, "b": bper, "stops": nstop})
    T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index); return T

def summarize(T, cost):
    T = T[(T.index >= "2006-01-01") & (T.index < "2010-01-01")]
    r = T.g - cost * T.to; x = r - T.b; w, wb = (1 + r / 100).cumprod(), (1 + T.b / 100).cumprod()
    yr = pd.DataFrame({"策略%": (1 + r / 100).groupby(T.index.year).prod() * 100 - 100, "0050%": (1 + T.b / 100).groupby(T.index.year).prod() * 100 - 100,
                       "全部股票%": (1 + T.u / 100).groupby(T.index.year).prod() * 100 - 100}).round(1)
    return {"每月減0050 平均/t": (round(float(x.mean()), 2), round(nw_t(x, 1), 2)), "每月減等權 平均/t": (round(float((r - T.u).mean()), 2), round(nw_t(r - T.u, 1), 2)),
            "月數": int(len(x)), "每月贏0050%": round(float((x > 0).mean() * 100)), "最大回落 策略/0050": (round(float((w / w.cummax() - 1).min() * 100), 1), round(float((wb / wb.cummax() - 1).min() * 100), 1)),
            "逐年": yr.to_dict(orient="index"), "平均分析範圍檔數": round(float(T.N.mean()))}, x, r, T

if __name__ == "__main__":
    out = {}
    T = run(F)
    for cost in (0.6, 1.1):
        s, x, r, Tt = summarize(T, cost); out[f"全部 成本{cost}%"] = s
        if cost == 0.6:
            ok = s["每月減0050 平均/t"][0] > 0 and s["每月減0050 平均/t"][1] >= 2.0; out["新期間也成立"] = bool(ok)
            # V1：σ 用 2005-07 起的整段（只用過去）
            x0 = T.g - cost * T.to - T.b
            sig = pd.Series([x0.iloc[max(0, i - 7):max(0, i - 1)].std(ddof=1) if i >= 7 else np.nan for i in range(len(x0))], index=x0.index)
            tg = sig.expanding(min_periods=6).median(); w = (tg / sig).clip(upper=1.0).fillna(1.0); x1 = (w * x0 - 0.5 * cost * w.diff().abs().fillna(0))
            x1 = x1[(x1.index >= "2006-01-01") & (x1.index < "2010-01-01")]; out["另報 V1 每月減0050 平均/t"] = (round(float(x1.mean()), 2), round(nw_t(x1, 1), 2))
        print(f"== 成本 {cost}%", json.dumps(s, ensure_ascii=False), flush=True)
    # 10% 停損版（11/23 補充登錄的網站決策）
    TX = run_x1(F); TX = TX[(TX.index >= "2006-01-01") & (TX.index < "2010-01-01")]; xx = TX.r - TX.b
    wx, wb_ = (1 + TX.r / 100).cumprod(), (1 + TX.b / 100).cumprod()
    s0, x0b, r0b, _ = summarize(T, 0.6); w0 = (1 + r0b / 100).cumprod()
    dd = lambda w: round(float((w / w.cummax() - 1).min() * 100), 1)
    out["X1 10%停損"] = {"每月減0050 平均/t": (round(float(xx.mean()), 2), round(nw_t(xx, 1), 2)), "最大回落 X1/基準/0050": (dd(wx), dd(w0), dd(wb_)),
                       "逐年 X1/0050": {int(k): (round(float(v.r)), round(float(v.b))) for k, v in pd.DataFrame({"r": (1 + TX.r / 100).groupby(TX.index.year).prod() * 100 - 100, "b": (1 + TX.b / 100).groupby(TX.index.year).prod() * 100 - 100}).iterrows()},
                       "平均每月停損檔數": round(float(TX.stops.mean()), 1)}
    add = xx.mean() > 0 and dd(wx) > dd(w0) and xx.mean() >= x0b.mean() - 0.3
    out["網站加停損"] = bool(add); print("== X1", out["X1 10%停損"], "→ 網站加停損：", add, flush=True)
    Ft = F[F.code.map(mk).eq("TWSE")]; Tt = run(Ft); s, _, _, _ = summarize(Tt, 0.6); out["另報 只用上市股 成本0.6%"] = s
    print("== 只用上市股", json.dumps({k: v for k, v in s.items() if k != "逐年"}, ensure_ascii=False))
    print("新期間也成立：", out["新期間也成立"], "；V1：", out["另報 V1 每月減0050 平均/t"])
    json.dump(out, open(f"{REPO}/research/results/oos_2006.json", "w"), ensure_ascii=False, indent=1, default=str)
