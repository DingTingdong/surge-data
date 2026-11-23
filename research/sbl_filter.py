"""借券賣出（第 25 輪；2026/11/23 12:55 登錄，資料還沒抓下來、看結果前寫下，看到結果後不修改）

起因：借券賣出主要是法人放空，研究上借券賣出增加的股票之後表現偏弱；網站穩健版 3 檔想排除「法人正在放空」的股票。
資料：TWSE TWT93U（2005–2026，借券賣出當日餘額），TPEx 借券賣出（抓得到的年份；沒有資料的股票視為缺值、不排除）→ data/sbl/。
特性（換股日收盤已知）：
  sbl_flow20 ＝（借券賣出餘額 − 20 個交易日前餘額）÷ 20 日平均成交股數（成交值 ÷ 收盤價近似）；
  sbl_ratio ＝ 借券賣出餘額 ÷ 20 日平均成交股數（要幾天才回補得完）。
檢驗：
  (1) 全部股票每月 Fama-MacBeth：下個月報酬減 0050（y20 − b20）對兩個特性的同月百分位，前後兩半（2006–2015、2016–2026）各報 t。
  (2) 穩健版 3 檔加過濾：新買進時跳過 sbl_flow20 在當天全部股票前 10% 的（續抱的不受影響），其餘規則同穩健版。
採用條件（全部符合才在網站加過濾）：
  ① 2011–2026 每月減 0050 > 穩健版，且 Newey-West t ≥ 2.9；② 五段中至少四段不比穩健版差；③ 最大回落不比穩健版深；
  ④ sbl_flow20 的 Fama-MacBeth 係數前後兩半都 < 0 且 t ≤ −2。
"""
import glob, json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import robust_select as RS
from vs0050 import nw_t
from calm3 import sim_calm
from wr_confirm import stats

def sbl_feats(path, S):
    X = pd.read_pickle(path); days, cols = X["days"], X["cols"]
    bal = S.pivot(index="date", columns="code", values="sbl_bal").reindex(index=days, columns=cols)
    A = pd.DataFrame(X["A"], index=days, columns=cols).astype(float); C = pd.DataFrame(X["C"], index=days, columns=cols).astype(float)
    vol = (A / C).rolling(20, min_periods=15).mean()
    flow = (bal - bal.shift(20)) / vol; ratio = bal / vol
    return X, flow, ratio

def attach(D, X, flow, ratio):
    pos = {d: i for i, d in enumerate(X["days"])}; cj = {c: j for j, c in enumerate(X["cols"])}; F = D["F"]
    ti = F.date.map(pos).values; ji = F.code.map(cj).values
    F["flow"] = flow.values[ti, ji]; F["ratio"] = ratio.values[ti, ji]
    F["p_flow"] = F.groupby("date").flow.rank(pct=True); F["p_ratio"] = F.groupby("date").ratio.rank(pct=True); return D

if __name__ == "__main__":
    S = pd.concat([pd.read_csv(f, dtype={"code": str}) for f in sorted(glob.glob(f"{REPO}/data/sbl/*.csv.gz"))], ignore_index=True)
    S["date"] = pd.to_datetime(S.date); S = S.drop_duplicates(["date", "code"], keep="last")
    out = {"每年有資料的股數（中位）": S.groupby(S.date.dt.year).apply(lambda z: int(z.groupby("date").code.nunique().median())).to_dict()}
    Dn = RS.load("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01"); Do = RS.load("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01")
    for D, path in ((Dn, "/tmp/model_v2/vs0050_extra.pkl"), (Do, "/tmp/model_v2/vs0050_oldestonly_extra.pkl")):
        X, fl, ra = sbl_feats(path, S); attach(D, X, fl, ra); del X, fl, ra
    F = pd.concat([Do["F"], Dn["F"]], ignore_index=True); F["x"] = F.y20 - F.b20
    for k in ("flow", "ratio"):
        co = []
        for d, g in F.dropna(subset=[k, "x"]).groupby("date"):
            if len(g) > 50: zz = (g["p_" + k] - g["p_" + k].mean()) / g["p_" + k].std(); co.append((d, float(np.polyfit(zz, g.x, 1)[0])))
        co = pd.DataFrame(co, columns=["date", "b"]).set_index("date").b; h1, h2 = co[co.index < "2016-01-01"], co[co.index >= "2016-01-01"]
        out[f"FM {k} 2006–2015"] = (round(float(h1.mean()), 3), round(nw_t(h1.reset_index(drop=True), 1), 2)); out[f"FM {k} 2016–2026"] = (round(float(h2.mean()), 3), round(nw_t(h2.reset_index(drop=True), 1), 2))
        z = F.dropna(subset=[k, "x"]); out[f"{k} 五等分 下月減0050（低→高）"] = z.groupby(pd.qcut(z["p_" + k], 5, labels=False)).x.mean().round(2).to_dict()
        print(k, out[f"FM {k} 2006–2015"], out[f"FM {k} 2016–2026"], out[f"{k} 五等分 下月減0050（低→高）"], flush=True)
    def filt(D):
        E = dict(D); G = D["F"].copy(); G["rk"] = G.groupby("date").cs.rank(ascending=False); b = (G.rk <= 20) & (G.p_flow > 0.9); G.loc[b, "vol20"] = np.inf; E["F"] = G; return E, int(b.sum())
    (Eo, bo), (En, bn) = filt(Do), filt(Dn)
    s0 = stats(pd.concat([sim_calm(Do, 20), sim_calm(Dn, 20)])); s1 = stats(pd.concat([sim_calm(Eo, 20), sim_calm(En, 20)]))
    out["穩健版"] = s0; out["穩健版＋排除借券增加前10%"] = s1; out["擋掉次數"] = bo + bn; print("穩健版", s0, "\n加過濾", s1, "擋掉", bo + bn)
    c = {"①": s1["2011–2026"][0] > s0["2011–2026"][0] and s1["2011–2026"][1] >= 2.9, "②": sum(s1[s] >= s0[s] for s in RS.SEGS) >= 4, "③": s1["最大回落"] >= s0["最大回落"],
         "④": all(out[f"FM flow {h}"][0] < 0 and out[f"FM flow {h}"][1] <= -2 for h in ("2006–2015", "2016–2026"))}
    out["各項"] = c; out["通過"] = all(c.values()); print(c, "通過" if all(c.values()) else "不通過")
    json.dump(out, open(f"{REPO}/research/results/sbl_filter.json", "w"), ensure_ascii=False, indent=1, default=str)
