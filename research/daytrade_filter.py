"""現股當沖太熱的股票（第 27 輪；2026/11/23 13:45 登錄，資料還沒抓下來、看結果前寫下，看到結果後不修改）

起因：當沖比重高代表短線投機資金集中，研究上這類股票之後報酬偏弱；想知道網站穩健版 3 檔要不要排除當沖太熱的股票。
資料：TWSE TWTB4U 個股當沖量值（2014-01 起；2014 年只有約 150 檔可當沖、2017 年降稅後才普及）→ data/daytrade/。上櫃沒有個股當沖量值。
特性：dtr20 ＝ 近 20 個交易日當沖買進成交金額合計 ÷ 同期成交值合計（換股日收盤已知）；沒有當沖資料的股票為缺值。
檢驗期間：2017-06～2026-11（只用上市股票算特性；上櫃股票特性缺值）。
  (1) 上市股票每月 Fama-MacBeth：下個月報酬減 0050 對 dtr20 同月百分位；前後兩半（2017-06～2021-12、2022-01～2026-11）各報 t。
  (2) 穩健版 3 檔加過濾：新買進時跳過 dtr20 在當天上市股票前 10% 的（缺值不排除、續抱不受影響），其餘同穩健版。
採用條件（全部符合才在網站加過濾）：
  ① 2017-06～2026-11 每月減 0050 平均 > 穩健版；② 2022–2026 也 > 穩健版；③ 最大回落不比穩健版深；
  ④ dtr20 的 Fama-MacBeth 係數前後兩半都 < 0 且 t ≤ −2。
另報：過濾實際擋掉幾次。
結果（11/23 17:10）：不通過 → 網站不加。方向和假設相反：當沖比重越高，下個月反而略好（五等分減 0050 −0.97／−0.68／−0.58／−0.30／−0.10），
  Fama-MacBeth 前後兩半 +0.36（t 1.35）、+0.27（t 1.04），都不顯著。穩健版排除當沖前 10%：2017-06～2026 每月 +2.40 vs +2.50，2022–2026 +2.20 vs +2.37，回落一樣 −23%。
"""
import glob, json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import robust_select as RS
from vs0050 import nw_t
from calm3 import sim_calm
from wr_confirm import stats
START, MID = "2017-06-01", "2022-01-01"

def dtr_matrix(X):
    days, cols = X["days"], X["cols"]
    D = pd.concat([pd.read_csv(f, dtype={"code": str}) for f in sorted(glob.glob(f"{REPO}/data/daytrade/*.csv.gz"))], ignore_index=True)
    D["date"] = pd.to_datetime(D.date); D = D.drop_duplicates(["date", "code"], keep="last")
    B = D.pivot(index="date", columns="code", values="dt_buy").reindex(index=days, columns=cols)
    A = pd.DataFrame(X["A"], index=days, columns=cols).astype(float)
    tw = pd.Series(cols.isin(D.code.unique()), index=cols)  # 有當沖資料的（上市）
    num = B.fillna(0).where(A.notna()).rolling(20, min_periods=15).sum(); den = A.rolling(20, min_periods=15).sum()
    R = (num / den).where(B.notna().rolling(20, min_periods=1).max().astype(bool))  # 期間內完全沒當沖資料（不是上市或不可當沖）→ 缺值
    return R

if __name__ == "__main__":
    X = pd.read_pickle("/tmp/model_v2/vs0050_extra.pkl"); R = dtr_matrix(X); pos = {d: i for i, d in enumerate(X["days"])}; cj = {c: j for j, c in enumerate(X["cols"])}
    D = RS.load("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01"); F = D["F"]
    F["dtr"] = [R.values[pos[d], cj[c]] for d, c in zip(F.date, F.code)]
    F["p_dtr"] = F.groupby("date").dtr.rank(pct=True)
    out = {}
    # (1) Fama-MacBeth：上市股票（有 dtr 的），下個月報酬減 0050 對 dtr 百分位
    Z = F[(F.date >= START) & F.dtr.notna()].copy(); Z["x"] = Z.y20 - Z.b20; co = []
    for d, g in Z.groupby("date"):
        g = g.dropna(subset=["x"])
        if len(g) > 50: zz = (g.p_dtr - g.p_dtr.mean()) / g.p_dtr.std(); co.append((d, float(np.polyfit(zz, g.x, 1)[0])))
    co = pd.DataFrame(co, columns=["date", "b"]).set_index("date").b
    h1, h2 = co[co.index < MID], co[co.index >= MID]
    out["FM 前半 2017-06～2021"] = (round(float(h1.mean()), 3), round(nw_t(h1.reset_index(drop=True), 1), 2)); out["FM 後半 2022～2026"] = (round(float(h2.mean()), 3), round(nw_t(h2.reset_index(drop=True), 1), 2))
    q = Z.assign(qq=pd.qcut(Z.p_dtr, 5, labels=False)).groupby("qq").x.mean().round(2).to_dict(); out["dtr 五等分 下月減0050（低→高）"] = q
    print("FM", out["FM 前半 2017-06～2021"], out["FM 後半 2022～2026"], "五等分", q, flush=True)
    # (2) 穩健版加過濾：新買進跳過 dtr 在當天上市股票前 10%
    F["p_dtr_tw"] = F.groupby("date").dtr.rank(pct=True)
    E = dict(D); G = F.copy(); G["rk"] = G.groupby("date").cs.rank(ascending=False); blocked = (G.rk <= 20) & (G.p_dtr_tw > 0.9) & (G.date >= START)
    G.loc[blocked, "vol20"] = np.inf; E["F"] = G
    T0, T1 = sim_calm(D, 20), sim_calm(E, 20); T0, T1 = T0[T0.index >= START], T1[T1.index >= START]
    def st(T):
        x = T.r - T.b; w = (1 + T.r / 100).cumprod()
        return {"2017-06～2026": (round(float(x.mean()), 2), round(nw_t(x, 1), 2)), "2022～2026": round(float(x[x.index >= MID].mean()), 2), "最大回落": round(float((w / w.cummax() - 1).min() * 100))}
    s0, s1 = st(T0), st(T1); out["穩健版"] = s0; out["穩健版＋排除當沖前10%"] = s1
    out["擋掉的次數（前20名中被標記）"] = int(blocked.sum())
    print("穩健版", s0, "\n加過濾", s1, "擋掉", out["擋掉的次數（前20名中被標記）"], flush=True)
    c = {"①": s1["2017-06～2026"][0] > s0["2017-06～2026"][0], "②": s1["2022～2026"] > s0["2022～2026"], "③": s1["最大回落"] >= s0["最大回落"],
         "④": out["FM 前半 2017-06～2021"][0] < 0 and out["FM 前半 2017-06～2021"][1] <= -2 and out["FM 後半 2022～2026"][0] < 0 and out["FM 後半 2022～2026"][1] <= -2}
    out["各項"] = c; out["通過"] = all(c.values()); print(c, "通過" if all(c.values()) else "不通過")
    json.dump(out, open(f"{REPO}/research/results/daytrade_filter.json", "w"), ensure_ascii=False, indent=1, default=str)
