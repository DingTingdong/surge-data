"""季報毛利率／營益率改善、月份效應（第 41、42 輪；2026/11/24 00:05 登錄，看結果前寫下，看到結果後不修改）

第 41 輪 季報品質：data/fin（2013Q1 起，累計數 → 換成單季），最近一季「毛利率、營益率」與去年同季比。
  資料可用日用法定最晚期限（Q1 5/15、Q2 8/14、Q3 11/14、Q4 隔年 3/31）。
  Q1 規則：穩健版，新買進時跳過「最近一季毛利率和營益率都比去年同季差」的股票（沒資料不排除；續抱不受影響）。
  判斷（2015–2026；季報資料 2014 起才有年比）：① 每月減 0050 > 穩健版同期，且 t ≥ 2.9；② 2016–19、2020–23、2024–26 三段中至少兩段比較好，
    且沒有一段差超過 0.2；③ 最大回落不比穩健版深。另報：前 20 名中被排除的比例、被排除與沒排除的下個月減 0050。
第 42 輪 月份效應：穩健版每月減 0050 依換股月份分組。前半（2006–2015）找出平均為負的月份，看後半（2016–2026）是否也為負。
  只有「兩半都為負、且這些月份改抱 0050 時兩半都變好」才採用（那幾個月換股日改買 0050）。另報：兩半各月平均的相關係數。
"""
import glob, json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import robust_select as RS
from vs0050 import nw_t
from calm3 import sim_calm
from wr_confirm import stats
GM = ["營業毛利（毛損）淨額", "營業毛利（毛損）"]; OP = ["營業利益（損失）"]; REV = ["營業收入"]

def quarterly():
    rows = []
    for fn in sorted(glob.glob(f"{REPO}/data/fin/*.csv.gz")):
        try: d = pd.read_csv(fn, dtype={"code": str})
        except Exception: continue
        y, s = int(os.path.basename(fn)[:4]), int(os.path.basename(fn)[5]); d = d[d.stmt == "is"]; d["code"] = d.code.str[:4]
        p = d.pivot_table(index="code", columns="field", values="value", aggfunc="first")
        g = lambda names: p[[n for n in names if n in p.columns]].bfill(axis=1).iloc[:, 0] if any(n in p.columns for n in names) else pd.Series(np.nan, index=p.index)
        rows.append(pd.DataFrame({"code": p.index, "y": y, "s": s, "rev": g(REV).values, "gm": g(GM).values, "op": g(OP).values}))
    Q = pd.concat(rows, ignore_index=True).sort_values(["code", "y", "s"])
    for k in ("rev", "gm", "op"):  # 累計 → 單季
        prev = Q.groupby(["code", "y"])[k].shift(1); Q[k + "_q"] = np.where(Q.s == 1, Q[k], Q[k] - prev)
    Q["gmr"] = Q.gm_q / Q.rev_q; Q["opr"] = Q.op_q / Q.rev_q; Q.loc[Q.rev_q <= 0, ["gmr", "opr"]] = np.nan
    Q["qi"] = Q.y * 4 + Q.s; L = Q[["code", "qi", "gmr", "opr"]].rename(columns={"gmr": "gmr4", "opr": "opr4"}); L["qi"] += 4
    Q = Q.merge(L, on=["code", "qi"], how="left"); Q["worse"] = (Q.gmr < Q.gmr4) & (Q.opr < Q.opr4); Q["ok"] = Q.gmr.notna() & Q.gmr4.notna() & Q.opr.notna() & Q.opr4.notna()
    dl = {1: (5, 15), 2: (8, 14), 3: (11, 14), 4: (3, 31)}
    Q["avail"] = [pd.Timestamp(y + (s == 4), *dl[s]) for y, s in zip(Q.y, Q.s)]
    return Q[Q.ok][["code", "avail", "worse"]]

if __name__ == "__main__":
    out = {}
    Dn = RS.load("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01"); Do = RS.load("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01")
    Q = quarterly(); F = Dn["F"]
    F = pd.merge_asof(F.sort_values("date"), Q.sort_values("avail"), left_on="date", right_on="avail", by="code", direction="backward")
    F.loc[(F.date - F.avail).dt.days > 200, "worse"] = np.nan; F["worse"] = F.worse.fillna(False).astype(bool)
    base = dict(Dn); base["F"] = F.copy(); E = dict(Dn); G = F.copy(); G["rk"] = G.groupby("date").cs.rank(ascending=False); G.loc[(G.rk <= 20) & G.worse, "vol20"] = np.inf; E["F"] = G
    T0, T1 = sim_calm(base, 20), sim_calm(E, 20); T0, T1 = T0[T0.index >= "2015-01-01"], T1[T1.index >= "2015-01-01"]
    def st(T):
        x = T.r - T.b; sg = RS.seg(x.index); w = (1 + T.r / 100).cumprod()
        return {"2015–2026": (round(float(x.mean()), 2), round(nw_t(x, 1), 2)), **{s: round(float(x[sg == s].mean()), 2) for s in ("2016-19", "2020-23", "2024-26")}, "最大回落": round(float((w / w.cummax() - 1).min() * 100))}
    s0, s1 = st(T0), st(T1); out["第41輪 穩健版（2015起）"] = s0; out["第41輪 排除毛利營益都變差"] = s1
    top = F[F.date >= "2015-01-01"].sort_values("cs", ascending=False).groupby("date").head(20); out["前20名被排除%"] = round(float(top.worse.mean() * 100), 1)
    out["前20名 被排除/沒排除 下月減0050"] = (round(float((top.y20 - top.b20)[top.worse].mean()), 2), round(float((top.y20 - top.b20)[~top.worse].mean()), 2))
    d = [s1[k] - s0[k] for k in ("2016-19", "2020-23", "2024-26")]
    out["第41輪 通過"] = bool(s1["2015–2026"][0] > s0["2015–2026"][0] and s1["2015–2026"][1] >= 2.9 and sum(v > 0 for v in d) >= 2 and min(d) >= -0.2 and s1["最大回落"] >= s0["最大回落"])
    # 第 42 輪 月份效應
    T = pd.concat([sim_calm(Do, 20), sim_calm(Dn, 20)]); x = T.r - T.b; mo = x.index.month
    h1, h2 = x[x.index < "2016-01-01"], x[x.index >= "2016-01-01"]
    m1, m2 = h1.groupby(h1.index.month).mean(), h2.groupby(h2.index.month).mean()
    out["第42輪 各月 前半/後半"] = {int(m): (round(float(m1.get(m, np.nan)), 2), round(float(m2.get(m, np.nan)), 2)) for m in range(1, 13)}
    out["第42輪 兩半相關"] = round(float(np.corrcoef(m1.reindex(range(1, 13)), m2.reindex(range(1, 13)))[0, 1]), 2)
    bad = [m for m in range(1, 13) if m1.get(m, 0) < 0 and m2.get(m, 0) < 0]; out["第42輪 兩半都為負的月份"] = bad
    out["第42輪 通過"] = bool(bad) and all(x[(x.index.month.isin(bad)) & (x.index < "2016-01-01")].mean() < 0 for _ in [0]) and all(x[(x.index.month.isin(bad)) & (x.index >= "2016-01-01")].mean() < 0 for _ in [0])
    for k, v in out.items(): print(k, json.dumps(v, ensure_ascii=False, default=str), flush=True)
    json.dump(out, open(f"{REPO}/research/results/quality_season.json", "w"), ensure_ascii=False, indent=1, default=str)
