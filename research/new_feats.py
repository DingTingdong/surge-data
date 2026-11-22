"""新特性：季盈餘驚奇（SUE）、波動收縮（VCP）——控制月選綜合分數後，還有沒有額外預測力？（第 8 輪；2026/11/23 00:50 登錄，看結果前寫下）

依據：盈餘公布後漂移（Bernard & Thomas 1989 等，最穩健的異常報酬之一）；Minervini 的波動收縮型態（實戰派常用，學術證據少）。
樣本：月選的換股日（每月 11 日後第一個交易日）、分析範圍（20 日均額 ≥ 1,000 萬）的每月快照，2011–2026（SUE 約 2015 下半年起才算得出）。
目標：之後 20 個交易日報酬在當月的百分位（同 ml_rank.py）。
特性：
  S SUE：單季母公司淨利（季報是累計值，相減得單季），SUE＝（本季 − 去年同季）÷ 過去 8 季「季對去年同季變化」的標準差（至少 6 季）。
    可用日＝法定申報期限隔天：Q1 5/16、Q2 8/15、Q3 11/15、Q4 隔年 4/1；換股日用「可用日 ≤ 換股日」的最新一季。
  V1 收縮比：近 10 日報酬波動 ÷ 近 60 日報酬波動，越小代表越收縮（取負號後算百分位）。
  V2 VCP：收縮比的百分位（越收縮越高）× 「距 250 日高點 15% 以內」（不在高點附近的給 0）。
檢驗（Fama-MacBeth）：每月把目標對「綜合分數百分位、新特性百分位（缺值給 0.5）」做橫斷面迴歸，看新特性係數。
  判斷：係數平均 > 0、Newey-West t（落後 1）≥ 2.9、各段期間（S：2015 下–2019、2020–23、2024–26；V：2011–15、2016–19、2020–23、2024–26）平均都 > 0。
  通過的特性再做一次實用檢驗：把它加成綜合分數的第 13 個特性（等權），前 20（月選持有方式、成本 0.6%）每月減「原本前 20」平均 > 0 且 t ≥ 2，
  而且 2006–2009 年（data/oldest）也不比原本差，才放上網站。
累計檢驗 18 個規則；這一輪 3 個，門檻 t ≥ 2.9。另報：五等分（新特性在綜合分數前 20% 的股票裡）每月報酬。
結果（11/23 01:00）：網站維持現狀。
  S SUE：係數 +0.81（t 1.05）→ 不通過（月營收已先反映）。
  V1 收縮比：+1.93（t 3.86），四段都正 → 通過；但實用檢驗（加成第 13 個特性）新前20 − 原前20 每月 −0.31（t −1.78），四段都負 → 不採用。
  V2 VCP：+3.38（t 4.40），四段都正 → 通過；實用檢驗 −0.25（t −1.57），四段都負 → 不採用。
  解讀：收縮能幫一般股票排序，但月選挑的強勢股本來就波動大，加進去會把最強的股票往後擠（綜合前 20% 內五等分沒有單調關係）。
"""
import glob, json, os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
import top20_goal as TG
from vs0050 import COMP, nw_t, per, PER

def fin_ni():
    names = ["淨利（淨損）歸屬於母公司業主", "淨利（損）歸屬於母公司業主", "本期淨利（淨損）", "本期稅後淨利（淨損）", "本期淨利(淨損)"]
    rows = []
    for fn in sorted(glob.glob(f"{REPO}/data/fin/*.csv.gz")):
        y, s = int(os.path.basename(fn)[:4]), int(os.path.basename(fn)[5])
        d = pd.read_csv(fn, dtype={"code": str}); d["code"] = d.code.str[:4]
        z = d[d.field.isin(names)].copy(); z["pr"] = z.field.map({n: i for i, n in enumerate(names)}); z = z.sort_values("pr").drop_duplicates("code")
        rows.append(pd.DataFrame({"code": z.code.values, "ytd": z.value.values, "y": y, "s": s}))
    Q = pd.concat(rows, ignore_index=True); Q["qi"] = Q.y * 4 + Q.s - 1
    prev = Q[["code", "qi", "ytd", "y"]].rename(columns={"ytd": "ytd_p", "y": "y_p"}); prev["qi"] += 1
    Q = Q.merge(prev, on=["code", "qi"], how="left")
    Q["ni"] = np.where(Q.s == 1, Q.ytd, np.where(Q.y_p == Q.y, Q.ytd - Q.ytd_p, np.nan))
    Q = Q.sort_values(["code", "qi"]); lag4 = Q[["code", "qi", "ni"]].rename(columns={"ni": "ni4"}); lag4["qi"] += 4
    Q = Q.merge(lag4, on=["code", "qi"], how="left"); Q["d4"] = Q.ni - Q.ni4
    Q = Q.sort_values(["code", "qi"]); Q["sd"] = Q.groupby("code").d4.transform(lambda s: s.rolling(8, min_periods=6).std())
    Q["sue"] = Q.d4 / Q.sd
    avail = {1: (5, 16, 0), 2: (8, 15, 0), 3: (11, 15, 0), 4: (4, 1, 1)}
    Q["avail"] = [pd.Timestamp(y + avail[s][2], avail[s][0], avail[s][1]) for y, s in zip(Q.y, Q.s)]
    return Q[["code", "qi", "sue", "avail"]].replace([np.inf, -np.inf], np.nan).dropna()

def fm(F, col, periods):
    co = []
    for d, g in F.groupby("date"):
        g = g[g.tgt.notna()]
        if len(g) < 50: continue
        Xm = np.column_stack([np.ones(len(g)), g.cs_p.values, g[col].values]); b = np.linalg.lstsq(Xm, g.tgt.values, rcond=None)[0]; co.append((d, b[2]))
    s = pd.Series(dict(co)).sort_index(); s.index = pd.DatetimeIndex(s.index); pr = per(s.index)
    return s, {"平均": round(float(s.mean() * 100), 2), "t": round(nw_t(s, 1), 2), "月數": int(len(s)), "各段": {p: round(float(s[pr == p].mean() * 100), 2) for p in periods if (pr == p).any()}}

def practical(F, col, cost=0.6):
    F = F.copy(); F["cs13"] = (F.cs * 12 + F[col]) / 13
    TG.G = {d: g.sort_values("cs", ascending=False).reset_index(drop=True) for d, g in F.groupby("date")}
    Tn = TG.simulate(20, score="cs13"); To = TG.simulate(20)
    d = (Tn.g - cost * Tn.to) - (To.g - cost * To.to); en = TG.evaluate(Tn, cost)
    return d, en, Tn


if __name__ == "__main__":
    F = TG.F.copy(); X = TG.X; days, cols = X["days"], X["cols"]
    F["tgt"] = F.groupby("date").y20.rank(pct=True); F["cs_p"] = F.groupby("date").cs.rank(pct=True)
    # SUE：每個換股日用可用日 ≤ 換股日的最新一季
    Q = fin_ni().sort_values("avail"); F = F.sort_values("date")
    F = pd.merge_asof(F, Q[["code", "avail", "sue"]].sort_values("avail"), left_on="date", right_on="avail", by="code", direction="backward")
    F.loc[(F.date - F.avail).dt.days > 200, "sue"] = np.nan   # 太舊（漏報）就不用
    # 收縮比、VCP
    C = pd.DataFrame(X["C"], index=days, columns=cols); r1 = C.pct_change(fill_method=None)
    vcr = r1.rolling(10, min_periods=8).std() / r1.rolling(60, min_periods=45).std()
    pos = {d: i for i, d in enumerate(days)}; cj = {c: j for j, c in enumerate(cols)}
    F["vcr"] = vcr.values[F.date.map(pos).values, F.code.map(cj).values]; F = F.replace([np.inf, -np.inf], np.nan)
    F["S"] = F.groupby("date").sue.rank(pct=True).fillna(0.5)
    F["V1"] = F.groupby("date").vcr.rank(pct=True, ascending=False).fillna(0.5)
    F["V2"] = np.where(F.d_hi250 >= -0.15, F.V1, 0.0)
    out = {}
    for col, pers, start in (("S", ("2011-15", "2016-19", "2020-23", "2024-26"), "2015-07-01"), ("V1", PER, "2011-01-01"), ("V2", PER, "2011-01-01")):
        Fs = F[F.date >= start]; s, r = fm(Fs, col, pers)
        segs = list(r["各段"].values()); ok = r["平均"] > 0 and r["t"] >= 2.9 and all(v > 0 for v in segs)
        # 另報：綜合分數前 20% 裡，依新特性分五等分的 20 日報酬（每月平均）
        top = Fs[Fs.cs_p >= 0.8].copy(); top["q"] = top.groupby("date")[col].transform(lambda z: pd.qcut(z.rank(method="first"), 5, labels=False))
        qq = top.groupby(["date", "q"]).y20.mean().unstack().mean().round(2).to_dict()
        out[col] = {**r, "通過": bool(ok), "綜合前20%內 五等分 20日報酬%（低→高）": qq, "有值比例%": round(float(Fs[{"S": "sue", "V1": "vcr", "V2": "vcr"}[col]].notna().mean() * 100), 1)}
        print(col, "通過" if ok else "不通過", json.dumps(out[col], ensure_ascii=False), flush=True)
    for col in [c for c in ("V1", "V2") if out[c]["通過"]]:
        d, en, Tn = practical(F, col)
        ok = d.mean() > 0 and nw_t(d, 1) >= 2
        out[col]["實用檢驗 加成第13個特性"] = {"新前20−原前20 每月 平均/t": (round(float(d.mean()), 2), round(nw_t(d, 1), 2)), "各段": {p: round(float(d[per(d.index) == p].mean()), 2) for p in PER},
                                        "新前20 減0050": {k: en[k] for k in ("減0050 全期", "四段", "發現期外", "年化 策略/0050", "最大回落", "贏0050年數")}, "通過（還要 2006–2009 不比原本差）": bool(ok)}
        print(col, "實用檢驗", json.dumps(out[col]["實用檢驗 加成第13個特性"], ensure_ascii=False), flush=True)
    json.dump(out, open(f"{REPO}/research/results/new_feats.json", "w"), ensure_ascii=False, indent=1, default=str)

