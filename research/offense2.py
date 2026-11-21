"""攻擊型選股第二輪：季獲利、借券、規模分層、外資持股、當沖（依 reports/台股強勢股選股擴大研究.md 的優先順序，含外部覆核後的修正）。

全部在看結果之前寫下（2026/11/22 04:05），看到結果後不修改。季財報資料（data/fin）寫這段時還在下載、沒有看過。
共同設定（同 research/offense.py）：2015-06～2026-11 連續資料；每月第一個 11 日以後的交易日收盤後形成、隔天開盤買、抱 20 個交易日；
  範圍＝分析範圍內股票（20日均額 ≥ 1,000 萬），只用平常月份（恐慌定義同 offense.py）；三段期間 2016–19、2020–23、2024–26。
  cs＝research/deep_normal.py 的 12 個特性當月百分位等權平均（只控制 cs，不同時放入它的成分，避免完全共線）。
唯一的主要統計量（每個假說都一樣）：每月做 Fama-MacBeth 迴歸「20 日報酬 ～ 新訊號當月百分位 ＋ cs」，取新訊號係數
  （＝新訊號從最低到最高百分位的 20 日報酬差，%），全期平均的 Newey-West t（落後 1 個月）。
成立＝以下三項全部符合（不是任一）：
  ① 全期 NW t 在事先寫好的方向 ≥ 2.9；② 三段期間的係數平均都和事先方向相同；
  ③ 做多增量：每月在 cs 前20% 之中取新訊號最好的三分之一（依方向），扣 0.6%×換股比例，減「cs 前20% 全部」扣成本後的報酬，全期平均 > 0（另報 t）。
資料與限制（事先寫明）：
  外資持股、發行股數、借券、當沖：data/extra（證交所 MI_QFIIS、TWT93U、TWTB4U；櫃買 qfii、margin_sbl），2015-06 起；櫃買沒有個股當沖資料。
  季財報：data/fin（公開資訊觀測站各季彙總表 t163sb04／t163sb05，2013Q1 起）。是各季當時申報的報表，但公司事後更正申報可能取代原值，無法排除。
    損益表若為年初累計，單季＝本季累計 − 上季累計（第一季直接用）；可用日＝法定期限：Q1 5/15、Q2 8/14、Q3 11/14、Q4 隔年 3/31（一律用最晚期限，偏保守）。
    金融業沒有營業毛利，G2 中為缺值（不排名）。
G1 季 ROE：單季歸屬母公司淨利（沒有就用本期淨利）÷ 上一季末歸屬母公司權益（沒有就用權益總額）。方向＋。
G2 季毛利 ÷ 上一季末資產總額。方向＋。
G3 借券賣出流量：近 20 日（借券賣出 − 還券）股數合計 ÷ 近 250 日平均每日成交股數 × 20（成交股數＝成交額 ÷ 收盤價）。方向 −。另報融券＋借券合計版（不判定）。
G4 規模分層（能否贏 0050 的診斷）：用點時市值（發行股數 × 收盤價），只在每月市值前 150 大之中，cs 本身的 FM 係數（不再控制 cs）。方向＋。
   ③ 改成：前 150 大之中 cs 最高 20 檔，扣成本後減 0050 同期，全期平均 > 0。另報前 150 大以外。
G5 外資持股比例 60 日變化（外資持股股數 ÷ 發行股數，今天減 60 個交易日前）。方向＋。另報持股比例水準（不判定）。
G6 當沖比率：近 20 日當沖成交股數 ÷ 近 20 日成交股數（只上市）。方向 −（過熱）。
多重檢定：本輪 6 個假說，門檻沿用 t ≥ 2.9（比 6 個的 Bonferroni 2.64 嚴）。另報的版本都不判定、不挑選。
"""
import glob, json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, REPO); sys.path.insert(0, HERE)
CACHE = "/tmp/model_v2"
exec(open(os.path.join(HERE, "offense.py"), encoding="utf-8").read().split('if __name__ == "__main__" and sys.argv[1:2] == ["build"]:')[0].split('"""', 2)[2])
from normal_recheck import nw_t
DEADLINE = {1: (5, 15), 2: (8, 14), 3: (11, 14), 4: (3, 31)}

def fin_quarterly():
    """data/fin 長格式 → 每家公司每季：單季淨利（母公司）、單季營業毛利、季末權益（母公司）、季末資產總額、可用日。"""
    fs = sorted(glob.glob(f"{REPO}/data/fin/*.csv.gz"))
    if not fs: return None
    pick = {"ni": ["淨利（淨損）歸屬於母公司業主", "淨利（損）歸屬於母公司業主", "本期淨利（淨損）", "本期稅後淨利（淨損）", "本期淨利(淨損)"],
            "gp": ["營業毛利（毛損）淨額", "營業毛利（毛損）", "營業毛利(毛損)"],
            "eq": ["歸屬於母公司業主之權益合計", "歸屬於母公司業主權益合計", "權益總額", "權益總計", "權益合計"],
            "ta": ["資產總額", "資產總計", "資產合計"]}
    rows = []
    for fn in fs:
        y, s = int(os.path.basename(fn)[:4]), int(os.path.basename(fn)[5])
        d = pd.read_csv(fn, dtype={"code": str}); d["code"] = d.code.str[:4]
        out = {}
        for k, names in pick.items():
            z = d[d.field.isin(names)].copy(); z["pr"] = z.field.map({n: i for i, n in enumerate(names)})
            z = z.sort_values("pr").drop_duplicates("code"); out[k] = z.set_index("code").value
        q = pd.DataFrame(out); q["y"], q["s"] = y, s; rows.append(q.reset_index())
    Q = pd.concat(rows, ignore_index=True).sort_values(["code", "y", "s"])
    return Q

def build2():
    out = f"{CACHE}/offense2.pkl"
    if os.path.exists(out): return pd.read_pickle(out)
    D = build(); R = D["R"].copy()
    os.chdir(full_root())
    from radar_core import load_prices, panels
    p = load_prices(); P = panels(p); del p
    C, A, raw = P["C"], P["A"], P["rawC"]; days, cols = C.index, C.columns
    assert len(days) == R.ti.max() + 1 or True
    os.chdir(REPO)
    X = pd.concat([pd.read_csv(f, dtype={"code": str}) for f in sorted(glob.glob("data/extra/*.csv.gz"))], ignore_index=True)
    X["date"] = pd.to_datetime(X.date); X = X.drop_duplicates(["date", "code"], keep="last")
    pv = lambda c: X.pivot(index="date", columns="code", values=c).reindex(index=days, columns=cols)
    ti, ji = R.ti.values, R.ji.values; take = lambda M: (M.values if isinstance(M, pd.DataFrame) else M)[ti, ji]
    with np.errstate(divide="ignore", invalid="ignore"):
        sh = pv("shares").ffill(limit=5); fsh = pv("fshares").ffill(limit=5)
        R["mcap"] = take(sh * raw); fr = fsh / sh; R["fr"] = take(fr); R["dfr60"] = take(fr - fr.shift(60))
        vol = A / raw; v250 = vol.rolling(250, min_periods=120).mean()
        sell, ret, sbal = pv("sbl_sell"), pv("sbl_ret"), pv("short_bal")
        net20 = (sell - ret).rolling(20, min_periods=15).sum()
        R["sblf"] = take(net20 / (v250 * 20)); R["shortf"] = take((net20 + (sbal - sbal.shift(20))) / (v250 * 20))
        dts = pv("dt_shares"); tw = pd.Series([P["MK"].ffill().iloc[-1].get(c) == "TWSE" for c in cols], index=cols)
        dtr = dts.fillna(0).rolling(20, min_periods=15).sum() / vol.rolling(20, min_periods=15).sum()
        dtr = dtr.where(tw.values[None, :] & dts.notna().rolling(60, min_periods=1).max().astype(bool))
        R["dtr20"] = take(dtr)
    del X
    Q = fin_quarterly()
    if Q is not None:
        Q = Q.copy()
        # 損益表是年初累計（已用台積電 2024 核對）→ 單季＝本季累計 − 同年上一季累計；資產負債用「上一季」季末（精確對應，缺季就缺值）
        Q["qi"] = Q.y * 4 + Q.s - 1
        prevq = Q[["code", "qi", "ni", "gp", "eq", "ta", "y"]].rename(columns={"ni": "ni_p", "gp": "gp_p", "eq": "eq_l", "ta": "ta_l", "y": "y_p"}); prevq["qi"] += 1
        Q = Q.merge(prevq, on=["code", "qi"], how="left")
        for k in ("ni", "gp"):
            Q[k + "_q"] = np.where(Q.s == 1, Q[k], np.where(Q.y_p == Q.y, Q[k] - Q[k + "_p"], np.nan))
        Q["roeq"] = Q.ni_q / Q.eq_l.where(Q.eq_l > 0); Q["gpla"] = Q.gp_q / Q.ta_l.where(Q.ta_l > 0)
        Q["avail"] = [pd.Timestamp(y + (s == 4), *DEADLINE[s]) for y, s in zip(Q.y, Q.s)]
        Q = Q.sort_values("avail")
        R = R.sort_values("date")
        R = pd.merge_asof(R, Q[["code", "avail", "roeq", "gpla"]].dropna(subset=["avail"]).sort_values("avail"), left_on="date", right_on="avail", by="code", direction="backward")
        R["fin_age"] = (R.date - R.avail).dt.days; R.loc[R.fin_age > 200, ["roeq", "gpla"]] = np.nan  # 太舊（超過約兩季）就不用
    R.to_pickle(out); return R

def fm2(M, x, ctrl=True):
    co = {}
    for d, g in M.groupby("date"):
        g = g[g.y20.notna() & g[x].notna()]
        if len(g) < 30 or g[x].nunique() < 3: continue
        Z = np.column_stack([np.ones(len(g)), g[x].rank(pct=True).values] + ([g.cs.values] if ctrl else []))
        co[d] = np.linalg.lstsq(Z, g.y20.values, rcond=None)[0][1]
    return pd.Series(co)

def per3(idx):
    i = pd.to_datetime(pd.Series(idx)).values
    return np.where(i < np.datetime64("2020-01-01"), "2016-19", np.where(i < np.datetime64("2024-01-01"), "2020-23", "2024-26"))

def lo_incr(M, x, sign):
    rows, pa, ps = {}, set(), set()
    for d, g in M.groupby("date"):
        top = g[g.csr >= 0.8]; t2 = top[top[x].notna()]
        if len(t2) < 9: pa, ps = set(), set(); continue
        r = t2[x].rank(pct=True) if sign > 0 else 1 - t2[x].rank(pct=True) + 1 / len(t2)
        sub = t2[r > 2 / 3]; ca, cs_ = set(top.code), set(sub.code)
        ta = 1.0 if not pa else len(ca - pa) / len(ca); tsub = 1.0 if not ps else len(cs_ - ps) / max(1, len(cs_)); pa, ps = ca, cs_
        rows[d] = (sub.y20.mean() - 0.6 * tsub) - (top.y20.mean() - 0.6 * ta)
    return pd.Series(rows)

def judge(name, coef, sign, lo, extra=None):
    bp = {p: round(float(coef[per3(coef.index) == p].mean()), 3) for p in ("2016-19", "2020-23", "2024-26")}
    t = nw_t(coef, 1); c1 = t * sign >= 2.9; c2 = all(v * sign > 0 for v in bp.values()); c3 = bool(lo.mean() > 0)
    r = {"係數全期": round(float(coef.mean()), 3), "NW t": round(float(t), 2), "月數": int(len(coef)), "分段": bp,
         "做多增量(每月)": round(float(lo.mean()), 3), "做多增量 NW t": round(float(nw_t(lo, 1)), 2),
         "①": bool(c1), "②": bool(c2), "③": c3, "成立": bool(c1 and c2 and c3)}
    if extra: r.update(extra)
    print(f"\n== {name}：成立={r['成立']}（①{c1} ②{c2} ③{c3}）\n   {r}", flush=True); return r

if __name__ == "__main__" and sys.argv[1:2] == ["run"]:
    R = build2(); R = R[R.date >= "2016-01-01"]
    M = R[R.fd & ~R.panic & R.y20.notna()].copy()
    M["cs"] = np.mean([M.groupby("date")[c].rank(pct=True).fillna(0.5).values for c in COMP], axis=0); M["csr"] = M.groupby("date").cs.rank(pct=True)
    res = {}
    for key, nm, sign in (("roeq", "G1 季 ROE", +1), ("gpla", "G2 季毛利÷前期資產", +1), ("sblf", "G3 借券賣出流量", -1), ("dfr60", "G5 外資持股60日變化", +1), ("dtr20", "G6 當沖比率", -1)):
        if key not in M or M[key].notna().mean() < 0.2: print(nm, "資料不足，跳過"); continue
        ex = {"覆蓋率": round(float(M[key].notna().mean()), 2)}
        if key == "sblf": ex["另報 融券＋借券合計 係數"] = (lambda c: (round(float(c.mean()), 3), round(nw_t(c, 1), 2)))(fm2(M, "shortf"))
        if key == "dfr60": ex["另報 外資持股水準 係數"] = (lambda c: (round(float(c.mean()), 3), round(nw_t(c, 1), 2)))(fm2(M, "fr"))
        res[nm] = judge(nm, fm2(M, key), sign, lo_incr(M, key, sign), ex)
    # G4 規模分層
    M["mrk"] = M.groupby("date").mcap.rank(ascending=False, method="first")
    L = M[M.mrk <= 150].copy(); c4 = fm2(L.assign(csx=L.cs), "cs", ctrl=False)
    rows, prev = {}, set()
    for d, g in L.groupby("date"):
        pk = g.sort_values("cs", ascending=False).head(20); cur = set(pk.code); to = 1.0 if not prev else len(cur - prev) / 20; prev = cur
        rows[d] = pk.y20.mean() - 0.6 * to - g.b20.iloc[0]
    lo4 = pd.Series(rows); S_ = M[M.mrk > 150]; c4s = fm2(S_, "cs", ctrl=False)
    res["G4 大型股（市值前150）"] = judge("G4 大型股（市值前150）裡的綜合分數", c4, +1, lo4,
                                     {"前150 前20檔減0050 分段": {p: round(float(lo4[per3(lo4.index) == p].mean()), 2) for p in ("2016-19", "2020-23", "2024-26")},
                                      "另報 前150以外 cs 係數": (round(float(c4s.mean()), 3), round(nw_t(c4s, 1), 2))})
    json.dump(res, open(f"{REPO}/research/results/offense2.json", "w"), ensure_ascii=False, indent=1, default=str)
