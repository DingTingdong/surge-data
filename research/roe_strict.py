"""季 ROE 口徑檢查（2026/11/22，外部覆核後；探索，不改 offense2.py 的事先登錄判斷）。
① 分子、分母有多少是用備用欄位（本期淨利／權益總額）而非「歸屬母公司」；② 只用兩者都是歸屬母公司的嚴格口徑，重算 G1。
資料可得日仍為法定最晚期限（Q4 隔年 3/31）；前季末權益 ≤ 0 不算。"""
import os, sys, glob; sys.argv = ["x"]; __file__ = os.path.abspath(__file__)
exec(open(os.path.join(os.path.dirname(__file__), "offense2.py"), encoding="utf-8").read().split('if __name__ == "__main__" and sys.argv[1:2] == ["run"]:')[0].split('"""', 2)[2])
rows = []
for fn in sorted(glob.glob(f"{REPO}/data/fin/*.csv.gz")):
    y, s = int(os.path.basename(fn)[:4]), int(os.path.basename(fn)[5]); d = pd.read_csv(fn, dtype={"code": str}); d["code"] = d.code.str[:4]
    has = lambda names: set(d[d.field.isin(names)].code)
    ni_p = has(["淨利（淨損）歸屬於母公司業主", "淨利（損）歸屬於母公司業主"]); eq_p = has(["歸屬於母公司業主之權益合計", "歸屬於母公司業主權益合計"])
    allc = set(d.code); rows.append((y, s, len(allc), len(ni_p) / len(allc), len(eq_p) / len(allc), len(ni_p & eq_p) / len(allc)))
T = pd.DataFrame(rows, columns=["y", "s", "家數", "淨利為母公司%", "權益為母公司%", "兩者都是%"])
print("口徑覆蓋（各季平均）：", T[["淨利為母公司%", "權益為母公司%", "兩者都是%"]].mean().round(3).to_dict(), "；最低一季", T[["兩者都是%"]].min().round(3).to_dict())
# 嚴格口徑：分子分母都要是歸屬母公司
strict = set()
for fn in sorted(glob.glob(f"{REPO}/data/fin/*.csv.gz")):
    y, s = int(os.path.basename(fn)[:4]), int(os.path.basename(fn)[5]); d = pd.read_csv(fn, dtype={"code": str}); d["code"] = d.code.str[:4]
    a = set(d[d.field.isin(["淨利（淨損）歸屬於母公司業主", "淨利（損）歸屬於母公司業主"])].code) & set(d[d.field.isin(["歸屬於母公司業主之權益合計", "歸屬於母公司業主權益合計"])].code)
    strict |= {(c, y, s) for c in a}
R = build2(); R = R[R.date >= "2016-01-01"]
Q = fin_quarterly(); Q["qi"] = Q.y * 4 + Q.s - 1
Q["ok"] = [(c, y, s) in strict for c, y, s in zip(Q.code, Q.y, Q.s)]
pq = Q[["code", "qi", "ni", "eq", "y", "ok"]].rename(columns={"ni": "ni_p", "eq": "eq_l", "y": "y_p", "ok": "ok_p"}); pq["qi"] += 1
Q = Q.merge(pq, on=["code", "qi"], how="left")
Q["ni_q"] = np.where(Q.s == 1, Q.ni, np.where(Q.y_p == Q.y, Q.ni - Q.ni_p, np.nan))
Q["roe_s"] = np.where(Q.ok & Q.ok_p.fillna(False).astype(bool) & (Q.eq_l > 0), Q.ni_q / Q.eq_l, np.nan)
Q["avail"] = [pd.Timestamp(y + (s == 4), *DEADLINE[s]) for y, s in zip(Q.y, Q.s)]
R = pd.merge_asof(R.sort_values("date"), Q[["code", "avail", "roe_s"]].sort_values("avail").rename(columns={"avail": "av2"}), left_on="date", right_on="av2", by="code", direction="backward")
R.loc[(R.date - R.av2).dt.days > 200, "roe_s"] = np.nan
M = R[R.fd & ~R.panic & R.y20.notna()].copy()
M["cs"] = np.mean([M.groupby("date")[c].rank(pct=True).fillna(0.5).values for c in COMP], axis=0); M["csr"] = M.groupby("date").cs.rank(pct=True)
for lab, ctrl in (("控制綜合分數", True), ("不控制", False)):
    c = fm2(M, "roe_s", ctrl=ctrl); print(f"嚴格口徑季 ROE（{lab}）：係數", round(c.mean(), 3), "NW t", round(nw_t(c, 1), 2), {p: round(float(c[per3(c.index) == p].mean()), 2) for p in ("2016-19", "2020-23", "2024-26")}, "覆蓋率", round(M.roe_s.notna().mean(), 2))
lo = lo_incr(M, "roe_s", +1); print("做多增量（每月）", round(lo.mean(), 3), "NW t", round(nw_t(lo, 1), 2))
