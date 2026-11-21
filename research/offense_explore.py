"""探索（看過 offense.py 結果後才做，不是事先登錄）：成立假說的補充數字。"""
import os, sys; sys.argv = ["x"]; __file__ = os.path.abspath(__file__)
exec(open(os.path.join(os.path.dirname(__file__), "offense.py")).read().split('if __name__ == "__main__" and sys.argv[1:2] == ["run"]:')[0])
D = build(); R = D["R"]; R = R[R.date >= "2016-01-01"]
M = R[R.fd & ~R.panic & R.y20.notna()].copy()
M["cs"] = np.mean([M.groupby("date")[c].rank(pct=True).fillna(0.5).values for c in COMP], axis=0); M["csr"] = M.groupby("date").cs.rank(pct=True)
top = lambda g: g.csr >= 0.8
A_ = longonly(M, top); B_ = longonly(M, lambda g: top(g) & (g.sur.rank(pct=True) >= 2 / 3)); C_ = longonly(M, lambda g: top(g) & (g.sur.rank(pct=True) < 1 / 3))
print("H1 綜合前20% 全部 相對等權（每月、扣成本）", tser(A_.ex_u), by_period(A_.ex_u))
print("H1 綜合前20% 且 SUR 前1/3", tser(B_.ex_u), by_period(B_.ex_u), "；且 SUR 後1/3", tser(C_.ex_u), by_period(C_.ex_u))
d = (B_.ex_u - A_.ex_u).dropna(); print("H1 加上SUR 比只看綜合多賺（每月）", tser(d), by_period(d))
# 加入 SUR 的新綜合分數（13 個特性）前20檔
M["cs13"] = np.mean([M.groupby("date")[c].rank(pct=True).fillna(0.5).values for c in COMP + ["sur"]], axis=0)
t12 = longonly(M, lambda g: g.cs.rank(ascending=False, method="first") <= 20); t13 = longonly(M, lambda g: g.cs13.rank(ascending=False, method="first") <= 20)
print("前20檔 12特性 vs 13特性（含SUR），相對等權每月：", tser(t12.ex_u), tser(t13.ex_u), "；相對0050：", tser(t12.ex_b), tser(t13.ex_b))
dd = (t13.ex_u - t12.ex_u).dropna(); print("   差", tser(dd), by_period(dd))
# H5 細節
E5 = D["E5"]; E5 = E5[E5.date >= "2016-01-01"].copy(); E5["m"] = E5.date.dt.to_period("M")
for lab in ("非漲停", "漲停"):
    z = E5[(E5.type == lab) & (E5.h == 60)].copy(); z["xu"] = z.y - 0.6 - z.u; z["xb"] = z.y - 0.6 - z.b
    s = z.groupby("m").xb.mean(); s.index = z.groupby("m").date.first().values
    print(f"H5 {lab} 60日：相對0050", tser(s), by_period(s), "；每筆賺錢%", round(float((z.y - 0.6 > 0).mean() * 100), 1), "；贏等權%", round(float((z.xu > 0).mean() * 100), 1),
          "；逐年相對等權", z.groupby(z.date.dt.year).xu.mean().round(2).to_dict())
# H5 事件當月的綜合分數（用事件前最近一次每月快照）
snap = M[["date", "code", "csr"]].sort_values("date")
z = E5[(E5.type == "非漲停") & (E5.h == 60)].sort_values("date").copy(); z["xu"] = z.y - 0.6 - z.u
z = pd.merge_asof(z, snap.rename(columns={"date": "sd"}), left_on="date", right_on="sd", by="code", direction="backward")
z["cq"] = pd.cut(z.csr, [0, 0.5, 0.8, 1.01], labels=["<50%", "50-80%", "前20%"])
print("H5 非漲停事件依前一次綜合分數排名（60日相對等權）：\n", z.groupby("cq", observed=True).xu.agg(["size", "mean"]).round(2).to_string(), "\n   無快照", int(z.csr.isna().sum()))
