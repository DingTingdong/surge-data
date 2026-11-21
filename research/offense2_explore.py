"""探索（看過 offense2.py 結果後才做，不是事先登錄）：季 ROE 為什麼和文獻相反？虧損公司的今日發動是否更差？"""
import os, sys; sys.argv = ["x"]; __file__ = os.path.abspath(__file__)
exec(open(os.path.join(os.path.dirname(__file__), "offense2.py"), encoding="utf-8").read().split('if __name__ == "__main__" and sys.argv[1:2] == ["run"]:')[0].split('"""', 2)[2])
R = build2(); R = R[R.date >= "2016-01-01"]
M = R[R.fd & ~R.panic & R.y20.notna()].copy()
M["cs"] = np.mean([M.groupby("date")[c].rank(pct=True).fillna(0.5).values for c in COMP], axis=0)
c = fm2(M, "roeq", ctrl=False); print("季ROE 不控制綜合分數：係數", round(c.mean(), 3), "NW t", round(nw_t(c, 1), 2), {p: round(float(c[per3(c.index) == p].mean()), 2) for p in ("2016-19", "2020-23", "2024-26")})
q = M[M.roeq.notna()].assign(q=lambda z: np.minimum(5, np.floor(z.groupby("date").roeq.rank(pct=True, method="first") * 5 - 1e-9) + 1)).groupby(["date", "q"]).y20.mean().unstack()
print("季ROE 五等分 20日平均（每月等權）：", q.mean().round(2).to_dict(), "；第5−第1組 NW t", round(nw_t(q[5.0] - q[1.0], 1), 2))
print("季ROE 與綜合分數的相關（月內等級相關平均）：", round(float(M.groupby("date").apply(lambda g: g[["roeq", "cs"]].corr(method="spearman").iloc[0, 1]).mean()), 2))
neg = M.assign(neg=M.roeq < 0).groupby(["date", "neg"]).y20.mean().unstack()
print("虧損（單季ROE<0）vs 獲利：", neg.mean().round(2).to_dict(), "差 NW t", round(nw_t(neg[True] - neg[False], 1), 2))
K = R[R.pk & ~R.panic & R.y20.notna()].copy(); K["x"] = K.y20 - 1.1 - K.b20; K["m"] = K.date.dt.to_period("M")
for lab, z in (("今日發動 虧損公司", K[K.roeq < 0]), ("今日發動 獲利公司", K[K.roeq >= 0])):
    s = z.groupby(["m", "date"]).x.mean().groupby(level=0).mean()
    print(lab, "20日減0050（每天先平均·每月等權）", round(s.mean(), 2), "NW t", round(nw_t(s, 1), 2), "筆數", len(z), {p: round(float(s[per3(s.index.to_timestamp()) == p].mean()), 2) for p in ("2016-19", "2020-23", "2024-26")})
