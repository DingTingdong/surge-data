"""探索（看過結果後）：「綜合分數前20檔每月勝等權 +2.9%」這個好得可疑的數字，做穩健性檢查。
分期間、成本提高到 1.1%／1.5%、只看 20 日均額 ≥ 5,000 萬／1 億的股票、前20／50／100 檔、拿掉發現期 2020–23。"""
import os, sys; sys.argv = ["x"]; __file__ = os.path.abspath(__file__)
exec(open(os.path.join(os.path.dirname(__file__), "offense.py")).read().split('if __name__ == "__main__" and sys.argv[1:2] == ["run"]:')[0])
D = build(); R = D["R"]; R = R[R.date >= "2016-01-01"]
M0 = R[R.fd & ~R.panic & R.y20.notna()].copy()
def run(M, K=20, cost=0.6):
    M = M.copy(); M["cs"] = np.mean([M.groupby("date")[c].rank(pct=True).fillna(0.5).values for c in COMP], axis=0)
    rows, prev = {}, set()
    for d, g in M.groupby("date"):
        pk = g.sort_values("cs", ascending=False).head(K); cur = set(pk.code); to = 1.0 if not prev else len(cur - prev) / len(cur); prev = cur
        rows[d] = (pk.y20.mean() - cost * to - g.y20.mean(), pk.y20.mean() - cost * to - g.b20.iloc[0], to)
    T = pd.DataFrame(rows, index=["u", "b", "to"]).T
    return tser(T.u), by_period(T.u), tser(T.b), round(float(T.to.mean()), 2)
for lab, M, K, c in (("基準 前20 成本0.6", M0, 20, 0.6), ("成本1.1", M0, 20, 1.1), ("成本1.5", M0, 20, 1.5),
                     ("均額≥5千萬 前20", M0[M0.amt >= 50], 20, 0.6), ("均額≥1億 前20", M0[M0.amt >= 100], 20, 0.6), ("均額≥3億 前20", M0[M0.amt >= 300], 20, 0.6),
                     ("前50", M0, 50, 0.6), ("前100", M0, 100, 0.6)):
    u, bp, b, to = run(M, K, c)
    print(f"{lab}: 勝等權(平均,t,月數) {u} 分段 {bp} ｜勝0050 {b} ｜換股 {to}", flush=True)
