"""探索（看過 deep_normal.py A/B/C 結果後才做，不是事先登錄；數字只當參考，要靠前瞻驗證確認）。
1) 綜合分數拆成只用價格、只用營收；換股緩衝；前10；抱60日；恐慌月份。
2) 今日發動依「當月全市場綜合分數」分組；每月綜合前20檔依當天狀態分組。
"""
import os, sys; _me = os.path.abspath(__file__); sys.argv=["x"]; exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "deep_normal.py")).read())
R = load(); res, sel, _ = study_a(R)
PRICE = ["r20", "mom6_1", "d_hi250", "m60", "am", "rs_g20"]; REV = ["yoy1", "yoy3", "rev_acc", "rev_hi12", "mom1"]
def score(M, fs):
    sc = np.zeros(len(M))
    for fc in fs: sc += M.groupby("date")[fc].rank(pct=True).fillna(0.5).values
    return sc / len(fs)
def run(M, fs, K=20, buf=None, cost=0.6, yk="y20", bk="b20"):
    M = M.assign(s=score(M, fs)); M["rk"] = M.groupby("date").s.rank(ascending=False, method="first")
    prev = []; rows = []
    for d, g in M.groupby("date"):
        g = g.sort_values("rk")
        if buf and prev:
            keep = [c for c in prev if c in set(g[g.rk <= buf].code)]
            cur = keep + [c for c in g.code if c not in keep][:K - len(keep)]
        else: cur = list(g.code[:K])
        to = 1.0 if not prev else len(set(cur) - set(prev)) / K; prev = cur
        gg = g.set_index("code").loc[cur]
        rows.append({"date": d, "net": gg[yk].mean() - cost * to, "b": g[bk].iloc[0], "to": to})
    T = pd.DataFrame(rows); T["per"] = period(T.date)
    out = {}
    for per, g in T.groupby("per"):
        n = len(g); f_ = 12 / n if yk == "y20" else None
        if yk == "y20":
            out[per] = (round(float((np.prod(1 + g.net / 100) ** f_ - 1) * 100), 1), round(float((np.prod(1 + g.b / 100) ** f_ - 1) * 100), 1), round(float(g.to.mean() * 100)))
        else:
            out[per] = (round(float(g.net.mean()), 2), round(float(g.b.mean()), 2), round(float((g.net > g.b).mean() * 100)))
    return out
Mn = R[R.fd & ~R.panic & R.y20.notna()].copy()
ALL = list(sel)
print("（年化%、0050年化%、平均換股%）只算平常月份")
print("基準 全部入選特性 前20：", run(Mn, ALL))
print("只用價格特性 前20：", run(Mn, PRICE))
print("只用營收特性 前20：", run(Mn, REV))
print("價格＋營收（不含融資）前20：", run(Mn, PRICE + REV))
print("緩衝：排名掉出前60才換 前20：", run(Mn, ALL, buf=60))
print("緩衝：排名掉出前40才換 前20：", run(Mn, ALL, buf=40))
print("前10：", run(Mn, ALL, K=10))
M60 = R[R.fd & ~R.panic & R.y60.notna()].copy()
print("抱60日（每月新一批，每筆平均%、0050同期%、贏0050月份%）：", run(M60, ALL, yk="y60", bk="b60"))
Mp = R[R.fd & R.panic & R.y20.notna()].copy()
print("恐慌月份（前20 每月平均 vs 0050）：", {p: (round(float(g.y20.mean()), 2)) for p, g in Mp.assign(s=score(Mp, ALL)).sort_values("s", ascending=False).groupby("date").head(20).groupby(period(Mp.assign(s=score(Mp, ALL)).sort_values("s", ascending=False).groupby("date").head(20).date))}, "恐慌月數", Mp.date.nunique())

R = load(); res, sel, _ = study_a(R)
F = list(sel)
U = R[R.fd & R.y20.notna()].copy(); U["m"] = U.date.dt.to_period("M")
K = R[R.pk & ~R.panic & R.y20.notna()].copy(); K["m"] = K.date.dt.to_period("M")
# 今日發動的每個特性，換算成「當月全市場快照」裡的百分位，再等權平均成綜合分數
sc = np.zeros(len(K)); 
for fc in F:
    pr = np.full(len(K), 0.5)
    for m, g in K.groupby("m"):
        ref = np.sort(U.loc[U.m == m, fc].dropna().values)
        if len(ref) < 50: continue
        v = g[fc].values; p = np.searchsorted(ref, v, side="right") / len(ref)
        pr[K.index.get_indexer(g.index)] = np.where(np.isnan(v), 0.5, p)
    sc += pr
K["s"] = sc / len(F); K["y"] = K.y20 - 1.1; K["x"] = K.y - K.b20
K["q"] = pd.cut(K.s, [0, 0.5, 0.6, 0.7, 0.8, 1.01], labels=["<0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "≥0.8"])
t = K.groupby([period(K.date), "q"], observed=True).agg(n=("y", "size"), 平均=("y", "mean"), 賺錢=("y", lambda v: (v > 0).mean() * 100), 減0050=("x", "mean")).round(2)
print("平常時期今日發動，依「全市場綜合分數」分組（抱20日、扣1.1%）：\n", t.to_string())
yr = K[K.s >= 0.8].groupby(K.date.dt.year).y.agg(["size", "mean"]).round(2); print("綜合分數≥0.8 逐年：", yr.to_dict("index"))
# 每月快照：綜合前20檔裡，當天剛好是今日發動／仍在動／都不是
Mn = R[R.fd & ~R.panic & R.y20.notna()].copy(); s2 = np.zeros(len(Mn))
for fc in F: s2 += Mn.groupby("date")[fc].rank(pct=True).fillna(0.5).values
Mn["s"] = s2 / len(F); top = Mn.sort_values(["date", "s"], ascending=[True, False]).groupby("date").head(20)
top["kind"] = np.where(top.pk, "當天今日發動", np.where(top.hot, "仍在動", "都不是"))
print("每月綜合前20檔，依當天狀態（抱20日未扣成本）：\n", top.groupby([period(top.date), "kind"]).y20.agg(["size", "mean"]).round(2).to_string())
