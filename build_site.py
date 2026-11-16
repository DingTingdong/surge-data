"""讀取 data/prices.csv，計算最新交易日的活躍度指標，輸出 docs/index.html。"""
import json, os, numpy as np, pandas as pd

p = pd.read_csv("data/prices.csv", dtype={"code": str}); p["date"] = pd.to_datetime(p.date)
p = p[~p.code.str.startswith("0")]
pv = lambda c: p.pivot(index="date", columns="code", values=c).sort_index()
C, H, L, A = pv("close"), pv("high"), pv("low"), pv("amount")
last = C.index[-1]
info = p.sort_values("date").drop_duplicates("code", keep="last").set_index("code")
r1 = C.pct_change(); R = lambda df, n, f: getattr(df.rolling(n, min_periods=n), f)()
tr = pd.concat([(H - L), (H - C.shift()).abs(), (C.shift() - L).abs()]).groupby(level=0).max()
rg10 = (R(H, 10, "max") - R(L, 10, "min")) / C * 100
f = {}
f["v"] = R(r1, 20, "std") * 100; f["rg"] = rg10; f["atr"] = R(tr, 14, "mean") / C * 100
f["lu"] = R((r1 >= 0.09).astype(float).where(C.notna()), 20, "sum")
f["mx"] = (C / C.shift(5) - 1).rolling(40, min_periods=40).max() * 100
f["am"] = R(A, 5, "mean") / R(A, 20, "mean"); f["amt"] = R(A, 20, "mean") / 1e6
f["ch"] = r1 * 100; f["r5"] = (C / C.shift(5) - 1) * 100; f["r20"] = (C / C.shift(20) - 1) * 100
f["dh"] = (C / R(C, 60, "max") - 1) * 100; f["c"] = C; f["n60"] = C.rolling(60, min_periods=60).count()
f["ma20"] = R(C, 20, "mean")
_n = len(C); _i = np.where((r1 >= 0.09).values, np.arange(_n)[:, None], -1).max(axis=0)
f["sn"] = pd.DataFrame([np.where(_i < 0, 99, (_n - 1) - _i)], index=C.index[-1:], columns=C.columns).reindex(C.index)
f["bk"] = pd.concat([(C.shift(5 * k) / C.shift(5 * k + 5) - 1) * 100 for k in range(4)], keys=range(4)).groupby(level=1).max().reindex(C.index)
d = pd.DataFrame({k: v.loc[last] for k, v in f.items()}).dropna(subset=["v", "rg", "atr", "lu", "mx", "am", "n60", "sn", "bk"])
d = d[(d.amt >= 10)].copy()
d["mk"] = info.market.reindex(d.index).map({"TPEX": "櫃", "TWSE": "市"}); d["n"] = info.name.reindex(d.index)
g = d.groupby("mk")
rk = lambda c: g[c].rank(pct=True)
d["rv"] = rk("v"); d["rc"] = pd.concat([rk(c) for c in ["v", "rg", "atr", "lu", "mx"]], axis=1).mean(axis=1)
d["rc"] = d.groupby("mk").rc.rank(pct=True)
cols = ["n", "mk", "c", "ch", "v", "rg", "atr", "lu", "am", "rc", "rv", "r5", "r20", "dh", "amt", "sn", "bk"]
out = d.reset_index().rename(columns={"index": "t", "code": "t"})
data = [{k: (round(float(r[k]), 2) if k not in ("n", "mk", "t") else r[k]) for k in ["t"] + cols} for _, r in out.iterrows()]
for r in data: r["lu"] = int(r["lu"]); r["amt"] = int(r["amt"]); r["sn"] = int(r["sn"])
allc = pd.DataFrame({"r5": f["r5"].loc[last], "a": (C.loc[last] > f["ma20"].loc[last])}).dropna()
meta = {"date": last.strftime("%Y/%m/%d"), "m5": round(float(allc.r5.median()), 2), "above": int(round(allc.a.mean() * 100))}
html = open("site_template.html", encoding="utf-8").read()
html = html.replace("/*DATA*/[]", json.dumps(data, ensure_ascii=False, separators=(",", ":"))).replace("/*META*/{}", json.dumps(meta))
os.makedirs("docs", exist_ok=True); open("docs/index.html", "w", encoding="utf-8").write(html)
print(meta, len(data), "stocks", len(html) // 1024, "KB")
