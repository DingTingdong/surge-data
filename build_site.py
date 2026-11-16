"""讀取 data/prices.csv，計算每日指標與分級名單，寫紀錄並輸出 docs/index.html。

紀錄（只新增，不改舊資料）：
- data/signals.csv：每天「仍在動」全部名單，含分級（精選／觀察／仍在動）與規則版本
- data/days.csv：每天一列，含各級檔數（零檔的日子也記），用來區分「沒有訊號」與「沒有跑」
"""
import json, os, numpy as np, pandas as pd

RULE_VERSION = "v1.1"        # 門檻 11/16 定案；11/17 修正缺值與歷史市場別後重算紀錄
FORWARD_START = "2026-11-16" # 這天（含）以後的名單才算真正的新資料驗證；之前的是規則制定期間的回測
SEED_DAYS = 40               # 第一次建立紀錄時往回補算的交易日數（標為回測）
HOLD = 5                     # 隔日開盤買、第5個交易日收盤賣
COST = 1.1                   # 手續費＋稅＋滑價，約1.1%
LOCK = 0.095                 # 一價到底且漲跌幅達此比例 → 視為漲跌停鎖死
AM_MIN = 2.0
SIG, DAYS = "data/signals.csv", "data/days.csv"

p = pd.read_csv("data/prices.csv", dtype={"code": str}); p["date"] = pd.to_datetime(p.date)
p = p[~p.code.str.startswith("0")].drop_duplicates(["date", "code"], keep="last")
pv = lambda c: p.pivot(index="date", columns="code", values=c).sort_index()
C, O, H, L, A, MK = pv("close"), pv("open"), pv("high"), pv("low"), pv("amount"), pv("market")
days = C.index; last = days[-1]
name = p.sort_values("date").drop_duplicates("code", keep="last").set_index("code").name
r1 = C.pct_change(fill_method=None)   # 停牌中斷不補價，避免把兩天的漲跌算成一天
R = lambda df, n, f: getattr(df.rolling(n, min_periods=n), f)()
tr = pd.concat([(H - L), (H - C.shift()).abs(), (C.shift() - L).abs()]).groupby(level=0).max()

# ---- 每日指標（全部日期一次算好，只往回看）----
f = {}
f["v"] = R(r1, 20, "std") * 100
f["rg"] = (R(H, 10, "max") - R(L, 10, "min")) / C * 100
f["atr"] = R(tr, 14, "mean") / C * 100
f["lu"] = R((r1 >= 0.09).astype(float).where(r1.notna()), 20, "sum")
f["mx"] = (C / C.shift(5) - 1).rolling(40, min_periods=40).max() * 100
f["am"] = R(A, 5, "mean") / R(A, 20, "mean"); f["amt"] = R(A, 20, "mean") / 1e6
f["ch"] = r1 * 100; f["r5"] = (C / C.shift(5) - 1) * 100; f["r20"] = (C / C.shift(20) - 1) * 100
f["dh"] = (C / R(C, 60, "max") - 1) * 100; f["c"] = C
f["n60"] = C.rolling(60, min_periods=60).count(); f["ma20"] = R(C, 20, "mean")
f["bk"] = pd.concat([(C.shift(5 * k) / C.shift(5 * k + 5) - 1) * 100 for k in range(4)], keys=range(4)).groupby(level=1).max().reindex(days)
idx = np.arange(len(days))[:, None]
pos = pd.DataFrame(np.where((r1 >= 0.09).values, idx, np.nan), index=days, columns=C.columns).ffill()
f["sn"] = pd.DataFrame(idx - pos.values, index=days, columns=C.columns).fillna(99)  # 0 = 今天漲停

S = pd.DataFrame({k: v.stack(future_stack=True) for k, v in f.items()})
S.index.names = ["date", "code"]
S = S.dropna(subset=["v", "rg", "atr", "lu", "mx", "am", "bk", "c"])
S = S[(S.n60 >= 55) & (S.amt >= 10)].reset_index()  # 允許近60日偶有停牌缺幾天
S["mk"] = MK.stack().reindex(pd.MultiIndex.from_frame(S[["date", "code"]])).map({"TPEX": "櫃", "TWSE": "市"}).values  # 當天的市場別
g = S.groupby(["date", "mk"])
S["rv"] = g.v.rank(pct=True)
S["rc"] = pd.concat([g[c].rank(pct=True) for c in ["v", "rg", "atr", "lu", "mx"]], axis=1).mean(axis=1)
S["rc"] = S.groupby(["date", "mk"]).rc.rank(pct=True)
# 分級：仍在動 = 活躍度前10% 且 今天或前10個交易日內有漲停；精選再加 當天上漲、量能比≥2；觀察 = 量能比 1～2
S["hot"] = (S.rc >= 0.9) & (S.sn <= 10)
S["pk"] = S.hot & (S.ch > 0) & (S.am >= AM_MIN)
S["wt"] = S.hot & (S.ch > 0) & (S.am >= 1) & ~S.pk
S["tier"] = np.select([S.pk, S.wt, S.hot], ["精選", "觀察", "仍在動"], "")

# ---- 紀錄：只新增，不改舊的 ----
os.makedirs("data", exist_ok=True)
ds = lambda x: x.dt.strftime("%Y-%m-%d")
sig = pd.read_csv(SIG, dtype={"code": str}) if os.path.exists(SIG) else None
dlog = pd.read_csv(DAYS) if os.path.exists(DAYS) else None
lastlog = None if dlog is None or not len(dlog) else pd.Timestamp(dlog.date.max())
want = days[-SEED_DAYS:] if lastlog is None else days[days > lastlog]
if len(want):
    kind = lambda d: np.where(d < pd.Timestamp(FORWARD_START), "回測", "實際")
    new = S[S.hot & S.date.isin(want)].copy()
    new = new.assign(ver=RULE_VERSION, kind=kind(new.date), name=new.code.map(name), close=new.c.round(2),
                     rc=new.rc.round(3), am=new.am.round(2), ch=new.ch.round(2), date=ds(new.date))
    new = new[["date", "ver", "kind", "tier", "code", "name", "mk", "close", "rc", "am", "ch"]]
    sig = new if sig is None else pd.concat([sig, new], ignore_index=True)
    sig.to_csv(SIG, index=False)
    cnt = S[S.date.isin(want)].groupby("date").agg(universe=("code", "size"), hot=("hot", "sum"), watch=("wt", "sum"), pick=("pk", "sum")).reindex(want, fill_value=0)
    cnt = cnt.reset_index().rename(columns={"index": "date"})
    cnt = cnt.assign(ver=RULE_VERSION, kind=kind(cnt.date), date=ds(cnt.date))[["date", "ver", "kind", "universe", "hot", "watch", "pick"]]
    dlog = cnt if dlog is None else pd.concat([dlog, cnt], ignore_index=True)
    dlog.to_csv(DAYS, index=False)

# ---- 追蹤報酬 ----
dpos = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(days)}
Cf = C.ffill(limit=HOLD)  # 持有中停牌：用最後可得收盤，不把它丟掉
def locked(row_o, row_h, row_l, prev_c, up):
    if not (row_o == row_o and prev_c == prev_c and prev_c > 0): return False
    flat = row_o == row_h == row_l
    chg = row_o / prev_c - 1
    return flat and (chg >= LOCK if up else chg <= -LOCK)
def outcome(i, code):
    """隔日開盤進場、第 HOLD 個交易日收盤出場，扣成本。回傳 (報酬%, 狀態)。"""
    if i + 1 >= len(days): return None, "待進場"
    k = C.columns.get_loc(code)
    o = O.iat[i + 1, k]
    if not o == o or o <= 0: return None, "無法進場"
    if locked(o, H.iat[i + 1, k], L.iat[i + 1, k], C.iat[i, k], True): return None, "漲停買不到"
    j = i + HOLD
    if j >= len(days): return round((Cf.iat[-1, k] / o - 1) * 100 - COST, 2), "進行中"
    while j + 1 < len(days) and j < i + HOLD + 5 and locked(O.iat[j, k], H.iat[j, k], L.iat[j, k], Cf.iat[j - 1, k], False):
        j += 1  # 出場日跌停鎖死 → 順延，最多5天
    c = Cf.iat[j, k]
    if not c == c: return None, "缺資料"
    return round((c / o - 1) * 100 - COST, 2), "完成"

def stats(rs):
    a = np.array([r for r in rs if r is not None], float)
    if not len(a): return None
    q = np.quantile(a, 0.05)
    return {"n": int(len(a)), "avg": round(float(a.mean()), 2), "med": round(float(np.median(a)), 2),
            "win": round(float((a > 0).mean() * 100)), "up": round(float((a >= 10).mean() * 100)),
            "dn": round(float((a <= -10).mean() * 100)), "w5": round(float(a[a <= q].mean()), 1)}

univ = S[["date", "code", "hot", "pk"]].assign(date=ds(S.date))
track, pool = [], {k: {"pk": [], "hot": [], "all": [], "blk": 0} for k in ("實際", "回測")}
for d, grp in sig.groupby("date", sort=False):
    if d not in dpos: continue
    i = dpos[d]; kd = grp.kind.iloc[0]
    pk = grp[grp.tier == "精選"]
    items = []
    for _, r in pk.iterrows():
        ret, stt = outcome(i, r.code); items.append({"t": r.code, "n": r["name"], "r": ret, "s": stt})
        if stt == "漲停買不到": pool[kd]["blk"] += 1
    other = [outcome(i, c)[0] for c in grp[grp.tier != "精選"].code]
    allr = [outcome(i, c)[0] for c in univ[univ.date == d].code] if i + 1 < len(days) else []
    done = i + HOLD < len(days)
    if done:
        pool[kd]["pk"] += [x["r"] for x in items]; pool[kd]["hot"] += other; pool[kd]["all"] += allr
    mean = lambda rs: (lambda s: s and s["avg"])(stats(rs))
    track.append({"d": d, "k": kd, "s": "待進場" if i + 1 >= len(days) else ("完成" if done else "進行中"),
                  "n": len(items), "avg": mean([x["r"] for x in items]), "hot": mean(other), "all": mean(allr),
                  "it": sorted(items, key=lambda x: -(x["r"] if x["r"] is not None else -999))})
for d in dlog[dlog.pick == 0].date:  # 零檔的日子也列出
    if d not in {t["d"] for t in track} and d in dpos:
        track.append({"d": d, "k": dlog.set_index("date").kind[d], "s": "無精選", "n": 0, "avg": None, "hot": None, "all": None, "it": []})
track.sort(key=lambda x: x["d"], reverse=True)
summary = {k: {"pk": stats(v["pk"]), "hot": stats(v["hot"]), "all": stats(v["all"]), "blk": v["blk"],
               "days": len({t["d"] for t in track if t["k"] == k and t["s"] in ("完成", "無精選")})} for k, v in pool.items()}

# ---- 今天的清單 ----
d = S[S.date == last].set_index("code").copy()
prev_pk = set(S[(S.date == days[-2]) & S.pk].code) if len(days) > 1 else set()
d["n"] = d.index.map(name); d["nw"] = d.pk & ~d.index.isin(prev_pk)
cols = ["n", "mk", "c", "ch", "v", "rg", "atr", "lu", "am", "rc", "rv", "r5", "r20", "dh", "amt", "sn", "bk"]
data = []
for code, r in d.iterrows():
    x = {"t": code}
    for k in cols: x[k] = r[k] if k in ("n", "mk") else round(float(r[k]), 2)
    x["lu"] = int(x["lu"]); x["amt"] = int(x["amt"]); x["sn"] = int(x["sn"]); x["nw"] = bool(r.nw)
    if r.rc >= 0.8:  # 只給可能顯示的股票附上走勢，控制檔案大小
        x["sp"] = [round(float(v), 2) for v in C[code].iloc[-30:].tolist() if v == v]
    data.append(x)
allc = pd.DataFrame({"r5": f["r5"].loc[last], "a": (C.loc[last] > f["ma20"].loc[last])}).dropna()
meta = {"date": last.strftime("%Y/%m/%d"), "m5": round(float(allc.r5.median()), 2),
        "above": int(round(allc.a.mean() * 100)), "fwd": FORWARD_START, "ver": RULE_VERSION, "hold": HOLD, "am": AM_MIN,
        "hv": round(float(d[d.rc >= 0.9].v.median()), 1), "hrg": round(float(d[d.rc >= 0.9].rg.median()), 1), "sum": summary}
html = open("site_template.html", encoding="utf-8").read()
J = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":"))
html = html.replace("/*DATA*/[]", J(data)).replace("/*META*/{}", J(meta)).replace("/*TRACK*/[]", J(track))
os.makedirs("docs", exist_ok=True); open("docs/index.html", "w", encoding="utf-8").write(html)
print({k: meta[k] for k in ("date", "ver", "above")}, len(data), "stocks,", int(d.pk.sum()), "picks,", len(track), "tracked days,", len(html) // 1024, "KB")
print(json.dumps(summary, ensure_ascii=False))
