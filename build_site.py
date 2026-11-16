"""讀取 data/prices.csv，計算每日活躍度指標與精選名單，輸出 docs/index.html。

精選名單每天寫進 data/picks.csv（只新增、不改舊紀錄），用來追蹤之後的真實表現。
"""
import json, os, numpy as np, pandas as pd

RULE_SET = "2026-11-13"   # 規則依此日以前的資料訂出；之後的名單才算真正的新資料驗證
SEED_DAYS = 40            # 第一次建立追蹤紀錄時，往回補算的交易日數（標為回測）
HOLD = 5                  # 追蹤報酬：隔日開盤買、第5個交易日收盤賣
COST = 1.1                # 手續費＋稅＋滑價，約1.1%
PICKS = "data/picks.csv"

p = pd.read_csv("data/prices.csv", dtype={"code": str}); p["date"] = pd.to_datetime(p.date)
p = p[~p.code.str.startswith("0")]
pv = lambda c: p.pivot(index="date", columns="code", values=c).sort_index()
C, O, H, L, A = pv("close"), pv("open"), pv("high"), pv("low"), pv("amount")
days = C.index; last = days[-1]
info = p.sort_values("date").drop_duplicates("code", keep="last").set_index("code")
mkt = info.market.map({"TPEX": "櫃", "TWSE": "市"})
r1 = C.pct_change(); R = lambda df, n, f: getattr(df.rolling(n, min_periods=n), f)()
tr = pd.concat([(H - L), (H - C.shift()).abs(), (C.shift() - L).abs()]).groupby(level=0).max()

# ---- 每日指標（全部日期一次算好）----
f = {}
f["v"] = R(r1, 20, "std") * 100
f["rg"] = (R(H, 10, "max") - R(L, 10, "min")) / C * 100
f["atr"] = R(tr, 14, "mean") / C * 100
f["lu"] = R((r1 >= 0.09).astype(float).where(C.notna()), 20, "sum")
f["mx"] = (C / C.shift(5) - 1).rolling(40, min_periods=40).max() * 100
f["am"] = R(A, 5, "mean") / R(A, 20, "mean"); f["amt"] = R(A, 20, "mean") / 1e6
f["ch"] = r1 * 100; f["r5"] = (C / C.shift(5) - 1) * 100; f["r20"] = (C / C.shift(20) - 1) * 100
f["dh"] = (C / R(C, 60, "max") - 1) * 100; f["c"] = C
f["n60"] = C.rolling(60, min_periods=60).count(); f["ma20"] = R(C, 20, "mean")
f["bk"] = pd.concat([(C.shift(5 * k) / C.shift(5 * k + 5) - 1) * 100 for k in range(4)], keys=range(4)).groupby(level=1).max().reindex(days)
idx = np.arange(len(days))[:, None]
pos = pd.DataFrame(np.where((r1 >= 0.09).values, idx, np.nan), index=days, columns=C.columns).ffill()
f["sn"] = pd.DataFrame(idx - pos.values, index=days, columns=C.columns).fillna(99)

S = pd.DataFrame({k: v.stack(future_stack=True) for k, v in f.items()})
S.index.names = ["date", "code"]
S = S.dropna(subset=["v", "rg", "atr", "lu", "mx", "am", "bk", "c"])
S = S[(S.n60 >= 55) & (S.amt >= 10)].reset_index()  # 允許近60日偶有停牌缺幾天
S["mk"] = S.code.map(mkt)
g = S.groupby(["date", "mk"])
S["rv"] = g.v.rank(pct=True)
S["rc"] = pd.concat([g[c].rank(pct=True) for c in ["v", "rg", "atr", "lu", "mx"]], axis=1).mean(axis=1)
S["rc"] = S.groupby(["date", "mk"]).rc.rank(pct=True)
# 精選門檻（2026/11/16 定案）：活躍度前10%、10日內有漲停、當天收紅、量能比（5日均額/20日均額）至少2
AM_MIN = 2.0
S["hot"] = (S.rc >= 0.9) & (S.sn <= 10)
S["pk"] = S.hot & (S.ch > 0) & (S.am >= AM_MIN)

# ---- 精選紀錄：只新增，不改舊的 ----
cols_log = ["date", "code", "name", "mk", "close", "kind"]
log = pd.read_csv(PICKS, dtype={"code": str}) if os.path.exists(PICKS) else pd.DataFrame(columns=cols_log)
logged = set(log.date)
want = days[-SEED_DAYS:] if not len(log) else days[days > pd.Timestamp(max(logged))]
new = S[S.pk & S.date.isin(want)].copy()
if len(new):
    new["name"] = new.code.map(info.name); new["close"] = new.c.round(2)
    new["kind"] = np.where(new.date <= pd.Timestamp(RULE_SET), "回測", "實際")
    new["date"] = new.date.dt.strftime("%Y-%m-%d")
    log = pd.concat([log, new[cols_log]], ignore_index=True)
    os.makedirs("data", exist_ok=True); log.to_csv(PICKS, index=False)

# ---- 追蹤報酬 ----
dpos = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(days)}
elig = S[["date", "code"]].assign(date=S.date.dt.strftime("%Y-%m-%d"))
def outcome(i, code):
    """回傳 (報酬%, 狀態)：隔日開盤進場、第 HOLD 個交易日收盤出場，扣成本。"""
    if i + 1 >= len(days): return None, "待進場"
    o = O.iat[i + 1, O.columns.get_loc(code)]
    if not o == o or o <= 0: return None, "無法進場"
    j = min(i + HOLD, len(days) - 1)
    c = C.iat[j, C.columns.get_loc(code)]
    if not c == c: return None, "缺資料"
    return round((c / o - 1) * 100 - COST, 2), ("完成" if i + HOLD < len(days) else "進行中")
track = []
for d, grp in log.groupby("date", sort=False):
    if d not in dpos: continue
    i = dpos[d]; items = []
    for _, r in grp.iterrows():
        ret, stt = outcome(i, r.code)
        items.append({"t": r.code, "n": r["name"], "mk": r.mk, "r": ret})
    rets = [x["r"] for x in items if x["r"] is not None]
    # 同期全體：當天所有符合流動性門檻的股票
    base = []
    if i + 1 < len(days):
        j = min(i + HOLD, len(days) - 1)
        cs = elig[elig.date == d].code
        b = (C.iloc[j][cs] / O.iloc[i + 1][cs] - 1) * 100 - COST
        base = b.replace([np.inf, -np.inf], np.nan).dropna()
    track.append({"d": d, "k": grp.kind.iloc[0], "s": "待進場" if i + 1 >= len(days) else ("完成" if i + HOLD < len(days) else "進行中"),
                  "n": len(items), "avg": round(float(np.mean(rets)), 2) if rets else None,
                  "all": round(float(base.mean()), 2) if len(base) else None,
                  "it": sorted(items, key=lambda x: -(x["r"] if x["r"] is not None else -999))})
track.sort(key=lambda x: x["d"], reverse=True)

# ---- 今天的清單 ----
d = S[S.date == last].set_index("code").copy()
prev_pk = set(S[(S.date == days[-2]) & S.pk].code) if len(days) > 1 else set()
d["n"] = d.index.map(info.name); d["nw"] = d.pk & ~d.index.isin(prev_pk)
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
        "above": int(round(allc.a.mean() * 100)), "rule": RULE_SET, "hold": HOLD, "am": AM_MIN,
        "hv": round(float(d[d.rc >= 0.9].v.median()), 1), "hrg": round(float(d[d.rc >= 0.9].rg.median()), 1)}
html = open("site_template.html", encoding="utf-8").read()
J = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":"))
html = html.replace("/*DATA*/[]", J(data)).replace("/*META*/{}", J(meta)).replace("/*TRACK*/[]", J(track))
os.makedirs("docs", exist_ok=True); open("docs/index.html", "w", encoding="utf-8").write(html)
print(meta, len(data), "stocks,", int(d.pk.sum()), "picks,", len(track), "tracked days,", len(html) // 1024, "KB")
