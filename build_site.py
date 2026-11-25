"""讀取 data/prices.csv，計算每日指標與分級名單，寫紀錄並輸出 docs/index.html。

紀錄（只新增，不改舊資料）：
- data/signals.csv：每天「今日發動」與「仍在動」全部名單，含分級與規則版本
- data/panic_log.csv：上市空頭家數近10天最高 ≥ 750 的每一天，全部跌深股與名單股，只增不改。ts=1 為撿跌深名單（11/19 為舊定義：市值≤100億；11/20 起為這一波的前波強勢股），top=1 為網站顯示的前20名（11/20 起）
- data/days.csv：每天一列，含各級檔數（零檔的日子也記），用來區分「沒有訊號」與「沒有跑」
規則改版時，舊紀錄移到 data/archive/，新版本重新開始記錄（v1 的紀錄在 data/archive/*_v1.csv）。
"""
import json, os, numpy as np, pandas as pd
from radar_core import load_prices, panels, features, tiers, RULE_VERSION

FORWARD_START = "2026-11-17" # v2.0 上線日：這天（含）以後的名單才算真正的新資料驗證；之前的是回測
SEED_DAYS = 40               # 第一次建立紀錄時往回補算的交易日數（標為回測）
HOLD = 5                     # 隔日開盤買、第5個交易日收盤賣
COST = 1.1                   # 手續費＋稅＋滑價，約1.1%
LOCK = 0.095                 # 一價到底且漲跌幅達此比例 → 視為漲跌停鎖死
SIG, DAYS = "data/signals.csv", "data/days.csv"

p = load_prices(years_back=1)
# 最新一天如果只抓到一個市場（例如太早手動執行、櫃買還沒公布），先不算這天，等下次執行再補
cnt = p[p.date == p.date.max()].groupby("market").size()
if cnt.get("TWSE", 0) < 300 or cnt.get("TPEX", 0) < 300:
    print("latest day incomplete, skipped:", p.date.max().date(), cnt.to_dict())
    p = p[p.date < p.date.max()]
P = panels(p)                                       # 已做除權息／減資調整
C, O, H, L = P["C"], P["O"], P["H"], P["L"]
days = C.index; last = days[-1]
name = p.sort_values("date").drop_duplicates("code", keep="last").set_index("code").name
f = features(P)
S = tiers(f, P["MK"], since=days[max(0, len(days) - 70)])  # 網站只需近期；指標仍用完整歷史算

# ---- 大盤狀態：分析範圍內股票的等權指數是否在60日均線下（「逆勢」，探索性標示，見 research/results/README.md）----
inb = (f["amt"] >= 10) & (f["n60"] >= 55)
for k in ("v", "rg", "atr", "lu", "mx", "am", "bk"): inb &= f[k].notna()  # 與 radar_core 的分析範圍相同
ewi = (1 + C.pct_change(fill_method=None).where(inb).mean(axis=1).fillna(0)).cumprod()  # 分析範圍內股票的等權指數
weak = (ewi < ewi.rolling(60, min_periods=60).mean())
S["weak"] = S.date.map(weak).fillna(False).astype(bool)

# 空頭家數（看盤網站的「短均線空頭排列」：5日線 < 10日線 < 20日線）。沒成交的日子沿用前一天收盤，和看盤網站一樣把冷門股也算進去。
# 主要看「上市」家數 ≥ 800（老師的理論；2020年以來 5 次，見 research/results/README.md），另外列上櫃與合計。
_alive = (C.notna().cumsum() > 0) & (C.notna()[::-1].cumsum()[::-1] > 0)
_Cf = C.ffill(limit=20).where(_alive)
_m = {n: _Cf.rolling(n, min_periods=n).mean() for n in (5, 10, 20, 60)}
_ok = _Cf.notna() & _m[60].notna()
_sh = (_m[5] < _m[10]) & (_m[10] < _m[20]) & _ok
_mk = P["MK"].ffill().iloc[-1]
_tw = [c for c in C.columns if _mk.get(c) == "TWSE"]; _tp = [c for c in C.columns if _mk.get(c) == "TPEX"]
bear = _sh[_tw].sum(axis=1)
bear_tot = _ok[_tw].sum(axis=1)
d20 = (C.iloc[-1] / C.iloc[-20:].max() - 1) * 100  # 離自己近20日最高收盤
above10 = C.iloc[-1] >= C.iloc[-10:].mean()  # 收盤在10日線上（仍在動分頁只放這種、而且當天沒跌停）
bear_meta = {"n": int(bear.iloc[-1]), "tot": int(bear_tot.iloc[-1]), "h": [int(x) for x in bear.iloc[-40:]], "dd": [x.strftime("%m/%d") for x in days[-40:]],
             "d0": days[-40].strftime("%Y-%m-%d"), "hi": int(bear.iloc[-40:].max()),
             "otc": int(_sh[_tp].iloc[-1].sum()), "otc_tot": int(_ok[_tp].iloc[-1].sum()),
             "all": int(_sh.iloc[-1].sum()), "all_tot": int(_ok.iloc[-1].sum())}

# ---- 大跌事件（定義同 research/drops.py）：等權指數比近20日最高收盤低 ≥ 6% 觸發，回到離高點不到 3% 才結束 ----
ddi = (ewi / ewi.rolling(20, min_periods=1).max() - 1).values
drop_now = None; i = 0
while i < len(ddi):
    if ddi[i] <= -0.06:
        j = i
        while j < len(ddi) and ddi[j] <= -0.03: j += 1
        if j >= len(ddi):  # 還沒結束
            drop_now = (int(np.argmax(ewi.values[max(0, i - 19):i + 1])) + max(0, i - 19), i)
        i = j
    else: i += 1
dk = (C.iloc[-1] / C.iloc[drop_now[0]] - 1) * 100 if drop_now else None
drop_meta = {"pk": days[drop_now[0]].strftime("%Y-%m-%d"), "t": days[drop_now[1]].strftime("%Y-%m-%d"),
             "idx": round(float((ewi.iloc[-1] / ewi.iloc[drop_now[0]] - 1) * 100), 1)} if drop_now else None

# ---- 紀錄：只新增，不改舊的 ----
os.makedirs("data", exist_ok=True)
ds = lambda x: x.dt.strftime("%Y-%m-%d")
sig = pd.read_csv(SIG, dtype={"code": str}) if os.path.exists(SIG) else None
dlog = pd.read_csv(DAYS) if os.path.exists(DAYS) else None
lastlog = None if dlog is None or not len(dlog) else pd.Timestamp(dlog.date.max())
want = days[-SEED_DAYS:] if lastlog is None else days[days > lastlog]
if len(want):
    kind = lambda d: np.where(d < pd.Timestamp(FORWARD_START), "回測", "實際")
    new = S[(S.hot | S.pk) & S.date.isin(want)].copy()
    new = new.assign(ver=RULE_VERSION, kind=kind(new.date), name=new.code.map(name), close=new.c.round(2),
                     rc=new.rc.round(3), am=new.am.round(2), ch=new.ch.round(2), date=ds(new.date))
    new = new[["date", "ver", "kind", "tier", "code", "name", "mk", "close", "rc", "am", "ch"]]
    sig = new if sig is None else pd.concat([sig, new], ignore_index=True)
    sig.to_csv(SIG, index=False)
    cnt = S[S.date.isin(want)].groupby("date").agg(universe=("base", "sum"), hot=("hot", "sum"), pick=("pk", "sum"), weak=("weak", "max")).reindex(want, fill_value=0)
    cnt = cnt.reset_index().rename(columns={"index": "date"})
    cnt = cnt.assign(ver=RULE_VERSION, kind=kind(cnt.date), date=ds(cnt.date), weak=cnt.weak.astype(int))[["date", "ver", "kind", "universe", "hot", "pick", "weak"]]
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

TRAIL, CAP = 0.10, 20  # 移動停損：收盤比買進後最高收盤低10%就隔天開盤賣，最多持有20個交易日（research/exits.py 的 X6）
def outcome_trail(i, code):
    """隔日開盤進場，移動停損出場。回傳 (報酬%, 狀態, 持有天數)。"""
    if i + 1 >= len(days): return None, "待進場", 0
    k = C.columns.get_loc(code); e = i + 1
    o = O.iat[e, k]
    if not o == o or o <= 0: return None, "無法進場", 0
    if locked(o, H.iat[e, k], L.iat[e, k], C.iat[i, k], True): return None, "漲停買不到", 0
    end, peak = e + CAP - 1, o
    for t in range(e, min(end, len(days) - 1) + 1):
        c = C.iat[t, k]
        if not c == c: continue
        peak = max(peak, c)
        if c <= peak * (1 - TRAIL) and t < end:
            if t + 1 >= len(days): return round((c / o - 1) * 100 - COST, 2), "明天賣", t - e + 1
            for s in range(t + 1, min(t + 6, len(days))):
                if not locked(O.iat[s, k], H.iat[s, k], L.iat[s, k], Cf.iat[s - 1, k], False):
                    return round((O.iat[s, k] / o - 1) * 100 - COST, 2), "完成", s - e + 1
            return round((Cf.iat[-1, k] / o - 1) * 100 - COST, 2), "進行中", len(days) - e
    if end < len(days): return round((Cf.iat[end, k] / o - 1) * 100 - COST, 2), "完成", CAP
    return round((Cf.iat[-1, k] / o - 1) * 100 - COST, 2), "進行中", len(days) - e

def stats(rs):
    a = np.array([r for r in rs if r is not None], float)
    if not len(a): return None
    q = np.quantile(a, 0.05)
    return {"n": int(len(a)), "avg": round(float(a.mean()), 2), "med": round(float(np.median(a)), 2),
            "win": round(float((a > 0).mean() * 100)), "up": round(float((a >= 10).mean() * 100)),
            "dn": round(float((a <= -10).mean() * 100)), "w5": round(float(a[a <= q].mean()), 1)}

univ = S[S.base][["date", "code"]].assign(date=ds(S[S.base].date))
track, pool = [], {k: {"pk": [], "hot": [], "all": [], "trail": [], "blk": 0} for k in ("實際", "回測")}
for d, grp in sig.groupby("date", sort=False):
    if d not in dpos: continue
    i = dpos[d]; kd = grp.kind.iloc[0]
    pk = grp[grp.tier == "今日發動"]
    items = []
    for _, r in pk.iterrows():
        ret, stt = outcome(i, r.code); tr, ts, td = outcome_trail(i, r.code)
        items.append({"t": r.code, "n": r["name"], "r": ret, "s": stt, "x": tr, "xs": ts, "xd": td})
        if i + CAP < len(days) and tr is not None: pool[kd]["trail"].append(tr)  # 只算20天期限已過的，避免只算到先停損出場的
        if stt == "漲停買不到": pool[kd]["blk"] += 1
    other = [outcome(i, c)[0] for c in grp[grp.tier == "仍在動"].code]
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
        track.append({"d": d, "k": dlog.set_index("date").kind[d], "s": "無訊號", "n": 0, "avg": None, "hot": None, "all": None, "it": []})
track.sort(key=lambda x: x["d"], reverse=True)
summary = {k: {"pk": stats(v["pk"]), "hot": stats(v["hot"]), "all": stats(v["all"]), "trail": stats(v["trail"]), "blk": v["blk"],
               "days": len({t["d"] for t in track if t["k"] == k and t["s"] in ("完成", "無訊號")})} for k, v in pool.items()}

# ---- 持有追蹤：近20個交易日的今日發動，照移動停損規則算現在的狀態 ----
def hold_state(i, code):
    k = C.columns.get_loc(code); e = i + 1
    if e >= len(days): return {"s": "明天開盤買"}
    o = O.iat[e, k]
    if not o == o or o <= 0: return None
    if locked(o, H.iat[e, k], L.iat[e, k], C.iat[i, k], True): return {"s": "漲停買不到"}
    end, peak, lastc = e + CAP - 1, o, o
    for t in range(e, min(end, len(days) - 1) + 1):
        c = C.iat[t, k]
        if not c == c: continue
        peak, lastc = max(peak, c), c
        if c <= peak * (1 - TRAIL) and t < end:
            if t == len(days) - 1:
                return {"s": "明天開盤賣", "e": o, "c": c, "pk": peak, "st": peak * (1 - TRAIL), "dd": t - e + 1}
            return {"s": "已賣出", "e": o, "c": c, "pk": peak, "st": peak * (1 - TRAIL), "dd": t - e + 2, "sd": days[t + 1].strftime("%m/%d")}
    if end <= len(days) - 1:
        return {"s": "到期賣出", "e": o, "c": lastc, "pk": peak, "st": peak * (1 - TRAIL), "dd": CAP}
    return {"s": "持有中", "e": o, "c": lastc, "pk": peak, "st": peak * (1 - TRAIL), "dd": len(days) - e, "left": end - (len(days) - 1)}
holds = []
recent = set(days[-(CAP + 1):].strftime("%Y-%m-%d"))
for r in sig[(sig.tier == "今日發動") & sig.date.isin(recent)].itertuples():
    if r.date not in dpos: continue
    h = hold_state(dpos[r.date], r.code)
    if not h: continue
    h.update({"t": r.code, "n": r.name, "d": r.date[5:].replace("-", "/")})
    for kk in ("e", "c", "pk", "st"):
        if kk in h: h[kk] = round(float(h[kk]), 2)
    if "e" in h and "c" in h: h["r"] = round((h["c"] / h["e"] - 1) * 100, 2)
    holds.append(h)
order = {"明天開盤賣": 0, "持有中": 1, "明天開盤買": 2, "已賣出": 3, "到期賣出": 4, "漲停買不到": 5}
holds.sort(key=lambda h: (order.get(h["s"], 9), h["d"]), reverse=False)
active_stop = {h["t"]: h["st"] for h in holds if h["s"] in ("持有中", "明天開盤賣")}

# ---- 籌碼：今天的法人買賣超；融資用最近有資料的一天（融資約晚上才公布）----
import glob
chip_info, chip_meta = {}, {}
cfiles = sorted(glob.glob("data/chips/*.csv*"), key=lambda x: (os.path.basename(x)[:4], x.endswith(".csv")))[-2:]
if cfiles:
    ch = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in cfiles], ignore_index=True)
    ch = ch[ch.date <= last.strftime("%Y-%m-%d")].drop_duplicates(["date", "code"], keep="last")
    cdays = sorted(ch.date.unique())
    t0 = last.strftime("%Y-%m-%d")
    today_c = ch[ch.date == t0].set_index("code") if t0 in cdays else None
    mdays = sorted(ch[ch.margin.notna()].date.unique())
    if len(mdays) > 5:
        md, md5 = mdays[-1], mdays[-6]
        m1, m0 = ch[ch.date == md].set_index("code").margin, ch[ch.date == md5].set_index("code").margin
        mchg = ((m1 - m0) / m0.where(m0 > 0) * 100).dropna()
        chip_meta = {"md": md[5:].replace("-", "/")}
    else:
        mchg = pd.Series(dtype=float)
    for code in S[S.date == last].code:
        x = {}
        if today_c is not None and code in today_c.index:
            r = today_c.loc[code]
            for k, kk in (("foreign", "fi"), ("trust", "tr")):
                if r[k] == r[k]: x[kk] = int(round(r[k]))
        if code in mchg.index: x["mg"] = round(float(mchg[code]), 1)
        if x: chip_info[code] = x
    chip_meta["inst"] = t0 if today_c is not None else None

# ---- 今天的清單 ----
d = S[S.date == last].set_index("code").copy()
prev_pk = set(S[(S.date == days[-2]) & S.pk].code) if len(days) > 1 else set()
d["n"] = d.index.map(name); d["nw"] = d.pk & ~d.index.isin(prev_pk)
cols = ["n", "mk", "c", "ch", "v", "rg", "atr", "lu", "am", "rc", "rv", "r5", "r20", "dh", "amt", "sn", "bk", "vr", "a1"]
# 處置股：今天在處置期間，或已公布、下個交易日起處置（分盤交易，部分需預收款券）
disp = {}
if os.path.exists("data/disposal.csv"):
    dp = pd.read_csv("data/disposal.csv", dtype={"code": str})
    t0 = last.strftime("%Y-%m-%d")
    for r in dp[(dp.end >= t0) & (dp.announced <= t0)].itertuples():
        disp[r.code] = ("處置中 至" if r.start <= t0 else "處置 自") + " " + (r.start if r.start > t0 else r.end)[5:].replace("-", "/")
# 近20個交易日進過今日發動的紀錄（從存檔讀，不重算）
rs_days = set(days[-20:].strftime("%Y-%m-%d"))
rh = sig[sig.date.isin(rs_days) & (sig.tier == "今日發動")].sort_values("date")
recent_hist = {c: [[x.date[5:].replace("-", "/"), x.tier] for x in g.itertuples()] for c, g in rh.groupby("code")}
ru = (C.iloc[-1] / C.iloc[-20:].min() - 1) * 100  # 從近20日最低收盤算起的漲幅（調整後價格）
# 產業別與股本（只是標示，不影響分級）
ind = pd.read_csv("data/industry.csv", dtype={"code": str, "ind": str}).set_index("code") if os.path.exists("data/industry.csv") else pd.DataFrame(columns=["ind_name", "cap"])
grp = []
if len(ind):
    gi = d.assign(g=d.index.map(ind.ind_name)).dropna(subset=["g"])
    base_rate = max(1e-9, float((gi.pk | gi.hot).mean()))
    for g, x in gi.groupby("g"):
        ne, nh = int(x.pk.sum()), int((x.hot & ~x.pk).sum())
        lift = (ne + nh) / len(x) / base_rate  # 這個產業的活躍比例是整體的幾倍
        if ne + nh >= 3 and lift >= 1.5: grp.append([g, ne, nh, int(len(x)), round(lift, 1)])
    grp.sort(key=lambda r: (-r[4], -r[1], r[0]))
# 撿跌深名單（2026/11/20 使用者定義，research/panic_wave.py）：以「這一波」為準，找被打下來的前波強勢股，不限市值。
#   這一波的起點：近60個交易日等權指數最高的那天
#   前波強勢：起點之前60個交易日的漲幅在分析範圍前30%，而且那段期間漲停過
#   誤殺：從起點前5天起算的自己最高收盤，到今天跌了 ≥ 20%
_ewv0 = ewi.values; _wpk = int(np.argmax(_ewv0[-61:])) + len(_ewv0) - 61
wave_start = days[_wpk].strftime("%Y-%m-%d")
_pre_ret = (C.iloc[_wpk] / C.iloc[_wpk - 60] - 1).reindex(d.index) * 100  # 這一波之前60日的漲幅
_pre = _pre_ret.rank(pct=True)
_dw = (C.iloc[-1] / C.iloc[max(_wpk - 5, 0):].max() - 1).reindex(d.index) * 100  # 這一波自己的跌幅
# 大跌期間的小反彈（research/panic_rebound_strength.py）：近30日指數最高點之後，指數漲最多的連續3個交易日，每檔在那3天漲多少
_ewv = ewi.values; _pk = int(np.argmax(_ewv[-31:])) + len(_ewv) - 31; _lo = min(_pk + 5, len(_ewv) - 2)
_w3 = (ewi / ewi.shift(3) - 1).iloc[_lo:len(_ewv) - 1]
if len(_w3) and _w3.max() > 0:
    _e = days.get_loc(_w3.idxmax()); _s = _e - 3
    bounce = ((C.iloc[_e] / C.iloc[_s] - 1) * 100).reindex(d.index)
    bounce_meta = {"s": days[_s].strftime("%m/%d"), "e": days[_e].strftime("%m/%d"), "idx": round(float(_w3.max() * 100), 1)}
else:
    bounce, bounce_meta = pd.Series(dtype=float), None
_lim = (f["ch"].iloc[_wpk - 60:_wpk + 1] >= 9.5).any()
_mv = (ind["cap"] * P["rawC"].iloc[-1].reindex(ind.index) / 10) if len(ind) else pd.Series(dtype=float)  # 市值（億）＝股本×股價÷面額10元
# 11/20 改用每檔自己的漲勢與高點（research/panic_own_high.py；使用者指出安碁資訊這類恐慌中逆勢大漲的股票被漏掉）：
#   自己這段漲勢＝近20日最高收盤 ÷ 前100～20日最低收盤 − 1，排分析範圍前30%，且近60日漲停過；從自己近20日最高收盤跌 ≥ 20%
_hi20 = C.iloc[-20:].max()
_run = (_hi20 / C.iloc[-100:-19].min() - 1).reindex(d.index) * 100
_runr = _run.rank(pct=True)
_lim60 = (f["ch"].iloc[-60:] >= 9.5).any()
_dw = ((C.iloc[-1] / _hi20 - 1) * 100).reindex(d.index)
theme = {c for c in d.index if _runr.get(c, 0) > 0.7 and bool(_lim60.get(c, False)) and _dw.get(c, 0) <= -20}
top20 = set(_run.reindex(sorted(theme)).dropna().sort_values(ascending=False).index[:20])  # 依自己這段漲勢排序取前20（research/panic_own_high.py）
# ---- 撿跌深名單紀錄（只增不改）：上市空頭家數近10天最高 ≥ 750 的每一天，記下全部跌深股（離20日高 ≥ 20%），ts=1 是網站名單 ----
# 研究只有 4～7 次歷史樣本，所以從 2026/11/19 起每天記下來，之後用真實結果檢驗。
# ---- 再發動（research/relaunch.py）：前10日內漲停過、今天前曾從近20日最高拉回 ≥15%、今天又漲 ≥9%、成交額 ≥2,000萬 ----
# 整體不比今日發動好（事先標準未過），但恐慌期間（上市空頭家數近10日最高 ≥750）歷史上7次恐慌全為正（探索），所以只在恐慌期間顯示。
# 2026/11/21 修正（外部檢查指出）：舊寫法「前5日最低 ≤ 前20日最高×0.85」沒要求高點在低點之前，一路漲也會被算進去。
# 改成：前 5 日內某天的收盤，相對「到那天為止的近 20 日最高」跌了 ≥ 15%（research/panic_recheck.py 的修正定義）。
_dd = C.iloc[-40:] / C.iloc[-40:].rolling(20, min_periods=10).max() - 1; _pull = _dd.iloc[-6:-1].min() <= -0.15
_last = S[S.date == days[-1]].set_index("code")
relaunch = {c for c, r in _last.iterrows() if r.base and r.ch >= 9 and r.a1 >= 20 and r.snp <= 10 and bool(_pull.get(c, False))}
# ---- 模型評分（凍結的 normal_v1，tools/train_model.py；research/normal_model.py）：只是標示，前瞻驗證中 ----
# 每天替今日發動評分（高＝2023 分數前20%門檻以上、低＝後20%以下），記到 data/model_log.csv（只增不改）。
mscore = {}
try:
    import pickle
    from model_features import make_context, features_for
    _mdl = pickle.load(open("data/model/normal_v1.pkl", "rb")); _mcfg = json.load(open("data/model/normal_v1.json", encoding="utf-8"))
    _today_pk = S[(S.date == days[-1]) & S.pk].code.tolist()
    if _today_pk:
        _ctx = make_context(P, f, S, bear)
        _X = pd.DataFrame([features_for(_ctx, days[-1], c) for c in _today_pk]).reindex(columns=_mdl["feats"]).replace([np.inf, -np.inf], np.nan)
        for c, v in zip(_today_pk, _mdl["model"].predict(_X)):
            mscore[c] = (float(v), "高" if v >= _mcfg["hi"] else ("低" if v <= _mcfg["lo"] else "中"))
except Exception as _e:
    print("model scoring skipped:", _e)
ML = "data/model_log.csv"
_ml = pd.read_csv(ML, dtype={"code": str}) if os.path.exists(ML) else None
if mscore and (_ml is None or days[-1].strftime("%Y-%m-%d") not in set(_ml.date)):
    _new = pd.DataFrame([{"date": days[-1].strftime("%Y-%m-%d"), "ver": "normal_v1", "code": c, "name": name.get(c, ""), "score": round(v, 3), "band": b,
                          "panic": int(int(bear.iloc[-10:].max()) >= 750)} for c, (v, b) in sorted(mscore.items())])
    pd.concat([_ml, _new], ignore_index=True).to_csv(ML, index=False) if _ml is not None else _new.to_csv(ML, index=False)

# ---- 「趨勢＋營收」綜合分數（composite_score.py；research/deep_normal.py）：防守標籤，< 0.5 標「趨勢營收弱」 ----
# ---- 新高突破（new_high.py；research/offense.py H5，驗證中）：最近 10 個交易日、在分析範圍內 ----
nhev = {}
try:
    import new_high as _nh
    _ev = _nh.events(10); _bs = set(zip(S[S.base].date, S[S.base].code))
    for r in _ev.sort_values("date").itertuples():
        if (r.date, r.code) in _bs:
            nhev[r.code] = (r.date, float(r.close), round(float(r.ch), 2), int(int(bear.loc[:r.date].iloc[-10:].max()) >= 750))
except Exception as _e:
    print("new-high skipped:", _e)
NH_LOG = "data/newhigh_log.csv"
_nl = pd.read_csv(NH_LOG, dtype={"code": str}) if os.path.exists(NH_LOG) else None
_nt = [(c, v) for c, v in nhev.items() if v[0] == days[-1]]
if _nt and (_nl is None or days[-1].strftime("%Y-%m-%d") not in set(_nl.date)):
    _new = pd.DataFrame([{"date": days[-1].strftime("%Y-%m-%d"), "code": c, "name": name.get(c, ""), "close": v[1], "ch": v[2], "panic": v[3]} for c, v in _nt])
    pd.concat([_nl, _new], ignore_index=True).to_csv(NH_LOG, index=False) if _nl is not None else _new.to_csv(NH_LOG, index=False)

cscore = {}
try:
    import composite_score as _cs
    _pk_today = sorted(set(S[(S.date == days[-1]) & S.pk].code) | set(nhev))
    if _pk_today: cscore = _cs.scores(P, f, S, days[-1], _pk_today)
except Exception as _e:
    print("composite scoring skipped:", _e)
CL = "data/score_log.csv"
_cl = pd.read_csv(CL, dtype={"code": str}) if os.path.exists(CL) else None
if cscore and set(cscore) & set(S[(S.date == days[-1]) & S.pk].code) and (_cl is None or days[-1].strftime("%Y-%m-%d") not in set(_cl.date)):
    _pkset = set(S[(S.date == days[-1]) & S.pk].code)  # 只記今日發動（新高突破另有紀錄）
    _new = pd.DataFrame([{"date": days[-1].strftime("%Y-%m-%d"), "code": c, "name": name.get(c, ""), "score": round(v, 4), "weak": int(v < _cs.WEAK),
                          "panic": int(int(bear.iloc[-10:].max()) >= 750)} for c, v in sorted(cscore.items()) if c in _pkset])
    pd.concat([_cl, _new], ignore_index=True).to_csv(CL, index=False) if _cl is not None else _new.to_csv(CL, index=False)

# ---- 每月前20（research/top20_goal.py 的基準規則；research/top20_exact.py 回測）----
# 每月 11 日後第一個交易日收盤後，分析範圍內「趨勢＋營收」分數最高的 20 檔，隔天開盤買；下次換股日還在前 20 的續抱。
# 名單第一次算出（而且當天融資資料已更新）就記到 data/top20_log.csv，只增不改；之後的報酬才是真正的前瞻驗證。
T20_LOG = "data/top20_log.csv"; m20 = {}; m20_meta = None
try:
    import composite_score as _cs2
    _dsr = pd.Series(days); _fdm = _dsr[_dsr.dt.day >= 11].groupby(_dsr.dt.to_period("M")).first()
    _form = _fdm.iloc[-1]; _fs = _form.strftime("%Y-%m-%d")
    _t20 = pd.read_csv(T20_LOG, dtype={"code": str}) if os.path.exists(T20_LOG) else None
    import glob as _g2; _chfs = sorted(_g2.glob("data/chips/*.csv*"), key=lambda x: (os.path.basename(x)[:4], x.endswith(".csv"))); _chf = _chfs[-1] if _chfs else None
    _chip_ok = _chf is not None and pd.Timestamp(pd.read_csv(_chf, usecols=["date"]).date.max()) >= _form
    if _t20 is not None and _fs in set(_t20.date):
        _cur = _t20[_t20.date == _fs].sort_values("rank"); _logged = str(_cur.logged.iloc[0])
    else:
        _bs = S[(S.date == _form) & S.base].code.tolist()
        _sc = _cs2.scores(P, f, S, _form, _bs)
        _top = sorted(_sc.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
        _logged = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=8)).strftime("%Y-%m-%d")
        _cur = pd.DataFrame([{"date": _fs, "rank": i + 1, "code": c, "name": name.get(c, ""), "score": round(v, 4), "logged": _logged} for i, (c, v) in enumerate(_top)])
        if _chip_ok and len(_cur) == 20:
            pd.concat([_t20, _cur], ignore_index=True).to_csv(T20_LOG, index=False) if _t20 is not None else _cur.to_csv(T20_LOG, index=False)
    _ie = days.get_loc(_form) + 1; _rets = []
    for c, rk in zip(_cur.code, _cur["rank"]):
        m20[c] = {"m20": int(rk)}
        if _ie < len(days) and c in C.columns:
            _o, _h, _l = O.iloc[_ie][c], H.iloc[_ie][c], L.iloc[_ie][c]; _pc = C.iloc[_ie - 1][c]
            if not (_o == _o and _o > 0) or (_o == _h == _l and _pc == _pc and _o / _pc - 1 >= LOCK): m20[c]["m20x"] = 1; continue
            _v = C[c].iloc[_ie:].dropna()
            if len(_v): _r = (float(_v.iloc[-1]) / float(_o) - 1) * 100; m20[c]["m20r"] = round(_r, 1); _rets.append(_r)
    _b = None
    if _ie < len(days) and os.path.exists("data/etf.csv"):
        _e = pd.read_csv("data/etf.csv", dtype={"code": str}); _e = _e[_e.code == "0050"].assign(date=lambda z: pd.to_datetime(z.date)).drop_duplicates("date", keep="last").set_index("date").sort_index()
        _aj = pd.read_csv("data/adjust.csv", dtype={"code": str}); _aj = _aj[_aj.code == "0050"].assign(date=lambda z: pd.to_datetime(z.date))
        _d0 = days[_ie]
        if _d0 in _e.index and _e.open.get(_d0, np.nan) > 0:
            _div = float(_aj[(_aj.date > _d0) & (_aj.date <= days[-1])].ratio.prod()) if len(_aj) else 1.0
            _b = round((float(_e.close.loc[:days[-1]].iloc[-1]) / float(_e.open[_d0]) / _div - 1) * 100, 1)
    _nx = (_form + pd.offsets.MonthBegin(1)).replace(day=11)
    while _nx.weekday() >= 5: _nx += pd.Timedelta(days=1)
    m20_meta = {"form": _form.strftime("%m/%d"), "entry": days[_ie].strftime("%m/%d") if _ie < len(days) else None, "n": len(_rets), "nx": int(sum(1 for v in m20.values() if v.get("m20x"))),
                "avg": round(float(np.mean(_rets)), 1) if _rets else None, "b": _b, "next": _nx.strftime("%m/%d"), "logged": _logged, "late": _logged > _fs}
    # 歷次名單（data/top20_log.csv）：每期 換股日隔天開盤 → 下次換股日隔天開盤（進行中的到最新收盤），扣 1.1% × 換股比例；0050 同段含息
    _hist = []; _h3 = []; _sels = []
    def _calm(codes, i0):  # 穩健版：前 20 名依換股日（i0 − 1）收盤前 20 日波動由低到高排（第 24 輪，使用者 11/23 選定）
        cs_ = [c for c in codes if c in C.columns]; w_ = C[cs_].iloc[max(0, i0 - 21):i0].pct_change(fill_method=None)
        v_ = w_.std().where(w_.notna().sum() >= 15); rk_ = {c: k for k, c in enumerate(codes)}
        # 11/25 使用者要求：換股日收盤實際股價 ≥ 500 元（一張太貴買不起）直接跳過、排到最後（回測影響很小：research/results/README.md「千元股」）
        _px = P["rawC"].iloc[i0 - 1] if i0 - 1 < len(P["rawC"]) else P["rawC"].iloc[-1]
        return sorted(cs_, key=lambda c: (1 if float(_px.get(c, 0) or 0) >= 500 else 0, v_.get(c) if v_.get(c) == v_.get(c) else 9e9, rk_[c]))
    if os.path.exists(T20_LOG):
        _tl = pd.read_csv(T20_LOG, dtype={"code": str}); _fl = sorted(_tl.date.unique()); _prev = set()
        _e5 = pd.read_csv("data/etf.csv", dtype={"code": str}) if os.path.exists("data/etf.csv") else None
        if _e5 is not None: _e5 = _e5[_e5.code == "0050"].assign(date=lambda z: pd.to_datetime(z.date)).drop_duplicates("date", keep="last").set_index("date").sort_index()
        _aj5 = pd.read_csv("data/adjust.csv", dtype={"code": str}); _aj5 = _aj5[_aj5.code == "0050"].assign(date=lambda z: pd.to_datetime(z.date))
        # 只買 3 檔：上期持有的還在前 20 續抱，不足的由前 20 名裡波動最低、開盤買得到的遞補
        for _k, _f0 in enumerate(_fl):
            _t0 = pd.Timestamp(_f0)
            if _t0 not in days or days.get_loc(_t0) + 1 >= len(days): continue
            _i0 = days.get_loc(_t0) + 1; _i1 = None
            if _k + 1 < len(_fl) and pd.Timestamp(_fl[_k + 1]) in days and days.get_loc(pd.Timestamp(_fl[_k + 1])) + 1 < len(days): _i1 = days.get_loc(pd.Timestamp(_fl[_k + 1])) + 1
            _codes = _tl[_tl.date == _f0].code.tolist(); _rr = []
            for c in _codes:
                if c not in C.columns: continue
                _o = O.iloc[_i0][c]; _pc = C.iloc[_i0 - 1][c]
                if c not in _prev and (not (_o == _o and _o > 0) or (_o == H.iloc[_i0][c] == L.iloc[_i0][c] and _pc == _pc and _o / _pc - 1 >= LOCK)): continue
                _x = O.iloc[_i1][c] if _i1 is not None and O.iloc[_i1][c] == O.iloc[_i1][c] else C[c].iloc[:(_i1 if _i1 is not None else len(days))].dropna().iloc[-1]
                _rr.append((float(_x) / float(_o if _o == _o and _o > 0 else _pc) - 1) * 100)
            _keep3 = [c for c in _h3 if c in _codes]; _new3 = []
            for c in _calm(_codes, _i0):
                if len(_keep3) + len(_new3) >= 3: break
                if c in _keep3 or c not in C.columns: continue
                _o3 = O.iloc[_i0][c]; _pc3 = C.iloc[_i0 - 1][c]
                if not (_o3 == _o3 and _o3 > 0) or (_o3 == H.iloc[_i0][c] == L.iloc[_i0][c] and _pc3 == _pc3 and _o3 / _pc3 - 1 >= LOCK): continue
                _new3.append(c)
            _sel3 = _keep3 + _new3; _rr3 = []
            for c in _sel3:
                _o3 = O.iloc[_i0][c]; _pc3 = C.iloc[_i0 - 1][c]
                _x3 = O.iloc[_i1][c] if _i1 is not None and O.iloc[_i1][c] == O.iloc[_i1][c] else C[c].iloc[:(_i1 if _i1 is not None else len(days))].dropna().iloc[-1]
                _rr3.append((float(_x3) / float(_o3 if _o3 == _o3 and _o3 > 0 else _pc3) - 1) * 100)
            _to3 = 1.0 if not _h3 else len(_new3) / 3; _h3 = _sel3; _sels.append(_sel3)
            _to = 1.0 if not _prev else len(set(_codes) - _prev) / 20; _prev = set(_codes)
            _bb = None
            if _e5 is not None and days[_i0] in _e5.index:
                _xe = days[_i1] if _i1 is not None else days[-1]
                _px1 = float(_e5.open.get(_xe, np.nan)) if _i1 is not None else float(_e5.close.loc[:_xe].iloc[-1])
                _dv = float(_aj5[(_aj5.date > days[_i0]) & (_aj5.date <= _xe)].ratio.prod()) if len(_aj5) else 1.0
                if _px1 == _px1: _bb = round((_px1 / float(_e5.open[days[_i0]]) / _dv - 1) * 100, 1)
            _hist.append({"f": _t0.strftime("%y/%m/%d"), "r": round(float(np.mean(_rr)) - 1.1 * _to, 1) if _rr else None, "b": _bb, "done": _i1 is not None,
                          "r3": round(float(np.mean(_rr3)) - 1.1 * _to3, 1) if _rr3 else None, "s3": _sel3,
                          "late": str(_tl[_tl.date == _f0].logged.iloc[0]) > _f0})
    _s3 = None
    for h in _hist:
        _sx = h.pop("s3", None)
        if h["f"] == _form.strftime("%y/%m/%d"): _s3 = _sx
    if _s3 is None:  # 換股日當晚（還沒到買進日）：上期 3 檔還在前 20 的續抱，其餘依名次補（開盤漲停買不到就再往下補）
        _cc = _cur.code.tolist(); _s3 = [c for c in _h3 if c in _cc]; _s3 += [c for c in _calm(_cc, days.get_loc(_form) + 1) if c not in _s3][:3 - len(_s3)]
    for c in _s3:
        if c in m20: m20[c]["m3"] = 1
    # 換股提醒：上一期的 3 檔 → 這一期的 3 檔
    _incur = bool(_hist) and _hist[-1]["f"] == _form.strftime("%y/%m/%d")
    _prev3 = (_sels[-2] if len(_sels) >= 2 else []) if _incur else (_sels[-1] if _sels else [])
    _nm = lambda c: f"{name.get(c, c)}（{c}）"
    m20_meta["chg"] = {"sell": [_nm(c) for c in _prev3 if c not in _s3], "buy": [_nm(c) for c in _s3 if c not in _prev3], "keep": [_nm(c) for c in _s3 if c in _prev3]}
    _r3 = [m20[c]["m20r"] for c in _s3 if c in m20 and "m20r" in m20[c]]
    m20_meta["avg3"] = round(float(np.mean(_r3)), 1) if _r3 else None
    m20_meta["hist"] = _hist
    _ok = [h for h in _hist if h["r"] is not None and h["b"] is not None and not h["late"]]
    m20_meta["fwd"] = {"n": len(_ok), "r": round((np.prod([1 + h["r"] / 100 for h in _ok]) - 1) * 100, 1), "b": round((np.prod([1 + h["b"] / 100 for h in _ok]) - 1) * 100, 1)} if _ok else None
    _ok3 = [h for h in _ok if h.get("r3") is not None]
    m20_meta["fwd3"] = {"n": len(_ok3), "r": round((np.prod([1 + h["r3"] / 100 for h in _ok3]) - 1) * 100, 1), "b": round((np.prod([1 + h["b"] / 100 for h in _ok3]) - 1) * 100, 1)} if _ok3 else None
except Exception as _e:
    print("top20 skipped:", _e)

RL_LOG = "data/relaunch_log.csv"  # 每天全部記下（不論恐慌與否），之後驗證；只增不改
_rl = pd.read_csv(RL_LOG, dtype={"code": str}) if os.path.exists(RL_LOG) else None
_rlpmax = int(bear.iloc[-10:].max())
if _rl is None or days[-1].strftime("%Y-%m-%d") not in set(_rl.date):
    _new = pd.DataFrame([{"date": days[-1].strftime("%Y-%m-%d"), "code": c, "name": name.get(c, ""), "close": round(float(P["rawC"].iloc[-1][c]), 2),
                          "ch": round(float(_last.at[c, "ch"]), 2), "panic": int(_rlpmax >= 750), "pmax": _rlpmax} for c in sorted(relaunch)],
                        columns=["date", "code", "name", "close", "ch", "panic", "pmax"])
    pd.concat([_rl, _new], ignore_index=True).to_csv(RL_LOG, index=False) if _rl is not None else _new.to_csv(RL_LOG, index=False)

PANIC_LOG = "data/panic_log.csv"
_pmax = int(bear.iloc[-10:].max()); _today = days[-1].strftime("%Y-%m-%d")
_plog = pd.read_csv(PANIC_LOG, dtype={"code": str}) if os.path.exists(PANIC_LOG) else None
if _pmax >= 750 and (_plog is None or _today not in set(_plog.date)):
    _rows = [{"date": _today, "code": c, "close": round(float(P["rawC"].iloc[-1][c]), 2), "d20": round(float(d20[c]), 1),
              "mv": None if _mv.get(c, np.nan) != _mv.get(c, np.nan) else round(float(_mv[c]), 1), "ts": int(c in theme), "top": int(c in top20),
              "bear": int(bear.iloc[-1]), "pmax": _pmax} for c in d.index if d20.get(c, 0) <= -20 or c in theme]
    _new = pd.DataFrame(_rows, columns=["date", "code", "close", "d20", "mv", "ts", "top", "bear", "pmax"])
    pd.concat([_plog, _new], ignore_index=True).to_csv(PANIC_LOG, index=False) if _plog is not None else _new.to_csv(PANIC_LOG, index=False)

# ---- 月營收年增率（只是標示；research/panic_revenue.py 驗證過，加進撿跌深條件沒有明顯幫助）----
import glob as _glob
_rf = sorted(_glob.glob("data/revenue/*.csv*"))[-2:]
_rv = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in _rf], ignore_index=True) if _rf else pd.DataFrame(columns=["ym", "code", "yoy"])
_rv = _rv.dropna(subset=["yoy"]).sort_values("ym")
rev_last = _rv.groupby("code").tail(1).set_index("code")  # 每檔最近一個已公布月份
rev_3 = _rv.groupby("code").tail(3).groupby("code").yoy.mean()
# 營收創 12 個月新高（第 32 輪「營收強」＝年增 ≥ 20% 且創新高；至少 6 個月可比）
rev_hi = set()
if "rev" in _rv.columns:
    for _c, _g in _rv.dropna(subset=["rev"]).sort_values("ym").groupby("code"):
        _p = _g.rev.iloc[-13:-1]
        if len(_p) >= 6 and _g.rev.iloc[-1] >= _p.max(): rev_hi.add(_c)
rev_ym = _rv.ym.max() if len(_rv) else None

# ---- 族群（官方產業）走勢（research/group_trend.py）：同產業其他股票等權指數，近5日漲且站上10日線＝已反彈；在10日線下＝仍弱 ----
# 恐慌期間，族群仍弱時的今日發動（7次恐慌全部較好）比族群已反彈的好很多，所以恐慌期間標示出來。
_inb = S[S.base & S.date.isin(days[-16:])].assign(v=True).pivot(index="date", columns="code", values="v").reindex(index=days[-16:], columns=C.columns).fillna(False).astype(bool)
_r1 = (C.iloc[-17:].pct_change(fill_method=None).iloc[1:]).where(_inb)
_g = pd.Series(ind["ind_name"]).reindex(C.columns) if len(ind) else pd.Series(dtype=object)
gstate = {}
for _gn, _cs in _g.dropna().groupby(_g.dropna()).groups.items():
    _s, _n = _r1[_cs].sum(axis=1, min_count=1), _r1[_cs].notna().sum(axis=1)
    for c in _cs:
        if c not in d.index: continue
        _m = ((_s - _r1[c].fillna(0)) / (_n - _r1[c].notna().astype(int)).replace(0, np.nan)).fillna(0)
        _ix = (1 + _m).cumprod(); _ma = _ix.rolling(10).mean()
        gstate[c] = 2 if (_ix.iloc[-1] > _ix.iloc[-6] and _ix.iloc[-1] > _ma.iloc[-1]) else (1 if _ix.iloc[-1] < _ma.iloc[-1] else 0)

# ---- 借券賣出餘額比（第 33 輪另報：今日發動借券比前 20% 的，比其他今日發動差 1～4 個百分點，五段中四段）----
sbr = {}
try:
    _sf = sorted(_glob.glob("data/sbl/*.csv.gz"))[-2:]
    if _sf:
        _sb = pd.concat([pd.read_csv(f, dtype={"code": str}, usecols=["date", "code", "sbl_bal"]) for f in _sf], ignore_index=True); _sb["date"] = pd.to_datetime(_sb.date)
        _ld = _sb.date.max()
        if (pd.Timestamp(days[-1]) - _ld).days <= 7:
            _b = _sb[_sb.date == _ld].drop_duplicates("code").set_index("code").sbl_bal
            _sh = (P["A"] / P["rawC"]).iloc[-20:].mean()
            _r = (_b / _sh.reindex(_b.index)).replace([np.inf, -np.inf], np.nan).dropna(); _r = _r[_r.index.isin(d.index)]
            sbr = {c: int(round(v * 100)) for c, v in _r.rank(pct=True).items()}
except Exception as _e:
    print("sbl skipped:", _e)

# ---- 可轉債觀察名單（使用者原本的可轉債選股法；11/23 起，還沒回測，歷史名單回補中）----
# 每檔取最近發行、仍有餘額的一檔；轉換價用發行時的價格（除權息後可能已往下調）。
# 放量站上轉換價＝收盤從轉換價以下站上、當天成交值 ≥ 前 20 日平均 2 倍（近 30 個交易日內）；之後每天收盤都沒跌回＝守住天數（三日法則：≥ 3 天才算確認）。
cbinfo = {}
try:
    _cbf = sorted(_glob.glob("data/cb/issbd5_*.csv*"))
    if _cbf:
        _cb = pd.read_csv(_cbf[-1], dtype=str)
        _cb["cp"] = pd.to_numeric(_cb["Conversion/ExchangePriceAtIssuance"], errors="coerce"); _cb["iss"] = pd.to_numeric(_cb.IssueAmount, errors="coerce"); _cb["out"] = pd.to_numeric(_cb.OutstandingAmount, errors="coerce")
        _cb = _cb[(_cb.cp > 0) & (_cb.out > 0) & ~_cb.ShortName.astype(str).str.contains("私") & (_cb.OfferingMethod.astype(str) != "8")].sort_values("IssueDate")  # 私募債不在市場交易，不列
        _rc, _am = P["rawC"], P["A"]; _todays = pd.Timestamp(days[-1])
        for _code, _g in _cb.groupby("IssuerCode"):
            if _code not in _rc.columns: continue
            _z = _g.iloc[-1]; _cp = float(_z.cp); _s = _rc[_code].dropna()
            if len(_s) < 45: continue
            _px = float(_s.iloc[-1]); _a = _am[_code].reindex(_s.index); _a20 = _a.shift(1).rolling(20, min_periods=15).mean()
            _info = {"cb": str(_z.ShortName).strip(), "cbp": round(_cp, 2), "cbd": round((_px / _cp - 1) * 100, 1), "cb20": round((_px / float(_s.iloc[-21]) - 1) * 100, 1),
                     "cbc": int(round((1 - float(_z.out) / float(_z.iss)) * 100)) if _z.iss > 0 else None}
            _st = pd.to_datetime(str(_z["Conversion/ExchangePeriodStartDate"]), format="%Y%m%d", errors="coerce")
            if _st == _st and _st > _todays: _info["cbs"] = _st.strftime("%m/%d")
            _isd = pd.to_datetime(str(_z.IssueDate), format="%Y%m%d", errors="coerce")  # 第 50 輪 H4：發行後 60 個交易日內，比一般股票好（但沒贏 0050）
            if _isd == _isd:
                _nd = int((pd.DatetimeIndex(days) > _isd).sum())
                if _nd <= 60: _info["cbn"] = _nd
            _up = (_s > _cp).values
            for _i in range(len(_s) - 1, max(len(_s) - 30, 1) - 1, -1):
                if _up[_i] and not _up[_i - 1]:
                    if _a.iloc[_i] >= 2 * _a20.iloc[_i]:
                        _after = _up[_i + 1:]; _info["cbb"] = _s.index[_i].strftime("%m/%d"); _info["cbk"] = int(len(_after)) if _after.all() else -1
                    break
            cbinfo[_code] = _info
except Exception as _e:
    print("cb list skipped:", _e)

# 封單過強（第 37 輪：今日發動收盤鎖死、封單 ÷ 成交量在近一年上市鎖死漲停的前 1/3，之後明顯較差；只有上市有資料）
lockx = {}
try:
    _bf = sorted(_glob.glob("data/bidvol/*.csv.gz"))[-2:]
    if _bf:
        _bv = pd.concat([pd.read_csv(f_, dtype={"code": str}) for f_ in _bf], ignore_index=True); _bv["date"] = pd.to_datetime(_bv.date)
        _bv = _bv[(_bv.market == "TWSE") & (_bv.chg / (_bv.close - _bv.chg) >= 0.09)].copy()
        _bv["lk"] = (_bv.bid_px.round(2) == _bv.close.round(2)) & ~(_bv.ask_vol.fillna(0) > 0); _bv["L"] = np.where(_bv.lk, _bv.bid_vol / (_bv.vol / 1000).replace(0, np.nan), 0.0)
        _h = _bv[(_bv.date > pd.Timestamp(days[-1]) - pd.Timedelta(days=365)) & (_bv.L > 0)]
        if len(_h) > 100 and (_bv.date == pd.Timestamp(days[-1])).any():
            _th = float(_h.L.quantile(2 / 3)); _t = _bv[(_bv.date == pd.Timestamp(days[-1])) & (_bv.L >= _th)]
            lockx = {c: round(float(v), 2) for c, v in zip(_t.code, _t.L)}
except Exception as _e:
    print("lock skipped:", _e)
_u10 = (P["C"].pct_change(fill_method=None).iloc[-10:] >= 0.09).sum()
try:  # 盤中通知名單（data/notify_flags.json）：分數 ≥ 0.8、營收年增 ≥ 20% 且創 12 個月新高、借券不多、前 10 天沒漲停過
    import composite_score as _cs3
    _base3 = S[(S.date == days[-1]) & S.base].code.tolist(); _all3 = _cs3.scores(P, f, S, days[-1], _base3)
    _fl = {c: name.get(c, "") for c, v in _all3.items() if v >= 0.8 and c in rev_last.index and float(rev_last.at[c, "yoy"]) >= 20 and c in rev_hi and sbr.get(c, 0) < 80 and int(_u10.get(c, 0)) == 0}
    _hold3 = {c: name.get(c, "") for c, v in m20.items() if v.get("m3")}  # 月選本期買：盤中碰跌停就提醒（資訊，不是賣出訊號）
    json.dump({"date": days[-1].strftime("%Y-%m-%d"), "codes": _fl, "hold": _hold3}, open("data/notify_flags.json", "w", encoding="utf-8"), ensure_ascii=False)
except Exception as _e:
    print("notify flags skipped:", _e)  # 盤中：今天 ≥9% 且這個是 0 ＝第一根漲停（今日發動）
data = []
for code, r in d.iterrows():
    x = {"t": code}
    for k in cols: x[k] = r[k] if k in ("n", "mk") else (round(float(r[k]), 2) if r[k] == r[k] else None)
    x["lu"] = int(x["lu"] or 0); x["amt"] = int(x["amt"] or 0); x["sn"] = int(x["sn"]); x["nw"] = bool(r.nw); x["e"] = bool(r.pk); x["u5"] = bool(r.up5 == 1); x["h"] = bool(r.hot)
    x["ru"] = round(float(ru.get(code, np.nan)), 1) if ru.get(code, np.nan) == ru.get(code, np.nan) else None
    if code in recent_hist: x["hs"] = recent_hist[code]
    if d20.get(code) == d20.get(code): x["d20"] = round(float(d20[code]), 1)
    if code in top20: x["ts"] = 1
    if code in relaunch: x["rl"] = 1
    if code in mscore: x["ms"] = mscore[code][1]
    if code in cscore: x["cs"] = round(cscore[code], 2)
    if code in m20: x.update(m20[code])
    if code in cbinfo: x.update(cbinfo[code])
    if code in sbr: x["sbr"] = sbr[code]
    x["u10"] = int(_u10.get(code, 0))
    if code in lockx: x["lkx"] = lockx[code]
    if code in nhev:
        _d, _c0, _, _pp = nhev[code]; x["nh"] = _d.strftime("%m/%d"); x["nhp"] = _pp
        if _c0 > 0 and code in P["rawC"].columns and P["rawC"][code].iloc[-1] == P["rawC"][code].iloc[-1]: x["nhr"] = round((float(P["rawC"][code].iloc[-1]) / _c0 - 1) * 100, 1)
    if gstate.get(code): x["gw"] = gstate[code]
    if _run.get(code) == _run.get(code): x["pr"] = round(float(_run[code]), 1)
    if _dw.get(code) == _dw.get(code): x["dw"] = round(float(_dw[code]), 1)
    if bounce.get(code) == bounce.get(code): x["bo"] = round(float(bounce[code]), 1)
    if code in rev_last.index:
        x["ry"] = round(float(rev_last.at[code, "yoy"]), 1); x["rm"] = int(rev_last.at[code, "ym"][5:])
        if code in rev_3.index: x["ry3"] = round(float(rev_3[code]), 1)
        if code in rev_hi: x["rh"] = 1
    if bool(above10.get(code, False)) and x["ch"] > -9.4: x["s10"] = 1
    if dk is not None and dk.get(code) == dk.get(code): x["dk"] = round(float(dk[code]), 1)
    if code in disp: x["dp"] = disp[code]
    if code in active_stop: x["stp"] = active_stop[code]
    if code in chip_info: x.update(chip_info[code])
    if code in ind.index:
        x["g"] = ind.at[code, "ind_name"]
        if ind.at[code, "cap"] == ind.at[code, "cap"]: x["cap"] = round(float(ind.at[code, "cap"]), 1)
    if True:  # 每檔都附近30日走勢（所有分頁都會顯示）
        x["sp"] = [round(float(v), 2) for v in C[code].iloc[-30:].tolist() if v == v]
    data.append(x)
allc = pd.DataFrame({"r5": f["r5"].loc[last], "a": (C.loc[last] > f["ma20"].loc[last])}).dropna()
# ---- 撿跌深真實紀錄（data/panic_log.csv）vs 正2：每個名單日，隔天開盤等權買進前20名（top=1；11/19 只有舊定義全名單 ts=1），看至今／5日／20日 ----
def _etf(code):
    if not os.path.exists("data/etf.csv"): return None, None
    e = pd.read_csv("data/etf.csv"); e = e[e.code == code].copy(); e["date"] = pd.to_datetime(e.date)
    e = e.set_index("date").reindex(days)[["open", "close"]].ffill(limit=10)
    r = e.close.pct_change(); fac = pd.Series(1.0, index=days)
    for dd_ in r.index[r.abs() > 0.35]:  # 分割（例：00631L 2026/3/31）
        fac[days < dd_] *= e.close[dd_] / e.close.shift(1)[dd_]
    return e.open * fac, e.close * fac
_o2, _c2 = _etf("00631L")
ptrack = []
if os.path.exists(PANIC_LOG):
    _pl = pd.read_csv(PANIC_LOG, dtype={"code": str})
    for _d, _g in _pl.groupby("date"):
        _t = pd.Timestamp(_d)
        if _t not in days: continue
        _i = days.get_loc(_t)
        _sel = _g[_g.top == 1].code.tolist() if "top" in _g and (_g.top == 1).any() else _g[_g.ts == 1].code.tolist()
        _row = {"d": _d, "n": len(_sel), "old": not ("top" in _g and (_g.top == 1).any()), "held": max(0, len(days) - 1 - _i)}
        if _i + 1 < len(days) and _sel:
            _o = O.iloc[_i + 1].reindex(_sel)
            def _ret(k):  # 買進後第 k 個交易日收盤；k=None → 最新
                j = len(days) - 1 if k is None else _i + k
                if j >= len(days): return None
                v = (C.iloc[j].reindex(_sel) / _o - 1).dropna()
                return round(float(v.mean() * 100), 2) if len(v) else None
            def _ret2(k):
                if _o2 is None: return None
                j = len(days) - 1 if k is None else _i + k
                if j >= len(days) or _o2.iloc[_i + 1] != _o2.iloc[_i + 1]: return None
                return round(float((_c2.iloc[j] / _o2.iloc[_i + 1] - 1) * 100), 2)
            _row.update({"now": _ret(None), "d5": _ret(5), "d20": _ret(20), "e_now": _ret2(None), "e5": _ret2(5), "e20": _ret2(20)})
        ptrack.append(_row)

# ---- 這一波恐慌「第一天」與「第二次碰到」選出的撿跌深前 3 名（第 43、44 輪：回測是在這兩個時點選才有效；事後補算） ----
pwave = None
try:
    _bv = bear.values; _N = len(days); _t0 = None
    for _t in range(max(_N - 60, 21), _N):
        if _bv[_t] >= 750 and (_bv[_t - 20:_t] < 750).all(): _t0 = _t
    if _t0 is not None:
        _s2, _bl = None, 0
        for _t in range(_t0 + 1, min(_t0 + 41, _N)):
            if _bv[_t] < 750: _bl += 1
            else:
                if _bl >= 3: _s2 = _t; break
                _bl = 0
        _o50, _c50 = _etf("0050")
        def _w2top(j, k=3):
            bs = set(S[(S.date == days[j]) & S.base].code); Cj = C.iloc[:j + 1]
            hi = Cj.iloc[-20:].max(); rv = (hi / Cj.iloc[-100:-19].min() - 1); rv = rv[rv.index.isin(bs)].dropna()
            rr = rv.rank(pct=True); l60 = (f["ch"].iloc[j - 59:j + 1] >= 9.5).any(); dw = Cj.iloc[-1] / hi - 1
            on = O.iloc[j + 1] if j + 1 < _N else pd.Series(dtype=float)
            cs = [c for c in rv.sort_values(ascending=False).index if rr[c] > 0.7 and bool(l60.get(c, False)) and dw.get(c, 0) <= -0.2 and (j + 1 >= _N or on.get(c, 0) > 0)]
            return cs[:k]
        def _pack(j, lab):
            cs = _w2top(j); r = {"lab": lab, "d": days[j].strftime("%m/%d"), "s": [[c, str(name.get(c, c))] for c in cs]}
            if j + 1 < _N and cs:
                rs = [(float(C.iloc[-1][c]) / float(O.iloc[j + 1][c]) - 1) * 100 for c in cs if O.iloc[j + 1][c] > 0]
                for (c, _n), x in zip(r["s"], rs): pass
                r["r"] = round(float(np.mean(rs)), 1) if rs else None
                r["each"] = [round((float(C.iloc[-1][c]) / float(O.iloc[j + 1][c]) - 1) * 100, 1) for c in cs]
                if _o50 is not None and _o50.iloc[j + 1] == _o50.iloc[j + 1]: r["b"] = round(float((_c50.iloc[-1] / _o50.iloc[j + 1] - 1) * 100), 1)
            return r
        pwave = [_pack(_t0, "第一天")] + ([_pack(_s2, "第二次碰到 750")] if _s2 is not None else [])
except Exception as _e:
    print("pwave skipped:", _e)

# ---- 今日發動「營收強＋分數高」往前記錄（11/24 起，只增不改；不再回測，用新資料驗證）----
# 每天收盤記下全部今日發動與是否符合（分數 ≥ 0.8、營收年增 ≥ 20% 且創 12 個月新高、借券餘額比不在前 20%）；
# 報酬：隔天開盤買、抱 20 個交易日（未滿到最新收盤），扣 0.6%；0050 同段。
GS_LOG = "data/gs_log.csv"; pkf = None
try:
    _gl = pd.read_csv(GS_LOG, dtype={"code": str}) if os.path.exists(GS_LOG) else None
    _tdk = days[-1].strftime("%Y-%m-%d"); _pkx = [x for x in data if x.get("e")]
    if _pkx and (_gl is None or _tdk not in set(_gl.date)):
        _pan = int(int(bear.iloc[-10:].max()) >= 750)
        _new = pd.DataFrame([{"date": _tdk, "code": x["t"], "name": x["n"], "gs": int((x.get("cs") or 0) >= 0.8 and (x.get("ry") or 0) >= 20 and bool(x.get("rh")) and x.get("sbr", 0) < 80),
                              "cs": x.get("cs"), "ry": x.get("ry"), "rh": int(bool(x.get("rh"))), "sbr": x.get("sbr"), "panic": _pan} for x in _pkx])
        _gl = pd.concat([_gl, _new], ignore_index=True) if _gl is not None else _new; _gl.to_csv(GS_LOG, index=False)
    if _gl is not None and len(_gl):
        _o50, _c50 = _etf("0050"); _pos = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(days)}; _N = len(days); _rows = []
        for r in _gl.itertuples():
            i = _pos.get(r.date)
            if i is None or i + 1 >= _N or r.code not in C.columns: continue
            o = O.iloc[i + 1][r.code]; x = min(i + 20, _N - 1); cx = C.iloc[x][r.code]
            if not (o > 0 and cx == cx): continue
            _rows.append({"gs": r.gs, "panic": r.panic, "r": (cx / o - 1) * 100 - 0.6, "b": (_c50.iloc[x] / _o50.iloc[i + 1] - 1) * 100, "done": int(i + 20 <= _N - 1)})
        _R = pd.DataFrame(_rows, columns=["gs", "panic", "r", "b", "done"])
        pkf = {"since": str(_gl.date.min())[5:].replace("-", "/"), "days": int(_gl.date.nunique()), "lg": int((_gl.gs == 1).sum()), "lo": int((_gl.gs == 0).sum())}
        if len(_R):
            for k, g in (("gs", _R[_R.gs == 1]), ("other", _R[_R.gs == 0])):
                pkf[k] = {"n": int(len(g)), "done": int(g.done.sum()), "r": round(float(g.r.mean()), 2) if len(g) else None, "ex": round(float((g.r - g.b).mean()), 2) if len(g) else None,
                          "win": int(round((g.r > g.b).mean() * 100)) if len(g) else None}
except Exception as _e:
    print("gs log skipped:", _e)

# 可轉債代收公告後 10 天（第 57 輪往後記錄，11/25 起）：公告日隔天開盤買、抱 10 個交易日收盤，扣 0.6%；0050 同一天開盤買、扣 0.18%。
# 去重同 research/cb_holdout.py：抓得到「第 N 次」照次數，抓不到的同公司 90 天內一次；同公司 30 天內只留一筆。只記錄，不提醒（2006–2010 驗證 t 1.71 未過 2.0）。
cbf = None
try:
    if os.path.exists("data/cb_live.csv"):
        _k = pd.read_csv("data/cb_live.csv", dtype=str); _k["d"] = pd.to_datetime(_k.date.map(lambda s: f"{int(s.split('/')[0]) + 1911}-{s.split('/')[1]}-{s.split('/')[2]}" if "/" in s else s), errors="coerce")
        _k = _k.dropna(subset=["d"]); _k = _k[_k.d >= "2026-11-25"].sort_values("d"); _k["n"] = _k.subject.str.replace(r"\s+", "", regex=True).str.extract(r"第([一二三四五六七八九十\d]+)次")[0]
        _a = _k[_k.n.notna()].drop_duplicates(["code", "n"]); _rw, _ls = [], {}
        for _d, _c, _nm in zip(_k[_k.n.isna()].d, _k[_k.n.isna()].code, _k[_k.n.isna()].name):
            if _c in _ls and (_d - _ls[_c]).days < 90: continue
            _ls[_c] = _d; _rw.append((_d, _c, _nm))
        _ev = pd.concat([_a[["d", "code", "name"]], pd.DataFrame(_rw, columns=["d", "code", "name"])]).sort_values("d"); _kp, _lc = [], {}
        for _d, _c, _nm in zip(_ev.d, _ev.code, _ev.name):
            if _c in _lc and (_d - _lc[_c]).days < 30: continue
            _lc[_c] = _d; _kp.append((_d, _c, _nm))
        _o50, _c50 = _etf("0050"); _N = len(days); _rs = []
        for _d, _c, _nm in _kp:
            i = int(days.searchsorted(_d)); ie = i + 1
            if ie >= _N or _c not in C.columns or not (O.iloc[ie][_c] > 0): _rs.append({"d": _d.strftime("%m/%d"), "c": _c, "n": _nm}); continue
            x = min(ie + 9, _N - 1); r = (C.iloc[x][_c] / O.iloc[ie][_c] - 1) * 100 - 0.6; bb = (_c50.iloc[x] / _o50.iloc[ie] - 1) * 100 - 0.18
            _rs.append({"d": _d.strftime("%m/%d"), "c": _c, "n": _nm, "r": round(float(r), 1), "x": round(float(r - bb), 1), "done": int(ie + 9 <= _N - 1)})
        _dn = [z for z in _rs if z.get("done")]
        cbf = {"n": len(_rs), "done": len(_dn), "ex": round(float(np.mean([z["x"] for z in _dn])), 2) if _dn else None,
               "win": int(round(np.mean([z["x"] > 0 for z in _dn]) * 100)) if _dn else None, "last": _rs[-5:]}
except Exception as _e:
    print("cb forward skipped:", _e)

meta = {"date": last.strftime("%Y/%m/%d"), "m5": round(float(allc.r5.median()), 2),
        "above": int(round(allc.a.mean() * 100)), "fwd": FORWARD_START, "ver": RULE_VERSION, "hold": HOLD,
        "hv": round(float(d[d.rc >= 0.9].v.median()), 1), "hrg": round(float(d[d.rc >= 0.9].rg.median()), 1), "sum": summary, "holds": holds, "weak": bool(weak.iloc[-1]), "chip": chip_meta, "grp": grp[:8], "drop": drop_meta, "bear": bear_meta, "rvm": rev_ym, "ptrack": ptrack, "bounce": bounce_meta, "wave": wave_start, "pwave": pwave, "pkf": pkf, "cbf": cbf, "m20": m20_meta,
        "built": (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=8)).strftime("%m/%d %H:%M")}
html = open("site_template.html", encoding="utf-8").read()
J = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":"))
html = html.replace("/*DATA*/[]", J(data)).replace("/*META*/{}", J(meta)).replace("/*TRACK*/[]", J(track))
os.makedirs("docs", exist_ok=True); open("docs/index.html", "w", encoding="utf-8").write(html)
print({k: meta[k] for k in ("date", "ver", "above")}, len(data), "stocks,", int(d.pk.sum()), "picks,", len(track), "tracked days,", len(html) // 1024, "KB")
print(json.dumps(summary, ensure_ascii=False))

# ---- Telegram 通知（11/23 起；只在 GitHub Actions 有設定 token 時才發，同一件事只發一次）----
try:
    import sys as _sys; _sys.path.insert(0, "tools"); import notify as _nt
    _td = days[-1]; _tds = _td.strftime("%m/%d"); _url = "https://dingtingdong.github.io/surge-data/"
    if m20_meta and m20_meta.get("chg") and m20_meta.get("form") == _tds and not m20_meta.get("entry"):
        _c = m20_meta["chg"]
        _msg = f"【月選換股】{_tds} 收盤選出，明天開盤：" + "".join(f"\n{k} " + "、".join(v) for k, v in (("賣", _c["sell"]), ("買", _c["buy"]), ("續抱", _c["keep"])) if v)
        _nt.send(f"m20-{_td:%Y-%m-%d}", _msg + "\n（開盤前掛限價單；開盤漲停買不到就換候補下一檔）\n" + _url)
    # 月選週報（11/25 使用者要求）：每週最後一個交易日收盤後（週五；週五休市就看下週一前最後一天）傳一次
    if days[-1].weekday() == 4:
        _h3w = [x for x in data if x.get("m3")]
        if _h3w and m20_meta.get("entry"):
            _ln = "、".join(f"{x['n']} {('+' if (x.get('m20r') or 0) >= 0 else '')}{x.get('m20r')}%" for x in _h3w)
            _f3 = m20_meta.get("fwd3"); _rec = f"\n正式紀錄 {_f3['n']} 期累計 {_f3['r']:+.1f}%，0050 {_f3['b']:+.1f}%。" if _f3 else "\n正式紀錄從 12/11 換股那期開始累計。"
            _nt.send(f"weekly-{_td:%Y-%m-%d}", f"【月選週報】{_tds}\n本期（{m20_meta.get('entry')} 進場）：{_ln}\n3 檔平均 {m20_meta.get('avg3'):+.1f}%，同期 0050 {m20_meta.get('b'):+.1f}%。{_rec}\n下次換股：{m20_meta.get('next')} 收盤後出名單。\n" + _url)
    # 月選本期買的「爆雷」資訊提醒（11/24 使用者要求；不是賣出訊號：回測裡大跌時賣掉換 0050 反而較差）
    _h3 = {c: v for c, v in m20.items() if v.get("m3")}
    if _h3 and os.path.exists("data/disposal.csv"):
        _dp = pd.read_csv("data/disposal.csv", dtype={"code": str}); _dp = _dp[(_dp.announced == _td.strftime("%Y-%m-%d")) & _dp.code.isin(list(_h3))]
        for _r in _dp.itertuples():
            _nt.send(f"hold-dp-{_r.code}-{_r.announced}", f"【持股注意】月選持股 {_r.name}（{_r.code}）今天被公告{_r.measure}（{_r.start}～{_r.end}），處置期間要預收款券、流動性變差。\n這是資訊提醒，不是賣出訊號；請自己判斷是不是公司出問題。\n" + _url)
    for _x in data:
        if _x.get("t") in _h3 and (_x.get("ch") or 0) <= -9.5:
            _nt.send(f"hold-ld-{_x['t']}-{_td:%Y-%m-%d}", f"【持股注意】月選持股 {_x['n']}（{_x['t']}）今天收跌停 {_x.get('ch')}%。\n回測裡月中停損沒有幫助，照規則抱到換股日；但如果是公司爆雷（財報、掏空、違約），請先查公開資訊觀測站重大訊息。\n" + _url)
    _gs = [_x for _x in data if _x.get("e") and (_x.get("cs") or 0) >= 0.8 and (_x.get("ry") or 0) >= 20 and _x.get("rh") and _x.get("sbr", 0) < 80]
    if _gs and os.environ.get("NOTIFY_PK") == "1":  # 11/24 使用者決定：今日發動還沒達標，預設不通知（設 NOTIFY_PK=1 才恢復）
        _nt.send(f"pk-{_td:%Y-%m-%d}", f"【今日發動・營收強＋分數高】{_tds}：" + "、".join(f"{_x['n']}（{_x['t']}）" for _x in _gs) + "\n回測五段都贏 0050（每筆 +1.7%），但還沒達標，只當參考。\n" + _url)
    # 撿跌深（第 43～45 輪）：上市空頭家數「第一次」到 750（前 20 日都在 750 下）那天收盤後通知前 3；第 19 個交易日晚上提醒隔天收盤賣（回測抱到第 20 天收盤）
    _t0w = globals().get("_t0")
    if _t0w is not None and pwave:
        if _t0w == len(days) - 1:
            _nm = "、".join(f"{n_}（{c_}）" for c_, n_ in pwave[0]["s"])
            _nt.send(f"pick3-buy-{_td:%Y-%m-%d}", f"【撿跌深】{_tds} 上市空頭家數第一次到 {int(bear.iloc[-1])}（≥750）。前 3：{_nm}\n明天開盤買（開盤鎖漲停買不到就換撿跌深下一名），抱 20 個交易日。\n回測 2014–2026 這個時點 11 次都贏全部股票（平均多約 14%～20%），但挑法是看過這些恐慌才定的，用 2006–2019 驗證每次只多 2～5%；但單檔約 6 成賺錢、持有中最大回落平均 −11%，次數少，不要重壓。\n" + _url)
        if _t0w + 19 == len(days) - 1:
            _nm = "、".join(n_ for c_, n_ in pwave[0]["s"])
            _nt.send(f"pick3-sell-{days[_t0w]:%Y-%m-%d}", f"【撿跌深】{days[_t0w]:%m/%d} 選的前 3（{_nm}）明天是第 20 個交易日，回測是在第 20 天收盤賣出。\n" + _url)
    if len(bear) >= 2 and int(bear.iloc[-1]) >= 800 and int(bear.iloc[-2]) < 800:
        _nt.send(f"panic-{_td:%Y-%m-%d}", f"【大盤】{_tds} 上市空頭家數 {int(bear.iloc[-1])}，第一次衝上 800（深度恐慌）。月選照常抱到換股日；撿跌深分頁已出現。\n" + _url)
except Exception as _e:
    print("notify skipped:", type(_e).__name__)
