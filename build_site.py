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
_hi20s = C.iloc[-21:-1].max(); _lo5s = C.iloc[-6:-1].min()
_last = S[S.date == days[-1]].set_index("code")
relaunch = {c for c, r in _last.iterrows() if r.base and r.ch >= 9 and r.a1 >= 20 and r.snp <= 10 and _lo5s.get(c, np.nan) <= _hi20s.get(c, np.nan) * 0.85}
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
    if gstate.get(code): x["gw"] = gstate[code]
    if _run.get(code) == _run.get(code): x["pr"] = round(float(_run[code]), 1)
    if _dw.get(code) == _dw.get(code): x["dw"] = round(float(_dw[code]), 1)
    if bounce.get(code) == bounce.get(code): x["bo"] = round(float(bounce[code]), 1)
    if code in rev_last.index:
        x["ry"] = round(float(rev_last.at[code, "yoy"]), 1); x["rm"] = int(rev_last.at[code, "ym"][5:])
        if code in rev_3.index: x["ry3"] = round(float(rev_3[code]), 1)
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

meta = {"date": last.strftime("%Y/%m/%d"), "m5": round(float(allc.r5.median()), 2),
        "above": int(round(allc.a.mean() * 100)), "fwd": FORWARD_START, "ver": RULE_VERSION, "hold": HOLD,
        "hv": round(float(d[d.rc >= 0.9].v.median()), 1), "hrg": round(float(d[d.rc >= 0.9].rg.median()), 1), "sum": summary, "holds": holds, "weak": bool(weak.iloc[-1]), "chip": chip_meta, "grp": grp[:8], "drop": drop_meta, "bear": bear_meta, "rvm": rev_ym, "ptrack": ptrack, "bounce": bounce_meta, "wave": wave_start,
        "built": (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=8)).strftime("%m/%d %H:%M")}
html = open("site_template.html", encoding="utf-8").read()
J = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":"))
html = html.replace("/*DATA*/[]", J(data)).replace("/*META*/{}", J(meta)).replace("/*TRACK*/[]", J(track))
os.makedirs("docs", exist_ok=True); open("docs/index.html", "w", encoding="utf-8").write(html)
print({k: meta[k] for k in ("date", "ver", "above")}, len(data), "stocks,", int(d.pk.sum()), "picks,", len(track), "tracked days,", len(html) // 1024, "KB")
print(json.dumps(summary, ensure_ascii=False))
