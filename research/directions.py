"""三個新方向：能不能贏「一直抱 0050」（2011–2026）

全部在看結果之前寫下（2026/11/22 19:40），看到結果後不修改。
起因：research/vs0050.py 顯示選股訊號穩定贏「全部股票平均」，但輸 0050——差距來自 0050（台積電權重過半）大幅領先一般股票。
  所以這次不再找新的個股訊號，改測三種可能繞過這個問題的作法，每一種在文獻上都有依據：
  D1 族群動能（Moskowitz & Grinblatt 1999：產業動能解釋大部分個股動能）——強勢族群的龍頭，族群強時自然會含台積電、AI 股。
  D2 策略動能（Ehsani & Linnainmaa 2022：多數因子報酬有正自我相關）——選股贏 0050 的時期才選股，否則抱 0050。
  D3 讓贏家跑（Wilcox & Crittenden 2005；Zarattini, Pagani & Wilcox 2025：創歷史新高買進、移動停損出場，不設固定持有期）。
資料：同 research/vs0050.py（2010-01～2026-11 接成一段，0050 含息）。成本：來回 0.6%。不排除恐慌月份（真的投資人一直在場內）。
期間：2011–15、2016–19、2020–23、2024–26。三項的共同判斷（全部符合才成立）：
  ① 每月「策略報酬 − 0050」平均 > 0 且 Newey-West t ≥ 2.9；② 四段期間平均都 ≥ 0（D2 允許整段都抱 0050 時為 0），D1、D3 要 > 0。
D1：每月 11 日後第一個交易日收盤後，分析範圍（20 日均額 ≥ 1,000 萬）內的股票依官方產業別（data/industry.csv，用現在的分類）分組，
    至少 5 檔的產業才算；產業強度＝產業內股票過去 120 個交易日報酬的等權平均。取最強 3 個產業，各取 20 日均額最大的 5 檔，共 15 檔等權；
    隔天開盤買（開盤一價漲停買不到不計）、抱 20 個交易日；每月換股，成本 0.6% × 換股比例。NW 落後 1。
D2：每月同一天，先算「綜合分數前 20 檔」（12 個趨勢＋營收特性、當天分析範圍內百分位等權平均）每月扣成本後減 0050 的報酬 e；
    如果「前 2～13 個月」（跳過最近一個月，避免和還沒結束的持有期重疊）的 e 加總 > 0 → 這個月抱前 20 檔，否則抱 0050（與 0050 的差＝0）。
    從 0050 換回選股時成本算 0.6%。NW 落後 1。
D3：平常時期的新高突破（定義同 vs0050.py 的 N1），隔天開盤買；收盤比買進後最高收盤低 20% → 隔天開盤賣；不設最長持有（資料結束時還沒賣的用最後收盤計）。
    每天把所有持有中的部位等權平均成一個組合（沒有部位時抱 0050），買進日、賣出日各扣 0.3%；每月複利，減 0050 同月報酬。NW 落後 3。
另報（不判定）：三項的年化報酬與 0050 對比、D1 改成「產業內 20 日均額加權」、D2 打開選股的月份比例、D3 平均持有天數、同時持有檔數、逐筆中位數與賺錢比例。
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
from vs0050 import COMP, nw_t, per, PER
X = pd.read_pickle("/tmp/model_v2/vs0050_extra.pkl"); K = pd.read_pickle("/tmp/model_v2/vs0050.pkl")
days, cols = X["days"], X["cols"]; N, J = len(days), len(cols); bidx = X["bidx"]

def mstat(s, L, allow_zero=False):
    s = pd.Series(s).dropna(); idx = pd.DatetimeIndex(s.index)
    o = {"全期": (round(float(s.mean()), 2), round(nw_t(s, L), 2), int(len(s)))}
    for p in PER:
        q = s[per(idx) == p]; o[p] = (round(float(q.mean()), 2), round(nw_t(q, L), 2) if len(q) >= 5 else None, int(len(q)))
    ok = o["全期"][0] > 0 and (o["全期"][1] or 0) >= 2.9 and all((o[p][0] >= 0 if allow_zero else o[p][0] > 0) for p in PER)
    return o, bool(ok)
def cagr(r, b):  # r, b：每期報酬 %（同一組日期）
    yrs = (pd.Timestamp(r.index[-1]) - pd.Timestamp(r.index[0])).days / 365.25
    return round(((1 + r / 100).prod() ** (1 / yrs) - 1) * 100, 1), round(((1 + b / 100).prod() ** (1 / yrs) - 1) * 100, 1)

F = X["fd"].copy(); F = F[F.date >= "2011-01-01"]
ind = pd.read_csv(f"{REPO}/data/industry.csv", dtype={"code": str}).set_index("code")["ind_name"]; F["ind"] = F.code.map(ind)
out = {}
# ---- D1 族群動能
rows, prev, rows_w = {}, set(), {}
for d, g in F.groupby("date"):
    g = g[g.ind.notna()]; cnt = g.groupby("ind").size(); ok = cnt[cnt >= 5].index
    mom = g[g.ind.isin(ok)].groupby("ind").r120.mean().dropna().sort_values(ascending=False)
    top = list(mom.index[:3]); pk = pd.concat([g[g.ind == i].nlargest(5, "a20") for i in top])
    cur = set(pk.code); to = 1.0 if not prev else len(cur - prev) / max(len(cur), 1); prev = cur
    v = pk.dropna(subset=["y20"]); b = g.b20.iloc[0]
    if len(v) == 0 or not np.isfinite(b): continue
    rows[d] = (v.y20.mean() - 0.6 * to, b, "、".join(top)); w = v.a20 / v.a20.sum(); rows_w[d] = (float((v.y20 * w).sum()) - 0.6 * to)
D1 = pd.DataFrame(rows, index=["r", "b", "inds"]).T; D1[["r", "b"]] = D1[["r", "b"]].astype(float); x1 = D1.r - D1.b
o1, ok1 = mstat(x1, 1)
out["D1 族群動能（前3產業×各5檔龍頭）"] = {"減0050(每月)": o1, "成立": ok1, "年化 策略/0050": cagr(D1.r, D1.b),
    "另報 產業內20日均額加權 減0050": mstat(pd.Series(rows_w) - D1.b, 1)[0],
    "最常入選的產業": D1.inds.str.split("、").explode().value_counts().head(8).to_dict(),
    "最近6個月入選": {str(k.date()): v for k, v in D1.inds.tail(6).items()}}
print("\nD1", json.dumps(out["D1 族群動能（前3產業×各5檔龍頭）"], ensure_ascii=False), flush=True)
# ---- D2 策略動能
F["cs"] = np.mean([F.groupby("date")[c].rank(pct=True).fillna(0.5).values for c in COMP], axis=0)
e, prev = {}, set(); rr = {}
for d, g in F.groupby("date"):
    pk = g.nlargest(20, "cs"); cur = set(pk.code); to = 1.0 if not prev else len(cur - prev) / 20; prev = cur
    v = pk.y20.dropna(); b = g.b20.iloc[0]
    if len(v) == 0 or not np.isfinite(b): continue
    e[d] = v.mean() - 0.6 * to - b; rr[d] = (v.mean() - 0.6 * to, b)
e = pd.Series(e); RR = pd.DataFrame(rr, index=["r", "b"]).T
dd, on_prev, onlist = {}, False, {}
for i, d in enumerate(e.index):
    past = e.iloc[max(0, i - 13):max(0, i - 1)]
    on = len(past) == 12 and past.sum() > 0
    dd[d] = (e.iloc[i] - (0.6 if not on_prev else 0)) if on else 0.0
    onlist[d] = on; on_prev = on
x2 = pd.Series(dd); o2, ok2 = mstat(x2, 1, allow_zero=True)
r2 = RR.b + x2
out["D2 策略動能（選股贏0050時才選股）"] = {"減0050(每月)": o2, "成立": ok2, "年化 策略/0050": cagr(r2, RR.b),
    "選股月份比例%": {p: round(float(pd.Series(onlist)[per(pd.DatetimeIndex(list(onlist))) == p].mean() * 100), 0) for p in PER},
    "參考：一直選股 減0050": mstat(e, 1)[0], "一直選股 年化": cagr(RR.r, RR.b)}
print("\nD2", json.dumps(out["D2 策略動能（選股贏0050時才選股）"], ensure_ascii=False), flush=True)
# ---- D3 讓贏家跑
Cv, Ov, lockn = X["C"], X["O"], X["lockn"]
Cff = pd.DataFrame(Cv).ffill(limit=60).values
pos = {d: i for i, d in enumerate(days)}; cj = {c: j for j, c in enumerate(cols)}
E = K[(K.kind == "nh") & ~K.panic & (K.date >= "2011-01-01")]
daily = np.zeros(N); cntp = np.zeros(N); trades = []
for d, c in zip(E.date, E.code):
    t, j = pos[d], cj[c]
    if t + 1 >= N: continue
    o = Ov[t + 1, j]
    if not (o > 0) or lockn[t, j]: continue
    mx = -np.inf; k = t + 1; ex_k = None
    while k < N:
        ck = Cff[k, j]
        if not np.isfinite(ck): ex_k = ("c", k - 1); break      # 下市／停牌超過 60 日：用最後收盤
        mx = max(mx, ck)
        if ck <= 0.8 * mx and k + 1 < N: ex_k = ("o", k + 1); break
        k += 1
    if ex_k is None: ex_k = ("c", N - 1)
    kind, ke = ex_k
    path = []  # 每日報酬
    for q in range(t + 1, ke + 1 if kind == "c" else ke + 1):
        if q == t + 1: r = Cff[q, j] / o - 1 - 0.003
        elif kind == "o" and q == ke:
            op = Ov[q, j]; op = op if op > 0 else Cff[q, j]; r = op / Cff[q - 1, j] - 1 - 0.003
        else: r = Cff[q, j] / Cff[q - 1, j] - 1
        if kind == "c" and q == ke and ke < N - 1: r -= 0.003
        daily[q] += r; cntp[q] += 1; path.append(r)
    tot = (np.prod(1 + np.array(path)) - 1) * 100; bb = (bidx[ke if kind == "c" else ke - 1] / bidx[t] - 1) * 100
    trades.append({"date": d, "code": c, "days": len(path), "ret": tot, "b": bb, "open_end": kind == "c" and ke == N - 1})
br = np.concatenate([[0.0], bidx[1:] / bidx[:-1] - 1])
port = np.where(cntp > 0, daily / np.maximum(cntp, 1), br)
S = pd.DataFrame({"p": port, "b": br, "n": cntp}, index=days); S = S[S.index >= E.date.min()]
M = S.groupby(S.index.to_period("M")).agg(p=("p", lambda z: (np.prod(1 + z) - 1) * 100), b=("b", lambda z: (np.prod(1 + z) - 1) * 100))
M.index = M.index.to_timestamp(); x3 = M.p - M.b; o3, ok3 = mstat(x3, 3)
T = pd.DataFrame(trades)
out["D3 讓贏家跑（新高突破＋20%移動停損、不設最長持有）"] = {"減0050(每月)": o3, "成立": ok3, "年化 策略/0050": cagr(M.p, M.b),
    "四段年化 策略/0050": {p: cagr(M.p[per(M.index) == p], M.b[per(M.index) == p]) for p in PER},
    "逐筆": {"筆數": int(len(T)), "平均%": round(float(T.ret.mean()), 1), "中位數%": round(float(T.ret.median()), 1), "賺錢%": round(float((T.ret > 0).mean() * 100), 1),
            "平均持有天數": round(float(T.days.mean()), 0), "中位數持有天數": int(T.days.median()), "同期0050平均%": round(float(T.b.mean()), 1), "資料結束仍持有": int(T.open_end.sum())},
    "同時持有檔數 平均/中位數/最多": (round(float(S.n.mean()), 1), int(S.n.median()), int(S.n.max()))}
print("\nD3", json.dumps(out["D3 讓贏家跑（新高突破＋20%移動停損、不設最長持有）"], ensure_ascii=False), flush=True)
json.dump(out, open(f"{REPO}/research/results/directions.json", "w"), ensure_ascii=False, indent=1, default=str)
