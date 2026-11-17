"""（此檔評估的是 v1 的精選，欄位 pk_v1；v2.0 起「今日發動」改為 research/early.py 的 E4）
「最快發現已經發動的股票」的評估：雷達在一波漲勢的第幾天抓到、抓到後還剩多少。

定義（2026/11/17 跑之前寫下，不依結果修改）：
- 一波大漲：某天收盤是近20個交易日最低（起漲日 L），之後20個交易日內最高收盤 ≥ L × 1.5。
  頂點 = 那20天內的最高收盤日。同一檔找到一波後，從頂點隔天再往下找，不重複計算。
  價格用除權息／減資調整後的收盤。
- 抓到：起漲日（含）到頂點前一天之間，第一次進入該分級的那天。
- 抓到時已漲 = 抓到當天收盤 ÷ 起漲日收盤 − 1；
  還剩（收盤算）= 頂點收盤 ÷ 抓到當天收盤 − 1；還剩（隔天開盤買）= 頂點收盤 ÷ 隔天開盤 − 1；
  剩下比例 = （頂點 − 抓到收盤）÷（頂點 − 起漲收盤）。
- 看不到：抓到前一直不在分析範圍（20日均額 < 1,000 萬或歷史不足），或從沒進入該分級。
- 反過來看每次「新發動」（該股前20個交易日都沒進過該分級）之後的表現：隔天開盤買，
  之後10個交易日內最高收盤、最低收盤，以及第5、10個交易日收盤；扣 1.1% 成本只用在收盤報酬。
分級比較：今日發動（原「精選」）、仍在動。
結果寫到 research/results/detection.json。
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radar_core import load_prices, panels, features, tiers

RUN, WIN, LOOK, COST = 0.5, 20, 20, 1.1
p = load_prices()
P = panels(p)
C, O = P["C"], P["O"]
f = features(P)
S = tiers(f, P["MK"])
S = S[S.base]  # v1 的分析範圍
days = C.index
pos = {d: i for i, d in enumerate(days)}
flags = {k: S[S[k]].groupby("code").date.apply(lambda s: np.array(sorted(pos[d] for d in s))) for k in ("pk_v1", "hot")}
seen = S.groupby("code").date.apply(lambda s: np.array(sorted(pos[d] for d in s)))

# ---- 找出每一波大漲 ----
Cv = C.values
rmin = C.rolling(LOOK, min_periods=LOOK).min().values
runs = []
for j, code in enumerate(C.columns):
    c = Cv[:, j]; i = LOOK
    while i < len(days) - 1:
        if c[i] == c[i] and c[i] <= rmin[i, j] + 1e-9:
            fut = c[i + 1:i + 1 + WIN]
            if len(fut) and np.nanmax(fut) >= c[i] * (1 + RUN):
                k = i + 1 + int(np.nanargmax(fut))
                runs.append((code, i, k)); i = k + 1; continue
        i += 1
R = pd.DataFrame(runs, columns=["code", "a", "b"])
R["year"] = [days[a].year for a in R.a]
R["gain"] = [Cv[b, C.columns.get_loc(c)] / Cv[a, C.columns.get_loc(c)] - 1 for c, a, b in zip(R.code, R.a, R.b)]
R["len"] = R.b - R.a

def detect(tier):
    out = []
    for c, a, b in zip(R.code, R.a, R.b):
        fl = flags[tier].get(c, np.array([], int)); fl = fl[(fl >= a) & (fl < b)]
        j = C.columns.get_loc(c); L, Pk = Cv[a, j], Cv[b, j]
        if not len(fl):
            vis = seen.get(c, np.array([], int)); vis = vis[(vis >= a) & (vis < b)]
            out.append({"hit": False, "visible": bool(len(vis))}); continue
        t = int(fl[0]); ct = Cv[t, j]; ot = O.values[t + 1, j] if t + 1 < len(days) else np.nan
        out.append({"hit": True, "visible": True, "day": t - a, "len": b - a, "done": ct / L - 1, "left": Pk / ct - 1,
                    "left_open": Pk / ot - 1 if ot == ot else np.nan, "share_left": (Pk - ct) / (Pk - L)})
    return pd.DataFrame(out)

def summarize(D):
    h = D[D.hit]
    return {"runs": int(len(D)), "caught": round(float(D.hit.mean() * 100), 1),
            "invisible": round(float((~D.visible).mean() * 100), 1),
            "day_med": float(h.day.median()), "len_med": float(h.len.median()),
            "done_med": round(float(h.done.median() * 100), 1), "left_med": round(float(h.left.median() * 100), 1),
            "left_open_med": round(float(h.left_open.median() * 100), 1),
            "share_left_med": round(float(h.share_left.median() * 100), 1)}

det = {}
for tier, name in (("pk_v1", "今日發動"), ("hot", "仍在動")):
    D = detect(tier); D["year"] = R.year.values
    det[name] = {"全期": summarize(D), **{str(y): summarize(g) for y, g in D.groupby("year")}}

# ---- 每次新發動之後 ----
Hv, Lv = P["H"].values, P["L"].values
def after(tier):
    rows = []
    for c, fl in flags[tier].items():
        j = C.columns.get_loc(c); prev = -10 ** 9
        for t in fl:
            new = t - prev > WIN; prev = t
            if not new or t + 10 >= len(days): continue
            o = O.values[t + 1, j]
            if not o == o or o <= 0: continue
            seg = Cv[t + 1:t + 11, j]
            rows.append({"year": days[t].year, "gap": (o / Cv[t, j] - 1) * 100, "max10": (np.nanmax(seg) / o - 1) * 100,
                         "min10": (np.nanmin(seg) / o - 1) * 100,
                         "r5": (Cv[t + 5, j] / o - 1) * 100 - COST, "r10": (Cv[t + 10, j] / o - 1) * 100 - COST})
    A = pd.DataFrame(rows)
    s = lambda g: {"n": int(len(g)), "per_year": round(len(g) / g.year.nunique(), 0), "gap_med": round(g.gap.median(), 2),
                   "max10_med": round(g.max10.median(), 1), "min10_med": round(g.min10.median(), 1),
                   "max10_ge20": round((g.max10 >= 20).mean() * 100, 1), "min10_le-10": round((g.min10 <= -10).mean() * 100, 1),
                   "r5_med": round(g.r5.median(), 2), "r5_avg": round(g.r5.mean(), 2), "r10_med": round(g.r10.median(), 2), "r10_avg": round(g.r10.mean(), 2)}
    return {"全期": s(A), **{str(y): s(g) for y, g in A.groupby("year")}}

res = {"definition": {"run": f"起漲日為近{LOOK}日最低收盤，{WIN}日內最高收盤 ≥ +{int(RUN*100)}%"},
       "runs_total": int(len(R)), "runs_per_year": {str(k): int(v) for k, v in R.groupby("year").size().items()},
       "run_gain_med": round(float(R.gain.median() * 100), 1), "run_len_med": float(R.len.median()),
       "detection": det, "after_new_flag": {"今日發動": after("pk_v1"), "仍在動": after("hot")}}
os.makedirs("research/results", exist_ok=True)
json.dump(res, open("research/results/detection.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: res[k] for k in ("runs_total", "runs_per_year", "run_gain_med", "run_len_med")}, ensure_ascii=False))
for n in ("今日發動", "仍在動"):
    print(n, "抓到", res["detection"][n]["全期"]); print(n, "新發動後", res["after_new_flag"][n]["全期"])
