"""找真正的「今日發動」：哪個條件最早抓到一波大漲，又不會每天列出一大堆。

全部在跑之前寫下（2026/11/17 14:20），看到結果後不修改條件與挑選規則。

候選條件（當天收盤後判斷；量比 = 當天成交額 ÷ 前20個交易日平均成交額，不含當天）：
  E1 量比 ≥ 3 且 當天漲 ≥ 5%
  E2 量比 ≥ 3 且 當天漲 ≥ 9%
  E3 量比 ≥ 2 且 當天漲 ≥ 9%
  E4 當天漲 ≥ 9%，且前10個交易日都沒有漲 ≥ 9%（第一根漲停）
  E5 E1 且 收盤距近20日最低收盤 < 20%（還在低檔）
對照：仍在動、舊版精選（v1.2 的 pk）
E 系列的範圍：當天成交額 ≥ 2,000 萬、前20日平均成交額 ≥ 300 萬、近60日至少55天有行情。
價格用除權息／減資調整後的價格；漲幅用調整後收盤。

一波大漲：同 detection.py（起漲日為近20日最低收盤，之後20個交易日內最高收盤 ≥ +50%）。
評估：
  發現：每波第一次被抓到是第幾天、抓到時已漲多少、隔天開盤買到頂點還剩多少。
  誤報：「新訊號」= 該股前20個交易日都沒出現過同一訊號。每天平均幾檔新訊號；
        新訊號後隔天開盤買，之後20個交易日內最高收盤 ≥ +20% 的比例（hit20），
        10個交易日內最低收盤 ≤ −10% 的比例（drop10），第5、10日收盤報酬中位數（扣1.1%）。
期間：2020–2023 挑選，2024–2026/11 驗證。

挑選規則：在 2020–2023 年，E1–E5 中「每天新訊號 ≤ 15 檔」且「hit20 ≥ 仍在動的 hit20」者，
選抓到天數中位數最小的；同天數選「還剩」中位數較大的。都不符合 → 選 hit20 最高的，並註明未達標。
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radar_core import load_prices, panels, features, tiers

RUN, WIN, LOOK, COST = 0.5, 20, 20, 1.1
p = load_prices()
P = panels(p)
C, O, A = P["C"], P["O"], P["A"]
f = features(P)
S = tiers(f, P["MK"])
days = C.index; N = len(days)
TRAIN_END = pd.Timestamp("2023-12-31")

r1 = C.pct_change(fill_method=None) * 100
a20 = A.shift(1).rolling(20, min_periods=20).mean()
vr = A / a20
n60 = C.rolling(60, min_periods=60).count()
ru = (C / C.rolling(20, min_periods=20).min() - 1) * 100
lu = r1 >= 9
prior_lu = lu.shift(1).rolling(10, min_periods=1).max().fillna(0).astype(bool)
base = (A >= 2e7) & (a20 >= 3e6) & (n60 >= 55)
sig = {
    "E1": base & (vr >= 3) & (r1 >= 5),
    "E2": base & (vr >= 3) & (r1 >= 9),
    "E3": base & (vr >= 2) & (r1 >= 9),
    "E4": base & lu & ~prior_lu,
    "E5": base & (vr >= 3) & (r1 >= 5) & (ru < 20),
}
for k, col in (("仍在動", "hot"), ("舊版精選", "pk_v1")):
    m = S[S[col]].assign(v=True).pivot(index="date", columns="code", values="v")
    sig[k] = m.reindex(index=days, columns=C.columns).fillna(False).astype(bool)

Cv, Ov = C.values, O.values
rmin = C.rolling(LOOK, min_periods=LOOK).min().values
runs = []
for j in range(C.shape[1]):
    c = Cv[:, j]; i = LOOK
    while i < N - 1:
        if c[i] == c[i] and c[i] <= rmin[i, j] + 1e-9:
            fut = c[i + 1:i + 1 + WIN]
            if len(fut) and np.isfinite(fut).any() and np.nanmax(fut) >= c[i] * (1 + RUN):
                k = i + 1 + int(np.nanargmax(fut)); runs.append((j, i, k)); i = k + 1; continue
        i += 1
R = pd.DataFrame(runs, columns=["j", "a", "b"])
R["train"] = [days[a] <= TRAIN_END for a in R.a]

def evaluate(M):
    Mv = M.values
    det = {True: [], False: []}
    for j, a, b, tr in zip(R.j, R.a, R.b, R.train):
        hit = np.flatnonzero(Mv[a:b, j])
        if not len(hit): det[tr].append(None); continue
        t = a + int(hit[0]); L, Pk, ct = Cv[a, j], Cv[b, j], Cv[t, j]
        ot = Ov[t + 1, j] if t + 1 < N else np.nan
        det[tr].append((t - a, ct / L - 1, Pk / ot - 1 if ot == ot and ot > 0 else np.nan, (Pk - ct) / (Pk - L)))
    post = {True: [], False: []}
    for j in range(Mv.shape[1]):
        last = -10 ** 9
        for t in np.flatnonzero(Mv[:, j]):
            new = t - last > WIN; last = t
            if not new or t + 20 >= N: continue
            o = Ov[t + 1, j]
            if not o == o or o <= 0: continue
            post[days[t] <= TRAIN_END].append((np.nanmax(Cv[t + 1:t + 21, j]) / o - 1, np.nanmin(Cv[t + 1:t + 11, j]) / o - 1,
                                               (Cv[t + 5, j] / o - 1) * 100 - COST, (Cv[t + 10, j] / o - 1) * 100 - COST))
    out = {}
    for tr, name in ((True, "2020-2023"), (False, "2024-2026")):
        d = [x for x in det[tr] if x is not None]; D = np.array(d) if d else np.empty((0, 4))
        Q = np.array(post[tr]) if post[tr] else np.empty((0, 4))
        nd = int(((days <= TRAIN_END) if tr else (days > TRAIN_END)).sum())
        out[name] = {
            "runs": len(det[tr]), "caught": round(len(d) / max(1, len(det[tr])) * 100, 1),
            "day_med": float(np.median(D[:, 0])) if len(D) else None,
            "done_med": round(float(np.median(D[:, 1])) * 100, 1) if len(D) else None,
            "left_open_med": round(float(np.nanmedian(D[:, 2])) * 100, 1) if len(D) else None,
            "share_left_med": round(float(np.median(D[:, 3])) * 100, 1) if len(D) else None,
            "new_per_day": round(len(Q) / nd, 1), "hit20": round(float((Q[:, 0] >= 0.2).mean()) * 100, 1) if len(Q) else None,
            "drop10": round(float((Q[:, 1] <= -0.1).mean()) * 100, 1) if len(Q) else None,
            "r5_med": round(float(np.nanmedian(Q[:, 2])), 2) if len(Q) else None,
            "r10_med": round(float(np.nanmedian(Q[:, 3])), 2) if len(Q) else None}
    return out

res = {k: evaluate(M) for k, M in sig.items()}
tr = lambda k: res[k]["2020-2023"]
ok = [k for k in ("E1", "E2", "E3", "E4", "E5") if tr(k)["new_per_day"] <= 15 and tr(k)["hit20"] >= tr("仍在動")["hit20"]]
if ok:
    pick = sorted(ok, key=lambda k: (tr(k)["day_med"], -tr(k)["left_open_med"]))[0]; note = "符合挑選條件"
else:
    pick = max(("E1", "E2", "E3", "E4", "E5"), key=lambda k: tr(k)["hit20"]); note = "沒有候選同時符合兩個門檻，改選 hit20 最高者"
out = {"pick": pick, "note": note, "qualified": ok, "results": res}
os.makedirs("research/results", exist_ok=True)
json.dump(out, open("research/results/early.json", "w"), ensure_ascii=False, indent=1)
cols = ["caught", "day_med", "done_med", "left_open_med", "share_left_med", "new_per_day", "hit20", "drop10", "r5_med", "r10_med"]
for per in ("2020-2023", "2024-2026"):
    print("==", per)
    print(pd.DataFrame({k: v[per] for k, v in res.items()}).T[cols].to_string())
print("挑選:", pick, note, "合格:", ok)
