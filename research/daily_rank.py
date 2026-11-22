"""每日排名進出場（第 6 輪；2026/11/22 23:50 登錄，看結果前寫下，看到結果後不修改）

起因：第 4、5 輪都倒在「換股日晚兩天，2024–26 就輸 0050」。一個月只選一次，選的那天剛好好不好影響很大（換股時點運氣）。
作法：不再固定換股日，改成每天收盤後重算綜合分數（同月選的 12 個「趨勢＋營收」特性，當天分析範圍內百分位等權平均），每天決定進出：
  最多 20 格、每格等權（每日以等權計算組合報酬）；空格放 0050。
  進場：有空格時，買當天分數排名前 20、還沒持有、不在冷卻期的股票，名次高的先買；隔天開盤買，開盤一價漲停買不到就留空格（放 0050）。
  出場：R6a 名次掉到 40 名以外（或掉出分析範圍）→ 隔天開盤賣；
        R6b 同 R6a，另加停損：收盤比買進價低 10% → 隔天開盤賣，之後 20 個交易日不再買這檔（冷卻）。
  成本：每次賣股扣 0.6%（成本以使用者 2.8 折計），0050 進出各扣 0.1%。
資料與報酬：同 research/vs0050.py（2010-01～2026-11 接成一段），2011-01 起算；每日組合報酬按月複利，減 0050（含息）同月報酬。
判斷（全部符合才算「穩」）：① 每月減 0050 平均 > 0 且 Newey-West t（落後 1）≥ 2.9；② 四段（2011–15、2016–19、2020–23、2024–26）都 > 0；
  ③ 發現期（2020–23）以外 t ≥ 2；④ 執行晚兩天（訊號當天收盤 → 第 3 個交易日開盤才買賣）時，全期 t ≥ 2 且 2024–26 > 0；⑤ 最大回落不比 0050 深。
  累計檢驗 14 個規則，門檻維持 t ≥ 2.9。
網站：通過的規則（兩個都過用全期 t 高的）還要 2006–2009 年（data/oldest）每月減 0050 > 0，才把「月選」換成每日版；否則維持現狀。
另報：成本 1.1%、年化、每年換手率、平均持有天數、每月贏 0050 比例、16 年中贏的年數、有沒有連兩年輸、最差滾動 12 個月；
  另外核對：每日分數在每月換股日和月選的分數一致（前 20 名重疊）。
結果（11/23 00:00）：核對 24 次換股日前 20 全部 20/20 一致。兩個都不「穩」→ 網站維持現狀。
  R6a：①②③④ 過（晚兩天 +1.31，t 2.64，2024–26 +0.47），⑤ 沒過（最大回落 −36.8 vs 0050 −29.3）；全期 +1.63（t 3.24）；平均持有 12 天、每格一年換約 20 次，
       成本 1.1% 時只剩 +0.78（t 1.55）、2024–26 0.0 → 對成本太敏感。
  R6b（加停損）：+1.39（t 2.79），最大回落 −41.6 → 更差。
  解讀：每天評分消除了換股時點問題，但換手太多、回落更深；月營收公布後的月換股＋停損仍是最好的組合。
"""
import glob, json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, REPO); sys.path.insert(0, HERE)
import vs0050 as V
from vs0050 import COMP, nw_t, per, PER
CACHE = "/tmp/model_v2/daily_cs.pkl"

def build():
    if os.path.exists(CACHE): return pd.read_pickle(CACHE)
    os.chdir(V.make_root())
    from radar_core import load_prices, panels, features, tiers
    p = load_prices(); P = panels(p); del p
    C, O, H, L, A = P["C"], P["O"], P["H"], P["L"], P["A"]
    f = features(P); S = tiers(f, P["MK"]); days, cols = C.index, C.columns; N, J = len(days), len(cols)
    inb = S[S.base].assign(v=True).pivot(index="date", columns="code", values="v").reindex(index=days, columns=cols).fillna(False).astype(bool)
    prevc = C.ffill().shift(1).values
    thr = (V.LOCK - 0.03 * (days < pd.Timestamp("2015-06-01")))[:, None]
    locked = (O.values == H.values) & (H.values == L.values) & (O.values / prevc - 1 >= thr)
    bidx = V.bench(days).values
    X = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        X["r20"] = C / C.shift(20) - 1; X["mom6_1"] = C.shift(20) / C.shift(125) - 1
        X["d_hi250"] = C / C.rolling(250, min_periods=120).max() - 1; X["m60"] = C / C.rolling(60, min_periods=60).mean() - 1
        X["am"] = A.rolling(5, min_periods=5).mean() / A.rolling(20, min_periods=10).mean()
        chp = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in sorted(glob.glob("data/chips/*.csv*"))], ignore_index=True)
        chp["date"] = pd.to_datetime(chp.date); chp = chp.drop_duplicates(["date", "code"], keep="last")
        mg = chp.pivot(index="date", columns="code", values="margin").reindex(index=days, columns=cols); X["mg20"] = mg / mg.shift(20) - 1; del mg, chp
        rv = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in sorted(glob.glob("data/revenue/*.csv*"))], ignore_index=True)
        rv["pub"] = pd.to_datetime(rv.ym + "-01") + pd.DateOffset(months=1) + pd.Timedelta(days=10)
        piv = lambda v: rv.pivot_table(index="ym", columns="code", values=v).sort_index().reindex(columns=cols)
        yoy = piv("yoy"); rev = piv("rev").reindex(yoy.index); mom = piv("mom").reindex(yoy.index); pub = rv.groupby("ym").pub.first().sort_index().reindex(yoy.index)
        Y, R_, M_ = yoy.values, rev.values, mom.values; K = len(Y)
        Y3 = np.full_like(Y, np.nan); RH = np.full_like(Y, np.nan)
        for k in range(K):
            if k >= 2: Y3[k] = np.nanmean(Y[k - 2:k + 1], axis=0)
            prev = R_[max(0, k - 12):k]
            if len(prev): cnt = np.isfinite(prev).sum(axis=0); RH[k] = np.where(cnt >= 6, R_[k] / np.nanmax(np.where(np.isfinite(prev), prev, -np.inf), axis=0) - 1, np.nan)
        kn = np.searchsorted(pub.values, days.values, side="right") - 1
        pick = lambda Mx: pd.DataFrame(np.where((kn >= 0)[:, None], Mx[np.maximum(kn, 0)], np.nan), index=days, columns=cols)
        X["yoy1"] = pick(Y); X["yoy3"] = pick(Y3); X["rev_hi12"] = pick(RH); X["rev_acc"] = X["yoy1"] - X["yoy3"]; X["mom1"] = pick(M_)
        ind = pd.read_csv("data/industry.csv", dtype={"code": str}).set_index("code"); grp = pd.Series(ind["ind_name"]).reindex(cols)
        rb = C.pct_change(fill_method=None).where(inb); g20 = pd.DataFrame(np.nan, index=days, columns=cols)
        for gname, cs_ in grp.dropna().groupby(grp.dropna()).groups.items():
            ix = (1 + rb[list(cs_)].mean(axis=1).fillna(0)).cumprod(); g20[list(cs_)] = np.repeat((ix / ix.shift(20) - 1).values[:, None], len(cs_), axis=1)
        X["rs_g20"] = X["r20"] - g20
    tot = np.zeros((N, J), dtype=np.float32)
    for c in COMP:
        Z = X[c].replace([np.inf, -np.inf], np.nan).where(inb)
        tot += Z.rank(axis=1, pct=True).where(inb, np.nan).fillna(0.5).values.astype(np.float32)
    cs = np.where(inb.values, tot / len(COMP), np.nan).astype(np.float32)
    rank = pd.DataFrame(cs).rank(axis=1, ascending=False, method="first").values.astype(np.float32)
    out = {"days": days, "cols": cols, "cs": cs, "rank": rank, "C": C.ffill(limit=60).values, "O": O.values, "locked": locked, "bidx": bidx, "inb": inb.values}
    os.chdir(REPO); pd.to_pickle(out, CACHE); return out

def simulate(D, stop=False, delay=0, cost=0.6, start="2011-01-01", nslot=20, keep=40):
    days, Cf, Ov, lk, rank, bidx = D["days"], D["C"], D["O"], D["locked"], D["rank"], D["bidx"]; N = len(days)
    t0 = int(np.searchsorted(days.values, np.datetime64(start)))
    slots = [None] * nslot; cool = {}; orders = {}; daily = np.zeros(N); nexit = 0; hold_days = []
    def sched(d, kind, x):
        orders.setdefault(d, []).append((kind, x))
    pend_in, pend_out = set(), set()
    for d in range(t0, N):
        r = np.zeros(nslot); b1 = bidx[d] / bidx[d - 1] - 1
        todays = orders.pop(d, [])
        exits = {x for k, x in todays if k == "out"}; entries = [x for k, x in todays if k == "in"]
        freed = []
        for i, s in enumerate(slots):
            if s is None: r[i] = b1; continue
            j = s["j"]
            if j in exits:
                op = Ov[d, j]; op = op if op > 0 else Cf[d, j]
                r[i] = (op / Cf[d - 1, j] - 1 if np.isfinite(op) and Cf[d - 1, j] > 0 else 0.0) - cost / 100 - 0.001
                pend_out.discard(j); hold_days.append(d - s["t"]); nexit += 1; slots[i] = None; freed.append(i)
            else:
                c1, c0 = Cf[d, j], Cf[d - 1, j]
                if not np.isfinite(c1):  # 下市／停牌過久：以前一天收盤出場
                    r[i] = -cost / 100; slots[i] = None; pend_out.discard(j); nexit += 1; continue
                r[i] = c1 / c0 - 1
        for j in entries:
            pend_in.discard(j)
            if any(s is not None and s["j"] == j for s in slots): continue
            free = [i for i in freed] + [i for i, s in enumerate(slots) if s is None and i not in freed]
            if not free: continue
            o = Ov[d, j]
            if not (o > 0) or lk[d, j] or not np.isfinite(Cf[d, j]): continue
            i = free[0]
            if i in freed: freed.remove(i); r[i] = (1 + r[i]) * (Cf[d, j] / o) - 1
            else: r[i] = Cf[d, j] / o - 1 - 0.001
            slots[i] = {"j": j, "px": o, "t": d}
        daily[d] = r.mean()
        if d + 1 + delay >= N: continue
        # 收盤後決定（隔 1+delay 天開盤執行）
        ex_day = d + 1 + delay
        for s in slots:
            if s is None or s["j"] in pend_out: continue
            j = s["j"]; rk = rank[d, j]
            hit_stop = stop and np.isfinite(Cf[d, j]) and Cf[d, j] <= s["px"] * 0.9
            if not np.isfinite(rk) or rk > keep or hit_stop:
                sched(ex_day, "out", j); pend_out.add(j)
                if hit_stop: cool[j] = d + 20
        held = {s["j"] for s in slots if s is not None}
        nfree = nslot - (len(held) - len(pend_out & held)) - len(pend_in)
        if nfree > 0:
            cand = np.where(rank[d] <= nslot)[0]; cand = cand[np.argsort(rank[d, cand])]
            for j in cand:
                if nfree <= 0: break
                if j in held or j in pend_in or cool.get(j, -1) >= d: continue
                sched(ex_day, "in", j); pend_in.add(j); nfree -= 1
    s = pd.Series(daily[t0:], index=days[t0:]); bd = pd.Series(np.concatenate([[0], bidx[1:] / bidx[:-1] - 1])[t0:], index=days[t0:])
    M = pd.DataFrame({"p": (1 + s).groupby(s.index.to_period("M")).prod() * 100 - 100, "b": (1 + bd).groupby(bd.index.to_period("M")).prod() * 100 - 100})
    M.index = M.index.to_timestamp(); yrs = (days[-1] - days[t0]).days / 365.25
    return M, {"每年換手（次/格）": round(nexit / nslot / yrs, 1), "平均持有天數": round(float(np.mean(hold_days)), 0) if hold_days else None}

def judge(M):
    x, r, b = M.p - M.b, M.p, M.b; pr = per(x.index)
    s = {"全期": (round(float(x.mean()), 2), round(nw_t(x, 1), 2)), "四段": {p: round(float(x[pr == p].mean()), 2) for p in PER},
         "發現期外": (round(float(x[pr != "2020-23"].mean()), 2), round(nw_t(x[pr != "2020-23"], 1), 2))}
    yr = pd.DataFrame({"s": (1 + r / 100).groupby(x.index.year).prod(), "b": (1 + b / 100).groupby(x.index.year).prod()}); lose = (yr.s <= yr.b).values
    rel = ((1 + r / 100).rolling(12).apply(np.prod, raw=True) / (1 + b / 100).rolling(12).apply(np.prod, raw=True) - 1).dropna() * 100
    w, wb = (1 + r / 100).cumprod(), (1 + b / 100).cumprod(); yrs = len(x) / 12
    s.update({"贏0050年數": f"{int((~lose).sum())}/{len(lose)}", "連兩年輸": bool(any(lose[i] and lose[i + 1] for i in range(len(lose) - 1))),
              "每月贏0050%": round(float((x > 0).mean() * 100)), "最差滾動12月%": round(float(rel.min()), 1),
              "最大回落 策略/0050": (round(float((w / w.cummax() - 1).min() * 100), 1), round(float((wb / wb.cummax() - 1).min() * 100), 1)),
              "年化 策略/0050": (round(float(w.iloc[-1] ** (1 / yrs) - 1) * 100, 1), round(float(wb.iloc[-1] ** (1 / yrs) - 1) * 100, 1)),
              "逐年": {int(k): (round(float(v.s * 100 - 100)), round(float(v.b * 100 - 100))) for k, v in yr.iterrows()}})
    return s

if __name__ == "__main__":
    D = build()
    # 核對：每月換股日的每日分數前 20 和月選（vs0050 快照）一致
    Xv = pd.read_pickle("/tmp/model_v2/vs0050_extra.pkl"); F = Xv["fd"]; dpos = {d: i for i, d in enumerate(D["days"])}; ov = []
    for dte in sorted(F.date.unique())[-24:]:
        g = F[F.date == dte].copy(); g["cs"] = np.mean([g[c].rank(pct=True).fillna(0.5).values for c in COMP], axis=0)
        a = set(g.nlargest(20, "cs").code); i = dpos[dte]; top = D["cols"][np.where(D["rank"][i] <= 20)[0]]; ov.append(len(a & set(top)))
    print("核對 最近 24 次換股日 前20 重疊：", ov, flush=True)
    out = {"核對 前20重疊": ov}; best = None
    for lab, kw in (("R6a 前20進、跌出前40出", {}), ("R6b 再加 10% 停損", {"stop": True})):
        M, info = simulate(D, **kw); s = judge(M)
        M2, _ = simulate(D, delay=2, **kw); s2 = judge(M2); M11, _ = simulate(D, cost=1.1, **kw); s11 = judge(M11)
        c = {"①": s["全期"][0] > 0 and s["全期"][1] >= 2.9, "②": all(v > 0 for v in s["四段"].values()), "③": s["發現期外"][0] > 0 and s["發現期外"][1] >= 2,
             "④": s2["全期"][0] > 0 and s2["全期"][1] >= 2 and s2["四段"]["2024-26"] > 0, "⑤": s["最大回落 策略/0050"][0] >= s["最大回落 策略/0050"][1]}
        ok = all(c.values()); out[lab] = {"成本0.6%": s, "晚兩天執行": {k: s2[k] for k in ("全期", "四段", "發現期外")}, "成本1.1%": {k: s11[k] for k in ("全期", "四段", "年化 策略/0050")}, "各項": c, "穩": ok, **info}
        if ok and (best is None or s["全期"][1] > out[best]["成本0.6%"]["全期"][1]): best = lab
        print("==", lab, "穩" if ok else "", c, info, "\n  ", {k: v for k, v in s.items() if k != "逐年"}, "\n   晚兩天", out[lab]["晚兩天執行"], "\n   1.1%", out[lab]["成本1.1%"], flush=True)
    out["候選（需 2006–2009 也 > 0）"] = best; print("候選：", best)
    json.dump(out, open(f"{REPO}/research/results/daily_rank.json", "w"), ensure_ascii=False, indent=1, default=str)
