"""今日分數與飆股偵測（第 72 輪；2026/11/27 14:05 登錄，看結果前寫下，看到結果後不修改）

起因：使用者 11/27 決定月選定型，研究轉向今日分數；第十二份覆核建議三層：偵測（不漏掉）→ 篩選 → 交易，各自評估。
第一部分 每日分數歷史（重建，快取 /tmp/model_v2/daily_v2.pkl）：
  同月選 12 個特性、等權百分位（pandas 平均名次），分析範圍同網站（20 日均額 ≥ 1,000 萬等）但排除 91xx TDR；除權息合併 older／old／根目錄。
  營收可知時間改保守：YYYY-MM 的營收從「下個月營收期限日（10 日起第一個交易日）的下一個交易日」收盤起才用（不假設任何公司提早公布）。
第二部分 偵測（2011–2026，股票範圍：普通股、非 TDR、前 20 日均額 ≥ 300 萬）：
  大漲事件：某天收盤 t，之後 20 個交易日內最高收盤 ≥ +20%（另報 +30%）；同一檔 20 個交易日內只算第一次。
  訊號（收盤後產生）：
    A 現行今日發動：當天漲 ≥ 9%（2015/6 前 ≥ 6%）、前 10 日沒有、當天成交值 ≥ 2,000 萬。
    B 同 A，不設當天成交值。
    C A ＋「第一根沒到 2,000 萬，之後 10 個交易日內出現成交值 ≥ 2,000 萬且收盤 ≥ 第一根收盤的那天」（每個第一根只算一次）。
    D 第一次進入今日分數前 20 名（前 20 個交易日都不在前 20）。
    E B 且當天今日分數排名 ≤ 100。
  指標：捕捉率（事件起點前 5 天到後 10 天內有訊號，且訊號日收盤到事件高點還有 ≥ 10%）、第一次捕捉距起點天數中位數、捕捉時剩餘漲幅中位數；
    命中率（訊號後隔天開盤買得到，之後 20 日內最高收盤 ≥ +20%／+30%）與「全部股票日」的基準比例 → 倍數（lift，處理訊號數量不同）；
    每天平均訊號數；隔天開盤買不到比例；訊號後 20 日減 0050（隔天開盤 → 第 20 天收盤，扣 0.6%）平均與中位數；最大不利變動中位數。
  （偵測只描述，不設通過門檻。）
第三部分 交易（資金限制：分 H 批、每天一批到期，到期那天開盤賣掉、用前一天收盤的訊號買最多 5 檔等權；不足 5 檔的部分放 0050；
    同一檔已在其他批持有就跳過；隔天開盤漲 ≥ 9.5%（2015/6 前 6.5%）買不到就換下一檔；依今日分數排名優先）：
    T1 今日分數前 5 名；T2 第一次進入前 20 名（同 D）；T3 前 20 名且當天第一根大漲（B）；
    T4 前 20 名且「第一根大漲（B）後 3～10 個交易日、期間收盤都不低於第一根的開盤、今天收盤 ≥ 第一根收盤」（每個第一根只算一次）。
  主要：H＝10，成本 0.6%（個股來回）與 1.1%；另報 H＝5、20。比較：每月減 0050、NW t（落後 3）、四段、2024–26、每日淨值最大回落（含 0050 同口徑）、年化、平均持有個股比例。
  判斷（H＝10，全部符合才「通過」→ 網站列為今日分數的訊號並往後驗證，不當實盤買法）：
    成本 0.6% 每月減 0050 > 0 且 t ≥ 2.9、四段都 > 0；成本 1.1% 每月減 0050 > 0 且 t ≥ 2；每日最大回落不比 0050 深 5 個百分點以上。
  累計：這輪 4 個交易規則（主要）＋ 8 個敏感度。2024–26 已被反覆研究過，不算樣本外。
結果（11/27 14:25）：
  偵測（事件 +20%：38,402 次；+30%：17,677 次；基準：任一股票日之後 20 日 ≥ +20% 的比例 8.0%、≥ +30% 3.5%）：
    捕捉率 +20%／+30%｜每天訊號｜命中 +20%（倍數）｜20 日減 0050 平均／中位
    A 現行今日發動 23.5／33.6｜9.6｜17.0%（2.11）｜−1.25／−3.73
    B 不設成交值   29.0／40.5｜11.2｜17.0%（2.12）｜−1.17／−3.55
    C 第一根量不足後放量 25.3／36.6｜10.3｜16.9%（2.10）｜−1.26／−3.71
    D 第一次進前 20   4.9／7.1｜2.1｜19.0%（2.36；+30% 2.95 倍）｜+0.72／−2.01
    E 第一根大漲且排名 ≤100  6.1／8.7｜2.4｜19.3%（2.40；+30% 3.05 倍）｜+0.16／−2.76
    捕捉時大多在起點後 2～3 天、還剩約 20～25% 漲幅。
  交易（H10，每月減 0050／t，成本 0.6｜1.1；每日回落；0050 每日回落 −34.0%、年化 15.9%）：
    T1 前 5 名 +0.41／3.55（四段都正）｜+0.18／1.59；−27.9%；平均只有 22% 資金在個股（前 5 名變動慢，已持有的跳過）
    T2 第一次進前 20 +0.27／1.55｜−0.10；T3 前 20＋第一根大漲 +0.21／2.46｜+0.09；T4 前 20＋發動後回測 +0.28／2.35｜+0.08
    → 都不通過（T1 只差成本 1.1% 那一條）。
"""
import glob, json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, REPO); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import vs0050 as V
from vs0050 import COMP, nw_t
import robust_select as RS
CACHE = "/tmp/model_v2/daily_v2.pkl"; SEG4 = ["2011-15", "2016-19", "2020-23", "2024-26"]

def build():
    if os.path.exists(CACHE): return pd.read_pickle(CACHE)
    os.chdir(V.make_root())
    from radar_core import load_prices, panels, features, tiers
    p = load_prices(); P = panels(p); del p
    C, O, H, L, A, raw = P["C"], P["O"], P["H"], P["L"], P["A"], P["rawC"]
    f = features(P); S = tiers(f, P["MK"]); days, cols = C.index, C.columns; N, J = len(days), len(cols)
    inb = S[S.base].assign(v=True).pivot(index="date", columns="code", values="v").reindex(index=days, columns=cols).fillna(False).astype(bool)
    tdr = pd.Index(cols).str.match(r"^91\d\d$"); inb.loc[:, tdr] = False; del S, f
    X = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        X["r20"] = C / C.shift(20) - 1; X["mom6_1"] = C.shift(20) / C.shift(125) - 1
        X["d_hi250"] = C / C.rolling(250, min_periods=120).max() - 1; X["m60"] = C / C.rolling(60, min_periods=60).mean() - 1
        X["am"] = A.rolling(5, min_periods=5).mean() / A.rolling(20, min_periods=10).mean()
        chp = pd.concat([pd.read_csv(fn, dtype={"code": str}, usecols=["date", "code", "margin"]) for fn in sorted(glob.glob("data/chips/*.csv*"))], ignore_index=True)
        chp["date"] = pd.to_datetime(chp.date); chp = chp.drop_duplicates(["date", "code"], keep="last")
        mg = chp.pivot(index="date", columns="code", values="margin").reindex(index=days, columns=cols); X["mg20"] = mg / mg.shift(20) - 1; del mg, chp
        rv = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in sorted(glob.glob("data/revenue/*.csv*"))], ignore_index=True)
        ds = pd.Series(days)
        def pubday(ym):   # 下個月營收期限日（10 日起第一個交易日）的下一個交易日
            m1 = pd.Timestamp(ym + "-01") + pd.DateOffset(months=1); k = int(np.searchsorted(days.values, np.datetime64(m1 + pd.Timedelta(days=9))))
            return days[k + 1] if k + 1 < N else pd.NaT
        yms = sorted(rv.ym.unique()); pubm = {ym: pubday(ym) for ym in yms}
        piv = lambda v: rv.pivot_table(index="ym", columns="code", values=v).sort_index().reindex(columns=cols)
        yoy = piv("yoy"); rev = piv("rev").reindex(yoy.index); mom = piv("mom").reindex(yoy.index); pub = pd.Series([pubm[y] for y in yoy.index], index=yoy.index)
        Y, R_, M_ = yoy.values, rev.values, mom.values; K = len(Y)
        Y3 = np.full_like(Y, np.nan); RH = np.full_like(Y, np.nan)
        for k in range(K):
            if k >= 2: Y3[k] = np.nanmean(Y[k - 2:k + 1], axis=0)
            prev = R_[max(0, k - 12):k]
            if len(prev): cnt = np.isfinite(prev).sum(axis=0); RH[k] = np.where(cnt >= 6, R_[k] / np.nanmax(np.where(np.isfinite(prev), prev, -np.inf), axis=0) - 1, np.nan)
        pv = pub.values.astype("datetime64[ns]"); ok = ~np.isnat(pv)
        kn = np.searchsorted(np.where(ok, pv, np.datetime64("2100-01-01")), days.values, side="right") - 1
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
    prevc_raw = raw.shift(1).values
    out = {"days": days, "cols": cols, "cs": cs, "rank": rank, "C": C.values.astype(np.float64), "O": O.values.astype(np.float64), "A": A.values.astype(np.float64),
           "rawO": (O * (raw / C)).values, "rawC": raw.values, "prevc_raw": prevc_raw, "inb": inb.values, "bidx": V.bench(days).values}
    os.chdir(REPO); pd.to_pickle(out, CACHE); return out

def e50_open(days, bidx):
    from mp_exec import E50
    e = E50.reindex(days); return bidx * (e.open / e.close).where((e.open > 0) & (e.close > 0)).fillna(1.0).values

if __name__ == "__main__":
    D = build(); days, cols = D["days"], D["cols"]; N, J = len(days), len(cols)
    C, O, A, rank, inb, b = D["C"], D["O"], D["A"], D["rank"], D["inb"], D["bidx"]; bO = e50_open(days, b)
    Cf = pd.DataFrame(C).ffill(limit=10).values
    common = ~pd.Index(cols).str.match(r"^91\d\d$")
    r1 = np.vstack([np.full((1, J), np.nan), C[1:] / C[:-1] - 1])
    thr = np.where(days >= pd.Timestamp("2015-06-01"), 0.09, 0.06)[:, None]; gap = np.where(days >= pd.Timestamp("2015-06-01"), 0.095, 0.065)
    up = (r1 >= thr) & np.isfinite(r1)
    prior = pd.DataFrame(up).shift(1).rolling(10, min_periods=1).max().fillna(0).values.astype(bool)
    a20p = pd.DataFrame(A).shift(1).rolling(20, min_periods=20).mean().values
    elig = (a20p >= 3e6) & common[None, :] & np.isfinite(C)
    first = up & ~prior & elig
    sig = {"A": first & (A >= 2e7), "B": first.copy()}
    Cn = first & (A < 2e7); conf = np.zeros_like(first)
    for t, j in zip(*np.nonzero(Cn)):
        for s in range(t + 1, min(t + 11, N)):
            if A[s, j] >= 2e7 and np.isfinite(C[s, j]) and C[s, j] >= C[t, j]: conf[s, j] = True; break
    sig["C"] = sig["A"] | conf
    top20 = (rank <= 20); prior20 = pd.DataFrame(top20).shift(1).rolling(20, min_periods=1).max().fillna(0).values.astype(bool)
    sig["D"] = top20 & ~prior20; sig["E"] = first & (rank <= 100)
    t0 = int(np.searchsorted(days.values, np.datetime64("2011-01-01"))); tN = N - 21
    # 大漲事件
    fmax = np.full((N, J), np.nan)
    Cmx = pd.DataFrame(Cf[::-1]).rolling(20, min_periods=1).max().values[::-1]   # Cmx[t] = max(Cf[t..t+19])
    fmax[:-1] = Cmx[1:] / Cf[:-1] - 1            # 之後 20 日最高收盤 / 今天收盤 − 1
    res = {}
    for X_ in (0.20, 0.30):
        big = (fmax >= X_) & elig; ev = []
        for j in range(J):
            last = -99
            for t in np.nonzero(big[t0:tN, j])[0] + t0:
                if t - last >= 20: ev.append((t, j)); last = t
        res[f"事件數 +{int(X_*100)}%"] = len(ev)
        for k, M_ in sig.items():
            cap, dl, rem = 0, [], []
            for t, j in ev:
                pk = np.nanmax(Cf[t + 1:t + 21, j])
                for s in range(max(t - 5, 0), min(t + 11, N)):
                    if M_[s, j] and np.isfinite(Cf[s, j]) and pk / Cf[s, j] - 1 >= 0.10: cap += 1; dl.append(s - t); rem.append((pk / Cf[s, j] - 1) * 100); break
            res.setdefault(k, {})[f"捕捉率 +{int(X_*100)}%"] = round(cap / len(ev) * 100, 1)
            res[k][f"捕捉距起點天數中位 +{int(X_*100)}%"] = float(np.median(dl)) if dl else None
            res[k][f"捕捉時剩餘漲幅中位 +{int(X_*100)}%"] = round(float(np.median(rem)), 1) if rem else None
    # 命中率、訊號品質
    base = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        eo = np.vstack([O[1:], np.full((1, J), np.nan)])              # 隔天開盤（還原）
        okb = (D["rawO"][1:] > 0) & ~(D["rawO"][1:] / D["rawC"][:-1] - 1 >= gap[1:, None] - 1e-9); okb = np.vstack([okb, np.zeros((1, J), bool)])
        fm2 = np.full((N, J), np.nan); fm2[:-1] = Cmx[1:] / eo[:-1] - 1   # 隔天開盤買、20 日內最高收盤
        Cmn = pd.DataFrame(Cf[::-1]).rolling(20, min_periods=1).min().values[::-1]; mae = np.full((N, J), np.nan); mae[:-1] = Cmn[1:] / eo[:-1] - 1
        c20 = np.vstack([Cf[20:], np.full((20, J), np.nan)]); x20 = (c20 / eo - 1) * 100 - 0.6
        b20 = np.concatenate([(b[20:] / bO[1:N - 19]) * 100 - 100, np.full(20, np.nan)]); b20 = np.concatenate([b20[: N - 1], [np.nan]])
    rows_m = slice(t0, tN); E_ = elig[rows_m] & okb[rows_m] & np.isfinite(fm2[rows_m])
    for X_ in (0.20, 0.30): base[X_] = float((fm2[rows_m][E_] >= X_).mean())
    res["基準（全部股票日）"] = {"+20%": round(base[0.20] * 100, 2), "+30%": round(base[0.30] * 100, 2)}
    nd = tN - t0
    for k, M_ in sig.items():
        m = M_[rows_m] & elig[rows_m]; tt = m & okb[rows_m] & np.isfinite(fm2[rows_m])
        h20, h30 = float((fm2[rows_m][tt] >= 0.20).mean()), float((fm2[rows_m][tt] >= 0.30).mean())
        xx = (x20[rows_m] - b20[rows_m][:, None])[tt]
        res[k].update({"每天訊號數": round(m.sum() / nd, 2), "隔天買不到%": round(float(1 - (m & okb[rows_m]).sum() / max(m.sum(), 1)) * 100, 1),
                       "命中率 +20%": round(h20 * 100, 1), "倍數 +20%": round(h20 / base[0.20], 2), "命中率 +30%": round(h30 * 100, 1), "倍數 +30%": round(h30 / base[0.30], 2),
                       "20日減0050 平均": round(float(np.nanmean(xx)), 2), "20日減0050 中位": round(float(np.nanmedian(xx)), 2), "最大不利變動中位%": round(float(np.nanmedian(mae[rows_m][tt]) * 100), 1)})
        print(k, json.dumps(res[k], ensure_ascii=False), flush=True)
    print("事件與基準", res["事件數 +20%"], res["事件數 +30%"], res["基準（全部股票日）"], flush=True)
    # 第三部分 交易
    T4 = np.zeros_like(first)
    for t, j in zip(*np.nonzero(sig["B"])):
        o_u, c_u = O[t, j], C[t, j]
        for s in range(t + 3, min(t + 11, N)):
            seg = Cf[t + 1:s + 1, j]
            if not np.all(np.isfinite(seg)) or np.nanmin(seg) < o_u: break
            if Cf[s, j] >= c_u and rank[s, j] <= 20: T4[s, j] = True; break
    TS = {"T1 前5名": rank <= 5, "T2 第一次進前20": sig["D"], "T3 前20＋第一根大漲": top20 & sig["B"], "T4 前20＋發動後回測": T4}
    rawO, prc = D["rawO"], D["prevc_raw"]
    def trade(M_, H, cost):
        nav_t = [None] * H; port = np.zeros(N); held = {}   # 每批：{"pos": [(j, 投入值, 進場價)], "etf": 投入值, "b0": bO 進場}
        total = 1.0; nav = np.full(N, np.nan); expo = []
        tr = [{"pos": [], "etf": 1.0 / H, "e0": bO[t0]} for _ in range(H)]
        for d in range(t0, N):
            k = (d - t0) % H; T_ = tr[k]
            # 到期：開盤賣
            val = sum(v * ((O[d, j] if O[d, j] > 0 else Cf[d, j]) / px) for j, v, px in T_["pos"]) * (1 - cost / 200) + T_["etf"] * (bO[d] / T_["e0"])
            for j, _, _ in T_["pos"]: held[j] = held.get(j, 1) - 1
            held = {j: c for j, c in held.items() if c > 0}
            # 用前一天收盤的訊號買
            pos = []
            if d - 1 >= 0 and d + H <= N:
                cand = np.nonzero(M_[d - 1])[0]; cand = cand[np.argsort(rank[d - 1, cand])]
                for j in cand:
                    if len(pos) >= 5: break
                    if j in held or not (O[d, j] > 0) or not (rawO[d, j] > 0) or (prc[d, j] > 0 and rawO[d, j] / prc[d, j] - 1 >= gap[d] - 1e-9): continue
                    pos.append(j)
            each = val / 5; T_["pos"] = [(j, each * (1 - cost / 200), O[d, j]) for j in pos]
            for j in pos: held[j] = held.get(j, 0) + 1
            T_["etf"] = val - each * len(pos); T_["e0"] = bO[d]
            # 當天收盤市值
            tv = 0.0
            for q in tr: tv += sum(v * (Cf[d, j] / px if np.isfinite(Cf[d, j]) else 1.0) for j, v, px in q["pos"]) + q["etf"] * (b[d] / q["e0"])
            nav[d] = tv; expo.append(sum(len(q["pos"]) for q in tr) / (5 * H))
        s = pd.Series(nav[t0:], index=days[t0:]); bb = pd.Series(b[t0:], index=days[t0:])
        Mr = s.resample("ME").last().pct_change() * 100; Mb = bb.resample("ME").last().pct_change() * 100
        Mr.iloc[0] = (s.resample("ME").last().iloc[0] / 1.0 - 1) * 100; Mb.iloc[0] = (bb.resample("ME").last().iloc[0] / bb.iloc[0] - 1) * 100
        x = (Mr - Mb).dropna(); sg = RS.seg(x.index); v = s.values
        return {"每月減0050": round(float(x.mean()), 2), "t": round(nw_t(x, 3), 2), "四段": [round(float(x[sg == q].mean()), 2) for q in SEG4],
                "2024–26": round(float(x[x.index >= "2024-01-01"].mean()), 2), "每日回落": round(float((v / np.maximum.accumulate(v) - 1).min() * 100), 1),
                "年化%": round(float(v[-1] ** (252 / len(v)) - 1) * 100, 1), "平均持股比例%": round(float(np.mean(expo)) * 100)}
    bv = b[t0:] / b[t0]; res["0050"] = {"每日回落": round(float((bv / np.maximum.accumulate(bv) - 1).min() * 100), 1), "年化%": round(float(bv[-1] ** (252 / len(bv)) - 1) * 100, 1)}
    print("0050", res["0050"], flush=True)
    for name, M_ in TS.items():
        out = {}
        for H, cost in ((10, 0.6), (10, 1.1), (5, 0.6), (20, 0.6)):
            out[f"H{H} 成本{cost}"] = trade(M_, H, cost); print(name, f"H{H} 成本{cost}", json.dumps(out[f"H{H} 成本{cost}"], ensure_ascii=False), flush=True)
        a, c = out["H10 成本0.6"], out["H10 成本1.1"]
        ok = [a["每月減0050"] > 0 and a["t"] >= 2.9, all(v > 0 for v in a["四段"]), c["每月減0050"] > 0 and c["t"] >= 2, a["每日回落"] >= res["0050"]["每日回落"] - 5]
        out["判斷"] = {"各條件": ok, "通過": all(ok)}; res[name] = out; print(name, out["判斷"], flush=True)
    json.dump(res, open(f"{REPO}/research/results/today_v2.json", "w"), ensure_ascii=False, indent=1, default=float)
