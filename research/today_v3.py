"""今日分數 T1 續抱版與偵測算法修正（第 73 輪；2026/11/27 15:10 登錄，看結果前寫下，看到結果後不修改）

起因：第十三份覆核（固定 34173ec）指出第 72 輪：① 「平均持股比例 22%」是格數比例，不是市值比例；② T1 到期全部賣掉、又可能馬上買回同一檔，白付成本；
  ③ 偵測沒有要求訊號在波段高點之前，捕捉率可能高估；④ 沒做「相同每日訊號數」的比較；⑤ 0050 進出沒扣成本、沒開盤價時用補值收盤賣、回落沒含起點。
資料：第 72 輪的每日分數快取（/tmp/model_v2/daily_v2.pkl），2011–2026。
基礎修正（所有版本都套用，不算新規則）：0050 進出各扣 0.09%；賣出日沒有開盤價（停牌）就延到下一個有開盤價的日子；每日淨值含起點 1；
  另報真正的個股市值占比、每年換手（賣出金額 ÷ 平均淨值）。
交易（每批最多 5 檔，依今日分數排名，隔天開盤漲 ≥9.5%／2015/6 前 6.5% 買不到就換下一檔）：
  T1-A 同第 72 輪 T1（分 10 批、抱 10 天，到期全賣再買），但套用基礎修正。
  T1-B 同 T1-A，但到期時「前一天收盤仍在前 5 名」的股票續抱（不賣、不付成本，留在同一批再抱 10 天），其餘賣掉再補。（主要）
  T1-C（探索）單一組合 5 格：持有的股票前一天收盤掉到第 20 名以外就隔天開盤賣；空格用前 5 名裡還沒持有的補；沒得補就放 0050。
  成本 0.6% 與 1.1%（個股來回）。
  判斷 T1-B（同第 72 輪標準，全部符合才「通過」→ 網站列為今日分數的訊號、往後驗證，不當實盤）：
    成本 0.6% 每月減 0050 > 0 且 NW t（落後 3）≥ 2.9、四段都 > 0；成本 1.1% > 0 且 t ≥ 2；每日最大回落不比 0050 深 5 點以上。T1-C 只描述。
偵測修正（描述性）：訊號日必須在波段高點之前，剩餘漲幅＝訊號日之後到事件窗結束的最高收盤 ÷ 訊號日收盤 − 1（≥ 10% 才算捕捉）；
  另做「相同每日訊號數」：每個版本每天最多取 2 檔（依今日分數排名，沒有排名的排後面、再依當天成交值），比捕捉率與命中率。
結果（11/27 15:40）：
  偵測修正後捕捉率幾乎不變（B +20% 29.0%、+30% 40.5%；A 23.5／33.6；D 4.9／7.1）。每天最多 2 檔時各版本很接近：A／B／C 捕捉 4.5%（+20%）、命中 18.3～18.4%；
    D 3.4%、19.3%；E 3.5%、19.6%（倍數 2.3～2.4）→ 同樣的訊號數下，先用今日分數排序後，各種偵測方式差別不大。
  交易（0050 每日回落 −34.0%）：
    T1-A 到期全賣：+0.37（t 3.25）｜成本 1.1% +0.15（t 1.29）；回落 −28.0%；個股市值占比 22%；每年換手 5.6 倍。
    T1-B 仍在前 5 續抱：+0.46（t 3.76，四段 0.14／0.35／0.92／0.54）｜成本 1.1% +0.28（t 2.27）；回落 −27.9%；個股市值占比 23%；每年換手 4.6 倍 → 四個條件都過，「通過」。
    T1-C（探索）5 格掉出前 20 才賣：+1.94（t 2.44）、2024–26 −0.18、回落 −47.2%、每年換手 23 倍；成本 1.1% +1.10（t 1.37）、回落 −54.4%。
  注意：T1-B 是第 72 輪沒通過後，依覆核建議改的版本（多重檢定）；而且約 77% 資金其實在 0050，只能列入往後驗證，不當實盤買法。
"""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, REPO); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
from today_v2 import build, e50_open, SEG4
from vs0050 import nw_t
import robust_select as RS

if __name__ == "__main__":
    D = build(); days, cols = D["days"], D["cols"]; N, J = len(days), len(cols)
    C, O, A, rank, b = D["C"], D["O"], D["A"], D["rank"], D["bidx"]; bO = e50_open(days, b); rawO, prc = D["rawO"], D["prevc_raw"]
    Cf = pd.DataFrame(C).ffill(limit=10).values; common = ~pd.Index(cols).str.match(r"^91\d\d$")
    r1 = np.vstack([np.full((1, J), np.nan), C[1:] / C[:-1] - 1])
    thr = np.where(days >= pd.Timestamp("2015-06-01"), 0.09, 0.06)[:, None]; gap = np.where(days >= pd.Timestamp("2015-06-01"), 0.095, 0.065)
    up = (r1 >= thr) & np.isfinite(r1); prior = pd.DataFrame(up).shift(1).rolling(10, min_periods=1).max().fillna(0).values.astype(bool)
    a20p = pd.DataFrame(A).shift(1).rolling(20, min_periods=20).mean().values; elig = (a20p >= 3e6) & common[None, :] & np.isfinite(C)
    first = up & ~prior & elig; sig = {"A": first & (A >= 2e7), "B": first.copy()}
    Cn = first & (A < 2e7); conf = np.zeros_like(first)
    for t, j in zip(*np.nonzero(Cn)):
        for s in range(t + 1, min(t + 11, N)):
            if A[s, j] >= 2e7 and np.isfinite(C[s, j]) and C[s, j] >= C[t, j]: conf[s, j] = True; break
    sig["C"] = sig["A"] | conf; top20 = rank <= 20; prior20 = pd.DataFrame(top20).shift(1).rolling(20, min_periods=1).max().fillna(0).values.astype(bool)
    sig["D"] = top20 & ~prior20; sig["E"] = first & (rank <= 100)
    t0 = int(np.searchsorted(days.values, np.datetime64("2011-01-01"))); tN = N - 21; res = {}
    # ---- 偵測修正 ----
    Cmx = pd.DataFrame(Cf[::-1]).rolling(20, min_periods=1).max().values[::-1]; fmax = np.full((N, J), np.nan); fmax[:-1] = Cmx[1:] / Cf[:-1] - 1
    def cap2(M_):   # 每天最多 2 檔
        out = np.zeros_like(M_)
        for d in range(N):
            js = np.nonzero(M_[d])[0]
            if len(js) <= 2: out[d, js] = True; continue
            key = np.where(np.isfinite(rank[d, js]), rank[d, js], 1e6) - np.nan_to_num(A[d, js]) * 1e-15
            out[d, js[np.argsort(key)[:2]]] = True
        return out
    eo = np.vstack([O[1:], np.full((1, J), np.nan)]); okb = np.vstack([(rawO[1:] > 0) & ~(rawO[1:] / D["rawC"][:-1] - 1 >= gap[1:, None] - 1e-9), np.zeros((1, J), bool)])
    fm2 = np.full((N, J), np.nan); fm2[:-1] = Cmx[1:] / eo[:-1] - 1
    sl = slice(t0, tN); E_ = elig[sl] & okb[sl] & np.isfinite(fm2[sl]); base = {X_: float((fm2[sl][E_] >= X_).mean()) for X_ in (0.2, 0.3)}
    for X_ in (0.20, 0.30):
        big = (fmax >= X_) & elig; ev = []
        for j in range(J):
            last = -99
            for t in np.nonzero(big[t0:tN, j])[0] + t0:
                if t - last >= 20: ev.append((t, j)); last = t
        for tag, SG in (("全部", sig), ("每天最多2檔", {k: cap2(v) for k, v in sig.items()})):
            for k, M_ in SG.items():
                cap, dl, rem = 0, [], []
                for t, j in ev:
                    w = Cf[t + 1:t + 21, j]
                    if not np.isfinite(w).any(): continue
                    tp = t + 1 + int(np.nanargmax(w))          # 波段高點那天
                    for s in range(max(t - 5, 0), min(t + 11, tp)):   # 訊號要在高點之前
                        if M_[s, j] and np.isfinite(Cf[s, j]):
                            rm = np.nanmax(Cf[s + 1:t + 21, j]) / Cf[s, j] - 1
                            if rm >= 0.10: cap += 1; dl.append(s - t); rem.append(rm * 100); break
                r = res.setdefault(f"偵測（{tag}）", {}).setdefault(k, {})
                r[f"捕捉率 +{int(X_*100)}%"] = round(cap / len(ev) * 100, 1); r[f"距起點天數中位 +{int(X_*100)}%"] = float(np.median(dl)) if dl else None
                r[f"剩餘漲幅中位 +{int(X_*100)}%"] = round(float(np.median(rem)), 1) if rem else None
                if X_ == 0.20:
                    m = M_[sl] & elig[sl]; tt = m & okb[sl] & np.isfinite(fm2[sl])
                    r["每天訊號數"] = round(m.sum() / (tN - t0), 2); r["命中率 +20%"] = round(float((fm2[sl][tt] >= 0.2).mean()) * 100, 1); r["倍數 +20%"] = round(float((fm2[sl][tt] >= 0.2).mean()) / base[0.2], 2)
                    r["命中率 +30%"] = round(float((fm2[sl][tt] >= 0.3).mean()) * 100, 1)
    for tag in ("全部", "每天最多2檔"): print(tag, json.dumps(res[f"偵測（{tag}）"], ensure_ascii=False), flush=True)
    # ---- 交易 ----
    T1 = rank <= 5; ECOST = 0.09
    def etf_after(ev, cash_in, spent):   # 0050 只對淨買賣扣 0.09%
        net = cash_in - spent
        return ev + net * (1 - ECOST / 100) if net >= 0 else ev + net / (1 - ECOST / 100)
    def can_buy(d, j): return O[d, j] > 0 and rawO[d, j] > 0 and not (prc[d, j] > 0 and rawO[d, j] / prc[d, j] - 1 >= gap[d] - 1e-9)
    def stats(nav, sv, sold):
        s = pd.Series(nav[t0:], index=days[t0:]); bb = pd.Series(b[t0:], index=days[t0:])
        me = s.resample("ME").last(); mb = bb.resample("ME").last()
        Mr = me.pct_change() * 100; Mb = mb.pct_change() * 100; Mr.iloc[0] = (me.iloc[0] - 1) * 100; Mb.iloc[0] = (mb.iloc[0] / bb.iloc[0] - 1) * 100
        x = (Mr - Mb).dropna(); sg = RS.seg(x.index); v = np.r_[1.0, s.values]
        yrs = len(s) / 252
        return {"每月減0050": round(float(x.mean()), 2), "t": round(nw_t(x, 3), 2), "四段": [round(float(x[sg == q].mean()), 2) for q in SEG4],
                "2024–26": round(float(x[x.index >= "2024-01-01"].mean()), 2), "每日回落": round(float((v / np.maximum.accumulate(v) - 1).min() * 100), 1),
                "年化%": round(float(v[-1] ** (1 / yrs) - 1) * 100, 1), "個股市值占比%": round(float(np.nanmean(sv[t0:] / nav[t0:])) * 100), "每年換手": round(float(sold / np.nanmean(nav[t0:]) / yrs), 1)}
    def tranche(H, cost, keep):
        tr = [{"pos": [], "etf": 1.0 / H, "e0": bO[t0], "due": t0 + k} for k in range(H)]; held = {}; nav = np.full(N, np.nan); sv = np.zeros(N); sold = 0.0
        for d in range(t0, N):
            for T_ in tr:
                if T_["due"] != d: continue
                keepl, val = [], 0.0
                for (j, v, px) in T_["pos"]:
                    if keep and d >= 1 and T1[d - 1, j]: keepl.append((j, v, px)); continue
                    if not (O[d, j] > 0): keepl.append((j, v, px)); continue          # 停牌：延到有開盤價
                    x = v * O[d, j] / px * (1 - cost / 200); val += x; sold += x; held[j] -= 1
                held = {j: c for j, c in held.items() if c > 0}
                ev_ = T_["etf"] * (bO[d] / T_["e0"])
                free = 5 - len(keepl)
                tot = val + ev_ + sum(v * (O[d, j] if O[d, j] > 0 else Cf[d - 1, j]) / px for j, v, px in keepl)
                each = tot / 5; new = []
                if d + 1 < N and free > 0:
                    cand = np.nonzero(T1[d - 1])[0]; cand = cand[np.argsort(rank[d - 1, cand])]
                    for j in cand:
                        if len(new) >= free: break
                        if j in held or any(j == p[0] for p in keepl) or not can_buy(d, j): continue
                        new.append(j)
                pos = list(keepl); avail = val + ev_ * (1 - ECOST / 100); spent = 0.0
                for j in new:
                    amt = min(each, avail - spent)
                    if amt <= 0: break
                    pos.append((j, amt * (1 - cost / 200), O[d, j])); spent += amt; held[j] = held.get(j, 0) + 1
                T_["pos"] = pos; T_["etf"] = etf_after(ev_, val, spent); T_["e0"] = bO[d]; T_["due"] = d + H
            tv, st = 0.0, 0.0
            for q in tr:
                x = sum(v * (Cf[d, j] / px if np.isfinite(Cf[d, j]) else 1.0) for j, v, px in q["pos"]); st += x; tv += x + q["etf"] * (b[d] / q["e0"])
            nav[d] = tv; sv[d] = st
        return stats(nav, sv, sold)
    def single(cost):   # T1-C：5 格，掉出前 20 才賣
        pos = {}; etf = 1.0; e0 = bO[t0]; nav = np.full(N, np.nan); sv = np.zeros(N); sold = 0.0
        for d in range(t0, N):
            if d >= 1:
                out = [j for j in pos if not (rank[d - 1, j] <= 20) and O[d, j] > 0]
                cash = 0.0
                for j in out: v, px = pos.pop(j); x = v * O[d, j] / px * (1 - cost / 200); cash += x; sold += x
                free = 5 - len(pos)
                cand = [j for j in np.nonzero(T1[d - 1])[0][np.argsort(rank[d - 1, np.nonzero(T1[d - 1])[0]])] if j not in pos and can_buy(d, j)][:free]
                if cand or cash > 0:
                    ev_ = etf * (bO[d] / e0); tot = cash + ev_ + sum(v * (O[d, j] if O[d, j] > 0 else Cf[d - 1, j]) / px for j, (v, px) in pos.items())
                    avail = cash + ev_ * (1 - ECOST / 100); spent = 0.0
                    for j in cand:
                        amt = min(tot / 5, avail - spent)
                        if amt <= 0: break
                        pos[j] = (amt * (1 - cost / 200), O[d, j]); spent += amt
                    etf = etf_after(ev_, cash, spent); e0 = bO[d]
            st = sum(v * (Cf[d, j] / px if np.isfinite(Cf[d, j]) else 1.0) for j, (v, px) in pos.items()); nav[d] = st + etf * (b[d] / e0); sv[d] = st
        return stats(nav, sv, sold)
    bv = np.r_[1.0, b[t0:] / b[t0]]; res["0050"] = {"每日回落": round(float((bv / np.maximum.accumulate(bv) - 1).min() * 100), 1)}
    for lab, fn in (("T1-A 到期全賣", lambda c: tranche(10, c, False)), ("T1-B 仍在前5續抱", lambda c: tranche(10, c, True)), ("T1-C 掉出前20才賣（探索）", single)):
        out = {f"成本{c}": fn(c) for c in (0.6, 1.1)}; res[lab] = out; print(lab, json.dumps(out, ensure_ascii=False), flush=True)
    a, c = res["T1-B 仍在前5續抱"]["成本0.6"], res["T1-B 仍在前5續抱"]["成本1.1"]
    ok = [a["每月減0050"] > 0 and a["t"] >= 2.9, all(v > 0 for v in a["四段"]), c["每月減0050"] > 0 and c["t"] >= 2, a["每日回落"] >= res["0050"]["每日回落"] - 5]
    res["T1-B 判斷"] = {"各條件": ok, "通過": all(ok)}; print(res["T1-B 判斷"], res["0050"])
    json.dump(res, open(f"{REPO}/research/results/today_v3.json", "w"), ensure_ascii=False, indent=1, default=float)
