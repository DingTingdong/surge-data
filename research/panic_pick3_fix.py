"""第 43 輪修正重跑：開盤鎖漲停買不到＋交易成本（2026/11/24 10:20 登錄，看結果前寫下）

起因：外部覆核指出第 43 輪（panic_pick3.py）只檢查隔天有沒有開盤價，沒有排除開盤一價鎖漲停，也沒扣成本；
  第 43b 輪（panic_pick3_old.py）只扣 1.1%，沒有減全部股票平均（網站與回覆誤寫成「比全部股票多」）。
修正：
  鎖漲停＝隔天開盤＝最高＝最低，且開盤比前一天收盤（未還原價）漲 ≥ 9.4% → 買不到，依名次往下一檔遞補（前 3 名；前 20 名直接剔除）。
  成本：每筆來回 0.6%（同月選研究）；另報 1.1%。
  另報：減全部股票同期平均（可買到的股票、同樣扣成本）。
判斷：用第 43 輪同樣三個條件（①平均不比前 20 少 1 以上 ②最低點平均比隨機 3 檔中位數淺 ③最差一次比隨機最差一次中位數好），700 與 750 都過才保留網站「前3」標示；否則拿掉。
"""
import os, sys, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
exec(open("research/panic_winners.py", encoding="utf-8").read().split("out = {}")[0])
Hh, Ll = P["H"], P["L"]
H = 20; rng = np.random.default_rng(20261124); NR = 1000
out = {}
for COST in (0.6, 1.1):
    res = {}
    for th in (700, 750, 800):
        rows, rnd, nlock = [], [], 0
        for i in signals(th):
            if i + H >= N: continue
            m = inb.iloc[i] & C0.iloc[i].notna(); o = O.iloc[i + 1]; okb = o.notna()
            lk = (O.iloc[i + 1] == Hh.iloc[i + 1]) & (O.iloc[i + 1] == Ll.iloc[i + 1]) & (O.iloc[i + 1] / C0.iloc[i] - 1 >= 0.094)
            buy = okb & ~lk
            hi20 = C0.iloc[i - 19:i + 1].max(); run = (hi20 / C0.iloc[i - 100:i - 19].min() - 1).where(m)
            W2 = (run.rank(pct=True) > 0.7) & lim.iloc[i - 59:i + 1].any() & (C0.iloc[i] / hi20 - 1 <= -0.2) & m & okb
            ranked = list(run[W2].sort_values(ascending=False).index)
            top20 = ranked[:20]; nlock += int(lk[top20[:3]].sum())
            top20b = [c for c in top20 if buy[c]]
            top3 = [c for c in ranked if buy[c]][:3]
            R = (C0.iloc[i + H] / o - 1) * 100 - COST
            u = R[m & buy].mean()
            PATH = (C0.iloc[i + 1:i + H + 1].div(o) - 1) * 100
            st = lambda s: (R[s].mean(), PATH[s].mean(axis=1).min())
            r0, l0 = st(top20b); r3, l3 = st(top3)
            rows.append({"date": str(days[i].date()), "P0": r0, "P0_low": l0, "P2": r3, "P2_low": l3, "u": u, "P2_names": top3})
            rnd.append([st(list(rng.choice(top20b, 3, replace=False))) for _ in range(NR)])
        T = pd.DataFrame(rows); A = np.array(rnd); rm, rl, rw = A[:, :, 0].mean(0), A[:, :, 1].mean(0), A[:, :, 0].min(0)
        res[th] = {"次數": len(T), "前3原排名開盤鎖死檔數": nlock, "前3": round(T.P2.mean(), 2), "前3中位數": round(T.P2.median(), 2), "前3最差": round(T.P2.min(), 2),
                   "前3最低點": round(T.P2_low.mean(), 2), "前20": round(T.P0.mean(), 2), "全部股票": round(T.u.mean(), 2),
                   "前3減全部": round((T.P2 - T.u).mean(), 2), "前3減全部>0": f"{(T.P2 > T.u).sum()}/{len(T)}", "前20減全部": round((T.P0 - T.u).mean(), 2),
                   "隨機平均中位數": round(np.median(rm), 2), "隨機最低點中位數": round(np.median(rl), 2), "隨機最差中位數": round(np.median(rw), 2),
                   "①": bool(T.P2.mean() >= T.P0.mean() - 1), "②": bool(T.P2_low.mean() > np.median(rl)), "③": bool(T.P2.min() > np.median(rw))}
        print(f"成本 {COST} 門檻 {th}", json.dumps(res[th], ensure_ascii=False))
    out[COST] = res
keep = all(out[0.6][th][k] for th in (700, 750) for k in ("①", "②", "③"))
out["保留前3"] = keep; print("保留網站前3：", keep)
json.dump(out, open("research/results/panic_pick3_fix.json", "w"), ensure_ascii=False, indent=1, default=str)
