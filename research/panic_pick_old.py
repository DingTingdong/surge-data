"""撿跌深名單在 2006–2019 年的恐慌波段還成立嗎？（第 15 輪；2026/11/23 09:55 登錄，看結果前寫下，看到結果後不修改）

名單（同網站 11/20 起的定義，build_site.py）：自己這段漲勢＝近 20 日最高收盤 ÷ 前 100～20 日最低收盤 − 1，在分析範圍前 30%；近 60 日漲停過
  （2015/6 前漲幅 ≥ 6.5%、之後 ≥ 9.5%）；從自己近 20 日最高收盤跌 ≥ 20%；依漲勢排序取前 20。
恐慌：2020 前空頭排列比例 ≥ 0.7499（近 10 日最高）。波段＝連續的恐慌日，前後隔 20 個交易日以上算新的一波。
進場：每一波第一個恐慌日收盤後選，隔天開盤買（整天一價鎖死買不到）；抱 20 個交易日，扣 1.1%，減等權分析範圍同期。
判斷：2006–2009、2010–2019 兩段各自「各波平均 > 0 且至少 60% 的波段 > 0」→ 成立，網站撿跌深說明加註「2006–2019 也成立」；否則註明只在 2020 年後成立。
另報：每一波的日期、檔數、名單／全部股票報酬；改成「恐慌期間每天都選」的平均（每波先平均）。
結果（11/23 10:00）：2006–2009 6 波平均減等權 +2.19、67% 為正 → 成立；2010–2019 13 波 −0.22、54% → 不成立（2010–2015 多數為負、2016–2018 4 波都正）。
  → 不算「2006–2019 也成立」；網站撿跌深說明寫出分段結果、提醒不要重壓。
"""
import os, sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run(path, start, end, H=20):
    X = pd.read_pickle(path); days = X["days"]; N = len(days)
    C, O, prevc, pn, inb, lk = X["C"], X["O"], X["prevc"], X["panic"].astype(bool), X["inb"], X["lockn"]
    Cff = pd.DataFrame(C).ffill(limit=60).values; ch = (C / prevc - 1) * 100
    lim = np.where(days < pd.Timestamp("2015-06-01"), 6.5, 9.5)[:, None]; islu = ch >= lim
    Onext = np.vstack([O[1:], np.full((1, C.shape[1]), np.nan)]); fut = np.vstack([Cff[H:], np.full((H, C.shape[1]), np.nan)])
    y = (fut / Onext - 1) * 100; ok = (Onext > 0) & ~lk; u = np.nanmean(np.where(inb & ok, y, np.nan), axis=1)
    pidx = [t for t in range(N) if pn[t] and pd.Timestamp(start) <= days[t] < pd.Timestamp(end) and t >= 100 and t + H < N]
    waves, cur, last = [], [], -99
    for t in pidx:
        if t - last > 20 and cur: waves.append(cur); cur = []
        cur.append(t); last = t
    if cur: waves.append(cur)
    def pick(t):
        hi20 = np.nanmax(Cff[t - 19:t + 1], axis=0); lo = np.nanmin(Cff[t - 99:t - 19], axis=0); runv = hi20 / lo - 1
        b = inb[t] & np.isfinite(runv); rr = pd.Series(runv[b]).rank(pct=True).values; runr = np.full(len(runv), np.nan); runr[b] = rr
        lim60 = islu[t - 59:t + 1].any(axis=0); dw = Cff[t] / hi20 - 1
        cand = np.where((runr > 0.7) & lim60 & (dw <= -0.20) & b)[0]
        cand = cand[np.argsort(-runv[cand])][:20]; cand = [j for j in cand if ok[t, j] and np.isfinite(y[t, j])]
        return cand
    rows = []
    for w in waves:
        t = w[0]; c = pick(t)
        r1 = (np.mean(y[t, c]) - 1.1) if c else np.nan
        daily = [np.mean(y[s, pick(s)]) - 1.1 - u[s] for s in w if pick(s)]
        rows.append({"起點": str(days[t].date()), "恐慌天數": len(w), "檔數": len(c), "名單20日%": round(float(r1), 2) if c else None, "全部股票%": round(float(u[t]), 2),
                     "減等權": round(float(r1 - u[t]), 2) if c else None, "每天都選 減等權（波內平均）": round(float(np.mean(daily)), 2) if daily else None})
    return rows

if __name__ == "__main__":
    out = {"2006–2009": run("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01"),
           "2010–2019": run("/tmp/model_v2/vs0050_extra.pkl", "2010-01-01", "2020-01-01")}
    j = {}
    for k, rows in out.items():
        xs = [r["減等權"] for r in rows if r["減等權"] is not None]
        j[k] = {"波數": len(xs), "平均減等權": round(float(np.mean(xs)), 2) if xs else None, "正的比例%": round(float(np.mean([x > 0 for x in xs]) * 100)) if xs else None,
                "成立": bool(xs and np.mean(xs) > 0 and np.mean([x > 0 for x in xs]) >= 0.6)}
        print("==", k); [print("  ", r) for r in rows]; print("  ", j[k])
    out["判斷"] = j; out["2006–2019 也成立"] = all(v["成立"] for v in j.values()); print("2006–2019 也成立：", out["2006–2019 也成立"])
    json.dump(out, open(f"{REPO}/research/results/panic_pick_old.json", "w"), ensure_ascii=False, indent=1)
