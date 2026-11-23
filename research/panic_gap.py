"""恐慌期間「今日發動」隔天開盤價規則的驗證（第 14 輪；2026/11/23 09:35 登錄，看結果前寫下，看到結果後不修改）

起因：網站恐慌期間提示「隔天開平或開低 3% 以內再買最穩，開高 3% 以上別追」，來自 relaunch_gap.py 的探索（主要是 2020 年後的恐慌波段），
  沒有事先登錄。現在用 2006–2019 年的恐慌期間（2006–2009 來自 data/oldest，完全沒用過；2010–2019 在探索時沒有用到開盤價分組）驗證。
恐慌：同研究頁（2020 前用空頭排列比例 ≥ 0.7499，近 10 日最高）。
事件：今日發動＝當天漲停（2015/6 前漲幅 ≥ 6.5%、之後 ≥ 9%），前 10 天沒漲停，當天成交值 ≥ 2,000 萬、前 20 日均額 ≥ 300 萬（同網站），發生在恐慌期間。
進場：隔天開盤；開盤一價漲停視為買不到。分組依「隔天開盤 ÷ 前一天收盤 − 1」：
  A 開平或開低 3% 以內：−3% ≤ 跳空 ≤ +1%；B 開高 1–3%；C 開高 3% 以上（買得到的）；另報開低 3% 以上。
出場：抱 20 個交易日收盤，扣 1.1%；與等權分析範圍同期報酬相減。
判斷（三項都要）：① A 組平均 > C 組平均；② A 組減等權 > 0；③ 2006–2009 與 2010–2019 兩段各自都符合 ① 與 ②。
  符合 → 網站提示保留並把「探索性」改成「已驗證」；不符合 → 網站提示改寫成實際結果（或拿掉）。
另報：各組筆數、中位數、賺錢比例，逐段結果。
結果（11/23 09:45）：不成立 → 網站拿掉「開平或開低 3% 以內最穩」的建議，改寫實際結果。
  2006–2009：A 1,511 筆 20 日扣成本 −3.84（中位 −3.73、賺錢 41%），減等權 −0.90；C −2.27，減等權 −0.28。
  2010–2019：A 1,443 筆 −0.73（中位 −2.04、42%），減等權 −1.24；C −0.32，減等權 −1.05。
  參考 2020–2026：A 952 筆 +9.65（75%），減等權 +0.09；開低 3% 以上 518 筆 +20.29（多在 2025/4）。
  註：這裡的「買不到」只算整天一價鎖死（約 1%），和 relaunch_gap.py 的「開漲停 35%」定義不同；A 組不受影響。
"""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)

def events(path, start, end, H=20):
    X = pd.read_pickle(path); days, cols = X["days"], X["cols"]; N = len(days)
    C, O, A, prevc, pn, inb, lockn = X["C"], X["O"], X["A"].astype(float), X["prevc"], X["panic"].astype(bool), X["inb"], X["lockn"]
    Cff = pd.DataFrame(C).ffill(limit=60).values
    ch = (C / prevc - 1) * 100; lim = np.where(days < pd.Timestamp("2015-06-01"), 6.5, 9.0)[:, None]
    islu = ch >= lim; idx = np.arange(N)[:, None]
    last = pd.DataFrame(np.where(islu, idx, np.nan)).ffill().shift(1).values; snp = np.where(np.isnan(idx - last), 99, idx - last)
    a1 = A / 1e6; a20p = pd.DataFrame(A).shift(1).rolling(20, min_periods=20).mean().values / 1e6
    ev = islu & (snp > 10) & (a1 >= 20) & (a20p >= 3) & pn[:, None]
    Onext = np.vstack([O[1:], np.full((1, len(cols)), np.nan)])
    fut = np.vstack([Cff[H + 1:], np.full((H + 1, len(cols)), np.nan)])   # 隔天開盤買、第 H 個交易日收盤（t+H）
    fut = np.vstack([Cff[H:], np.full((H, len(cols)), np.nan)])
    y = (fut / Onext - 1) * 100; okb = (Onext > 0) & ~lockn
    u = np.nanmean(np.where(inb & okb, y, np.nan), axis=1)
    t_, j_ = np.nonzero(ev); rows = []
    for t, j in zip(t_, j_):
        if t + H >= N or not (Onext[t, j] > 0): continue
        dte = days[t]
        if dte < pd.Timestamp(start) or dte >= pd.Timestamp(end): continue
        rows.append({"date": dte, "code": cols[j], "gap": (Onext[t, j] / C[t, j] - 1) * 100, "locked": bool(lockn[t, j]), "y": y[t, j] - 1.1, "u": u[t]})
    E = pd.DataFrame(rows); E["x"] = E.y - E.u; return E

def grp(g):
    return np.select([g < -3, g <= 1, g < 3], ["開低3%以上", "A 開平或開低3%內", "B 開高1–3%"], "C 開高3%以上")

def summ(E):
    B = E[~E.locked].copy(); B["g"] = grp(B.gap); o = {}
    for k, z in B.groupby("g"):
        o[k] = {"筆數": int(len(z)), "20日扣成本平均": round(float(z.y.mean()), 2), "中位數": round(float(z.y.median()), 2), "減等權": round(float(z.x.mean()), 2), "賺錢%": round(float((z.y > 0).mean() * 100))}
    o["開盤漲停買不到%"] = round(float(E.locked.mean() * 100)); o["事件數"] = int(len(E)); o["恐慌日期數"] = int(E.date.nunique()); return o

if __name__ == "__main__":
    out = {}
    out["2006–2009"] = summ(events("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01"))
    Eb = events("/tmp/model_v2/vs0050_extra.pkl", "2010-01-01", "2027-01-01")
    out["2010–2019"] = summ(Eb[Eb.date < "2020-01-01"]); out["參考 2020–2026（當初探索期）"] = summ(Eb[Eb.date >= "2020-01-01"])
    j = {}
    for k in ("2006–2009", "2010–2019"):
        o = out[k]; a = o.get("A 開平或開低3%內"); c = o.get("C 開高3%以上")
        j[k] = bool(a and c and a["20日扣成本平均"] > c["20日扣成本平均"] and a["減等權"] > 0)
    out["判斷"] = {"各段": j, "已驗證": all(j.values())}
    for k, v in out.items(): print("==", k, json.dumps(v, ensure_ascii=False))
    json.dump(out, open(f"{REPO}/research/results/panic_gap.json", "w"), ensure_ascii=False, indent=1)
