"""「今日分數」平常日買法（第 63 輪；2026/11/26 22:10 登錄，看結果前寫下，看到結果後不修改）

起因：使用者要知道網站「今日分數」（非換股日重算的趨勢＋營收前 20 名）怎麼用、抱多久、績效如何，並要可選的排序方式。
資料：vs0050.py 以 FD_OFFSETS=5,10 重建的月中快照（換股日＝每月 11 日起第一個交易日之後第 5、第 10 個交易日；已在營收期限之後，沒有營收偷看問題），2011–2026。
每個月、每個時點 k（5 或 10）各一次：在當天的前 20 名（分數 cs）裡，用以下方法挑 5 檔：
  A 分數最高 5 檔；B 20 日波動最低 5 檔（月選規則）；C 混合：分數名次＋低波動名次相加最小的 5 檔；D 最新月營收年增最高 5 檔。另列 E 前 20 名全買（參考）。
  隔天開盤買；開盤比前收漲 ≥ 9.5%（2015/6 前 6.5%）、沒開盤、進場開盤原始價 ≥ 500 元的跳過、往下遞補。每檔等金額，扣 0.6%。
持有：5、10、20 個交易日收盤賣；或「抱到下次換股」（下個月第 58 輪換股日的隔天開盤賣）。
比較：同日開盤買 0050（含息、扣 0.18%）。報每次平均減 0050、NW t（落後 3，事件依月份排列）、贏的比例、五段、2024–26。
判斷（描述性）：回答使用者「抱多久、績效」；若某方法在 k＝10、抱到下次換股時「平均減 0050 > 0、t ≥ 2.9、四段都正」，網站把它列為今日分數的建議排序，否則只當參考。
結果（11/26 22:30）：主要判斷不通過（k＝10、抱到下次換股：A +0.36（t 0.59）、B +0.30（0.90）、C +0.12（0.30）、D +0.24（0.44），2024–26 全負）。網站「今日分數」維持只供參考。
  每次（不是每月）減 0050，k＝10（換股後約第 10 個交易日，例如 11/26）：抱 5 天 A +1.04（t 2.65）／B +0.43／C +0.64／D +0.93；抱 10 天 +0.72／+0.46／+0.45／+0.87；抱 20 天 +0.42／+0.25／+0.06／+0.03。
  k＝5：抱 10 天 A +2.28（t 3.74，四段都正，2024–26 +0.81）、C +1.82（t 3.79）、D +1.84（t 2.97）、B +1.03；但抱 20 天或到下次換股就降到 +0.7～1.3（t < 2）。
  → 這組（k＝5、抱 10 天）是 32 個組合裡最好的，屬探索結果，要往後驗證才算數；月中以後才買、抱到下次換股，幾乎沒有比 0050 好。

"""
import json, os, sys, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import robust_select as RS
from vs0050 import nw_t
from mp_exec import E50
from price_cap import addraw
COST, BCOST = 0.6, 0.18
PK = "/tmp/model_v2/vs0050_fdx_mid_extra.pkl"

if __name__ == "__main__":
    D = RS.load(PK, "2011-01-01", "2027-01-01"); X = pd.read_pickle(PK); days = pd.DatetimeIndex(X["days"]); N = len(days)
    D["cols"] = list(X["cols"]); D["days"] = days; D = addraw(D, PK)
    O, C, pc, POS, CJ = D["O"], D["Cff"], np.asarray(X["prevc"], float), D["POS"], D["CJ"]
    e5 = E50.reindex(days); b = np.asarray(X["bidx"], float); bO = b * (e5.open / e5.close).where((e5.open > 0) & (e5.close > 0)).fillna(1.0).values
    dl = sorted(pd.read_pickle("/tmp/model_v2/vs0050_dl_extra.pkl")["fd"].date.unique())   # 第 58 輪換股日
    ds = pd.Series(days); g11 = ds[ds.dt.day >= 11].groupby(ds.dt.to_period("M"))
    snaps = {k: [d for d in g11.nth(k) if d in POS and d >= pd.Timestamp("2011-01-01")] for k in (5, 10)}
    G = {d: g for d, g in D["F"].groupby("date")}
    def pick(a, how):
        g = G[a].sort_values("cs", ascending=False).head(20).copy(); g["rc"] = np.arange(len(g)); g["rv"] = g.vol20.rank(method="first").fillna(99)
        if how == "A": order = g.sort_values("cs", ascending=False)
        elif how == "B": order = g.assign(v=g.vol20.fillna(np.inf)).sort_values(["v", "cs"], ascending=[True, False])
        elif how == "C": order = g.assign(s=g.rc + g.rv).sort_values(["s", "cs"], ascending=[True, False])
        elif how == "D": order = g.assign(y=g.yoy1.fillna(-np.inf)).sort_values(["y", "cs"], ascending=[False, False])
        else: order = g
        ie = POS[a] + 1; out = []
        thr = 0.095 if days[ie] >= pd.Timestamp("2015-06-01") else 0.065
        for c in order.code:
            j = CJ[c]
            if not (O[ie, j] > 0) or (pc[ie, j] > 0 and O[ie, j] / pc[ie, j] - 1 >= thr - 1e-9) or (how != "E" and D["rawO"][ie, j] >= 500): continue
            out.append(j)
            if how != "E" and len(out) >= 5: break
        return ie, out
    res = {}
    for k, S_ in snaps.items():
        for how in "ABCDE":
            rows = []
            for a in S_:
                if a + pd.Timedelta(days=0) not in POS: continue
                ie, js = pick(a, how)
                if not js or ie >= N: continue
                nx = next((d for d in dl if d > a), None)
                r = {"date": a}
                for H in (5, 10, 20, "next"):
                    if H == "next":
                        if nx is None or POS.get(nx) is None or POS[nx] + 1 >= N: r["xnext"] = np.nan; continue
                        ix = POS[nx] + 1; px = np.array([O[ix, j] if O[ix, j] > 0 else C[ix, j] for j in js]); bb = (bO[ix] / bO[ie] - 1) * 100 - BCOST
                    else:
                        ix = ie + H - 1
                        if ix >= N: r[f"x{H}"] = np.nan; continue
                        px = C[ix, js]; bb = (b[ix] / bO[ie] - 1) * 100 - BCOST
                    rr = np.nanmean((px / O[ie, js] - 1) * 100) - COST; r[f"x{H}"] = rr - bb
                rows.append(r)
            T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index); sg = RS.seg(T.index); s = {"次數": len(T)}
            for H in ("5", "10", "20", "next"):
                x = T[f"x{H}"].dropna()
                if not len(x): continue
                s[f"抱{H if H != 'next' else '到下次換股'}"] = {"平均減0050": round(float(x.mean()), 2), "t": round(nw_t(x.reset_index(drop=True), 3), 2), "贏%": round(float((x > 0).mean() * 100)),
                    "2024–26": round(float(x[x.index >= "2024-01-01"].mean()), 2), "四段": [round(float(x[sg[T.index.isin(x.index)] == q].mean()), 2) for q in RS.SEGS[1:]]}
            key = f"k={k}｜{'ABCDE'.index(how)}{how}"; res[key] = s; print(key, json.dumps(s, ensure_ascii=False), flush=True)
    json.dump(res, open(f"{REPO}/research/results/today_score.json", "w"), ensure_ascii=False, indent=1)
