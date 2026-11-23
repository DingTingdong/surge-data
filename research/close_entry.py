"""換股日收盤就買賣 vs 隔天開盤（第 23 輪；2026/11/23 12:00 登錄，看結果前寫下，看到結果後不修改）

起因：第 21 輪發現月選的優勢在營收公布後幾天內就消退（晚 2～4 個交易日換股，每月少 0.3～1.1%）。現在是換股日收盤後算名單、隔天開盤買；
  如果改成換股日盤中（約 13:20，價格接近收盤）先算暫定名單、當天收盤買賣，可以早一個晚上。網站已有盤中收集（intraday_collect.py），做得到。
作法（月選分數；3 檔與 20 檔；續抱＝還在前 20 名；成本每次賣出 0.6%）：
  E0 現行：換股日隔天開盤買賣。
  E1 收盤：換股日收盤價買賣（分數仍用當天收盤資料，近似盤中 13:20 的名單）；當天收漲停的新股買不到 → 往下一名遞補；
     要賣的股票當天收跌停賣不掉 → 改隔天開盤賣。0050 報酬同樣用換股日收盤到下次換股日收盤。
資料：2006–2009＋2011–2026（五段）；換股時點檢查用 vs0050_fdx_extra.pkl（11 日後第 3、5 個交易日）。
採用條件（全部符合才在網站加「換股日盤中暫定名單、收盤買」）：
  ① 3 檔：E1 減 E0 每月平均 ≥ 0.3、Newey-West t ≥ 2（2011–2026）；② 五段中至少四段 E1 > E0；③ 20 檔也 E1 > E0（2011–2026）；
  ④ 換股日改第 3、第 5 個交易日時，3 檔 E1 都 > E0。
另報：入選新股換股日收盤到隔天開盤的平均跳空（個股本身，0050 只有收盤指數所以不扣）。
結果（11/23 12:10）：不通過 → 維持隔天開盤。3 檔 E1 減 E0 每月 −0.52（t −0.86），五段 +0.36／−1.37／−0.90／+0.41／+0.21；20 檔 −0.07；
  第 3／5 個交易日 −0.36／+0.04。新買進股收盤到隔天開盤平均只跳空 +0.4% → 早一個晚上沒有好處。
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import robust_select as RS
from vs0050 import nw_t, LOCK
from wr_confirm import sim2, stats, nth_set

def sim_close(D, score="cs", n=3, keep_rank=20, cost=0.6, gaps=None):
    O, Cff, bidx, POS, CJ, ND, days = D["O"], D["Cff"], D["bidx"], D["POS"], D["CJ"], D["ND"], D["days"]
    G = {d: g for d, g in D["F"].groupby("date")}; rows, held = [], []
    def lim(i, j, up):  # 當天收在漲停／跌停
        th = LOCK - (0.03 if days[i] < pd.Timestamp("2015-06-01") else 0)
        if not (np.isfinite(Cff[i, j]) and np.isfinite(Cff[i - 1, j]) and Cff[i - 1, j] > 0): return False
        r = Cff[i, j] / Cff[i - 1, j] - 1; return r >= th if up else r <= -th
    px = {}  # 持有中股票的買進（或上次結算）價
    for a, b_ in zip(D["FDS"][:-1], D["FDS"][1:]):
        ia, ib = POS[a], POS[b_]
        if ib + 1 >= ND: break
        g = G[a].sort_values(score, ascending=False); rank = {c: i + 1 for i, c in enumerate(g.code)}
        keep = [c for c in held if rank.get(c, 1e9) <= keep_rank][:n]; new = []
        for c in g.code:
            if len(keep) + len(new) >= n: break
            if c in keep: continue
            j = CJ[c]
            if np.isfinite(Cff[ia, j]) and not lim(ia, j, True): new.append(c)
        sells = [c for c in held if c not in keep]; to = 1.0 if not rows else len(sells) / n; rs = []
        # 賣不掉（跌停）的：損益延到隔天開盤，記在本期（近似）
        adj = []
        for c in sells:
            j = CJ[c]
            if lim(ia, j, False) and O[ia + 1, j] > 0: adj.append((O[ia + 1, j] / Cff[ia, j] - 1) * 100 / n)
        for c in keep + new:
            j = CJ[c]; o = Cff[ia, j]; ex = Cff[ib, j]
            if not np.isfinite(ex): ex = Cff[ib - 1, j]
            if np.isfinite(ex) and o > 0: rs.append((ex / o - 1) * 100)
            if gaps is not None and c in new and O[ia + 1, j] > 0: gaps.append((O[ia + 1, j] / o - 1) * 100)
        held = keep + new
        rows.append({"date": a, "r": (np.mean(rs) if rs else 0) + sum(adj) - cost * to, "b": (bidx[ib] / bidx[ia] - 1) * 100})
    T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index); return T

def load(path, s, e):
    D = RS.load(path, s, e); D["days"] = pd.read_pickle(path)["days"]; return D

if __name__ == "__main__":
    Dn = load("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01"); Do = load("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01"); out = {}
    for n in (3, 20):
        gp = [] if n == 3 else None
        T0 = pd.concat([sim2(Do, "cs", n), sim2(Dn, "cs", n)]); T1 = pd.concat([sim_close(Do, n=n, gaps=gp), sim_close(Dn, n=n, gaps=gp)])
        x0, x1 = T0.r - T0.b, T1.r - T1.b; dx = (x1 - x0).dropna(); sg = RS.seg(dx.index); m = dx.index >= "2011-01-01"
        out[f"{n} 檔"] = {"E0 隔天開盤": stats(T0), "E1 當天收盤": stats(T1), "E1減E0 2011–26": (round(float(dx[m].mean()), 2), round(nw_t(dx[m], 1), 2)),
                         "E1減E0 五段": {s: round(float(dx[sg == s].mean()), 2) for s in RS.SEGS}}
        if gp is not None: out[f"{n} 檔"]["新買進股 收盤→隔天開盤 平均跳空%"] = round(float(np.mean(gp)), 2)
        print(n, "檔", json.dumps(out[f"{n} 檔"], ensure_ascii=False), flush=True)
    Dx = load("/tmp/model_v2/vs0050_fdx_extra.pkl", "2011-01-01", "2027-01-01"); tim = {}
    for k in (1, 2):
        E = nth_set(Dx, k); a, b = sim2(E, "cs", 3), sim_close(E, n=3); d = ((b.r - b.b) - (a.r - a.b)).dropna()
        tim[f"第{2 * k + 1}個交易日"] = {"E0": stats(a), "E1": stats(b), "E1減E0": round(float(d.mean()), 2)}; print(f"第{2 * k + 1}個交易日", tim[f"第{2 * k + 1}個交易日"], flush=True)
    out["換股時點"] = tim; o3 = out["3 檔"]
    c = {"①": o3["E1減E0 2011–26"][0] >= 0.3 and o3["E1減E0 2011–26"][1] >= 2, "②": sum(v > 0 for v in o3["E1減E0 五段"].values()) >= 4,
         "③": out["20 檔"]["E1減E0 2011–26"][0] > 0, "④": all(v["E1減E0"] > 0 for v in tim.values())}
    out["各項"] = c; out["通過"] = all(c.values()); print("各項", c, "通過" if all(c.values()) else "不通過")
    json.dump(out, open(f"{REPO}/research/results/close_entry.json", "w"), ensure_ascii=False, indent=1, default=str)
