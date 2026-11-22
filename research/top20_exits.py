"""月選的進場與出場方法（第 5 輪；2026/11/22 23:30 登錄，看結果前寫下，看到結果後不修改）

起因：第 4 輪 V1 不穩——換股日晚兩天，2024–26 就輸 0050；贏 0050 靠每月抓到大漲股。使用者要「合理合適的進退場方法」讓績效穩定。
基準：月選（top20_goal.py 基準，換股日隔天開盤買、還在前 20 的續抱），成本 0.6%（使用者 2.8 折），報酬算法同 top20_exact.py。
規則（各自和基準一樣，只改進出場）：
  X1 固定停損 10%（Han, Zhou & Zhu 2016）：持股收盤比「這期起點價」（換股日隔天開盤價）低 10% → 隔天開盤賣，資金放 0050 到下次換股日。
  X2 移動停損 15%：收盤比這期起點後的最高收盤低 15% → 隔天開盤賣，放 0050。
  X3 大盤濾網（Faber 2007，200 日均線）：換股日 0050（含息）收盤低於 200 日均線 → 這個月不選股、全部放現金（報酬 0）；高於才照常選股。
  X4 不追高進場：新買的股票隔天開盤比前一天收盤高 ≥ 5% → 不買，那一格放 0050。
  X5 X1＋X3。
  停損賣出算一次換股（成本 0.6% ÷ 20），換到 0050 再扣 0.2%；X3 全部轉現金的月份照賣出檔數扣成本。
判斷（全部符合才算「穩」）：
  ① 全期每月減 0050 > 0 且 Newey-West t ≥ 2.9；② 四段（2011–15、2016–19、2020–23、2024–26）都 > 0；③ 發現期（2020–23）以外 t ≥ 2；
  ④ 換股日改成 11 日後第 3 個交易日時，全期 t ≥ 2 且 2024–26 > 0（第 4 輪就是倒在這裡）；⑤ 最大回落不比 0050 深。
  累計檢驗 12 個規則，門檻維持 t ≥ 2.9。
網站：有規則「穩」→ 用全期 t 最高的那個，但還要 2006–2009 年（data/oldest，資料到了才算）每月減 0050 > 0 才換上網站；
  都不穩 → 網站維持基準，並把各規則的結果寫在研究頁。
另報：成本 1.1%、年化、每月贏 0050 比例、16 年中贏的年數、有沒有連兩年輸、最差滾動 12 個月、平均每月停損檔數、X3 空手的月份比例。
"""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
import top20_goal as TG
from top20_goal import POS, CJ, Ov, Cff, lockn, bidx, ND, FDS, G, stability, per, nw_t, PER, COMP

def sim_exit(fds, Gd, rule, cost=0.6, n=20, off=1):
    rows, held = [], []
    for a, b_ in zip(fds[:-1], fds[1:]):
        ta, tb = POS[a], POS[b_]
        if tb + off >= ND: break
        ie, ix = ta + off, tb + off; first = not rows
        bper = (bidx[ix - 1] / bidx[ie - 1] - 1) * 100
        if rule in ("X3", "X5") and ie - 1 >= 199 and bidx[ie - 1] < bidx[ie - 200:ie].mean():
            sells = len(held); held = []
            rows.append({"date": a, "r": -cost * sells / n, "b": bper, "stops": 0, "cash": True}); continue
        g = Gd[a]; rank = {c: i + 1 for i, c in enumerate(g.code)}
        keep = [c for c in held if rank.get(c, 1e9) <= n]; new = [c for c in g.code if c not in keep][:n - len(keep)]
        sells = len([c for c in held if c not in keep]); rs, hn, nstop = [], [], 0
        for c in keep + new:
            j = CJ[c]; is_new = c in new
            if is_new and (not (Ov[ie, j] > 0) or lockn[ie - 1, j]): continue
            o = Ov[ie, j] if Ov[ie, j] > 0 else Cff[ie - 1, j]
            if not (np.isfinite(o) and o > 0): continue
            if rule == "X4" and is_new and Cff[ie - 1, j] > 0 and o / Cff[ie - 1, j] - 1 >= 0.05:
                rs.append(bper - 0.2); continue
            k_stop, exit_px = None, None
            if rule in ("X1", "X2", "X5"):
                mx = -np.inf
                for k in range(ie, ix - 1):
                    ck = Cff[k, j]
                    if not np.isfinite(ck): break
                    mx = max(mx, ck); lvl = o * 0.9 if rule in ("X1", "X5") else mx * 0.85
                    if ck <= lvl:
                        ep = Ov[k + 1, j]; exit_px = ep if ep > 0 else Cff[k + 1, j]; k_stop = k; break
            if k_stop is not None and np.isfinite(exit_px):
                rest = bidx[ix - 1] / bidx[k_stop] - 1
                rs.append(((exit_px / o) * (1 + rest) - 1) * 100 - 0.2); nstop += 1; continue
            ex = Ov[ix, j] if ix < ND and Ov[ix, j] > 0 else Cff[min(ix, ND - 1), j]
            if not np.isfinite(ex): ex = Cff[ix - 1, j]
            if np.isfinite(ex): rs.append((ex / o - 1) * 100); hn.append(c)
        held = hn; to = 1.0 if first else (sells + nstop) / n
        rows.append({"date": a, "r": (np.mean(rs) if rs else bper) - cost * to, "b": bper, "stops": nstop, "cash": False})
    T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index); return T

def day3_groups():
    XX = pd.read_pickle("/tmp/model_v2/vs0050_fdx_extra.pkl"); FF = XX["fd"].copy(); FF = FF[FF.date >= "2011-01-01"]
    FF["cs"] = np.mean([FF.groupby("date")[c].rank(pct=True).fillna(0.5).values for c in COMP], axis=0)
    dser = pd.Series(sorted(FF.date.unique())); f3 = [pd.Timestamp(d) for d in sorted(dser.groupby(dser.dt.to_period("M")).nth(1).values)]
    return f3, {d: g.sort_values("cs", ascending=False).reset_index(drop=True) for d, g in FF[FF.date.isin(f3)].groupby("date")}

if __name__ == "__main__":
    f3, G3 = day3_groups(); out = {}; best = None
    for rule in ("基準", "X1", "X2", "X3", "X4", "X5"):
        T = sim_exit(FDS, G, rule); x = T.r - T.b; s = stability(x, T.r, T.b)
        T3 = sim_exit(f3, G3, rule); x3 = T3.r - T3.b; s3 = stability(x3, T3.r, T3.b)
        T11 = sim_exit(FDS, G, rule, cost=1.1); s11 = stability(T11.r - T11.b, T11.r, T11.b)
        c = {"①": s["全期"][0] > 0 and s["全期"][1] >= 2.9, "②": all(v > 0 for v in s["四段"].values()), "③": s["發現期外"][0] > 0 and s["發現期外"][1] >= 2,
             "④": s3["全期"][0] > 0 and s3["全期"][1] >= 2 and s3["2024-26"] > 0, "⑤": s["最大回落 策略/0050"][0] >= s["最大回落 策略/0050"][1]}
        ok = all(c.values()) and rule != "基準"
        out[rule] = {"成本0.6%": s, "換股日第3個交易日": {k: s3[k] for k in ("全期", "2024-26", "發現期外", "贏0050年數")}, "成本1.1%": {k: s11[k] for k in ("全期", "2024-26", "發現期外", "年化 策略/0050")},
                     "各項": c, "穩": bool(ok), "每月贏0050%": round(float((x > 0).mean() * 100)), "平均每月停損檔數": round(float(T.stops.mean()), 1), "空手月份%": round(float(T.cash.mean() * 100))}
        if ok and (best is None or s["全期"][1] > out[best]["成本0.6%"]["全期"][1]): best = rule
        print(f"== {rule}", "穩" if ok else "", c, "\n   ", {k: v for k, v in s.items() if k != "四段"}, "四段", s["四段"],
              "\n    第3個交易日", out[rule]["換股日第3個交易日"], "| 1.1%", out[rule]["成本1.1%"], "| 每月贏", out[rule]["每月贏0050%"], "停損/月", out[rule]["平均每月停損檔數"], "空手%", out[rule]["空手月份%"], flush=True)
    out["候選（需 2006–2009 也 > 0 才上網站）"] = best; print("候選：", best)
    json.dump(out, open(f"{REPO}/research/results/top20_exits.json", "w"), ensure_ascii=False, indent=1, default=str)
