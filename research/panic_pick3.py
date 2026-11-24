"""撿跌深縮成 3 檔（第 43 輪；2026/11/24 09:35 事先登錄，看結果前寫下，看到結果後不修改）

起因：使用者只能買少數幾檔，問撿跌深能不能像月選穩健版一樣縮成 3 檔。
名單：網站現行撿跌深前 20（research/panic_own_high.py 的 W2，依自己漲勢取前 20）。
時點：上市空頭家數第一次到 700（12 次）／750（7 次）／800（4 次），訊號日收盤後選、隔天開盤買、抱 20 個交易日，等權。
比較：
  P0 前 20 全買（基準）
  P1 前 20 裡近 20 日波動（日報酬標準差）最低 3 檔（＝月選穩健版的挑法）
  P2 前 20 依自己漲勢前 3 名（＝網站現行排序的前 3）
  P3 前 20 裡大跌中途小反彈最大 3 檔
  R  前 20 隨機挑 3 檔 1,000 次（每次恐慌各自抽），看分布
看：20 日平均報酬、每次恐慌最差的一次、持有期間組合最低點（最多先虧多少）、贏 0050 的次數。
採用條件（P1／P2／P3 各自判斷；700 與 750 兩個門檻都要符合）：
  ① 20 日平均不比 P0 少 1 個百分點以上；
  ② 持有期間最低點平均，比隨機 3 檔的中位數淺；
  ③ 最差一次恐慌的 20 日報酬，比隨機 3 檔「最差一次」的中位數好。
  都過 → 網站撿跌深標出那 3 檔（參考，次數少）；多個過選最低點最淺的。都沒過 → 網站不變，只寫結論。
限制：只有 7～12 次恐慌、且彼此不獨立，任何結果都只是參考。
"""
import os, sys, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
exec(open("research/panic_winners.py", encoding="utf-8").read().split("out = {}")[0])
ew = (1 + C0.pct_change(fill_method=None).where(inb).mean(axis=1).fillna(0)).cumprod()
et = pd.read_csv("data/etf.csv"); et["date"] = pd.to_datetime(et.date)
def adj_series(code, ref=None):
    x = et[et.code == code].set_index("date").reindex(days)
    x[["open", "close"]] = x[["open", "close"]].ffill(limit=10)
    fct = pd.Series(1.0, index=days)
    a = pd.read_csv("data/adjust.csv", dtype={"code": str}); a = a[a.code == code]
    for _, z in a.iterrows(): fct[days < pd.Timestamp(z.date)] *= z.ratio
    r = (x.close * fct).pct_change(); exp = ref if ref is not None else pd.Series(0.0, index=days)
    for d in r.index[(r - exp).abs() > 0.3]:
        ratio = (x.close.shift(1)[d] * fct.shift(1)[d]) / (x.close[d] * fct[d]) * (1 + exp[d]); fct[days < d] /= ratio
    return x.open * fct, x.close * fct
o50, c50 = adj_series("0050")
H = 20; rng = np.random.default_rng(20261124); NR = 1000
rows, rnd = [], {}
for th in (700, 750, 800):
    rnd[th] = []
    for i in signals(th):
        if i + H >= N: continue
        m = inb.iloc[i] & C0.iloc[i].notna(); o = O.iloc[i + 1]; okb = o.notna()
        hi20 = C0.iloc[i - 19:i + 1].max(); run = (hi20 / C0.iloc[i - 100:i - 19].min() - 1).where(m)
        W2 = (run.rank(pct=True) > 0.7) & lim.iloc[i - 59:i + 1].any() & (C0.iloc[i] / hi20 - 1 <= -0.2) & m & okb
        top = list(run[W2].sort_values(ascending=False).index[:20])
        vol = C0.iloc[i - 20:i + 1][top].pct_change(fill_method=None).std()
        pk = int(np.argmax(ew.values[i - 60:i + 1])) + i - 60; lo_ = min(pk + 5, i - 1); w3 = (ew / ew.shift(3) - 1).iloc[lo_:i]
        if len(w3) and w3.max() > 0: e = days.get_loc(w3.idxmax()); bo = (C0.iloc[e] / C0.iloc[e - 3] - 1)[top]
        else: bo = pd.Series(0.0, index=top)
        R = (C0.iloc[i + H] / o - 1)[top] * 100
        PATH = (C0.iloc[i + 1:i + H + 1][top].div(o[top]) - 1) * 100
        def st(sel):
            sel = [c for c in sel if R.get(c) == R.get(c)]
            return R[sel].mean(), PATH[sel].mean(axis=1).min()
        b50 = (c50.iloc[i + H] / o50.iloc[i + 1] - 1) * 100
        res = {"th": th, "date": str(days[i].date()), "n": len(top), "0050": b50}
        for nm, sel in (("P0", top), ("P1", list(vol.sort_values().index[:3])), ("P2", top[:3]), ("P3", list(bo.sort_values(ascending=False).index[:3]))):
            res[nm], res[nm + "_low"] = st(sel)
        res["P1_names"] = [name.get(c, c) if "name" in dir() and isinstance(name, dict) else c for c in vol.sort_values().index[:3]]
        rows.append(res)
        rnd[th].append([st(list(rng.choice(top, 3, replace=False))) for _ in range(NR)])
T = pd.DataFrame(rows); out = {}
for th in (700, 750, 800):
    g = T[T.th == th]; A = np.array(rnd[th])  # 次數 × NR × 2
    rm, rl, rw = A[:, :, 0].mean(0), A[:, :, 1].mean(0), A[:, :, 0].min(0)
    o_ = {"次數": len(g), "0050": round(g["0050"].mean(), 2),
          "隨機3檔": {"平均中位數": round(np.median(rm), 2), "平均5-95%": [round(np.percentile(rm, 5), 2), round(np.percentile(rm, 95), 2)],
                    "最低點中位數": round(np.median(rl), 2), "最差一次中位數": round(np.median(rw), 2)}}
    for nm in ("P0", "P1", "P2", "P3"):
        o_[nm] = {"平均": round(g[nm].mean(), 2), "中位數": round(g[nm].median(), 2), "最低點平均": round(g[nm + "_low"].mean(), 2),
                  "最差一次": round(g[nm].min(), 2), "贏0050": f"{(g[nm] > g['0050']).sum()}/{len(g)}", "賺錢": f"{(g[nm] > 0).sum()}/{len(g)}",
                  "平均在隨機的百分位": round((rm < g[nm].mean()).mean() * 100), "最低點在隨機的百分位": round((rl < g[nm + '_low'].mean()).mean() * 100)}
    out[th] = o_
    print(f"== {th}（{len(g)} 次）0050 {o_['0050']:+.2f}", json.dumps(o_, ensure_ascii=False))
chk = {}
for nm in ("P1", "P2", "P3"):
    c = {}
    for th in (700, 750):
        o_ = out[th]
        c[th] = {"①": o_[nm]["平均"] >= o_["P0"]["平均"] - 1, "②": o_[nm]["最低點平均"] > o_["隨機3檔"]["最低點中位數"], "③": o_[nm]["最差一次"] > o_["隨機3檔"]["最差一次中位數"]}
    chk[nm] = {"各項": c, "通過": all(all(v.values()) for v in c.values())}
passed = [nm for nm in chk if chk[nm]["通過"]]
best = max(passed, key=lambda nm: out[750][nm]["最低點平均"]) if passed else None
out["判斷"] = chk; out["採用"] = best; print(json.dumps(chk, ensure_ascii=False, default=str)); print("採用", best)
print(T[T.th == 750][["date", "P0", "P1", "P2", "P3", "0050"]].round(1).to_string(index=False))
json.dump({"summary": out, "rows": T.round(2).to_dict("records")}, open("research/results/panic_pick3.json", "w"), ensure_ascii=False, indent=1, default=str)
