"""(1) 撿跌深名單縮小：只取「大跌中途小反彈」前 N 名；(2) 接近底部改買正2（模擬）績效如何。

2026/11/20 使用者問，屬於探索。
時點：上市空頭家數第一次到 750（7 次）／800（4 次）／700（12 次），隔天開盤買。
名單：research/panic_wave.py 的新名單（這一波的前波強勢股），依小反彈（research/panic_rebound_strength.py）排序取前 10／20／30／50。
正2：有 data/etf.csv 時用 00631L 實際價格（正2實際）；另列 0050 每日漲跌 × 2 的模擬做對照，第一天用開盤到收盤，扣年化 1.5% 成本（管理費＋期貨轉倉等，概估）。
  0050 價格用除權息調整（data/adjust.csv）。
看：持有 20 日與 60 日報酬、持有期間最低點（最多先虧多少）。
"""
import gzip, glob, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
exec(open("research/panic_winners.py", encoding="utf-8").read().split("out = {}")[0])
ew = (1 + C0.pct_change(fill_method=None).where(inb).mean(axis=1).fillna(0)).cumprod()
# 0050：用 data/etf.csv（證交所 STOCK_DAY），除權息用 data/adjust.csv，分割（2025/6/18 一拆四）用單日跳動偵測還原
et = pd.read_csv("data/etf.csv"); et["date"] = pd.to_datetime(et.date)
def adj_series(code, ref=None):
    x = et[et.code == code].set_index("date").reindex(days)
    x[["open", "close"]] = x[["open", "close"]].ffill(limit=10)  # 0050 分割前停止交易 5 天（2025/6/11～17）
    f = pd.Series(1.0, index=days)
    a = pd.read_csv("data/adjust.csv", dtype={"code": str}); a = a[a.code == code]
    for _, z in a.iterrows(): f[days < pd.Timestamp(z.date)] *= z.ratio
    r = (x.close * f).pct_change()
    exp = ref if ref is not None else pd.Series(0.0, index=days)
    for d in r.index[(r - exp).abs() > 0.3]:   # 分割／反分割：單日價格跳動遠超過行情
        ratio = (x.close.shift(1)[d] * f.shift(1)[d]) / (x.close[d] * f[d]) * (1 + exp[d]); f[days < d] /= ratio
    return x.open * f, x.close * f
o50, c50 = adj_series("0050")
r50 = c50.pct_change()
FEE = 0.015 / 250
def lev_path(i, h, L=2.0):
    """i 是訊號日；隔天開盤買，持有 h 日；回傳每日淨值。"""
    v = [1 + L * (c50.iloc[i + 1] / o50.iloc[i + 1] - 1) - FEE]
    for k in range(i + 2, i + h + 1): v.append(v[-1] * (1 + L * r50.iloc[k] - FEE))
    return np.array(v)
# 真實正2（00631L）：data/etf.csv；2026/3/31 分割用單日跳動偵測還原（參考 0050×2）
REAL = True
o2, c2 = adj_series("00631L", ref=r50 * 2)
def real_path(i, h):
    v = c2.iloc[i + 1:i + h + 1] / o2.iloc[i + 1]
    return v.values
rows = []
for th in (750, 800, 700):
    for i in signals(th):
        if i + 60 >= N: continue
        m = inb.iloc[i] & C0.iloc[i].notna(); o = O.iloc[i + 1]; okb = o.notna()
        pk = int(np.argmax(ew.values[i - 60:i + 1])) + i - 60
        pre = (C0.iloc[pk] / C0.iloc[pk - 60] - 1).where(m)
        W = (pre.rank(pct=True) > 0.7) & lim.iloc[pk - 60:pk + 1].any() & (C0.iloc[i] / C0.iloc[max(pk - 5, 0):i + 1].max() - 1 <= -0.2) & m & okb
        lo_ = min(pk + 5, i - 1); w3 = (ew / ew.shift(3) - 1).iloc[lo_:i]
        if len(w3) and w3.max() > 0:
            e = days.get_loc(w3.idxmax()); bo = (C0.iloc[e] / C0.iloc[e - 3] - 1)
        else: bo = pd.Series(0.0, index=C0.columns)
        rank = bo[W].sort_values(ascending=False)
        # 11/20 新定義（research/panic_own_high.py）：每檔自己的漲勢與高點，依自己漲勢取前20
        hi20 = C0.iloc[i - 19:i + 1].max(); run = (hi20 / C0.iloc[i - 100:i - 19].min() - 1).where(m)
        W2 = (run.rank(pct=True) > 0.7) & lim.iloc[i - 59:i + 1].any() & (C0.iloc[i] / hi20 - 1 <= -0.2) & m & okb
        newtop = pd.Series(C0.columns.isin(run[W2].sort_values(ascending=False).index[:20]), index=C0.columns)
        def stock_stats(sel, h):
            r = (C0.iloc[i + h] / o - 1)[sel] * 100
            path = (C0.iloc[i + 1:i + h + 1].div(o)[sel.index[sel]] - 1).mean(axis=1) * 100  # 等權組合每天的報酬
            return r.mean(), path.min()
        res = {"th": th, "date": str(days[i].date()), "n_list": int(W.sum())}
        for h in (20, 60):
            for nm, sel in [("全部股票", m & okb), ("名單全部", W)] + [(f"名單前{k}", pd.Series(C0.columns.isin(rank.index[:k]), index=C0.columns)) for k in (10, 20, 30, 50)] + [("新前20", newtop)]:
                a, b = stock_stats(sel, h); res[f"{nm}_{h}"] = a; res[f"{nm}_{h}_low"] = b
            p1 = lev_path(i, h, 1.0); p2 = lev_path(i, h, 2.0)
            res[f"0050_{h}"] = (p1[-1] - 1) * 100; res[f"0050_{h}_low"] = (p1.min() - 1) * 100
            res[f"正2_{h}"] = (p2[-1] - 1) * 100; res[f"正2_{h}_low"] = (p2.min() - 1) * 100
            if REAL:
                pr = real_path(i, h)
                res[f"正2實際_{h}"] = (pr[-1] - 1) * 100 if pr[-1] == pr[-1] else np.nan; res[f"正2實際_{h}_low"] = (np.nanmin(pr) - 1) * 100
        rows.append(res)
T = pd.DataFrame(rows)
cols = ["全部股票", "新前20", "名單前20", "0050", "正2"] + (["正2實際"] if REAL else [])
for th in (750, 800, 700):
    g = T[T.th == th]
    print(f"== 門檻 {th}（{len(g)} 次），名單平均 {g.n_list.mean():.0f} 檔")
    out = pd.DataFrame({h: {c: f"{g[f'{c}_{h}'].mean():+.1f}%（最低 {g[f'{c}_{h}_low'].mean():+.1f}%，賺錢 {(g[f'{c}_{h}'] > 0).sum()}/{len(g)}）" for c in cols} for h in (20, 60)})
    print(out.to_string())
    if th == 750: print(g[["date"] + [f"{c}_20" for c in ("新前20", "正2實際", "0050")]].round(1).to_string(index=False))
T.round(2).to_json("research/results/panic_top_and_lev.json", orient="records", force_ascii=False, indent=1)
