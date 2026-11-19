"""撿跌深改成「以這一波為準」：每次恐慌，找這一波被打下來的前波強勢股，不限市值。

2026/11/20 使用者定義目的：找大跌前有好光景的強勢股，在這一波被跌深或誤殺。探索（看過前面的結果後才做）。
這一波的起點 pk：訊號日前 60 個交易日內，等權指數最高的那天。
  W 新名單（不限市值）：
    前波強勢：pk 之前 60 個交易日的漲幅在分析範圍前 30%，而且那段期間漲停過
    誤殺：從 pk 前 5 天起算的自己最高收盤，到訊號日跌了 ≥ 20%
  舊名單 R0：跌深（離20日高 ≥20%）＋市值≤100億＋大跌前強（前80→20日前30%）＋曾漲停
  W 的子集：W＋市值≤100億、W＋市值>100億
上市空頭家數第一次到 750（7 次）、700（12 次）、800（4 次），隔天開盤買、持有 20 日；看平均、比全部股票多多少、買後最低、檔數。
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
exec(open("research/panic_winners.py", encoding="utf-8").read().split("out = {}")[0])
ew = (1 + C0.pct_change(fill_method=None).where(inb).mean(axis=1).fillna(0)).cumprod()
rows = []
for th in (750, 700, 800):
    for i in signals(th):
        m = inb.iloc[i] & C0.iloc[i].notna(); o = O.iloc[i + 1]; okb = o.notna()
        pk = int(np.argmax(ew.values[i - 60:i + 1])) + i - 60
        pre = (C0.iloc[pk] / C0.iloc[pk - 60] - 1).where(m)
        strongW = (pre.rank(pct=True) > 0.7) & lim.iloc[pk - 60:pk + 1].any()
        hiW = C0.iloc[max(pk - 5, 0):i + 1].max(); deepW = (C0.iloc[i] / hiW - 1 <= -0.2) & m
        W = strongW & deepW
        mv = cap * raw.iloc[i] / 10
        hi20 = C0.iloc[i - 19:i + 1].max(); deep = (C0.iloc[i] / hi20 - 1 <= -0.2) & m
        strong = (C0.iloc[i - 20] / C0.iloc[i - 80] - 1).where(m).rank(pct=True) > 0.7
        R0 = deep & (mv <= 100) & strong & lim.iloc[i - 80:i - 19].any()
        r20 = (C0.iloc[i + 20] / o - 1) * 100; low = (C0.iloc[i + 1:i + 21].min() / o - 1) * 100; b = r20[m & okb].mean()
        for k, sel in {"全部股票": m, "舊名單（市值≤100億）": R0, "新名單 這一波的前波強勢股": W, "新名單中 市值≤100億": W & (mv <= 100), "新名單中 市值>100億": W & (mv > 100)}.items():
            sel = sel & okb
            rows.append({"th": th, "date": str(days[i].date()), "wave_start": str(days[pk].date()), "grp": k, "n": int(sel.sum()),
                         "r20": r20[sel].mean(), "ex20": r20[sel].mean() - b, "low20": low[sel].mean(), "win": (r20[sel] > 0).mean() * 100})
T = pd.DataFrame(rows)
for th in (750, 800, 700):
    g = T[T.th == th]
    s = g.groupby("grp", sort=False)[["n", "r20", "ex20", "low20", "win"]].mean().round(2)
    pv = g.pivot(index="date", columns="grp", values="ex20")
    s["贏舊名單次數"] = [int((pv[k] > pv["舊名單（市值≤100億）"]).sum()) for k in s.index]; s["次數"] = len(pv)
    print("== 門檻", th); print(s.to_string())
    if th == 750: print(g[g.grp == "新名單 這一波的前波強勢股"][["date", "wave_start", "n", "r20", "ex20"]].round(1).to_string(index=False))
T.round(2).to_json("research/results/panic_wave.json", orient="records", force_ascii=False, indent=1)
