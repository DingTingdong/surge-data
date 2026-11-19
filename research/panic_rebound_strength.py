"""撿跌深名單：(1) 大跌前漲幅用不同週期（40／60／80／120 日）有沒有差；
(2) 恐慌期間「前一次小反彈」反彈最強的股票，之後是不是反彈也最強。

2026/11/20 使用者問，屬於探索（看過 panic_winners.py 後才做）。
(2) 的「前一次小反彈」：訊號日前 30 個交易日內指數最高點之後（大跌期間），等權指數漲最多的連續 3 個交易日；看每檔股票在那 5 天漲多少。
都在撿跌深名單內分上下兩半，比平均 20 日報酬（隔天開盤買）。門檻 750（7 次）與 700（12 次）。
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
exec(open("research/panic_winners.py", encoding="utf-8").read().split("out = {}")[0])
ew = (1 + C0.pct_change(fill_method=None).where(inb).mean(axis=1).fillna(0)).cumprod()
rows = []
for th in (750, 700):
    for i in signals(th):
        m = inb.iloc[i] & C0.iloc[i].notna()
        hi20 = C0.iloc[i - 19:i + 1].max(); deep = (C0.iloc[i] / hi20 - 1 <= -0.2) & m
        strong = (C0.iloc[i - 20] / C0.iloc[i - 80] - 1).where(m).rank(pct=True) > 0.7
        pick = deep & (cap * raw.iloc[i] / 10 <= 100) & strong & lim.iloc[i - 80:i - 19].any() & O.iloc[i + 1].notna()
        r20 = ((C0.iloc[i + 20] / O.iloc[i + 1] - 1) * 100)[pick]
        if len(r20) < 10: continue
        # 前一次小反彈：訊號日前 30 日內，指數 5 日漲幅最大的那段（結束在訊號日前一天以前）
        # 只在「大跌期間」找：指數前 30 日內最高點之後，到訊號日前一天
        pk = int(np.argmax(ew.values[i - 30:i + 1])) + i - 30
        lo = min(pk + 5, i - 1)
        w5 = (ew / ew.shift(3) - 1).iloc[lo:i]; e = days.get_loc(w5.idxmax()); s = e - 3
        bounce_idx = float(w5.max() * 100)
        X = pd.DataFrame({
            "pre40": C0.iloc[i - 20] / C0.iloc[i - 40] - 1, "pre60": C0.iloc[i - 20] / C0.iloc[i - 60] - 1,
            "pre80": C0.iloc[i - 20] / C0.iloc[i - 80] - 1, "pre120": C0.iloc[i - 20] / C0.iloc[i - 120] - 1,
            "bounce": C0.iloc[e] / C0.iloc[s] - 1}).loc[r20.index]
        row = {"th": th, "date": str(days[i].date()), "n": len(r20), "指數小反彈%": round(bounce_idx, 1),
               "小反彈期間": f"{days[s].strftime('%m/%d')}→{days[e].strftime('%m/%d')}"}
        for k in X:
            hi = X[k] > X[k].median()
            row[k] = r20[hi].mean() - r20[~hi & X[k].notna()].mean()
        rows.append(row)
T = pd.DataFrame(rows)
for th in (750, 700):
    g = T[T.th == th].set_index("date").drop(columns="th")
    print("== 門檻", th); print(g.round(1).to_string())
    k = ["pre40", "pre60", "pre80", "pre120", "bounce"]
    print("平均（上半−下半，百分點）", g[k].mean().round(2).to_dict()); print("上半較好次數", (g[k] > 0).sum().to_dict(), "/", len(g))
T.round(2).to_json("research/results/panic_rebound_strength.json", orient="records", force_ascii=False, indent=1)
