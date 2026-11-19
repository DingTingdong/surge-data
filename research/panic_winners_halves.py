# 探索：撿跌深名單內，按特徵分上下兩半，比較平均20日報酬（research/panic_winners.py 的延伸）
import sys, os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
src = open("research/panic_winners.py").read().split("out = {}")[0]; exec(src)
for th in (750, 700):
    res = []
    for i in signals(th):
        m = inb.iloc[i] & C0.iloc[i].notna()
        hi20 = C0.iloc[i - 19:i + 1].max(); deep = (C0.iloc[i] / hi20 - 1 <= -0.2) & m
        strong = (C0.iloc[i - 20] / C0.iloc[i - 80] - 1).where(m).rank(pct=True) > 0.7
        pick = deep & (cap * raw.iloc[i] / 10 <= 100) & strong & lim.iloc[i - 80:i - 19].any() & O.iloc[i + 1].notna()
        r20 = ((C0.iloc[i + 20] / O.iloc[i + 1] - 1) * 100)[pick]
        if len(r20) < 10: continue
        F = feats(i).loc[r20.index]; row = {"date": str(days[i].date())}
        for k in ("pre", "vol", "amt20", "ry", "d20", "mv"):
            hi = F[k] > F[k].median() if k != "d20" else F[k] < F[k].median()
            row[k] = r20[hi].mean() - r20[~hi & F[k].notna()].mean()
            row[k + "_lo"] = r20[hi].quantile(0.25) - r20[~hi & F[k].notna()].quantile(0.25)
        res.append(row)
    T = pd.DataFrame(res).set_index("date")
    print("== 門檻", th, "（上半組平均 − 下半組平均，百分點；_lo 是第25百分位差：上半組的差勁股是否也比較差）")
    print(T.round(1).to_string()); print("平均", T.mean().round(2).to_dict()); print("上半組較好次數", (T[[c for c in T if not c.endswith('_lo')]] > 0).sum().to_dict())
