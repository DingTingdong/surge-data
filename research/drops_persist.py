"""探索性（看過 drops.py 的結果後才做）：大跌前段扛跌的股票，之後20天在大盤下跌日與上漲日各表現如何。用法：python research/drops_persist.py"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "drops.py")).read().split("# 事件")[0])
ev = []; i = 0; v = dd.values
while i < N:
    if v[i] <= -0.06:
        j = i
        while j < N and v[j] <= -0.03: j += 1
        pk = int(np.argmax(ew.values[max(0, i - 19):i + 1])) + max(0, i - 19); ev.append((i, pk)); i = j
    else: i += 1
er = ew.pct_change()
res = []
for t, pk in ev:
    if t + 20 >= N: continue
    m = inb.iloc[t]
    d = ((C.iloc[t] / C.iloc[pk] - 1)).where(m)
    R = C.pct_change(fill_method=None).iloc[t + 1:t + 21].loc[:, m.values]; d = d[m.values]
    dn = er.iloc[t + 1:t + 21] < 0
    sdn = R[dn.values].sum(); sup = R[~dn.values].sum()
    q = d.rank(pct=True)
    top, bot = q > 0.9, q <= 0.1
    res.append({"date": days[t].date(), "corr_dn": round(d.corr(sdn, method="spearman"), 2), "corr_up": round(d.corr(sup, method="spearman"), 2),
                "top_dn": round(sdn[top].mean() * 100, 1), "bot_dn": round(sdn[bot].mean() * 100, 1), "top_up": round(sup[top].mean() * 100, 1), "bot_up": round(sup[bot].mean() * 100, 1),
                "top_tot": round((sdn + sup)[top].mean() * 100, 1), "bot_tot": round((sdn + sup)[bot].mean() * 100, 1)})
R = pd.DataFrame(res); print(R.to_string()); print(R.drop(columns="date").mean().round(2).to_string())
print("corr_dn>0 share", (R.corr_dn > 0).mean(), "corr_up<0 share", (R.corr_up < 0).mean())
