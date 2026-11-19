"""撿跌深名單裡，事後漲最多的股票有什麼共通點（事後描述，探索）。

2026/11/19 使用者問。做法：上市空頭家數第一次到 750（7 次）與 700（12 次）的撿跌深名單，
每次按「隔天開盤買、20 日報酬」分成前 20%（贏家）與其他，比較訊號當天就知道的特徵（中位數）。
每個特徵算：贏家中位數 − 其他中位數，以及「幾次事件裡贏家比較高」。
為了避免找到巧合，先看 2021–2022 的事件，再看 2024–2026 的事件方向是否相同。
"""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radar_core import load_prices, panels, features, tiers

p = load_prices(); P = panels(p); C0, O, MK, raw, A = P["C"], P["O"], P["MK"], P["rawC"], P["A"]
f = features(P); S = tiers(f, MK)
days = C0.index; N = len(days)
alive = (C0.notna().cumsum() > 0) & (C0.notna()[::-1].cumsum()[::-1] > 0)
C = C0.ffill(limit=20).where(alive)
R = lambda n: C.rolling(n, min_periods=n).mean()
m5, m10, m20, m60 = R(5), R(10), R(20), R(60)
ok = C.notna() & m60.notna()
mk = MK.ffill().iloc[-1]; tw = [c for c in C.columns if mk.get(c) == "TWSE"]
bear = ((m5 < m10) & (m10 < m20) & ok)[tw].sum(axis=1)
inb = S[S.base].assign(v=True).pivot(index="date", columns="code", values="v").reindex(index=days, columns=C0.columns).fillna(False).astype(bool)
ind = pd.read_csv("data/industry.csv", dtype={"code": str}).set_index("code")
cap = pd.Series(ind["cap"]).reindex(C0.columns); grp = pd.Series(ind["ind_name"]).reindex(C0.columns)
ELEC = {"半導體", "電腦週邊", "光電", "通信網路", "電子零組件", "電子通路", "資訊服務", "其他電子"}
lim = f["ch"] >= 9.5
start = int(np.argmax(ok[tw].sum(axis=1).values >= 600)) + 80
ch = pd.concat([pd.read_csv(fn, dtype={"code": str}, usecols=["date", "code", "inst", "trust", "margin"]) for fn in sorted(glob.glob("data/chips/*.csv*"), key=lambda x: (os.path.basename(x)[:4], x.endswith(".csv")))], ignore_index=True)
ch["date"] = pd.to_datetime(ch.date); ch = ch.drop_duplicates(["date", "code"], keep="last")
piv = lambda c: ch.pivot(index="date", columns="code", values=c).reindex(index=days, columns=C0.columns)
inst, trust, margin = piv("inst"), piv("trust"), piv("margin")
rv = pd.concat([pd.read_csv(fn, dtype={"code": str}) for fn in sorted(glob.glob("data/revenue/*.csv*"))], ignore_index=True)
rv["pub"] = pd.to_datetime(rv.ym + "-01") + pd.DateOffset(months=1) + pd.Timedelta(days=10)
yoy = rv.pivot_table(index="ym", columns="code", values="yoy").sort_index(); pub = rv.groupby("ym").pub.first().sort_index()

NAMES = {"d20": "離20日高跌幅%", "dd60": "離60日高跌幅%", "fall_days": "從20日高跌了幾天", "r5": "近5日漲跌%", "ch0": "訊號當天漲跌%",
         "pre": "大跌前漲幅%（前80→20日）", "nlim": "大跌前漲停次數", "mv": "市值（億）", "price": "股價", "amt20": "20日均成交額（百萬）",
         "vr": "量能比（5日÷20日）", "vol": "20日波動%", "ry": "最近月營收年增%", "inst20": "近20日法人買超（張）", "trust20": "近20日投信買超（張）",
         "mg5": "融資5日變化%", "elec": "電子股（1=是）", "lim20": "近20日漲停次數"}

def feats(i):
    known = pub[pub <= days[i]].index
    hi20 = C0.iloc[i - 19:i + 1]
    return pd.DataFrame({
        "d20": (C0.iloc[i] / hi20.max() - 1) * 100, "dd60": (C0.iloc[i] / C0.iloc[i - 59:i + 1].max() - 1) * 100,
        "fall_days": 19 - hi20.values.argmax(axis=0) if False else pd.Series(19 - np.nanargmax(np.nan_to_num(hi20.values, nan=-1), axis=0), index=C0.columns),
        "r5": (C0.iloc[i] / C0.iloc[i - 5] - 1) * 100, "ch0": f["ch"].iloc[i], "pre": (C0.iloc[i - 20] / C0.iloc[i - 80] - 1) * 100,
        "nlim": lim.iloc[i - 80:i - 19].sum(), "mv": cap * raw.iloc[i] / 10, "price": raw.iloc[i], "amt20": A.iloc[i - 19:i + 1].mean() / 1e6,
        "vr": A.iloc[i - 4:i + 1].mean() / A.iloc[i - 19:i + 1].mean(), "vol": f["v"].iloc[i],
        "ry": yoy.loc[known[-1]].reindex(C0.columns) if len(known) else np.nan,
        "inst20": inst.iloc[i - 19:i + 1].sum(min_count=1), "trust20": trust.iloc[i - 19:i + 1].sum(min_count=1),
        "mg5": (margin.iloc[i] / margin.iloc[i - 5] - 1) * 100, "elec": grp.isin(ELEC).astype(float), "lim20": lim.iloc[i - 19:i + 1].sum()})

def signals(th): return [i for i in range(start, N) if bear.iloc[i] >= th and (bear.iloc[i - 20:i] < th).all() and i + 20 < N]

out = {}
for th in (750, 700):
    rec = []; samples = []
    for i in signals(th):
        m = inb.iloc[i] & C0.iloc[i].notna()
        hi20 = C0.iloc[i - 19:i + 1].max(); deep = (C0.iloc[i] / hi20 - 1 <= -0.2) & m
        strong = (C0.iloc[i - 20] / C0.iloc[i - 80] - 1).where(m).rank(pct=True) > 0.7
        pick = deep & (cap * raw.iloc[i] / 10 <= 100) & strong & lim.iloc[i - 80:i - 19].any() & O.iloc[i + 1].notna()
        r20 = ((C0.iloc[i + 20] / O.iloc[i + 1] - 1) * 100)[pick]
        if len(r20) < 10: continue
        F = feats(i).loc[r20.index]; win = r20 >= r20.quantile(0.8)
        row = {"date": str(days[i].date()), "n": int(len(r20)), "win_avg": round(r20[win].mean(), 1), "rest_avg": round(r20[~win].mean(), 1)}
        for k in NAMES:
            a, b = F.loc[win, k].median(), F.loc[~win, k].median()
            row[k] = a - b; row[k + "_w"] = a; row[k + "_r"] = b
        rec.append(row)
        samples.append(F.assign(win=win, r20=r20, date=str(days[i].date())))
    T = pd.DataFrame(rec); early = T.date < "2024"
    summ = []
    for k, nm in NAMES.items():
        summ.append({"特徵": nm, "贏家中位數": round(T[k + "_w"].mean(), 2), "其他中位數": round(T[k + "_r"].mean(), 2), "差": round(T[k].mean(), 2),
                     "贏家較高次數": f"{int((T[k] > 0).sum())}/{len(T)}", "2021-22 較高": f"{int((T[early][k] > 0).sum())}/{int(early.sum())}",
                     "2024-26 較高": f"{int((T[~early][k] > 0).sum())}/{int((~early).sum())}"})
    out[str(th)] = {"events": T[["date", "n", "win_avg", "rest_avg"]].to_dict("records"), "features": summ}
    print("== 門檻", th); print(T[["date", "n", "win_avg", "rest_avg"]].to_string(index=False)); print(pd.DataFrame(summ).to_string(index=False))
    if th == 750:
        SS = pd.concat(samples); top = SS[SS.win].sort_values("r20", ascending=False)
        top = top.assign(name=top.index.map(lambda c: p[p.code == c].name.iloc[-1] if (p.code == c).any() else ""), ind=top.index.map(grp))
        out["750_winners"] = top[["date", "name", "ind", "r20", "d20", "pre", "nlim", "mv", "ry"]].round(1).reset_index().rename(columns={"index": "code"}).to_dict("records")
        print(top[["date", "name", "ind", "r20", "d20", "pre", "nlim", "mv", "ry"]].round(1).head(40).to_string())
json.dump(out, open("research/results/panic_winners.json", "w"), ensure_ascii=False, indent=1, default=str)
