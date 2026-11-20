"""今日發動：看「族群」走向或「領頭股」能不能變好；以及今日發動賠錢的機率。

全部在看結果之前寫下（2026/11/21 02:15），看到結果後不修改。
族群＝證交所／櫃買官方產業（data/industry.csv，現在的分類；比較粗，不是題材）。產業指數＝分析範圍內同產業股票的等權平均（不含這檔自己）。
樣本：今日發動（v2），分「平常」與「恐慌期間」（上市空頭家數近10日最高 ≥750）。隔天開盤買。
訊號日就知道的條件：
  G1 族群走強：產業指數近5日漲、而且收在自己10日線上
  G2 族群走弱：產業指數在自己10日線下（G1 的對照）
  G3 領頭股：自己的近20日漲幅是同產業第一名
  G4 跟風股：同產業近10日內已經有別檔今日發動（自己不是第一個）
  G5 帶頭股：同產業近10日內沒有別檔今日發動（自己是第一個）
賣法：A 抱20日；B 照族群賣：產業指數收盤跌破自己10日線就隔天開盤賣（最多20日）；另附 X6 移動停損。每筆扣 1.1%。
判斷（平常時期）：某條件「兩段期間（2020–23、2024–26）抱20日平均都 > 0」而且「都比全部今日發動高 ≥ 1 個百分點」、保留 ≥ 20% 訊號 → 有用；
  照族群賣（B）兩段都 > 0 而且都比抱20日好 → 有用。否則照實報告沒有用。
另報：今日發動賠錢（扣成本後 < 0）的比例、賠超過 10% 的比例。
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
src = open("research/relaunch.py", encoding="utf-8").read().split("# 穩健性")[0]
for a in ('json.dump({"results"', 'for per in res: print(', 'print(yearly.to_string()); print("加進雷達：", add)', 'print("恐慌期間（True）vs 平常（False）："); print(q.to_string())'):
    src = src.replace(a, "pass  # " if a.startswith(("json", "for")) else "")
exec(src)
ind = pd.read_csv("data/industry.csv", dtype={"code": str}).set_index("code")
grp = pd.Series(ind["ind_name"]).reindex(C.columns)
inbase = S[S.base].assign(v=True).pivot(index="date", columns="code", values="v").reindex(index=days, columns=C.columns).fillna(False).astype(bool)
ret1 = C.pct_change(fill_method=None).where(inbase)
gsum = {g: ret1[cs].sum(axis=1, min_count=1) for g, cs in grp.dropna().groupby(grp.dropna()).groups.items()}
gcnt = {g: ret1[cs].notna().sum(axis=1) for g, cs in grp.dropna().groupby(grp.dropna()).groups.items()}
r20 = C / C.shift(20) - 1
pkd = S[S.pk][["date", "code"]].copy(); pkd["g"] = pkd.code.map(grp)
def ind_index_excl(g, code):
    s, n = gsum[g], gcnt[g]
    own = ret1[code].fillna(0); ownn = ret1[code].notna().astype(int)
    m = ((s - own) / (n - ownn).replace(0, np.nan)).fillna(0)
    return (1 + m).cumprod()
def sell_open(k, j):
    for s_ in range(k, min(k + 6, N)):
        if Ov[s_, j] == Ov[s_, j] and not locked(s_, j, False): return Ov[s_, j]
    return None
rows = []
F = B[B.grp == "F 今日發動"].copy(); F["g"] = F.code.map(grp); F = F.dropna(subset=["g"])
cache = {}
for r in F.itertuples():
    t, j = pos[r.date], col[r.code]; e = t + 1; end = e + 19
    if end >= N: continue
    o = Ov[e, j]
    if not o == o or o <= 0 or locked(e, j, True): continue
    key = (r.g, r.code)
    if key not in cache: cache[key] = ind_index_excl(r.g, r.code).values
    gi = cache[key]; gma10 = pd.Series(gi).rolling(10).mean().values
    G1 = gi[t] > gi[t - 5] and gi[t] > gma10[t]; G2 = gi[t] < gma10[t]
    peers = r20.loc[r.date, grp[grp == r.g].index][inbase.loc[r.date, grp[grp == r.g].index]]
    G3 = len(peers) > 1 and peers.idxmax() == r.code
    prior = pkd[(pkd.g == r.g) & (pkd.code != r.code) & (pkd.date < r.date) & (pkd.date >= days[max(0, t - 10)])]
    G4 = len(prior) > 0; G5 = not G4
    k = end
    while k > e and not Cv[k, j] == Cv[k, j]: k -= 1
    h20 = (Cv[k, j] / o - 1) * 100 - COST
    gb = None
    for k2 in range(e, end):
        if gi[k2] < gma10[k2]:
            px = sell_open(k2 + 1, j); gb = None if px is None else (px / o - 1) * 100 - COST; break
    if gb is None: gb = h20
    x = x6(t, j)
    rows.append({"date": r.date, "train": r.date <= pd.Timestamp("2023-12-31"), "panic": bool(panic.get(r.date, False)),
                 "G1": G1, "G2": G2, "G3": G3, "G4": G4, "G5": G5, "h20": h20, "gb": gb, "x6": x if isinstance(x, float) else np.nan})
T = pd.DataFrame(rows)
def st(g):
    return pd.Series({"n": len(g), "抱20日": g.h20.mean(), "照族群賣": g.gb.mean(), "移動停損": g.x6.mean(), "抱20日賠錢%": (g.h20 < 0).mean() * 100, "抱20日賠10%以上%": (g.h20 < -10).mean() * 100})
out = {}
for pn, lab in ((False, "平常"), (True, "恐慌期間")):
    for tr, per in ((True, "2020-2023"), (False, "2024-2026")):
        g0 = T[(T.panic == pn) & (T.train == tr)]
        tab = pd.DataFrame({"全部": st(g0), **{k: st(g0[g0[k]]) for k in ("G1", "G2", "G3", "G4", "G5")}}).T.round(2)
        tab["保留%"] = (tab.n / tab.at["全部", "n"] * 100).round(0)
        out[(lab, per)] = tab; print("==", lab, per); print(tab.to_string())
useful = [k for k in ("G1", "G2", "G3", "G4", "G5") if all(out[("平常", p)].at[k, "抱20日"] > 0 and out[("平常", p)].at[k, "抱20日"] - out[("平常", p)].at["全部", "抱20日"] >= 1 and out[("平常", p)].at[k, "保留%"] >= 20 for p in ("2020-2023", "2024-2026"))]
gb_ok = all(out[("平常", p)].at["全部", "照族群賣"] > 0 and out[("平常", p)].at["全部", "照族群賣"] > out[("平常", p)].at["全部", "抱20日"] for p in ("2020-2023", "2024-2026"))
print("平常時期有用的族群條件：", useful or "沒有", "；照族群賣有用：", gb_ok)
# 賠錢機率總表（全期間）
for pn, lab in ((False, "平常"), (True, "恐慌期間")):
    g = T[T.panic == pn]
    print(lab, "筆數", len(g), "抱20日：賠錢 %.0f%%、賠10%%以上 %.0f%%、賺20%%以上 %.0f%%；移動停損：賠錢 %.0f%%" % ((g.h20 < 0).mean() * 100, (g.h20 < -10).mean() * 100, (g.h20 > 20).mean() * 100, (g.x6 < 0).mean() * 100))
import json
json.dump({f"{a}|{b}": v.reset_index().to_dict("records") for (a, b), v in out.items()} | {"useful": useful, "group_exit_ok": gb_ok}, open("research/results/group_trend.json", "w"), ensure_ascii=False, indent=1)
# 探索：恐慌期間 G1（族群已轉強）vs G2（族群仍弱）分波段
wave_id = (panic != panic.shift()).cumsum()
P2 = T[T.panic].copy(); P2["wave"] = P2.date.map(wave_id)
w = P2.groupby("wave").apply(lambda g: pd.Series({"起": g.date.min().strftime("%Y-%m"), "族群仍弱n": int(g.G2.sum()), "族群仍弱": g[g.G2].h20.mean(),
                                                    "族群已轉強n": int(g.G1.sum()), "族群已轉強": g[g.G1].h20.mean()}), include_groups=False)
print(w.round(1).to_string())
