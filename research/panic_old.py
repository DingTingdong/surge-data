"""恐慌期間的發現，在從沒用過的 2016–2019 是否也成立。

全部在看結果之前寫下（2026/11/21 18:55），看到結果後不修改。之前恐慌期間的結果（今日發動 +7.7%、族群仍弱較好）都是 2020–2026 看過後的探索。
資料：data/old（research/model_v2.py build old 的列）。恐慌：「上市短均線空頭家數 ÷ 有60日線上市檔數」近10日最高 ≥ 0.7499（＝750 的同比例）。
樣本：今日發動（分析範圍內），隔天開盤買（開盤一價漲停買不到不計），抱 20 個交易日，扣 1.1%。期間 2016-01-01～2019-12-31。
  P1 恐慌期間今日發動平均 > 0，而且比同期平常的今日發動高 ≥ 3 個百分點 → 成立
  P2 恐慌期間「族群仍弱」（產業等權指數在10日線下）平均 > 「族群已反彈」（指數在10日線上且近5日上漲）→ 成立
另報：每一波恐慌各自的結果（波與波之間隔超過 10 個交易日算新的一波）。產業指數這裡含自己（網站不含自己），差別很小。
"""
import json, os, sys
import pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO)
R = pd.read_pickle("/tmp/model_v2/rows_old.pkl")
ratio = json.load(open(f"{REPO}/research/results/model_v2_final.json"))["ratio"]
R["panic"] = R.bearf10 >= ratio
F = R[R.pk & R.y.notna() & (R.date >= "2016-01-01") & (R.date <= "2019-12-31")].copy()
st = lambda y: {"n": int(len(y)), "平均": round(float(y.mean()), 2), "中位": round(float(y.median()), 2), "賺錢%": round(float((y > 0).mean() * 100), 1)}
pn, nm = F[F.panic], F[~F.panic]
weak = pn[pn.g_m10 < 0]; reb = pn[(pn.g_m10 > 0) & (pn.g_5 > 0)]
res = {"恐慌今日發動": st(pn.y), "平常今日發動": st(nm.y), "族群仍弱": st(weak.y), "族群已反彈": st(reb.y)}
res["P1成立"] = bool(pn.y.mean() > 0 and pn.y.mean() - nm.y.mean() >= 3); res["P2成立"] = bool(weak.y.mean() > reb.y.mean())
d = pd.Series(sorted(R[R.panic & (R.date >= "2016-01-01") & (R.date <= "2019-12-31")].date.unique()))
days = pd.Series(sorted(R.date.unique())); pos = {x: i for i, x in enumerate(days)}
wave = (d.map(pos).diff().fillna(99) > 10).cumsum(); wmap = dict(zip(d, wave))
pn = pn.assign(w=pn.date.map(wmap))
res["各波"] = {f"{g.date.min():%Y-%m-%d}～{g.date.max():%m-%d}": {**st(g.y), "仍弱": round(float(g[g.g_m10 < 0].y.mean()), 2), "已反彈": round(float(g[(g.g_m10 > 0) & (g.g_5 > 0)].y.mean()), 2)}
             for _, g in pn.groupby("w")}
res["恐慌天數"] = int(len(d)); res["恐慌期間"] = [f"{d[wave == w].min():%Y-%m-%d}～{d[wave == w].max():%Y-%m-%d}" for w in sorted(wave.unique())]
print(json.dumps(res, ensure_ascii=False, indent=1))
json.dump(res, open(f"{REPO}/research/results/panic_old.json", "w"), ensure_ascii=False, indent=1)
