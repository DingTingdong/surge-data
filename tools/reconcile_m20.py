"""對帳：網站（radar_core＋composite_score）與研究（vs0050.py 快照＋calm3 挑法）在過去換股日的前 20 名、分數、穩健版 3 檔是否一致。
換股日用第 58 輪規則（營收期限日的下一個交易日）。只讀本機資料、只印結果。"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "research"); sys.argv = sys.argv[:1]
from radar_core import load_prices, panels, features, tiers
import composite_score as CS, robust_select as RS
YB = int(os.environ.get("YB", "2")); p = load_prices(years_back=YB); P = panels(p)   # 網站用 years_back=1
f = features(P); days = P["C"].index
ds = pd.Series(days); dl = ds[ds.dt.day >= 10].groupby(ds.dt.to_period("M")).first(); ix = np.searchsorted(ds.values, dl.values) + 1
forms = [ds.iloc[k] for k in ix if k < len(ds) and ds.iloc[k] >= pd.Timestamp(os.environ.get("FROM", "2025-01-01")) and ds.iloc[k] <= pd.Timestamp("2026-11-20")]
D = RS.load("/tmp/model_v2/vs0050_dl_extra.pkl", "2024-06-01", "2027-01-01"); F = D["F"]
C = P["C"]; rows = []
for t in forms:
    S = tiers(f, P["MK"], since=t - pd.Timedelta(days=100))   # 和網站一樣：分析範圍只算近期（網站 70 個交易日）
    base = S[(S.date == t) & S.base].code.tolist(); sc = CS.scores(P, f, S, t, base)
    site = [c for c, _ in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))[:20]]
    g = F[F.date == t].sort_values("cs", ascending=False); res = g.code.head(20).tolist()
    # 穩健版 3 檔（不含開盤可買與續抱，純看前 20 名裡 20 日波動最低）
    i = days.get_loc(t); w = C[site].iloc[i - 20:i + 1].pct_change(fill_method=None); v = w.std().where(w.notna().sum() >= 15)
    s3 = v.sort_values().index[:3].tolist(); r3 = g.head(20).assign(v=lambda z: z.vol20.fillna(np.inf)).sort_values("v").code.head(3).tolist()
    com = set(site) & set(res); d = [abs(sc[c] - float(g.set_index("code").cs[c])) for c in com]
    rows.append({"換股日": t.date(), "前20重疊": len(com), "分數最大差": round(max(d), 4) if d else None, "網站3檔": s3, "研究3檔": r3, "3檔一致": set(s3) == set(r3)})
R = pd.DataFrame(rows); pd.set_option("display.width", 250); print(R.to_string(index=False))
print("前20 平均重疊", round(R["前20重疊"].mean(), 1), "／ 3 檔一致", int(R["3檔一致"].sum()), "/", len(R))
