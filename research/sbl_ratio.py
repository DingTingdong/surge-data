"""借券賣出餘額偏高的排除（第 33 輪；2026/11/23 17:35 登錄，看結果前寫下，看到結果後不修改）

起因：第 25 輪另報項目發現「借券賣出餘額 ÷ 20 日平均成交股數」（sbl_ratio，法人放空部位要幾天才回補得完）在 2006–2015、2016–2026
  兩半的 Fama-MacBeth 都是 t −3.77，最高五分之一下個月比 0050 少 1.10%。注意：這是在同一份資料上看到的，這輪的組合檢驗不是全新期間，
  所以門檻維持嚴格，並且之後要用前瞻資料確認。
規則：
  B1 穩健版 3 檔，新買進時跳過 sbl_ratio 在當天全部股票前 20% 的（缺值＝沒有借券資料，不排除；續抱不受影響）。
  B2 同上，但門檻改成前 10%（另報、不影響判斷）。
採用條件（B1 全部符合才在網站加上）：① 2011–2026 每月減 0050 > 穩健版，且 t ≥ 2.9；② 五段中至少四段不比穩健版差；③ 最大回落不比穩健版深。
另報：今日發動（平常）依 sbl_ratio 分兩組（前 20% vs 其他）的每筆減 0050；擋掉次數。
結果（11/23 17:40）：B1 不通過（② 2020–23、2024–26 較差，只有 3 段不差）→ 穩健版不加。B1 +2.40（t 4.43）vs +2.35；B2 +2.37；擋掉只有 122／42 次（穩健版很少挑到借券多的）。
  另報：今日發動（平常）借券比前 20% 的每筆減 0050，五段 −0.53／−1.59／−3.02／−2.88／−6.90，其他 −0.59／−0.76／−1.88／−1.58／−2.79
  → 四段明顯更差（差 1～4 個百分點）；網站今日發動加「借券放空多」提醒（描述性，非事先登錄的採用檢驗）。
"""
import glob, json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import robust_select as RS
from calm3 import sim_calm
from wr_confirm import stats
from sbl_filter import sbl_feats, attach
if __name__ == "__main__":
    S = pd.concat([pd.read_csv(f, dtype={"code": str}) for f in sorted(glob.glob(f"{REPO}/data/sbl/*.csv.gz"))], ignore_index=True); S["date"] = pd.to_datetime(S.date); S = S.drop_duplicates(["date", "code"], keep="last")
    Dn = RS.load("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01"); Do = RS.load("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01"); RT = {}
    for D, path in ((Dn, "/tmp/model_v2/vs0050_extra.pkl"), (Do, "/tmp/model_v2/vs0050_oldestonly_extra.pkl")):
        X, fl, ra = sbl_feats(path, S); attach(D, X, fl, ra); RT[path] = (X["days"], X["cols"], ra.rank(axis=1, pct=True)); del X, fl
    def filt(D, q):
        E = dict(D); G = D["F"].copy(); G["rk"] = G.groupby("date").cs.rank(ascending=False); b = (G.rk <= 20) & (G.p_ratio > 1 - q); G.loc[b, "vol20"] = np.inf; E["F"] = G; return E, int(b.sum())
    s0 = stats(pd.concat([sim_calm(Do, 20), sim_calm(Dn, 20)])); out = {"穩健版": s0}
    for lab, q in (("B1 排除前20%", 0.2), ("B2 排除前10%", 0.1)):
        (Eo, bo), (En, bn) = filt(Do, q), filt(Dn, q); s = stats(pd.concat([sim_calm(Eo, 20), sim_calm(En, 20)])); out[lab] = {**s, "擋掉次數": bo + bn}; print(lab, out[lab], flush=True)
    print("穩健版", s0)
    # 另報：今日發動（平常）依 sbl_ratio 分組
    parts = []
    for ev, path in (("/tmp/model_v2/vs0050_oldestonly.pkl", "/tmp/model_v2/vs0050_oldestonly_extra.pkl"), ("/tmp/model_v2/vs0050.pkl", "/tmp/model_v2/vs0050_extra.pkl")):
        E = pd.read_pickle(ev); E = E[(E.kind == "pk") & E.base & ~E.panic.astype(bool)].copy(); days, cols, PR = RT[path]
        pos = {d: i for i, d in enumerate(days)}; cj = {c: j for j, c in enumerate(cols)}; ti = E.date.map(pos); ji = E.code.map(cj); ok = (ti.notna() & ji.notna()).values
        v = np.full(len(E), np.nan); v[ok] = PR.values[ti[ok].astype(int), ji[ok].astype(int)]; E["pr"] = v; parts.append(E)
    E = pd.concat(parts); E["x"] = E.y20 - 0.6 - E.b20; E = E.dropna(subset=["x"]); E["seg"] = RS.seg(E.date)
    out["今日發動 依借券比分組（每筆減0050）"] = {s: {"前20%": round(float(E[(E.seg == s) & (E.pr > 0.8)].x.mean()), 2), "其他": round(float(E[(E.seg == s) & (E.pr <= 0.8)].x.mean()), 2), "無資料": round(float(E[(E.seg == s) & E.pr.isna()].x.mean()), 2)} for s in RS.SEGS}
    print(out["今日發動 依借券比分組（每筆減0050）"])
    s1 = out["B1 排除前20%"]
    c = {"①": s1["2011–2026"][0] > s0["2011–2026"][0] and s1["2011–2026"][1] >= 2.9, "②": sum(s1[k] >= s0[k] for k in RS.SEGS) >= 4, "③": s1["最大回落"] >= s0["最大回落"]}
    out["各項"] = c; out["通過"] = all(c.values()); print(c, "通過" if all(c.values()) else "不通過")
    json.dump(out, open(f"{REPO}/research/results/sbl_ratio.json", "w"), ensure_ascii=False, indent=1, default=str)
