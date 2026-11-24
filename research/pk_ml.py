"""今日發動：逐年滾動訓練的模型排序（第 47 輪；2026/11/24 17:40 事先登錄，看結果前寫下，看到結果後不修改）

起因：11/21 normal_model.py 的模型在 2024–26 排序很一致（最高一成 +5.35、最低一成 −4.22），但只用 2020–22 訓練、只檢驗一段；
  使用者要讓今日發動變好。這輪改成 2006–2026 逐年滾動、五段都檢驗，用和月選相同的標準。
樣本：vs0050 事件表的今日發動（kind=pk），分析範圍內、非恐慌期間；2005–2010（舊）與 2010–2026（新）兩份，重複的保留新的。
  報酬：隔天開盤買、抱 20 個交易日（y20），扣 0.6%，減 0050 同期（b20）＝x。
特徵（訊號日收盤就知道、2006–2026 都算得出來的，共 22 個；不用籌碼，因為早年沒有）：
  當天漲幅、開盤跳空、實體、上影線、成交值（log）、量比（÷前 20 日均值）、前 20 日日波動、近 5／20／60／120 日漲幅、
  離 60／250 日高點、離 20／60 日均線、近 60 日漲停次數、股價（log）、
  營收：最近已公布月年增、年增 ≥ 20%、創 12 個月新高；綜合分數 cs；
  大盤：等權指數近 20 日漲幅、站上 60 日線比例（分析範圍內）。
模型：梯度提升樹（HistGradientBoostingRegressor，max_depth 3、learning_rate 0.05、max_iter 200、min_samples_leaf 100，固定不調）；目標＝x。
滾動：測試年 Y＝2007…2026（2010 不算，同其他輪），用 Y 之前所有年份訓練，且剔除訓練樣本中訊號日在 Y 年前 30 個交易日內的（避免報酬期重疊）。
選取：每個測試年，門檻＝訓練樣本預測值的第 80 百分位（只看訓練資料），測試年預測 ≥ 門檻的就選。
判斷（同第 32 輪）：被選的那群 ① 五段（2006–09＝2007–09、2011–15、2016–19、2020–23、2024–26）每筆減 0050 平均都 > 0 且每段 ≥ 20 筆；
  ② 全部平均 > 0、以進場月份分群 Newey-West t ≥ 2.9。通過 → 網站今日發動改用模型分數排序、標「模型前 20%」，並往前記錄；不通過 → 照實報告。
另報：測試年預測值十分位的平均（看排序是否一致）、被選比例、和「營收強＋分數高」比。
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
from vs0050 import nw_t
from base_breakout_rev import revenue
from sklearn.ensemble import HistGradientBoostingRegressor
SEGS = ["2006-09", "2011-15", "2016-19", "2020-23", "2024-26"]
def seg(d): return np.where(d < pd.Timestamp("2010-01-01"), "2006-09", np.where(d < pd.Timestamp("2011-01-01"), "2010", np.where(d < pd.Timestamp("2016-01-01"), "2011-15", np.where(d < pd.Timestamp("2020-01-01"), "2016-19", np.where(d < pd.Timestamp("2024-01-01"), "2020-23", "2024-26")))))
FEATS = ["ch", "gap", "body", "upper", "lamt", "vr", "vol20", "r5", "r20", "r60", "r120", "dh60", "dh250", "m20", "m60", "nlim60", "lpx", "yoy", "r1", "r2", "cs", "mk20", "mkab"]

def feats(ev, extra):
    E = pd.read_pickle(ev); E = E[(E.kind == "pk") & E.base & ~E.panic.astype(bool)].copy()
    X = pd.read_pickle(extra); days, cols = X["days"], X["cols"]
    C = pd.DataFrame(X["C"], index=days, columns=cols); O = pd.DataFrame(X["O"], index=days, columns=cols)
    H = pd.DataFrame(X["H"], index=days, columns=cols); L = pd.DataFrame(X["L"], index=days, columns=cols)
    A = pd.DataFrame(np.asarray(X["A"], float), index=days, columns=cols); pc = pd.DataFrame(X["prevc"], index=days, columns=cols)
    inb = pd.DataFrame(np.asarray(X["inb"]).astype(bool), index=days, columns=cols)
    r = C.pct_change(fill_method=None); ch = (C / pc - 1) * 100
    lm = np.where(days < pd.Timestamp("2015-06-01"), 6.5, 9.5)[:, None]
    P = {"ch": ch, "gap": O / pc - 1, "body": C / O - 1, "upper": (H - C) / (H - L).replace(0, np.nan), "lamt": np.log(A.replace(0, np.nan)),
         "vr": A / A.shift(1).rolling(20, min_periods=10).mean(), "vol20": r.shift(1).rolling(20, min_periods=15).std(),
         "r5": C / C.shift(5) - 1, "r20": C / C.shift(20) - 1, "r60": C / C.shift(60) - 1, "r120": C / C.shift(120) - 1,
         "dh60": C / C.rolling(60, min_periods=30).max() - 1, "dh250": C / C.rolling(250, min_periods=120).max() - 1,
         "m20": C / C.rolling(20, min_periods=15).mean() - 1, "m60": C / C.rolling(60, min_periods=45).mean() - 1,
         "nlim60": pd.DataFrame((ch.values >= lm).astype(float), index=days, columns=cols).shift(1).rolling(60, min_periods=1).sum(), "lpx": np.log(pc)}
    ew = r.where(inb).mean(axis=1); ewi = (1 + ew.fillna(0)).cumprod()
    mk20 = ewi / ewi.shift(20) - 1; mkab = (C > C.rolling(60, min_periods=45).mean()).where(inb).mean(axis=1)
    pos = {d: i for i, d in enumerate(days)}; cj = {c: j for j, c in enumerate(cols)}
    ti = E.date.map(pos); ji = E.code.map(cj); ok = ti.notna() & ji.notna(); E = E[ok].copy(); ti = ti[ok].astype(int).values; ji = ji[ok].astype(int).values
    for k, M in P.items(): E[k] = np.asarray(M, float)[ti, ji]
    E["mk20"] = mk20.values[ti]; E["mkab"] = mkab.values[ti]
    return E

if __name__ == "__main__":
    Eo = feats("/tmp/model_v2/vs0050_oldestonly.pkl", "/tmp/model_v2/vs0050_oldestonly_extra.pkl")
    En = feats("/tmp/model_v2/vs0050.pkl", "/tmp/model_v2/vs0050_extra.pkl")
    E = pd.concat([Eo[Eo.date < En.date.min()], En], ignore_index=True).drop_duplicates(["date", "code"], keep="last")
    R = revenue(); E = pd.merge_asof(E.sort_values("date"), R.sort_values("pub"), left_on="date", right_on="pub", by="code", direction="backward")
    E["r1"] = E.r1.astype(float); E["r2"] = E.r2.astype(float)
    E = E.dropna(subset=["y20", "b20"]).reset_index(drop=True); E["x"] = E.y20 - 0.6 - E.b20; E["seg"] = seg(E.date); E["yr"] = E.date.dt.year
    E["gs"] = (E.cs >= 0.8) & (E.r1 > 0) & (E.r2 > 0)
    print("事件", len(E), E.date.min().date(), E.date.max().date(), flush=True)
    td = np.array(sorted(E.date.unique())); out_rows = []
    for Y in range(2007, 2027):
        if Y == 2010: continue
        start = pd.Timestamp(f"{Y}-01-01"); prior = td[td < np.datetime64(start)]
        cut = prior[-30] if len(prior) > 30 else prior[0]
        tr = E[E.date < cut]; te = E[E.yr == Y].copy()
        if len(tr) < 500 or not len(te): continue
        m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=100, random_state=0).fit(tr[FEATS], tr.x)
        th = np.percentile(m.predict(tr[FEATS]), 80); te["p"] = m.predict(te[FEATS]); te["sel"] = te.p >= th
        te["dec"] = pd.qcut(te.p.rank(method="first"), 10, labels=False) + 1
        out_rows.append(te); print(Y, "train", len(tr), "test", len(te), "sel", int(te.sel.sum()), "sel x", round(te[te.sel].x.mean(), 2), "all x", round(te.x.mean(), 2), flush=True)
    T = pd.concat(out_rows, ignore_index=True); T = T[T.seg != "2010"]
    def summ(z):
        mm = z.groupby(z.date.dt.to_period("M")).x.mean()
        sg = {s: (int((z.seg == s).sum()), round(float(z[z.seg == s].x.mean()), 2) if (z.seg == s).any() else None) for s in SEGS}
        return {"筆數": int(len(z)), "平均": round(float(z.x.mean()), 2), "中位": round(float(z.x.median()), 2), "贏0050%": round(float((z.x > 0).mean() * 100)),
                "t": round(nw_t(mm.reset_index(drop=True), 3), 2) if len(mm) > 10 else None, "五段": sg}
    res = {"全部": summ(T), "模型前20%": summ(T[T.sel]), "營收強＋分數高": summ(T[T.gs]), "模型前20%且營收強＋分數高": summ(T[T.sel & T.gs]),
           "十分位": {int(k): round(float(v), 2) for k, v in T.groupby("dec").x.mean().items()},
           "十分位（2024-26）": {int(k): round(float(v), 2) for k, v in T[T.seg == "2024-26"].groupby("dec").x.mean().items()}}
    s = res["模型前20%"]; ok = all(v[0] >= 20 and v[1] is not None and v[1] > 0 for v in s["五段"].values()) and s["平均"] > 0 and (s["t"] or 0) >= 2.9
    res["通過"] = bool(ok)
    for k, v in res.items(): print(k, json.dumps(v, ensure_ascii=False, default=str), flush=True)
    json.dump(res, open(f"{REPO}/research/results/pk_ml.json", "w"), ensure_ascii=False, indent=1, default=str)
