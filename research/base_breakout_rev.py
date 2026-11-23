"""低基期突破＋營收催化（第 30 輪；2026/11/23 14:55 登錄，看結果前寫下，看到結果後不修改）

起因：第 29 輪單靠價量型態沒有優勢（每筆減 0050 −1.3%）。使用者原本的方法要求「必須有明確產業或資金催化」「營收改善」；
  題材無法量化，改用營收當催化的代理。
訊號：第 29 輪同樣的「低基期第一根放量紅 K 突破」（放量 2 倍），再加上訊號日已公布的最近一個月營收（每月 10 日後才算公布）：
  R1 營收年增 ≥ 20%；R2 營收創 12 個月新高（≥ 前 12 個月最高，至少有 6 個月資料）；R3 兩者都要。
進出場同第 29 輪：E3 三日法則（主）、E1 隔天開盤；抱 20 個交易日；成本 0.6%；比 0050（含息）。
判斷（R3＋E3 為主；全部符合才在網站加分頁）：① 五段每筆減 0050 平均都 > 0；② 全部平均 > +1%、每月平均 NW t ≥ 2.9；③ 贏 0050 ≥ 50%。
另報：R1、R2、E1 的結果；每年訊號數；減同日全部股票平均。
"""
import glob, json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
from base_breakout import events, summarize, seg, SEGS

def revenue():
    fs = sorted(glob.glob(f"{REPO}/data/revenue/*.csv*") + glob.glob(f"{REPO}/data/old/revenue/*.csv*") + glob.glob(f"{REPO}/data/older/revenue/*.csv*") + glob.glob(f"{REPO}/data/oldest/revenue/*.csv*"))
    R = pd.concat([pd.read_csv(f, dtype={"code": str}, usecols=lambda c: c in ("ym", "code", "rev", "yoy")) for f in fs], ignore_index=True).drop_duplicates(["ym", "code"], keep="last")
    R = R.sort_values(["code", "ym"]); R["hi12"] = R.groupby("code").rev.transform(lambda s: s.shift(1).rolling(12, min_periods=6).max())
    R["pub"] = pd.to_datetime(R.ym + "-01") + pd.DateOffset(months=1) + pd.Timedelta(days=10)
    R["r1"] = R.yoy >= 20; R["r2"] = R.rev >= R.hi12; return R[["code", "pub", "r1", "r2", "yoy"]].dropna(subset=["pub"])

if __name__ == "__main__":
    E = pd.concat([events("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01"), events("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01")], ignore_index=True)
    E["seg"] = seg(E.date); R = revenue()
    E = pd.merge_asof(E.sort_values("date"), R.sort_values("pub"), left_on="date", right_on="pub", by="code", direction="backward")
    E["r1"] = E.r1.fillna(False).astype(bool); E["r2"] = E.r2.fillna(False).astype(bool); E["r3"] = E.r1 & E.r2; out = {"營收對到的比例%": round(float(E.pub.notna().mean() * 100), 1)}
    for rv, lab in (("r3", "R3 年增≥20%且創12月新高"), ("r1", "R1 年增≥20%"), ("r2", "R2 創12月新高")):
        for mode in ("E3", "E1"):
            z = E[E[rv] & (E["mode"] == mode)]; k = f"{lab} {mode}"; out[k] = summarize(z, "xH")
            out[k]["減同日全部股票"] = {s: round(float(z[z.seg == s].uH.mean()), 2) for s in SEGS if (z.seg == s).any()}
            print(k, json.dumps(out[k], ensure_ascii=False), flush=True)
    out["每年訊號數 R3 E1"] = E[E.r3 & (E["mode"] == "E1")].groupby(E.date.dt.year).size().to_dict()
    m = out["R3 年增≥20%且創12月新高 E3"]
    c = {"①": all(m[s]["平均"] > 0 for s in SEGS if s in m), "②": m["全部"]["平均"] > 1 and m["全部"]["每月平均 t"] >= 2.9, "③": m["全部"]["贏0050%"] >= 50}
    out["各項"] = c; out["通過"] = all(c.values()); print("每年", out["每年訊號數 R3 E1"], "\n各項", c, "通過" if all(c.values()) else "不通過")
    json.dump(out, open(f"{REPO}/research/results/base_breakout_rev.json", "w"), ensure_ascii=False, indent=1, default=str)
