"""今日發動「營收強＋分數高」：當天收盤前買 vs 隔天開盤買（第 38 輪；2026/11/23 22:25 登錄，看結果前寫下，看到結果後不修改）

起因：盤中通知已經會在這類股票碰到漲停時發出。使用者如果看到通知當天就買，價格大約是當天收盤（漲停價），
  不必等隔天開盤；過去發現今日發動隔天平均開高約 1%。但漲停「鎖死」時收盤買不到，要用封單資料分辨。
樣本：vs0050 事件表的今日發動（pk）、分析範圍內、非恐慌期間，且「營收強＋分數高」（營收年增 ≥ 20% 且創 12 個月新高、分數 ≥ 0.8）。
  封單資料只有上市（data/bidvol），所以只用上市股票。
比較（同樣抱到訊號日後第 20 個交易日收盤，成本 0.6%，減 0050 同期）：
  A 隔天開盤買（現行）；B 當天收盤買，但只算「收盤沒有鎖死」的（收盤最後揭示賣量 > 0，買得到）；B 的同一批股票用 A 的算法當對照。
判斷（全部符合才在網站和通知寫「可以當天收盤前買」）：① 同一批沒鎖死的股票，B 減 A 每筆平均 ≥ +1%，五段中至少四段 > 0；
  ② B 本身每筆減 0050 在五段中至少四段 > 0。另報：鎖死比例、鎖死的隔天開盤跳空。
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
from lock_strength import bidvol, events
from base_breakout_rev import revenue
from watch_filters import SEGS
if __name__ == "__main__":
    B = bidvol(); E = events().merge(B[["date", "code", "locked"]], on=["date", "code"], how="inner")
    R = revenue(); E = pd.merge_asof(E.sort_values("date"), R.sort_values("pub"), left_on="date", right_on="pub", by="code", direction="backward")
    E = E[(E.cs >= 0.8) & E.r1.fillna(False).astype(bool) & E.r2.fillna(False).astype(bool)].dropna(subset=["y20", "yc", "b20"]).copy()
    E["xa"] = E.y20 - 0.6 - E.b20; E["xb"] = E.yc - 0.6 - E.b20; out = {"筆數（上市、營收強＋分數高）": int(len(E)), "鎖死比例%": round(float(E.locked.mean() * 100), 1),
                                                                       "鎖死的隔天跳空%": round(float(E[E.locked].gap.mean()), 2), "沒鎖死的隔天跳空%": round(float(E[~E.locked].gap.mean()), 2)}
    U = E[~E.locked]; tab = {}
    for s in SEGS:
        z = U[U.seg == s]; tab[s] = {"筆數": int(len(z)), "A 隔天開盤": round(float(z.xa.mean()), 2) if len(z) else None, "B 當天收盤": round(float(z.xb.mean()), 2) if len(z) else None}
    out["沒鎖死的（每筆減0050）"] = tab; out["全部 B減A"] = round(float((U.xb - U.xa).mean()), 2)
    c = {"①": out["全部 B減A"] >= 1 and sum((tab[s]["B 當天收盤"] or 0) - (tab[s]["A 隔天開盤"] or 0) > 0 for s in SEGS if tab[s]["筆數"]) >= 4,
         "②": sum((tab[s]["B 當天收盤"] or -1) > 0 for s in SEGS if tab[s]["筆數"]) >= 4}
    out["各項"] = c; out["通過"] = all(c.values())
    for k, v in out.items(): print(k, json.dumps(v, ensure_ascii=False, default=str), flush=True)
    json.dump(out, open(f"{REPO}/research/results/gs_close_entry.json", "w"), ensure_ascii=False, indent=1, default=str)
