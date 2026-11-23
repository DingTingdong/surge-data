"""今日發動的漲停封單強度（第 37 輪；2026/11/23 18:35 登錄，封單資料還沒抓、看結果前寫下，看到結果後不修改）

起因：使用者把今日發動當主線之一；過去試過的價量、籌碼、營收、分數都沒讓平常時期的今日發動穩定賺錢。
  實戰常看「漲停封單」：收盤時漲停價上還掛著多少買單沒成交。這是之前沒用過的新資料。
資料：每日收盤「最後揭示買價／買量、賣價／賣量」（上市 MI_INDEX、上櫃 afterTrading/otc），只存漲幅 ≥ 6% 的股票 → data/bidvol/。
變數（今日發動當天）：封單比 L ＝ 收盤最後揭示買量（張）÷ 當天成交量（張）；收盤價不等於最後揭示買價、或最後揭示賣量 > 0（沒有鎖住）→ L ＝ 0。
樣本：vs0050 事件表的今日發動（pk）、分析範圍內、非恐慌期間；隔天開盤買、抱 20 個交易日；成本 0.6%；減 0050（含息）。
分組：每個日曆年內，依 L 分三等分（高／中／低；L ＝ 0 的都在低組）。
判斷（全部符合才在網站今日發動加「封單強」標記）：
  ① 高組減低組的每筆差，五段（2006–09、2011–15、2016–19、2020–23、2024–26）都 > 0，且以月份分群 NW t ≥ 2.9；
  ② 高組本身每筆減 0050 在五段中至少四段 > 0。
另報：三組隔天開盤跳空、20 天平均與中位數、贏 0050 比例；高組＋「營收強＋分數高」。
結果（11/24 01:00）：不通過，而且方向相反——封單越強越差。資料：上市 2006–2026（2013 年回補失敗，缺），今日發動 20,731 筆，84.8% 收盤鎖死。
  每筆減 0050（高／中／低）：2006–09 −1.02／+0.63／+0.07；2011–15 −2.16／−1.38／−0.89；2016–19 −3.35／−0.37／−2.24；2020–23 −2.76／−0.97／−0.44；
  2024–26 −5.80／−2.38／−0.67。高減低每月 −2.00（t −6.28），五段都為負；高組隔天平均開高 2.6%（低組 0.9%），贏 0050 只有 34%。
  → 隔天開盤才買時，封單強的已經開高、之後回落。事後發現（非事先假設的方向），但五段一致、t −6.3 → 網站今日發動加「封單過強」提醒（描述性）。
"""
import glob, json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
from vs0050 import nw_t
from watch_filters import seg, SEGS

def bidvol():
    B = pd.concat([pd.read_csv(f, dtype={"code": str}) for f in sorted(glob.glob(f"{REPO}/data/bidvol/*.csv.gz"))], ignore_index=True)
    B["date"] = pd.to_datetime(B.date); B = B[B.market == "TWSE"].drop_duplicates(["date", "code"], keep="last")
    B["locked"] = (B.bid_px.round(2) == B.close.round(2)) & ~(B.ask_vol.fillna(0) > 0)
    B["L"] = np.where(B.locked, B.bid_vol / (B.vol / 1000).replace(0, np.nan), 0.0); return B

def events():
    parts = []
    for ev, extra in (("/tmp/model_v2/vs0050_oldestonly.pkl", "/tmp/model_v2/vs0050_oldestonly_extra.pkl"), ("/tmp/model_v2/vs0050.pkl", "/tmp/model_v2/vs0050_extra.pkl")):
        E = pd.read_pickle(ev); E = E[(E.kind == "pk") & E.base & ~E.panic.astype(bool)].copy()
        X = pd.read_pickle(extra); pos = {d: i for i, d in enumerate(X["days"])}; cj = {c: j for j, c in enumerate(X["cols"])}
        C = pd.DataFrame(X["C"]).ffill(limit=60).values; O = X["O"]; N = len(X["days"])
        ti = E.date.map(pos).values; ji = E.code.map(cj).values; ok = ~pd.isna(ti) & ~pd.isna(ji); ti = np.where(ok, ti, 0).astype(int); ji = np.where(ok, ji, 0).astype(int)
        gap = np.where(ok & (ti + 1 < N), (O[np.minimum(ti + 1, N - 1), ji] / C[ti, ji] - 1) * 100, np.nan)
        yc = np.where(ok & (ti + 20 < N), (C[np.minimum(ti + 20, N - 1), ji] / C[ti, ji] - 1) * 100, np.nan)  # 當天收盤買、第 20 個交易日收盤賣
        E["gap"] = gap; E["yc"] = yc; parts.append(E)
    E = pd.concat(parts, ignore_index=True); E["seg"] = seg(E.date); return E[E.seg != "2010"].drop_duplicates(["date", "code"])

if __name__ == "__main__":
    B = bidvol(); E = events().merge(B[["date", "code", "locked", "L", "bid_vol", "ask_vol"]], on=["date", "code"], how="inner").dropna(subset=["y20", "b20"])
    E["x"] = E.y20 - 0.6 - E.b20; out = {"上市今日發動筆數（有封單資料）": int(len(E)), "鎖死比例%": round(float(E.locked.mean() * 100), 1)}
    E["yr"] = E.date.dt.year; E["grp"] = "低"
    for y, g in E.groupby("yr"):
        nz = g[g.L > 0]
        if len(nz) >= 6:
            q = nz.L.rank(method="first", pct=True); E.loc[nz.index, "grp"] = np.where(q > 2 / 3, "高", np.where(q > 1 / 3, "中", "低"))
    tab = {}
    for s in SEGS:
        z = E[E.seg == s]; tab[s] = {k: (int((z.grp == k).sum()), round(float(z[z.grp == k].x.mean()), 2) if (z.grp == k).any() else None) for k in ("高", "中", "低")}
    out["三組 每筆減0050（筆數, 平均）"] = tab
    mm = E.groupby([E.date.dt.to_period("M"), "grp"]).x.mean().unstack(); d = (mm["高"] - mm["低"]).dropna()
    out["高減低 每月 t"] = (round(float(d.mean()), 2), round(nw_t(d.reset_index(drop=True), 3), 2))
    out["高減低 五段"] = {s: (round(tab[s]["高"][1] - tab[s]["低"][1], 2) if tab[s]["高"][1] is not None and tab[s]["低"][1] is not None else None) for s in SEGS}
    out["三組 隔天跳空%"] = E.groupby("grp").gap.mean().round(2).to_dict(); out["三組 贏0050%"] = E.groupby("grp").x.apply(lambda v: round(float((v > 0).mean() * 100))).to_dict()
    c = {"①": all(v is not None and v > 0 for v in out["高減低 五段"].values()) and out["高減低 每月 t"][1] >= 2.9, "②": sum(tab[s]["高"][1] is not None and tab[s]["高"][1] > 0 for s in SEGS) >= 4}
    out["各項"] = c; out["通過"] = all(c.values())
    for k, v in out.items(): print(k, json.dumps(v, ensure_ascii=False, default=str), flush=True)
    json.dump(out, open(f"{REPO}/research/results/lock_strength.json", "w"), ensure_ascii=False, indent=1, default=str)
