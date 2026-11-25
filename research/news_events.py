"""公告事件：自結獲利、法說會、簽訂合約／訂單（第 55 輪；2026/11/25 17:40 事先登錄，看結果前寫下，看到結果後不修改）

起因：使用者要「能做就都做」；公告資料（data/mopsnews，2006–2026，上市＋上櫃）已補完，這是少數有完整資料、還沒測的方向。
事件（主旨去空白後判斷；同一公司同一類 20 個交易日內只算第一則）：
  E1 自結獲利：主旨含「自結」且含 盈餘／損益／獲利／淨利／財務報表／合併報表／EPS，排除 流動性、營收；
     E1+＝內容讀得到每股盈餘且為正（「每股盈餘」後 20 字內的第一個數字，前面沒有負號、括號、「虧」「損」）；E1−＝讀得到且為負。
  E2 法說會：主旨含「法人說明會」。
  E3 合約／訂單：主旨含（簽訂／簽署／得標／取得…訂單）且含（合約／契約／訂單／供應／銷售／採購），排除 解除、終止、報導、報載、澄清。
進場：公告日的下一個交易日開盤（公告日非交易日則用之後第一個交易日的下一日；開盤一價漲停買不到不算）；
  持有 20、60 個交易日；扣 0.6%；比較 0050（含息）與同日進場的全部股票平均。資料：2006–2009、2011–2026（2010 不算），分析範圍內股票。
判斷（每類各自）：60 日減全部股票平均 > 0，以公告月份分群 NW t ≥ 2.9，五段（2006–09、2011–15、2016–19、2020–23、2024–26）至少 4 段 > 0 → 成立。
  成立的類別 → 第二步（同樣事先寫下）：穩健版換股時，前 20 名裡換股日前 30 個交易日內有該事件的股票優先（在低波動排序前），
  用月選標準（每月比 0050 ≥ 穩健版 + 0.3、t ≥ 2.9、五段都正、回落不更深）判斷能否加入月選；沒成立的類別只報告。
"""
import json, os, sys, glob, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
from vs0050 import nw_t
COST = 0.6; SEGS = ["2006-09", "2011-15", "2016-19", "2020-23", "2024-26"]
def seg(d): return "2006-09" if d.year < 2010 else "2010" if d.year < 2011 else "2011-15" if d.year < 2016 else "2016-19" if d.year < 2020 else "2020-23" if d.year < 2024 else "2024-26"

def eps_sign(t):
    m = re.search(r"每股盈餘[^0-9\-－(（]{0,20}([\-－(（]?)\s*(\d+(?:\.\d+)?)", t)
    if not m: return np.nan
    pre = t[max(0, m.start(1) - 3):m.start(1) + 1]
    return -1.0 if (m.group(1) or "虧" in pre or "損" in pre) else 1.0

def events():
    N = pd.concat([pd.read_csv(f, dtype={"code": str, "date": str, "time": str}) for f in sorted(glob.glob(f"{REPO}/data/mopsnews/*.csv.gz"))]).drop_duplicates(["date", "time", "code", "subject"]).reset_index(drop=True)
    s = N.subject.fillna("").str.replace(r"\s+", "", regex=True); dt_ = N.detail.fillna("").str.replace(r"\s+", "", regex=True)
    N["d"] = pd.to_datetime(N.date, format="%Y%m%d", errors="coerce")
    e1 = s.str.contains("自結") & s.str.contains("盈餘|損益|獲利|淨利|財務報表|合併報表|EPS") & ~s.str.contains("流動性|營收")
    sign = pd.Series(np.nan, index=N.index); sign[e1] = (dt_[e1]).map(eps_sign)
    e2 = s.str.contains("法人說明會")
    e3 = s.str.contains("簽訂|簽署|得標|取得.{0,6}訂單|接獲") & s.str.contains("合約|契約|訂單|供應|銷售|採購") & ~s.str.contains("解除|終止|報導|報載|澄清")
    out = {"E1 自結獲利（全部）": N[e1], "E1+ 自結獲利為正": N[e1 & (sign > 0)], "E1− 自結為虧損": N[e1 & (sign < 0)], "E2 法說會": N[e2], "E3 合約／訂單": N[e3]}
    return {k: v[["d", "code"]].dropna() for k, v in out.items()}, {"E1 讀到正負的比例%": round(float(sign[e1].notna().mean() * 100), 1)}

def study(path, start, end, EV):
    X = pd.read_pickle(path); days = pd.DatetimeIndex(X["days"]); N = len(days); cj = {c: j for j, c in enumerate(X["cols"])}
    C = pd.DataFrame(X["C"]).ffill(limit=5).values; O = np.asarray(X["O"], float); H = np.asarray(X["H"], float); L = np.asarray(X["L"], float)
    inb = np.asarray(X["inb"]).astype(bool); b = np.asarray(X["bidx"], float)
    RU = {}
    for Hh in (20, 60):
        R = np.full(N, np.nan)
        for ie in range(1, N - Hh + 1):
            rr = C[ie + Hh - 1] / np.where(O[ie] > 0, O[ie], np.nan) - 1; m = inb[ie - 1] & np.isfinite(rr)
            if m.sum() > 50: R[ie] = np.nanmean(rr[m]) * 100 - COST
        RU[Hh] = R
    t0 = int(np.searchsorted(days.values, np.datetime64(start))); t1 = int(np.searchsorted(days.values, np.datetime64(end))); out = {}
    for k, ev in EV.items():
        rows, last = [], {}
        for d, c in sorted(zip(ev.d, ev.code)):
            j = cj.get(c)
            if j is None: continue
            i = int(np.searchsorted(days.values, np.datetime64(d)))   # 公告日（或之後第一個交易日）
            if not (t0 <= i < t1) or i + 61 >= N or not inb[min(i, N - 1), j]: continue
            if c in last and i - last[c] < 20: continue
            last[c] = i; ie = i + 1
            if not (O[ie, j] > 0) or (O[ie, j] == H[ie, j] == L[ie, j] and C[ie - 1, j] > 0 and O[ie, j] / C[ie - 1, j] > 1.094): continue
            r = {"date": days[i]}
            for Hh in (20, 60):
                ix = ie + Hh - 1; rH = (C[ix, j] / O[ie, j] - 1) * 100 - COST; r[f"x{Hh}"] = rH - (b[ix] / b[ie - 1] - 1) * 100; r[f"u{Hh}"] = rH - RU[Hh][ie]
            rows.append(r)
        out[k] = pd.DataFrame(rows)
    return out

def summ(Z):
    Z = Z[Z.seg != "2010"]
    if not len(Z): return {"筆數": 0}
    m = Z.groupby(Z.date.dt.to_period("M")).u60.mean()
    sg = {s: round(float(Z[Z.seg == s].u60.mean()), 2) if (Z.seg == s).any() else None for s in SEGS}
    r = {"筆數": int(len(Z)), "60日減全部": round(float(Z.u60.mean()), 2), "60日減全部 中位": round(float(Z.u60.median()), 2), "t": round(nw_t(m.reset_index(drop=True), 3), 2),
         "五段60日減全部": sg, "20日減全部": round(float(Z.u20.mean()), 2), "20日減0050": round(float(Z.x20.mean()), 2), "60日減0050": round(float(Z.x60.mean()), 2)}
    r["成立"] = bool(r["60日減全部"] > 0 and r["t"] >= 2.9 and sum(v is not None and v > 0 for v in sg.values()) >= 4)
    return r

if __name__ == "__main__":
    EV, info = events(); print({k: len(v) for k, v in EV.items()}, info, flush=True)
    A = study("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01", EV); B = study("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01", EV)
    res = {"資訊": info}
    for k in EV:
        Z = pd.concat([A[k], B[k]], ignore_index=True)
        if len(Z): Z["seg"] = Z.date.map(seg)
        res[k] = summ(Z) if len(Z) else {"筆數": 0}; print(k, json.dumps(res[k], ensure_ascii=False), flush=True)
    res["成立的類別"] = [k for k in EV if res[k].get("成立")]; print("成立", res["成立的類別"])
    json.dump(res, open(f"{REPO}/research/results/news_events.json", "w"), ensure_ascii=False, indent=1, default=str)
