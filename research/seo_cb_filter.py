"""現金增資、可轉債發行的排除（第 36 輪；2026/11/23 18:05 登錄，公告資料還沒抓、看結果前寫下，看到結果後不修改）

起因：強勢股常趁股價高時辦現金增資或發可轉債；研究上（seasoned equity offering／convertible issuance）發行後股價多半偏弱。
  想知道網站月選（穩健版 3 檔）要不要排除最近剛決議籌資的股票。
資料：觀測站重大訊息（data/mopsnews，2006–2026，依日期查詢的全文）。
事件（公告日，訊號只用公告日當天或之前）：
  SEO：主旨含「現金增資」，且主旨或內容含「董事會決議」「決議辦理」或「發行新股」（排除「盈餘轉增資」「資本公積轉增資」「員工」「私募」）；
  CB：主旨含「轉換公司債」，且主旨或內容含「董事會決議」「決議發行」或「募集」（排除私募）。
規則：
  X1 穩健版，新買進時跳過「換股日前 60 個交易日內有 SEO 或 CB 事件」的股票（續抱的不受影響）；
  X2 同 X1，但只看 SEO；X3 只看 CB（另報）。
採用條件（X1 全部符合才在網站加上）：① 2011–2026 每月減 0050 > 穩健版，且 t ≥ 2.9；② 五段中至少四段不比穩健版差；③ 最大回落不比穩健版深。
另報：月選前 20 名中有事件的比例；事件股與非事件股下個月減 0050；全部股票公告後 20／60 日減 0050（事件研究）。
實作補充（11/25 寫程式時、跑結果前寫下）：
  排除字詞（盈餘轉增資、資本公積轉增資、員工、私募）只檢查「主旨」：現金增資公告的內容幾乎都會寫到員工認購，若連內容一起排除會把全部 SEO 排掉。
  公告資料：TYPEK=sii 的查詢實際含上市＋上櫃（11/25 探測，otc 回傳相同內容），market 欄不可信；2025-05-28～06-30、2026-05-19～06-30 另外補抓。
  「60 個交易日內」：公告日在換股日（含）往前 60 個交易日內。報酬、持有規則、成本同 calm3.py 穩健版（等金額、0.6%）。
"""

import json, os, sys, glob, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
import robust_select as RS
from wr_confirm import stats

def events():
    N = pd.concat([pd.read_csv(f, dtype={"code": str, "date": str, "time": str}) for f in sorted(glob.glob(f"{REPO}/data/mopsnews/*.csv.gz"))], ignore_index=True)
    N = N.drop_duplicates(["date", "time", "code", "subject"]); N["detail"] = N.detail.fillna(""); N["subject"] = N.subject.fillna("")
    s, t = N.subject, N.subject + " " + N.detail
    seo = s.str.contains("現金增資") & t.str.contains("董事會決議|決議辦理|發行新股") & ~s.str.contains("盈餘轉增資|資本公積轉增資|員工|私募")
    cb = s.str.contains("轉換公司債") & t.str.contains("董事會決議|決議發行|募集") & ~s.str.contains("私募")
    N["d"] = pd.to_datetime(N.date, format="%Y%m%d", errors="coerce")
    return N[seo][["d", "code"]].dropna(), N[cb][["d", "code"]].dropna()

def evmap(D, days, ev):
    """code → 排序後的「事件所在交易日索引」陣列。"""
    di = np.searchsorted(days.values, ev.d.values)  # 公告日（非交易日就算下一個交易日）
    out = {}
    for c, i in zip(ev.code, di): out.setdefault(c, []).append(i)
    return {c: np.sort(np.array(v)) for c, v in out.items()}

def sim(D, skip=None, n=3, K=20, keep_rank=20, cost=0.6):
    O, Cff, lk, bidx, POS, CJ, ND = D["O"], D["Cff"], D["lockn"], np.asarray(D["bidx"], float), D["POS"], D["CJ"], D["ND"]
    G = {d: g for d, g in D["F"].groupby("date")}; rows, held, flag = [], [], []
    for a, b_ in zip(D["FDS"][:-1], D["FDS"][1:]):
        ie, ix = POS[a] + 1, POS[b_] + 1
        if ix >= ND: break
        g = G[a].sort_values("cs", ascending=False); rank = {c: i + 1 for i, c in enumerate(g.code)}; t = ie - 1
        hit = lambda c: skip is not None and c in skip and bool(((skip[c] <= t) & (skip[c] > t - 60)).any())
        for c in g.head(20).code: flag.append({"date": a, "code": c, "ev": hit(c) if skip is not None else False})
        keep = [c for c in held if rank.get(c, 1e9) <= keep_rank][:n]; new = []
        for c in g.head(K).assign(v=lambda z: z.vol20.fillna(np.inf)).sort_values(["v", "cs"], ascending=[True, False]).code.tolist() + g.code.iloc[K:].tolist():
            if len(keep) + len(new) >= n: break
            if c in keep or hit(c): continue
            j = CJ[c]
            if O[ie, j] > 0 and not lk[ie - 1, j]: new.append(c)
        to = 1.0 if not rows else len([c for c in held if c not in keep]) / n; rs, hn = [], []
        for c in keep + new:
            j = CJ[c]; o = O[ie, j] if O[ie, j] > 0 else Cff[ie - 1, j]; ex = O[ix, j] if O[ix, j] > 0 else Cff[ix, j]
            if not np.isfinite(ex): ex = Cff[ix - 1, j]
            if np.isfinite(ex) and np.isfinite(o) and o > 0: rs.append((ex / o - 1) * 100); hn.append(c)
        held = hn; rows.append({"date": a, "r": (np.mean(rs) if rs else 0) - cost * to, "b": (bidx[ix - 1] / bidx[ie - 1] - 1) * 100})
    T = pd.DataFrame(rows).set_index("date"); T.index = pd.DatetimeIndex(T.index); return T, pd.DataFrame(flag)

def top20_ev(D, skip):
    """月選前 20 名：有事件 vs 沒事件，下個月減 0050。"""
    O, Cff, bidx, POS, CJ, ND = D["O"], D["Cff"], np.asarray(D["bidx"], float), D["POS"], D["CJ"], D["ND"]; G = {d: g for d, g in D["F"].groupby("date")}; R = []
    for a, b_ in zip(D["FDS"][:-1], D["FDS"][1:]):
        ie, ix = POS[a] + 1, POS[b_] + 1
        if ix >= ND: break
        bm = (bidx[ix - 1] / bidx[ie - 1] - 1) * 100; t = ie - 1
        for c in G[a].sort_values("cs", ascending=False).head(20).code:
            j = CJ[c]; o = O[ie, j]; ex = O[ix, j] if O[ix, j] > 0 else Cff[ix - 1, j]
            if o > 0 and np.isfinite(ex): R.append({"ev": c in skip and bool(((skip[c] <= t) & (skip[c] > t - 60)).any()), "x": (ex / o - 1) * 100 - bm})
    return pd.DataFrame(R)

def evstudy(D, days, ev):
    Cff, bidx = D["Cff"], np.asarray(D["bidx"], float); CJ = D["CJ"]; out = {20: [], 60: []}
    for d, c in zip(ev.d, ev.code):
        if c not in CJ: continue
        i = int(np.searchsorted(days.values, np.datetime64(d))); j = CJ[c]
        for h in (20, 60):
            if i + 1 + h < len(days) and np.isfinite(Cff[i, j]) and np.isfinite(Cff[i + h, j]): out[h].append((Cff[i + h, j] / Cff[i, j] - 1) * 100 - (bidx[i + h] / bidx[i] - 1) * 100)
    return {f"{h}日": {"n": len(v), "平均": round(float(np.mean(v)), 2), "中位": round(float(np.median(v)), 2)} for h, v in out.items() if v}

if __name__ == "__main__":
    SEO, CB = events(); print("SEO 事件", len(SEO), "CB 事件", len(CB), flush=True)
    Dn = RS.load("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01"); Do = RS.load("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01")
    dn = pd.DatetimeIndex(pd.read_pickle("/tmp/model_v2/vs0050_extra.pkl")["days"]); do = pd.DatetimeIndex(pd.read_pickle("/tmp/model_v2/vs0050_oldestonly_extra.pkl")["days"])
    res = {"事件數": {"SEO": len(SEO), "CB": len(CB)}}
    for lab, ev in (("基準", None), ("X1 SEO或CB", pd.concat([SEO, CB])), ("X2 只SEO", SEO), ("X3 只CB", CB)):
        mo, mn = (evmap(Do, do, ev), evmap(Dn, dn, ev)) if ev is not None else (None, None)
        To, fo = sim(Do, mo); Tn, fn = sim(Dn, mn); s = stats(pd.concat([To, Tn])[["r", "b"]]); res[lab] = s
        if ev is not None:
            Z = pd.concat([top20_ev(Do, mo), top20_ev(Dn, mn)]); res[lab]["前20有事件比例%"] = round(float(Z.ev.mean() * 100), 1)
            res[lab]["前20 有事件／沒事件 下月減0050"] = (round(float(Z[Z.ev].x.mean()), 2), round(float(Z[~Z.ev].x.mean()), 2), int(Z.ev.sum()))
            res[lab]["事件研究（全部股票）"] = {"2006-09": evstudy(Do, do, ev[ev.d < "2010-01-01"]), "2011-26": evstudy(Dn, dn, ev[ev.d >= "2011-01-01"])}
        print(lab, json.dumps(res[lab], ensure_ascii=False, default=str), flush=True)
    b, x = res["基準"], res["X1 SEO或CB"]
    c = {"①": bool(x["2011–2026"][0] > b["2011–2026"][0] and x["2011–2026"][1] >= 2.9), "②": bool(sum(x[k] >= b[k] for k in RS.SEGS) >= 4), "③": bool(x["最大回落"] >= b["最大回落"])}
    res["判斷 X1"] = {"各項": c, "通過": all(c.values())}; print(res["判斷 X1"])
    json.dump(res, open(f"{REPO}/research/results/seo_cb_filter.json", "w"), ensure_ascii=False, indent=1, default=str)
