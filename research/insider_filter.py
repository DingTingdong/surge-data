"""內部人申報賣股的月選排除條件（第 16 輪；2026/11/23 10:35 登錄，資料回補之前寫下，看到結果後不修改）

依據：內部人（董監、經理人、大股東）比外部投資人更了解公司；台灣規定賣股要事前申報（證交法第 22 條之 2）。強勢股常見內部人趁高申報出脫。
資料：公開資訊觀測站「內部人持股轉讓事前申報」逐日資料（ajax_t56sb12），2006–2026，上市＋上櫃（tools/fetch_insider.py → data/insider/）。
訊號：換股日前 30 天內（申報日期），有內部人以「一般交易」（集中市場賣出）申報轉讓 → 標記「內部人申報賣股」。贈與、信託、鉅額、洽特定人等不算。
規則 IX：月選，但標記的股票不選、由下一名遞補；已持有的股票在換股日被標記也賣出。持有方式、成本 0.6%、五段期間同第 10 輪（robust_select.py）。
判斷同第 10 輪：① 2011–2026 每月減 0050 的 NW t ≥ 2.9；② 五段都 > 0；③ 最差段比月選高；④ 最大回落不比月選深 3 個百分點以上。符合 → 網站月選採用、卡片標「內部人申報賣股」。
補充登錄（11/23 13:50，資料仍未回補完、還沒看任何結果；網站已改成穩健版 3 檔）：
  規則 IX-C：穩健版 3 檔（前 20 名裡近 20 日波動最低 3 檔、還在前 20 名續抱），標記的股票不新買、已持有被標記也賣出。
  判斷：① 2011–2026 每月減 0050 > 穩健版，且 NW t ≥ 2.9；② 五段中至少四段不比穩健版差；③ 最大回落不比穩健版深。符合 → 網站採用。
另報：月選前 20 中被標記的比例、被標記與沒被標記的 20 日報酬（減 0050）分段比較；只算申報股數 ≥ 持股 10% 的版本。
結果（11/23 21:05）：資料限制——觀測站 report=SO 回傳的也是上市公司，上櫃完全沒抓到（38,139 筆，「一般交易」8,518 筆），只有上市股票會被標記。
  IX（月選 20 檔）：+1.96（t 3.73）vs +1.88，五段 1.12／1.68／1.35／3.92／0.57，最差段 0.57 > 0.42，回落 −31.4 vs −31.0 → 依原規則成立。
  IX-C（穩健版 3 檔，網站現行）：+2.56（t 4.34）vs +2.35，但 2016–19 1.74 < 1.96、2024–26 1.89 < 2.32（只有 3 段不差）→ 不成立。
  網站月選已改成穩健版，適用的是 IX-C → 網站不改。前 20 名中被標記的 3.3%；被標記的下個月減 0050 五段 +0.28／−1.56／−2.88／+9.16／−10.64，
  沒標記 +1.43／+2.19／+1.69／+5.44／+1.32（筆數 11～84，很少）。之後要補抓上櫃再驗一次。
加上櫃後（11/25 10:00，規則不改；上櫃 report=OY 補抓 2006–2026 完成，申報 55,994 筆、一般交易 12,611 筆）：
  IX（月選 20 檔）：+2.01（t 3.80）vs +1.88，五段 1.12／1.77／1.29／3.98／0.67，最差段 0.67 > 0.42，回落 −31.2 vs −31.0 → 仍成立（但網站用穩健版，不適用）。
  IX-C（穩健版，網站現行）：+2.38（t 4.19）vs +2.35，五段 2.52／2.94／1.44／3.20／1.57（穩健版 2.42／2.50／1.96／2.57／2.32）→ 2016–19、2024–26 較差，只有 3 段不差 → 仍不成立，網站不改。回落 −21 vs −23。
  前 20 名中被標記 4.5%；被標記的下月減 0050：+0.16／−1.26／+0.36／+4.24／−10.82，沒標記 +1.46／+2.24／+1.66／+5.53／+1.43（被標記筆數 17～93）。
  → 被標記的股票每段都比沒標記差（方向一致），但筆數少，放進穩健版 3 檔後效果不穩定。
"""
import glob, json, os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
sys.argv = sys.argv[:1]
import robust_select as RS

def load_insider():
    fs = sorted(glob.glob(f"{REPO}/data/insider/*.csv.gz")); D = pd.concat([pd.read_csv(f, dtype=str) for f in fs], ignore_index=True)
    col = lambda key, *also: [c for c in D.columns if key in c and all(a in c for a in also)][0]
    dc, cc, mc, sc, hc = col("申報日期"), col("代號"), col("轉讓方式"), col("轉讓股數", "預定轉讓方式"), col("目前持有股數", "自有")
    def roc(s):
        try: y, m, d = str(s).strip().split("/"); return pd.Timestamp(int(y) + 1911, int(m), int(d))
        except Exception: return pd.NaT
    num = lambda s: pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")
    out = pd.DataFrame({"date": D[dc].map(roc), "code": D[cc].astype(str).str.strip(), "method": D[mc].astype(str).str.strip(), "shares": num(D[sc]), "hold": num(D[hc])})
    out = out.dropna(subset=["date"]).drop_duplicates(); return out

def flag(F, I, frac=None):
    I = I[I.method.str.contains("一般交易")]
    if frac: I = I[I.shares >= frac * I.hold]
    F = F.copy(); F["ins"] = False
    for d in F.date.unique():
        hit = set(I[(I.date <= d) & (I.date >= d - pd.Timedelta(days=30))].code)
        F.loc[(F.date == d) & F.code.isin(hit), "ins"] = True
    return F

if __name__ == "__main__":
    I = load_insider(); print("申報筆數", len(I), "一般交易", int(I.method.str.contains("一般交易").sum()), "年份", I.date.dt.year.value_counts().sort_index().to_dict(), flush=True)
    A = RS.load("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01"); B = RS.load("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2005-07-01", "2010-03-01")
    res = {"月選（原本）": RS.evaluate(RS.sim(A, "cs"), RS.sim(B, "cs"))}
    for lab, frac in (("IX 排除內部人申報賣股", None), ("另報 只算申報 ≥ 持股10%", 0.10)):
        for D in (A, B): D["F"] = flag(D["F"], I, frac); D["F"]["csx"] = np.where(D["F"].ins, -1.0, D["F"].cs)
        res[lab] = RS.evaluate(RS.sim(A, "csx"), RS.sim(B, "csx"))
        if frac is None:
            FF = pd.concat([B["F"][(B["F"].date >= "2006-01-01") & (B["F"].date < "2010-01-01")], A["F"]]); top = FF.sort_values("cs", ascending=False).groupby("date").head(20)
            res["月選前20中被標記%"] = round(float(top.ins.mean() * 100), 1)
            by = top.assign(s=RS.seg(pd.DatetimeIndex(top.date))).groupby(["s", "ins"]).apply(lambda z: (round(float((z.y20 - z.b20).mean()), 2), int(len(z)))).unstack()
            res["分段 被標記/沒標記 20日減0050（筆數）"] = {k: (v.get(True), v.get(False)) for k, v in by.to_dict(orient="index").items()}
    b0, e = res["月選（原本）"], res["IX 排除內部人申報賣股"]
    res["IX 採用"] = bool(e["2011–2026 減0050"][1] >= 2.9 and all(v > 0 for v in e["五段"].values()) and e["最差段"] > b0["最差段"] and e["2011–2026 最大回落"] >= b0["2011–2026 最大回落"] - 3)
    # IX-C（11/23 13:50 補充登錄）：穩健版 3 檔，標記的不新買、已持有被標記也賣出（分數設成 −1 → 掉出前 20）
    from calm3 import sim_calm
    from wr_confirm import stats
    Dn = RS.load("/tmp/model_v2/vs0050_extra.pkl", "2011-01-01", "2027-01-01"); Do = RS.load("/tmp/model_v2/vs0050_oldestonly_extra.pkl", "2006-01-01", "2010-01-01")
    s0 = stats(pd.concat([sim_calm(Do, 20), sim_calm(Dn, 20)]))
    for D in (Dn, Do): D["F"] = flag(D["F"], I, None); D["F"]["cs"] = np.where(D["F"].ins, -1.0, D["F"].cs)
    s1 = stats(pd.concat([sim_calm(Do, 20), sim_calm(Dn, 20)]))
    res["穩健版"] = s0; res["IX-C 穩健版排除內部人申報賣股"] = s1
    res["IX-C 採用"] = bool(s1["2011–2026"][0] > s0["2011–2026"][0] and s1["2011–2026"][1] >= 2.9 and sum(s1[k] >= s0[k] for k in RS.SEGS) >= 4 and s1["最大回落"] >= s0["最大回落"])
    res["資料限制"] = "觀測站 report=SO 回傳的也是上市公司，上櫃完全沒抓到 → 只有上市股票會被標記"
    for k, v in res.items(): print("==", k, json.dumps(v, ensure_ascii=False, default=str))
    json.dump(res, open(f"{REPO}/research/results/insider_filter.json", "w"), ensure_ascii=False, indent=1, default=str)
