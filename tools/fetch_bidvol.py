"""抓每日收盤「最後揭示買量／賣量」（漲停封單）到 out/bidvol-YEAR.csv.gz（研究用）。只存當天漲幅 ≥ 6% 的股票（漲停附近），省空間。
上市：MI_INDEX（ALLBUT0999）；上櫃：afterTrading/otc（欄位名稱含「最後買量」才存）。
欄位：date, market, code, close, chg, vol（成交股數）, bid_px, bid_vol（張）, ask_px, ask_vol（張）。"""
import datetime as dt, os, re, time, requests, pandas as pd
YEAR = int(os.environ["YEAR"]); H = {"User-Agent": "Mozilla/5.0"}; os.makedirs("out", exist_ok=True)
def num(s):
    s = re.sub(r"<[^>]+>", "", str(s)).replace(",", "").strip()
    try: return float(s)
    except Exception: return None
def get(url):
    for k in range(4):
        try:
            r = requests.get(url, headers=H, timeout=60)
            if r.status_code == 200: return r.json()
        except Exception: pass
        time.sleep(20 * (k + 1))
    return None
def pick(f, *keys):
    for i, x in enumerate(f):
        if all(k in x for k in keys): return i
    return None
rows, failed, shown = [], [], False; d = dt.date(YEAR, 1, 1); end = min(dt.date(YEAR, 12, 31), dt.date.today())
while d <= end:
    if d.weekday() < 5:
        j = get(f"https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={d:%Y%m%d}&type=ALLBUT0999&response=json")
        if j is None: failed.append(f"TWSE {d}")
        else:
            for t in j.get("tables") or []:
                f = t.get("fields") or []
                if "證券代號" not in f: continue
                ic, icl, isg, idf, iv = f.index("證券代號"), f.index("收盤價"), pick(f, "漲跌(+/-)"), f.index("漲跌價差"), f.index("成交股數")
                ibp, ibv, iap, iav = f.index("最後揭示買價"), f.index("最後揭示買量"), f.index("最後揭示賣價"), f.index("最後揭示賣量")
                for r in t.get("data") or []:
                    c, df_ = num(r[icl]), num(r[idf])
                    if c is None or df_ is None or "+" not in str(r[isg]) or c - df_ <= 0 or df_ / (c - df_) < 0.06: continue
                    rows.append({"date": d.isoformat(), "market": "TWSE", "code": str(r[ic]).strip(), "close": c, "chg": df_, "vol": num(r[iv]), "bid_px": num(r[ibp]), "bid_vol": num(r[ibv]), "ask_px": num(r[iap]), "ask_vol": num(r[iav])})
        time.sleep(3.5)
        j = get(f"https://www.tpex.org.tw/www/zh-tw/afterTrading/otc?date={d:%Y/%m/%d}&type=EW&response=json")
        if j is None: failed.append(f"TPEx {d}")
        elif j.get("tables") and (j["tables"][0].get("data") or []):
            t = j["tables"][0]; f = t.get("fields") or []
            if not shown: print("TPEx fields", f, flush=True); shown = True
            ic, icl, idf, iv = pick(f, "代號"), pick(f, "收盤"), pick(f, "漲跌"), pick(f, "成交股數")
            ibp, ibv, iap, iav = pick(f, "最後買價"), pick(f, "最後買量"), pick(f, "最後賣價"), pick(f, "最後賣量")
            if None not in (ic, icl, idf, iv, ibp, ibv, iap, iav):
                for r in t["data"]:
                    c, df_ = num(r[icl]), num(r[idf])
                    if c is None or df_ is None or df_ <= 0 or c - df_ <= 0 or df_ / (c - df_) < 0.06: continue
                    rows.append({"date": d.isoformat(), "market": "TPEX", "code": str(r[ic]).strip(), "close": c, "chg": df_, "vol": num(r[iv]), "bid_px": num(r[ibp]), "bid_vol": num(r[ibv]), "ask_px": num(r[iap]), "ask_vol": num(r[iav])})
        time.sleep(3.5)
    d += dt.timedelta(days=1)
df = pd.DataFrame(rows); df.to_csv(f"out/bidvol-{YEAR}.csv.gz", index=False, compression="gzip"); open(f"out/bidvol-{YEAR}.failed.txt", "w").write("\n".join(failed))
print(YEAR, "rows", len(df), df.groupby("market").date.nunique().to_dict() if len(df) else {}, "failed", len(failed))
