"""抓期交所：臺指選擇權三大法人買賣權分計（下載只開放約 3 年）→ data/taifex/opt_inst.csv；Put/Call 比（2007 起）→ data/taifex/pcratio.csv。
每次抓一個月；已有的日期不重抓。"""
import datetime as dt, io, os, time, requests, pandas as pd
H = {"User-Agent": "Mozilla/5.0"}; os.makedirs("data/taifex", exist_ok=True)
def down(url, data):
    t = requests.post(url, data=data, headers=H, timeout=90).content.decode("cp950", errors="replace")
    if not t.startswith("日期"): return None
    return pd.read_csv(io.StringIO(t), dtype=str, index_col=False)
def months(a, b):
    d = a.replace(day=1)
    while d <= b:
        e = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1); yield d, min(e, b); d = e + dt.timedelta(days=1)
today = (dt.datetime.utcnow() + dt.timedelta(hours=8)).date()
for name, url, mk, start in (("opt_inst", "https://www.taifex.com.tw/cht/3/callsAndPutsDateDown", lambda a, b: {"queryStartDate": a, "queryEndDate": b, "commodityId": "TXO"}, dt.date(2023, 11, 1)),
                             ("pcratio", "https://www.taifex.com.tw/cht/3/pcRatioDown", lambda a, b: {"queryStartDate": a, "queryEndDate": b}, dt.date(2007, 1, 1))):
    f = f"data/taifex/{name}.csv"; old = pd.read_csv(f, dtype=str) if os.path.exists(f) else None; have = set(old["日期"]) if old is not None else set(); parts = [old] if old is not None else []; miss = []
    for a, b in months(start, today):
        if all(x in have for x in pd.bdate_range(a, b).strftime("%Y/%m/%d")) and b < today - dt.timedelta(days=7): continue
        df = down(url, mk(a.strftime("%Y/%m/%d"), b.strftime("%Y/%m/%d")))
        if df is None:
            # 一個月太長就拆成一週
            for w0 in pd.date_range(a, b, freq="7D"):
                w1 = min(w0 + pd.Timedelta(days=6), pd.Timestamp(b)); d2 = down(url, mk(w0.strftime("%Y/%m/%d"), w1.strftime("%Y/%m/%d")))
                if d2 is None: miss.append(f"{w0:%Y/%m/%d}")
                else: parts.append(d2)
                time.sleep(2)
        else: parts.append(df)
        time.sleep(2)
    if parts:
        D = pd.concat(parts, ignore_index=True); D.columns = [c.strip() for c in D.columns]; D = D.loc[:, ~D.columns.str.startswith("Unnamed")]
        D = D.drop_duplicates().sort_values("日期"); D.to_csv(f, index=False); print(name, "筆數", len(D), "日期", D["日期"].min(), "～", D["日期"].max(), "缺", miss[:10], len(miss))
