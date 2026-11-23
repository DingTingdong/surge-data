"""每天補抓上市「最後揭示買賣量」（只存漲幅 ≥ 6%）到 data/bidvol/YYYY.csv.gz，給網站今日發動的「封單過強」提醒用。上櫃沒有買量資料。"""
import datetime as dt, os, re, time, requests, pandas as pd
H = {"User-Agent": "Mozilla/5.0"}
num = lambda s: (lambda x: float(x) if re.fullmatch(r"-?\d+(\.\d+)?", x) else None)(re.sub(r"<[^>]+>", "", str(s)).replace(",", "").strip())
today = (dt.datetime.utcnow() + dt.timedelta(hours=8)).date(); fn = f"data/bidvol/{today.year}.csv.gz"
old = pd.read_csv(fn, dtype={"code": str}) if os.path.exists(fn) else pd.DataFrame()
have = set(old.date) if len(old) else set(); rows = []
for k in range(6, -1, -1):
    d = today - dt.timedelta(days=k)
    if d.weekday() >= 5 or d.isoformat() in have: continue
    try: j = requests.get(f"https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={d:%Y%m%d}&type=ALLBUT0999&response=json", headers=H, timeout=60).json()
    except Exception: continue
    for t in j.get("tables") or []:
        f = t.get("fields") or []
        if "證券代號" not in f or "最後揭示買量" not in f: continue
        I = {x: f.index(x) for x in ("證券代號", "收盤價", "漲跌價差", "成交股數", "最後揭示買價", "最後揭示買量", "最後揭示賣價", "最後揭示賣量")}; isg = [i for i, x in enumerate(f) if "漲跌(+/-)" in x][0]
        for r in t.get("data") or []:
            c, df_ = num(r[I["收盤價"]]), num(r[I["漲跌價差"]])
            if c is None or df_ is None or "+" not in str(r[isg]) or c - df_ <= 0 or df_ / (c - df_) < 0.06: continue
            rows.append({"date": d.isoformat(), "market": "TWSE", "code": str(r[I["證券代號"]]).strip(), "close": c, "chg": df_, "vol": num(r[I["成交股數"]]),
                         "bid_px": num(r[I["最後揭示買價"]]), "bid_vol": num(r[I["最後揭示買量"]]), "ask_px": num(r[I["最後揭示賣價"]]), "ask_vol": num(r[I["最後揭示賣量"]])})
    time.sleep(4)
if rows:
    os.makedirs("data/bidvol", exist_ok=True); pd.concat([old, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(["date", "code"], keep="last").to_csv(fn, index=False, compression="gzip")
print("bidvol daily +", len(rows))
