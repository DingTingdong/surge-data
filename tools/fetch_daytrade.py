"""抓上市現股當沖個股成交量值（TWSE TWTB4U，2014-01 起）到 out/daytrade-YEAR.csv.gz（研究用；網站不讀）。
欄位：date, code, dt_shares（當沖成交股數）, dt_buy（當沖買進成交金額）, dt_sell（當沖賣出成交金額）。上櫃沒有個股當沖量值，不抓。"""
import datetime as dt, os, time, requests, pandas as pd
YEAR = int(os.environ["YEAR"]); H = {"User-Agent": "Mozilla/5.0"}; os.makedirs("out", exist_ok=True)
num = lambda s: float(str(s).replace(",", "").strip()) if str(s).replace(",", "").strip().replace(".", "").isdigit() else None
def get(url):
    for k in range(4):
        try:
            r = requests.get(url, headers=H, timeout=40)
            if r.status_code == 200: return r.json()
        except Exception: pass
        time.sleep(20 * (k + 1))
    return None
rows, failed = [], []; d = dt.date(YEAR, 1, 1); end = min(dt.date(YEAR, 12, 31), dt.date.today())
while d <= end:
    if d.weekday() < 5:
        ds = d.strftime("%Y%m%d"); j = get(f"https://www.twse.com.tw/rwd/zh/dayTrading/TWTB4U?date={ds}&selectType=All&response=json")
        if j is None: failed.append(ds)
        elif j.get("stat") == "OK":
            for t in j.get("tables") or []:
                f = t.get("fields") or []
                if "證券代號" in f and "當日沖銷交易成交股數" in f:
                    ic, ish, ib, isl = f.index("證券代號"), f.index("當日沖銷交易成交股數"), f.index("當日沖銷交易買進成交金額"), f.index("當日沖銷交易賣出成交金額")
                    rows += [{"date": d.isoformat(), "code": str(r[ic]).strip(), "dt_shares": num(r[ish]), "dt_buy": num(r[ib]), "dt_sell": num(r[isl])} for r in t.get("data") or []]
        time.sleep(3.5)
    d += dt.timedelta(days=1)
df = pd.DataFrame(rows); df.to_csv(f"out/daytrade-{YEAR}.csv.gz", index=False, compression="gzip"); open(f"out/daytrade-{YEAR}.failed.txt", "w").write("\n".join(failed))
print(YEAR, "rows", len(df), "days", df.date.nunique() if len(df) else 0, "codes/day", round(len(df) / max(1, df.date.nunique())) if len(df) else 0, "failed", len(failed))
