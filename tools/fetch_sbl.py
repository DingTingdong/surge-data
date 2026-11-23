"""抓借券賣出餘額（TWSE TWT93U 全年、TPEx 借券賣出）到 out/sbl-YEAR.csv.gz（研究用；網站不讀）。
欄位：date, market, code, sbl_bal（借券賣出當日餘額，股）, sbl_sell（當日借券賣出，股）, ms_bal（融券今日餘額，張）。"""
import datetime as dt, gzip, io, os, time, requests, pandas as pd
YEAR = int(os.environ["YEAR"]); H = {"User-Agent": "Mozilla/5.0"}; os.makedirs("out", exist_ok=True)
num = lambda s: float(str(s).replace(",", "").strip() or 0) if str(s).replace(",", "").strip().lstrip("-").replace(".", "").isdigit() else None
def parse(fields, data, d, mkt):
    bal = [i for i, f in enumerate(fields) if f in ("今日餘額", "當日餘額")]; sel = [i for i, f in enumerate(fields) if f in ("賣出", "當日賣出")]
    if len(bal) < 2 or len(sel) < 2: return []
    return [{"date": d, "market": mkt, "code": str(r[0]).strip(), "sbl_bal": num(r[bal[-1]]), "sbl_sell": num(r[sel[-1]]), "ms_bal": num(r[bal[0]])} for r in data if len(r) > bal[-1]]
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
        ds = d.strftime("%Y%m%d"); j = get(f"https://www.twse.com.tw/rwd/zh/marginTrading/TWT93U?date={ds}&response=json")
        if j is None: failed.append("TWSE " + ds)
        elif j.get("stat") == "OK" and j.get("data"): rows += parse(j["fields"], j["data"], d.isoformat(), "TWSE")
        time.sleep(3.5)
        if YEAR >= 2010:
            roc = f"{d.year - 1911:03d}/{d.month:02d}/{d.day:02d}"; j = get(f"https://www.tpex.org.tw/web/stock/margin_trading/margin_sbl/margin_sbl_result.php?l=zh-tw&d={roc}&o=json")
            if j is None: failed.append("TPEx " + ds)
            else:
                t = (j.get("tables") or [{}])[0]
                if t.get("data"): rows += parse(t["fields"], t["data"], d.isoformat(), "TPEX")
            time.sleep(3.5)
    d += dt.timedelta(days=1)
df = pd.DataFrame(rows); df.to_csv(f"out/sbl-{YEAR}.csv.gz", index=False, compression="gzip")
open(f"out/sbl-{YEAR}.failed.txt", "w").write("\n".join(failed))
print(YEAR, "rows", len(df), "days", df.date.nunique() if len(df) else 0, "by market", df.groupby("market").date.nunique().to_dict() if len(df) else {}, "failed", len(failed))
