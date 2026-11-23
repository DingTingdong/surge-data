"""每天補抓借券賣出餘額（上市 TWT93U、上櫃）到 data/sbl/YYYY.csv.gz：從檔案最後一天之後到今天的平日都試；還沒公布的日子下次再補。"""
import datetime as dt, glob, os, sys, time, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("YEAR", "2000")
import importlib.util
spec = importlib.util.spec_from_file_location("fs", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetch_sbl.py"))
src = open(spec.origin, encoding="utf-8").read().split("rows, failed = [], []")[0]; ns = {}; exec(src, ns)
today = (dt.datetime.utcnow() + dt.timedelta(hours=8)).date(); fn = f"data/sbl/{today.year}.csv.gz"
old = pd.read_csv(fn, dtype={"code": str}) if os.path.exists(fn) else pd.DataFrame(columns=["date", "market", "code", "sbl_bal", "sbl_sell", "ms_bal"])
last = pd.to_datetime(old.date).max().date() if len(old) else dt.date(today.year, 1, 1) - dt.timedelta(days=1)
rows = []; d = last + dt.timedelta(days=1)
while d <= today:
    if d.weekday() < 5:
        ds = d.strftime("%Y%m%d"); j = ns["get"](f"https://www.twse.com.tw/rwd/zh/marginTrading/TWT93U?date={ds}&response=json")
        got = []
        if j and j.get("stat") == "OK" and j.get("data"): got += ns["parse"](j["fields"], j["data"], d.isoformat(), "TWSE")
        time.sleep(3)
        roc = f"{d.year - 1911:03d}/{d.month:02d}/{d.day:02d}"; j = ns["get"](f"https://www.tpex.org.tw/web/stock/margin_trading/margin_sbl/margin_sbl_result.php?l=zh-tw&d={roc}&o=json")
        t = (j or {}).get("tables") or [{}]
        if t[0].get("data"): got += ns["parse"](t[0]["fields"], t[0]["data"], d.isoformat(), "TPEX")
        time.sleep(3)
        if any(r["market"] == "TWSE" for r in got): rows += got  # 上市還沒公布就整天先不存，下次再補
        else: break
    d += dt.timedelta(days=1)
if rows:
    new = pd.concat([old, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(["date", "market", "code"], keep="last")
    os.makedirs("data/sbl", exist_ok=True); new.to_csv(fn, index=False, compression="gzip"); print("sbl daily +", len(rows), "rows through", rows[-1]["date"])
else: print("sbl daily: nothing new (last", last, ")")
