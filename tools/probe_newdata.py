"""試抓新資料來源在 2016、2020 年的歷史（只印欄位與第一列）。"""
import os, sys, json, datetime as dt
sys.path.insert(0, os.getcwd())
from fetch_chips import get
def show(name, u):
    j = get(u)
    if j is None: print(name, "FAILED", u); return
    if isinstance(j, dict):
        keys = list(j.keys())[:8]; f = j.get("fields"); data = j.get("data") or j.get("aaData")
        tabs = j.get("tables")
        if tabs: f = tabs[0].get("fields"); data = tabs[0].get("data")
        print(name, "keys", keys, "stat", j.get("stat"), "fields", f, "n", None if data is None else len(data), "row0", (data or [None])[0])
    else: print(name, type(j), str(j)[:200])
for d in (dt.date(2016, 1, 4), dt.date(2020, 3, 2)):
    r = f"{d.year-1911}/{d:%m/%d}"
    show(f"{d} TWSE 外資持股 MI_QFIIS", f"https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS?date={d:%Y%m%d}&selectType=ALLBUT0999&response=json")
    show(f"{d} TWSE 融券借券餘額 TWT93U", f"https://www.twse.com.tw/rwd/zh/marginTrading/TWT93U?date={d:%Y%m%d}&response=json")
    show(f"{d} TWSE 當沖 TWTB4U", f"https://www.twse.com.tw/rwd/zh/dayTrading/TWTB4U?date={d:%Y%m%d}&selectType=All&response=json")
    show(f"{d} TPEx 外資持股 新", f"https://www.tpex.org.tw/www/zh-tw/insti/qfii?date={d:%Y/%m/%d}&response=json")
    show(f"{d} TPEx 外資持股 舊", f"https://www.tpex.org.tw/web/stock/3insti/qfii/qfii_result.php?l=zh-tw&d={r}&o=json")
    show(f"{d} TPEx 借券 舊", f"https://www.tpex.org.tw/web/stock/margin_trading/margin_sbl/margin_sbl_result.php?l=zh-tw&d={r}&o=json")
    show(f"{d} TPEx 借券 新", f"https://www.tpex.org.tw/www/zh-tw/margin/sbl?date={d:%Y/%m/%d}&response=json")
    show(f"{d} TPEx 當沖 舊", f"https://www.tpex.org.tw/web/stock/trading/intraday_trading/intraday_trading_list_result.php?l=zh-tw&d={r}&o=json")
    show(f"{d} TPEx 當沖 新", f"https://www.tpex.org.tw/www/zh-tw/intraday/list?date={d:%Y/%m/%d}&response=json")
