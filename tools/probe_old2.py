"""舊日期的法人資料欄位：證交所 T86 欄位名稱、櫃買新舊網址的回應格式。"""
import os, sys, json, datetime as dt
sys.path.insert(0, os.getcwd())
from fetch_chips import get
for d in (dt.date(2015, 7, 1), dt.date(2017, 3, 1), dt.date(2017, 12, 20)):
    j = get(f"https://www.twse.com.tw/rwd/zh/fund/T86?date={d:%Y%m%d}&selectType=ALLBUT0999&response=json")
    print("T86", d, j and j.get("fields"), j and (j.get("data") or [None])[0])
for d in (dt.date(2015, 7, 1), dt.date(2017, 3, 1), dt.date(2018, 1, 2), dt.date(2018, 6, 1), dt.date(2019, 1, 2)):
    j = get(f"https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade?type=Daily&sect=EW&date={d:%Y/%m/%d}&response=json")
    t = (j or {}).get("tables") or [{}]
    print("TPEX new", d, j and {k: (v if k != "tables" else len(v)) for k, v in j.items() if k in ("stat", "date", "tables")}, t[0].get("fields"), (t[0].get("data") or [None])[0])
    j2 = get(f"https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php?l=zh-tw&se=EW&t=D&d={d.year-1911}/{d:%m/%d}&o=json")
    print("TPEX old", d, j2 and {k: v for k, v in j2.items() if k not in ("aaData",)} , j2 and (j2.get("aaData") or [None])[:2])
