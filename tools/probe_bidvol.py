"""探測：每日收盤行情裡的「最後揭示買量」（漲停封單）歷史資料（研究用；只印結果）。"""
import time, requests
H = {"User-Agent": "Mozilla/5.0"}
for d in ("20080303", "20160601", "20261120"):
    try:
        j = requests.get(f"https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={d}&type=ALLBUT0999&response=json", headers=H, timeout=60).json()
        for t in j.get("tables") or []:
            f = t.get("fields") or []
            if "證券代號" in f: print("## TWSE", d, f, len(t.get("data") or []), (t.get("data") or [])[:1], flush=True)
    except Exception as e: print("## TWSE", d, "ERR", e)
    time.sleep(5)
for d in ("097/03/03", "105/06/01", "115/11/20"):
    for url in (f"https://www.tpex.org.tw/web/stock/aftertrading/daily_close_quotes/stk_quote_result.php?l=zh-tw&d={d}&o=json",
                f"https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQ?date={d.replace('/', '%2F')}&response=json"):
        try:
            j = requests.get(url, headers=H, timeout=60).json(); t = (j.get("tables") or [{}])[0] if j.get("tables") else j
            print("## TPEx", d, url.split("/")[-1][:25], t.get("fields") or j.get("fields"), len(t.get("data") or j.get("aaData") or []), (t.get("data") or j.get("aaData") or [])[:1], flush=True)
        except Exception as e: print("## TPEx", d, "ERR", e)
        time.sleep(5)
