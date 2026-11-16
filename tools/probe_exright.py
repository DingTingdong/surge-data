"""印出除權息、減資相關官方 API 的回傳格式（看 Actions 的 log）。"""
import json, time, requests
H = {"User-Agent": "Mozilla/5.0 (compatible; personal-research)"}
URLS = [
 "https://www.twse.com.tw/rwd/zh/exRight/TWT49U?startDate=20200601&endDate=20200630&response=json",
 "https://www.twse.com.tw/exchangeReport/TWT49U?response=json&strDate=20200601&endDate=20200630",
 "https://www.twse.com.tw/rwd/zh/reducation/TWTAUU?startDate=20200101&endDate=20201231&response=json",
 "https://www.twse.com.tw/exchangeReport/TWTAUU?response=json&strDate=20200101&endDate=20201231",
 "https://www.tpex.org.tw/www/zh-tw/bulletin/exDailyQ?startDate=2020/06/01&endDate=2020/06/30&response=json",
 "https://www.tpex.org.tw/web/stock/exright/dailyquo/exDailyQ_result.php?l=zh-tw&d=109/06/01&ed=109/06/30&o=json",
 "https://www.tpex.org.tw/www/zh-tw/bulletin/revivt?startDate=2020/01/01&endDate=2020/12/31&response=json",
 "https://www.tpex.org.tw/web/bulletin/revivt/revivt_result.php?l=zh-tw&d=109/01/01&ed=109/12/31&o=json",
]
for u in URLS:
    try:
        r = requests.get(u, headers=H, timeout=30)
        try: body = json.dumps(r.json(), ensure_ascii=False)
        except Exception: body = r.text
        print("=====", r.status_code, u); print(body[:1800])
    except Exception as e:
        print("=====", "ERR", u, e)
    time.sleep(4)
