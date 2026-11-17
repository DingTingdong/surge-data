"""印出融資融券、三大法人官方 API 的回傳格式（看 Actions log）。"""
import json, time, requests
H = {"User-Agent": "Mozilla/5.0 (compatible; personal-research)"}
URLS = [
 "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date=20200603&selectType=ALL&response=json",
 "https://www.twse.com.tw/rwd/zh/fund/T86?date=20200603&selectType=ALLBUT0999&response=json",
 "https://www.twse.com.tw/rwd/zh/fund/T86?date=20261113&selectType=ALLBUT0999&response=json",
 "https://www.tpex.org.tw/www/zh-tw/margin/balance?date=2020/06/03&response=json",
 "https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade?type=Daily&sect=EW&date=2020/06/03&response=json",
 "https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade?type=Daily&sect=EW&date=2026/11/13&response=json",
]
for u in URLS:
    try:
        r = requests.get(u, headers=H, timeout=30)
        try:
            j = r.json()
            if "tables" in j:
                j = {"keys": list(j.keys()), "tables": [{"title": t.get("title"), "fields": t.get("fields"), "data": (t.get("data") or [])[:2], "n": len(t.get("data") or [])} for t in j["tables"]]}
            else:
                j = {k: (v[:2] if isinstance(v, list) else v) for k, v in j.items()} | {"n_data": len(j.get("data") or [])}
            body = json.dumps(j, ensure_ascii=False)
        except Exception:
            body = r.text
        print("=====", r.status_code, u); print(body[:2500])
    except Exception as e:
        print("=====", "ERR", u, e)
    time.sleep(4)
