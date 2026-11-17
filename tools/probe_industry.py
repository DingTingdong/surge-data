"""看公司基本資料 API 的欄位（產業別），只印出來不存檔。"""
import requests, json
H = {"User-Agent": "Mozilla/5.0 (compatible; personal-research)"}
for u in ["https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
          "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O",
          "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"]:
    try:
        r = requests.get(u, headers=H, timeout=60); j = r.json()
        print("==", u, r.status_code, len(j))
        print(json.dumps(j[:2], ensure_ascii=False)[:1500])
        k = [x for x in j[0] if "產業" in x or "Industry" in x]
        if k:
            from collections import Counter
            print(k, Counter(x[k[0]] for x in j).most_common(60))
    except Exception as e:
        print("ERR", u, e)
