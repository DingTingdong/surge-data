"""探測證交所、櫃買中心「處置股票」公告 API 的回傳格式，原樣存檔供檢查。"""
import json, os, time, requests
H = {"User-Agent": "Mozilla/5.0 (compatible; personal-research)"}
URLS = {
    "twse_rwd": "https://www.twse.com.tw/rwd/zh/announcement/punish?startDate=20261001&endDate=20261116&response=json",
    "twse_old": "https://www.twse.com.tw/announcement/punish?response=json&startDate=20261001&endDate=20261116",
    "twse_2020": "https://www.twse.com.tw/rwd/zh/announcement/punish?startDate=20200101&endDate=20200131&response=json",
    "tpex_www": "https://www.tpex.org.tw/www/zh-tw/bulletin/disposal?startDate=2026/10/01&endDate=2026/11/16&response=json",
    "tpex_www2": "https://www.tpex.org.tw/www/zh-tw/bulletin/disposal?date=2026/11/16&response=json",
    "tpex_old": "https://www.tpex.org.tw/web/bulletin/disposal_information/disposal_information_result.php?l=zh-tw&sd=115/10/01&ed=115/11/16&o=json",
    "tpex_2020": "https://www.tpex.org.tw/www/zh-tw/bulletin/disposal?startDate=2020/01/01&endDate=2020/01/31&response=json",
}
os.makedirs("probe", exist_ok=True)
for k, u in URLS.items():
    try:
        r = requests.get(u, headers=H, timeout=30)
        body = r.text
        try:
            j = r.json(); body = json.dumps(j, ensure_ascii=False)[:20000]
        except Exception:
            body = body[:5000]
        open(f"probe/{k}.txt", "w", encoding="utf-8").write(f"{r.status_code} {u}\n{body}")
    except Exception as e:
        open(f"probe/{k}.txt", "w", encoding="utf-8").write(f"ERROR {u}\n{e}")
    time.sleep(4)
