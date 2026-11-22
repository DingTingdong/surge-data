"""試抓臺灣指數公司網站：歷年「定期審核結果」公告的列表與內容格式。"""
import re, requests
H = {"User-Agent": "Mozilla/5.0"}
for u in ("https://taiwanindex.com.tw/news", "https://taiwanindex.com.tw/news?page=2", "https://taiwanindex.com.tw/indexes/t00050",
          "https://www.twse.com.tw/zh/indices/ftse/announcement.html", "https://taiwanindex.com.tw/announcements"):
    try:
        r = requests.get(u, headers=H, timeout=30); r.encoding = "utf-8"; t = r.text
        links = re.findall(r'href="([^"]+)"[^>]*>([^<]{0,80})<', t)
        hits = [(a, b.strip()) for a, b in links if any(k in b for k in ("審核", "成分股", "成份股", "定期"))]
        print(u, r.status_code, len(t), "相關連結", hits[:15])
        apis = sorted(set(re.findall(r'["\'](/api/[^"\']+)["\']', t)))[:20]; print("  api", apis)
    except Exception as e: print(u, "ERR", e)
