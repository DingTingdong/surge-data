"""試抓臺灣指數公司網站（第二次）：找頁面內嵌的資料或背後的 JSON 介面。"""
import re, requests, json
H = {"User-Agent": "Mozilla/5.0"}
r = requests.get("https://taiwanindex.com.tw/news", headers=H, timeout=30); r.encoding = "utf-8"; t = r.text
print("next", "__NEXT_DATA__" in t, "nuxt", "__NUXT__" in t)
for k in ("審核", "成分股", "成份股"):
    for m in list(re.finditer(k, t))[:3]: print(k, "…", t[max(0, m.start() - 300):m.start() + 200].replace("\n", " ")[:500])
srcs = re.findall(r'src="([^"]+\.js)"', t); print("scripts", srcs[:12])
for s in srcs[:8]:
    u = s if s.startswith("http") else "https://taiwanindex.com.tw" + s
    try:
        js = requests.get(u, headers=H, timeout=30).text
        eps = sorted(set(re.findall(r'["\'`](/?api/[A-Za-z0-9_/\-{}$.?=&]+)["\'`]', js)))[:30]
        if eps: print(u, eps)
    except Exception as e: print(u, "ERR", e)
for u in ("https://taiwanindex.com.tw/api/news?page=1", "https://taiwanindex.com.tw/api/v1/news", "https://taiwanindex.com.tw/api/news/list?lang=zh"):
    try:
        x = requests.get(u, headers=H, timeout=30); print(u, x.status_code, x.headers.get("content-type"), x.text[:300].replace("\n", " "))
    except Exception as e: print(u, "ERR", e)
