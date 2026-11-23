"""探測：可轉債發行資料（TPEx OpenAPI swagger 裡跟債券／轉換有關的端點、各端點筆數與欄位）（研究用；只印結果）。"""
import json, time, requests
H = {"User-Agent": "Mozilla/5.0"}
sw = requests.get("https://www.tpex.org.tw/openapi/swagger.json", headers=H, timeout=60).json()
paths = sw.get("paths", {}); print("## paths", len(paths))
cand = []
for p, v in paths.items():
    s = json.dumps(v, ensure_ascii=False)
    if any(k in s for k in ("轉換", "轉(交)換", "可轉", "增資", "Convertible", "convertible")): cand.append(p); print("##", p, (v.get("get") or {}).get("summary"), flush=True)
for p in cand[:12]:
    try:
        r = requests.get("https://www.tpex.org.tw/openapi/v1" + p if not p.startswith("/v1") else "https://www.tpex.org.tw/openapi" + p, headers=H, timeout=60); j = r.json()
        print("## GET", p, r.status_code, len(j), list(j[0].keys()) if j else None, j[:1], flush=True)
    except Exception as e: print("## GET", p, "ERR", e)
    time.sleep(2)
print("## servers", sw.get("servers"))
