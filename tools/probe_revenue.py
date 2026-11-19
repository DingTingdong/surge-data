"""試抓公開資訊觀測站的月營收彙總表（歷史每個月一頁），只印出解析結果。"""
import re, requests
H = {"User-Agent": "Mozilla/5.0 (compatible; personal-research)"}
def parse(html):
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S | re.I):
        cells = [re.sub(r"<[^>]+>|&nbsp;", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S | re.I)]
        if len(cells) >= 7 and re.fullmatch(r"[1-9]\d{3}", cells[0]): rows.append(cells)
    return rows
for host in ("https://mopsov.twse.com.tw", "https://mops.twse.com.tw"):
    for mk in ("sii", "otc"):
        for ym in ((115, 10), (109, 1)):
            u = f"{host}/nas/t21/{mk}/t21sc03_{ym[0]}_{ym[1]}_0.html"
            try:
                r = requests.get(u, headers=H, timeout=60)
                for enc in ("utf-8", "big5", "cp950"):
                    try: txt = r.content.decode(enc); break
                    except UnicodeDecodeError: continue
                rows = parse(txt)
                print(u, r.status_code, len(r.content), "rows", len(rows), rows[:2])
            except Exception as e:
                print(u, "ERR", e)
