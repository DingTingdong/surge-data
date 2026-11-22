"""試抓 2005–2009 舊資料（研究用：想多一段完全沒用過、含 2008 金融海嘯的檢驗期）。只印筆數，不寫檔。"""
import os, re, sys, time, datetime as dt, requests
sys.path.insert(0, os.getcwd())
import fetch_prices as fp, fetch_chips as fc, fetch_adjust as fa, fetch_revenue as fr
H = {"User-Agent": "Mozilla/5.0"}
for d in (dt.date(2005, 6, 1), dt.date(2007, 1, 3), dt.date(2007, 8, 1), dt.date(2008, 10, 1), dt.date(2009, 6, 1)):
    t, o = fp.twse(d), fp.tpex(d)
    print(d, "價格 上市", None if t is None else len(t), "上櫃", None if o is None else len(o),
          "| 0050", None if not t else [r for r in t if r[2] == "0050"][:1], flush=True)
    for fn in (fc.twse_margin, fc.tpex_margin):
        r = fn(d); print("  ", fn.__name__, None if r is None else len(r), flush=True)
    time.sleep(2)
for y, m in ((2005, 7), (2008, 7)):
    a, b = dt.date(y, m, 1), dt.date(y, m, 28)
    print("除權息", y, m, [None if (r := fn(a, b)) is None else len(r) for fn in (fa.twse_ex, fa.twse_cut, fa.tpex_ex, fa.tpex_cut)], flush=True)
for y, m in ((2005, 6), (2008, 10), (2009, 5)):
    rs, ok = fr.month_rows(y, m); print("營收 現行網址", y, m, ok, len(rs), rs[:1], flush=True)
    for mk in ("sii", "otc"):
        for host in ("mopsov.twse.com.tw", "mops.twse.com.tw"):
            u = f"https://{host}/nas/t21/{mk}/t21sc03_{y - 1911}_{m}.html"
            try:
                r = requests.get(u, headers=H, timeout=60); txt = r.content.decode("big5", "replace")
                rows = [c for c in (re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S | re.I) for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", txt, flags=re.S | re.I)) if len(c) >= 10]
                print("  ", u, r.status_code, len(r.content), "資料列", len(rows), [re.sub(r"<[^>]+>|&nbsp;", "", x).strip() for x in rows[0]][:7] if rows else "", flush=True)
            except Exception as e: print("  ", u, "ERR", e, flush=True)
            time.sleep(2)
