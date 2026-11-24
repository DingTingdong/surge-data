"""補抓重大訊息回補時被工作時限截斷的兩段（2025-05-28～06-30、2026-05-19～06-30），併進 data/mopsnews 的半年檔。
11/25 探測：TYPEK=sii／otc 回傳的內容相同（都含上市＋上櫃），所以只查一次；market 欄一律寫 all。"""
import datetime as dt, os, sys, time, pandas as pd
sys.argv = [sys.argv[0]]; os.environ.setdefault("YEAR", "2025")
src = open("tools/fetch_mopsnews.py", encoding="utf-8").read().split("HALF = int(")[0]
exec(src)  # 取得 get()、parse()、KEYS
for tag, a, b in (("2025h1", dt.date(2025, 5, 28), dt.date(2025, 6, 30)), ("2026h1", dt.date(2026, 5, 19), dt.date(2026, 6, 30))):
    rows, failed, d = [], [], a
    while d <= b:
        if d.weekday() < 5:
            txt = get("sii", d)
            if txt is None: failed.append(str(d))
            elif "<form" in txt: rows += parse(txt, "all")
            print(tag, d, len(rows), flush=True); time.sleep(5)
        d += dt.timedelta(days=1)
    f = f"data/mopsnews/{tag}.csv.gz"; old = pd.read_csv(f, dtype={"code": str, "date": str, "time": str})
    new = pd.concat([old, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(subset=["date", "time", "code", "subject"])
    new.to_csv(f, index=False, compression="gzip"); print(tag, "加入", len(new) - len(old), "筆，失敗", failed, flush=True)
