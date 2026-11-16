"""補抓一段期間的上市＋上櫃四碼個股日行情，寫成壓縮 CSV（欄位同 data/prices.csv）。

用法：python fetch_history.py 2020-01-01 2020-12-31 out.csv.gz
- 只抓平日；兩個市場都回空資料的日子當作休市
- 連線失敗的日子最後再重試一次，仍失敗就列在 out.csv.gz.failed.txt
"""
import csv, gzip, sys, time, datetime as dt
from fetch_prices import twse, tpex, COLS

def main(start, end, out):
    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    days = [s + dt.timedelta(n) for n in range((e - s).days + 1)]
    days = [d for d in days if d.weekday() < 5]
    rows, failed, closed = [], [], []

    def grab(d):
        got = {}
        for m, fn in (("TWSE", twse), ("TPEX", tpex)):
            r = fn(d); time.sleep(3.5)
            got[m] = r
        return got

    def handle(d, got, final):
        if any(v is None for v in got.values()):
            if final:
                failed.append(d.isoformat())
                for v in got.values():  # 只抓到一邊也先留著
                    if v: rows.extend(v)
            return False
        if not any(got.values()):
            closed.append(d.isoformat()); return True
        for v in got.values(): rows.extend(v)
        return True

    retry = []
    for n, d in enumerate(days):
        if not handle(d, grab(d), final=False):
            retry.append(d)
        if n % 20 == 0:
            print(d, len(rows), "rows", flush=True)
    for d in retry:
        time.sleep(10)
        handle(d, grab(d), final=True)

    with gzip.open(out, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(COLS); w.writerows(rows)
    with open(out + ".closed.txt", "w") as f: f.write("\n".join(closed))
    if failed:
        with open(out + ".failed.txt", "w") as f: f.write("\n".join(failed))
    print("done", start, end, len(rows), "rows,", len(closed), "closed days,", len(failed), "failed")

if __name__ == "__main__":
    main(*sys.argv[1:4])
