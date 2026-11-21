"""試抓 2010–2014 舊資料：各來源在更早的日期是否還有資料（只印筆數）。"""
import os, sys, datetime as dt
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
import fetch_prices as fp, fetch_chips as fc, fetch_adjust as fa, fetch_disposal as fd, fetch_revenue as fr
import old_era as oe
for d in (dt.date(2010, 3, 2), dt.date(2012, 3, 1), dt.date(2012, 6, 1), dt.date(2014, 3, 3)):
    t, o = fp.twse(d), fp.tpex(d)
    print(d, "價格 上市", None if t is None else len(t), "上櫃", None if o is None else len(o))
    for fn in (oe.twse_inst, oe.tpex_inst, fc.twse_margin, fc.tpex_margin):
        r = fn(d); print("  ", fn.__name__, None if r is None else len(r), None if not r else list(r.items())[:1])
for y, m in ((2010, 6), (2012, 6), (2013, 1), (2014, 6)):
    a, b = dt.date(y, m, 1), dt.date(y, m, 28)
    print("除權息", y, m, [None if (r := fn(a, b)) is None else len(r) for fn in (fa.twse_ex, fa.twse_cut, fa.tpex_ex, fa.tpex_cut)],
          "處置", [None if (r := fn(a, b)) is None else len(r) for fn in (fd.twse, fd.tpex)])
    rs, ok = fr.month_rows(y, m); print("營收", y, m, len(rs), ok, rs[:1])
