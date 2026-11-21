"""試抓 2015–2019 舊資料，確認各來源在舊日期還有資料（只印筆數，不寫檔）。"""
import os, sys, datetime as dt
sys.path.insert(0, os.getcwd())
import fetch_prices as fp, fetch_chips as fc, fetch_adjust as fa, fetch_disposal as fd
for d in (dt.date(2015, 7, 1), dt.date(2017, 3, 1), dt.date(2019, 6, 3)):
    t, o = fp.twse(d), fp.tpex(d)
    print(d, "價格 上市", None if t is None else len(t), "上櫃", None if o is None else len(o), (t or [None])[:1], (o or [None])[:1])
    for fn in (fc.twse_inst, fc.tpex_inst, fc.twse_margin, fc.tpex_margin):
        r = fn(d); print("  ", fn.__name__, None if r is None else len(r), None if not r else list(r.items())[:1])
a, b = dt.date(2017, 7, 1), dt.date(2017, 7, 31)
for fn in (fa.twse_ex, fa.twse_cut, fa.tpex_ex, fa.tpex_cut):
    r = fn(a, b); print("除權息", fn.__name__, None if r is None else len(r), (r or [None])[:1])
for fn in (fd.twse, fd.tpex):
    r = fn(a, b); print("處置", fn.__name__, None if r is None else len(r), (r or [None])[:1])
import fetch_revenue as fr
print([x for x in dir(fr) if not x.startswith("_")])
