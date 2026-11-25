"""選擇權三大法人分計表猜下週方向（第 52 輪；2026/11/25 12:30 事先登錄，資料還沒抓完、看結果前寫下，看到結果後不修改）

來源：課程（風潮）文件「如何正確看待分計表格」：外資買買權（BC）＋賣賣權（SP）＝看多、買賣權（BP）＋賣買權（SC）＝看空；
  看未平倉（中線）、契約金額；週三結算後的未平倉最準，「星期四五看表格猜下周，大概 7 成勝率」。自營商是「黑手」，常和外資對作。
資料：期交所臺指選擇權三大法人買賣權分計（data/taifex/opt_inst.csv；下載只開放約 3 年，2023-11 起）；報酬用 0050（還原含息）。
訊號（每週最後一個交易日收盤後；用當天的未平倉）：
  F1 外資口數：外資「買權 未平倉口數買賣淨額」－「賣權 未平倉口數買賣淨額」> 0 → 看多，否則看空。
  F2 外資金額：同 F1，改用「未平倉契約金額買賣淨額」。
  F3 外資週變化：F1 的值比上一週增加 → 看多。
  D1 自營商口數（另報，不判斷）：同 F1 的算法用自營商，報「同向」與「反向」兩種命中率。
目標：下一週（本週最後交易日收盤 → 下週最後交易日收盤）0050 漲跌。對照：永遠看多的命中率。
判斷（F1～F3 各自）：① 命中率 ≥ 60%；② 比永遠看多高 5 個百分點以上；③ 看多那幾週的下週報酬平均 − 看空那幾週 > 0，t ≥ 2.9；
  ④ 前後兩半（2023-11～2025-04、2025-05～2026-11）③ 的差都 > 0。全過 → 雷達加「選擇權外資方向」大盤訊號；否則照實報告、只當資料顯示。
注意：只有約 150 週、只有一種盤勢（大多頭），就算通過也要往前記錄驗證。
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)

def p0050():
    et = pd.read_csv(f"{REPO}/data/etf.csv"); et["date"] = pd.to_datetime(et.date); x = et[et.code == "0050"].set_index("date").sort_index().close.astype(float)
    a = pd.read_csv(f"{REPO}/data/adjust.csv", dtype={"code": str}); a = a[a.code == "0050"]; f = pd.Series(1.0, index=x.index)
    for _, z in a.iterrows(): f[x.index < pd.Timestamp(z.date)] *= z.ratio
    c = x * f; r = c.pct_change()
    for d in r.index[r.abs() > 0.3]: f[x.index < d] /= (c.shift(1)[d] / c[d]); c = x * f; r = c.pct_change()
    return x * f

def load():
    D = pd.read_csv(f"{REPO}/data/taifex/opt_inst.csv", dtype=str); D.columns = [c.strip() for c in D.columns]
    D["date"] = pd.to_datetime(D["日期"].str.strip()); num = lambda s: pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")
    D["cp"] = D["買賣權別"].str.strip(); D["who"] = D["身份別"].str.strip()
    D["oi_n"] = num(D["未平倉口數買賣淨額"]); D["oi_v"] = num(D["未平倉契約金額買賣淨額(千元)"])
    return D

def net(D, who, col):
    z = D[D.who == who].pivot_table(index="date", columns="cp", values=col, aggfunc="sum")
    return (z["買權"] - z["賣權"]).sort_index()

if __name__ == "__main__":
    D = load(); px = p0050()
    days = pd.Series(D.date.unique()).sort_values(); wk = days.dt.to_period("W-FRI"); lastday = days.groupby(wk.values).max().values
    S = {"F1 外資口數": net(D, "外資", "oi_n"), "F2 外資金額": net(D, "外資", "oi_v"), "D1 自營商口數": net(D, "自營商", "oi_n")}
    S["F3 外資週變化"] = S["F1 外資口數"].reindex(lastday).diff()
    pxl = px.reindex(pd.DatetimeIndex(lastday), method="ffill"); fwd = (pxl.shift(-1) / pxl - 1) * 100
    res = {"週數": int(fwd.notna().sum()), "期間": f"{pd.Timestamp(lastday[0]).date()}～{pd.Timestamp(lastday[-1]).date()}", "永遠看多命中率": round(float((fwd.dropna() > 0).mean() * 100), 1)}
    half = pd.Timestamp("2025-05-01")
    for k, s in S.items():
        v = s.reindex(pd.DatetimeIndex(lastday)); ok = v.notna() & fwd.notna(); up = v[ok] > 0; y = fwd[ok]
        hit = float(((up & (y > 0)) | (~up & (y <= 0))).mean() * 100)
        a, b = y[up], y[~up]; diff = a.mean() - b.mean(); se = np.sqrt(a.var() / max(len(a), 1) + b.var() / max(len(b), 1))
        h = {}
        for lab, m in (("前半", y.index < half), ("後半", y.index >= half)):
            aa, bb = y[m & up.reindex(y.index).values], y[m & ~up.reindex(y.index).values]; h[lab] = round(float(aa.mean() - bb.mean()), 2) if len(aa) and len(bb) else None
        r = {"週數": int(ok.sum()), "看多週數": int(up.sum()), "命中率": round(hit, 1), "看多後下週平均%": round(float(a.mean()), 2), "看空後下週平均%": round(float(b.mean()), 2),
             "差": round(float(diff), 2), "t": round(float(diff / se), 2) if se > 0 else None, "兩半的差": h}
        if k.startswith("D1"): r["反向命中率"] = round(100 - hit, 1)
        else: r["通過"] = bool(hit >= 60 and hit >= res["永遠看多命中率"] + 5 and diff > 0 and (r["t"] or 0) >= 2.9 and all(x is not None and x > 0 for x in h.values()))
        res[k] = r; print(k, json.dumps(r, ensure_ascii=False), flush=True)
    print(json.dumps({k: res[k] for k in ("週數", "期間", "永遠看多命中率")}, ensure_ascii=False))
    json.dump(res, open(f"{REPO}/research/results/opt_inst_week.json", "w"), ensure_ascii=False, indent=1, default=str)
