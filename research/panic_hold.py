"""撿跌深前 3：抱幾天比較好（第 62 輪；2026/11/26 12:05 登錄，描述性，看結果前寫下）

起因：使用者持有大甲（撿跌深前 3），問「原本 20 天，如果縮短呢」。
事件與前 3 名：沿用獨立覆核重算的當天市場別、排除 TDR 版本（research/audit/out/B_all_pit_notdr.json，2018–2026 共 10 次）。
進場：t0 隔天開盤（同原研究；開盤鎖漲停的名單已由原研究遞補）。抱 3、5、10、15、20、40 個交易日收盤，扣 0.6%。
比較：同日開盤買的分析範圍等權。報每個持有天數的前 3 平均、減分析範圍、贏的次數。
注意：前 3 的排序本來就是看過這些事件才定的（樣本內），這裡只描述，不改規則；要改持有天數需要另外事先登錄並用新資料驗證。
"""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
E = json.load(open(f"{HERE}/audit/out/B_all_pit_notdr.json"))["750"]["events"]
X = pd.read_pickle("/tmp/model_v2/vs0050_extra.pkl"); days = pd.DatetimeIndex(X["days"]); cj = {c: j for j, c in enumerate(X["cols"])}
C = pd.DataFrame(X["C"]).ffill(limit=5).values; O = np.asarray(X["O"], float); inb = np.asarray(X["inb"]).astype(bool); lk = np.asarray(X["lockn"]).astype(bool)
HS = (3, 5, 10, 15, 20, 40); rows = []
for e in E:
    t0 = pd.Timestamp(e["t0"]); i = int(np.searchsorted(days.values, np.datetime64(t0))); ie = i + 1
    codes = [s[:4] for s in (e.get("P3_fill") or e["top3"])] if isinstance(e.get("P3_fill"), list) else [s[:4] for s in e["top3"]]
    r = {"t0": e["t0"], "top3": e["top3"]}
    for H in HS:
        ix = ie + H - 1
        if ix >= len(days): r[f"p{H}"] = r[f"u{H}"] = np.nan; continue
        pr = [(C[ix, cj[c]] / O[ie, cj[c]] - 1) * 100 - 0.6 for c in codes if c in cj and O[ie, cj[c]] > 0]
        rr = (C[ix] / np.where(O[ie] > 0, O[ie], np.nan) - 1) * 100 - 0.6; m = inb[i] & np.isfinite(rr) & ~lk[i]
        r[f"p{H}"] = float(np.mean(pr)); r[f"u{H}"] = float(np.nanmean(rr[m]))
    rows.append(r)
R = pd.DataFrame(rows); out = {}
for H in HS:
    x = R[f"p{H}"] - R[f"u{H}"]; out[f"{H}天"] = {"前3": round(R[f"p{H}"].mean(), 2), "分析範圍": round(R[f"u{H}"].mean(), 2), "差": round(x.mean(), 2), "贏": f"{int((x > 0).sum())}/{int(x.notna().sum())}", "前3賠錢次數": int((R[f"p{H}"] < 0).sum())}
print(json.dumps(out, ensure_ascii=False, indent=0)); print(R[["t0"] + [f"p{H}" for H in HS]].round(1).to_string(index=False))
json.dump(out, open(f"{REPO}/research/results/panic_hold.json", "w"), ensure_ascii=False, indent=1)
