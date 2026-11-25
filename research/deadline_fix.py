"""月營收法定期限遇假日順延：換股日改到期限日的下一個交易日（第 58 輪；2026/11/25 20:50 登錄，看結果前寫下）

起因：第四份外部覆核指出，每月 10 日遇週末／假日時，月營收申報期限順延到下一個營業日；原規則「11 日起第一個交易日」那天就是期限日，
  回測在那天收盤就用全部營收，等於偷看當晚才公布的營收。2011 起 191 個月有 72 個月（38%）受影響（每年 10 月都是，國慶日）。
修正：期限日＝該月 10 日起第一個交易日；換股日＝期限日的下一個交易日（10 日是交易日時與原規則相同）。
這是修正偷看未來，不是找更好的規則：結果不論好壞都採用，網站換股日同步改，月選對外數字改用修正後的。
做法：vs0050.py 以 FD_RULE=deadline 重建快照（2011–2026 與 2006–2010 兩段），用 calm3.sim_calm（穩健版 3 檔）與 wr_confirm.stats 算，和原規則並列。
另報：前 20 名等權（sim2 n=20）。
"""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.argv = sys.argv[:1]
import robust_select as RS
from wr_confirm import sim2, stats
from calm3 import sim_calm
if __name__ == "__main__":
    out = {}
    for lab, a, b in (("原規則", "/tmp/model_v2/vs0050_oldestonly_extra.pkl", "/tmp/model_v2/vs0050_extra.pkl"), ("期限順延修正", "/tmp/model_v2/vs0050_oldestonly_dl_extra.pkl", "/tmp/model_v2/vs0050_dl_extra.pkl")):
        Do, Dn = RS.load(a, "2006-01-01", "2010-01-01"), RS.load(b, "2011-01-01", "2027-01-01")
        T = pd.concat([sim_calm(Do, 20), sim_calm(Dn, 20)]); T20 = pd.concat([sim2(Do, "cs", 20), sim2(Dn, "cs", 20)])
        out[lab] = {"穩健版3檔": stats(T), "前20等權": stats(T20), "換股日數": int(len(T))}
        print(lab, json.dumps(out[lab], ensure_ascii=False), flush=True)
    json.dump(out, open(f"{REPO}/research/results/deadline_fix.json", "w"), ensure_ascii=False, indent=1)
