"""抓上市櫃公司的產業別與實收資本額 → data/industry.csv（只在內容有變時改檔）。

欄位：code, market, name（簡稱）, ind（產業代碼）, ind_name（產業名稱）, cap（實收資本額，億元）
來源：證交所 openapi t187ap03_L（上市）、櫃買中心 openapi mopsfin_t187ap03_O（上櫃）。
"""
import csv, os, re, time, requests

IND = {"01": "水泥", "02": "食品", "03": "塑膠", "04": "紡織纖維", "05": "電機機械", "06": "電器電纜", "08": "玻璃陶瓷",
       "09": "造紙", "10": "鋼鐵", "11": "橡膠", "12": "汽車", "14": "建材營造", "15": "航運", "16": "觀光餐旅",
       "17": "金融保險", "18": "貿易百貨", "19": "綜合", "20": "其他", "21": "化學", "22": "生技醫療", "23": "油電燃氣",
       "24": "半導體", "25": "電腦週邊", "26": "光電", "27": "通信網路", "28": "電子零組件", "29": "電子通路",
       "30": "資訊服務", "31": "其他電子", "32": "文化創意", "33": "農業科技", "34": "電子商務", "35": "綠能環保",
       "36": "數位雲端", "37": "運動休閒", "38": "居家生活", "80": "管理股票", "91": "存託憑證"}
H = {"User-Agent": "Mozilla/5.0 (compatible; personal-research)"}
OUT = "data/industry.csv"

def get(url):
    for i in range(4):
        try:
            r = requests.get(url, headers=H, timeout=60)
            if r.status_code == 200: return r.json()
        except Exception: pass
        time.sleep(5 * (i + 1))
    return None

def cap(x):
    try: return round(float(str(x).replace(",", "")) / 1e8, 2)
    except ValueError: return ""

def main():
    rows = {}
    src = (("TWSE", "https://openapi.twse.com.tw/v1/opendata/t187ap03_L", "公司代號", "公司簡稱", "產業別", "實收資本額"),
           ("TPEX", "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", "SecuritiesCompanyCode", "CompanyAbbreviation", "SecuritiesIndustryCode", "Paidin.Capital.NTDollars"))
    ok = 0
    for mk, url, kc, kn, ki, kp in src:
        j = get(url)
        if not j: print("failed", mk); continue
        ok += 1
        for r in j:
            c = str(r.get(kc, "")).strip()
            if not re.fullmatch(r"[1-9]\d{3}", c): continue
            i = str(r.get(ki, "")).strip()
            rows[c] = [c, mk, str(r.get(kn, "")).strip(), i, IND.get(i, i), cap(r.get(kp))]
    if ok < 2 and os.path.exists(OUT):
        # 有一邊沒抓到：保留舊資料裡那一邊的公司
        with open(OUT, encoding="utf-8") as f:
            for r in list(csv.reader(f))[1:]:
                rows.setdefault(r[0], r)
    if not rows: return
    new = [["code", "market", "name", "ind", "ind_name", "cap"]] + [rows[c] for c in sorted(rows)]
    old = list(csv.reader(open(OUT, encoding="utf-8"))) if os.path.exists(OUT) else None
    if old == [[str(x) for x in r] for r in new]:
        print("unchanged", len(rows)); return
    with open(OUT, "w", newline="", encoding="utf-8") as f: csv.writer(f).writerows(new)
    print("wrote", len(rows))

if __name__ == "__main__":
    main()
