# 強勢股雷達

網站：https://dingtingdong.github.io/surge-data/ （手機可加到主畫面）

每個平日台北 18:20 起到隔天 06:50，GitHub Actions 每1～2小時自動檢查一次（GitHub 排程常延遲或漏跑，所以排很多個時間），抓證交所＋櫃買中心收盤行情、處置股、除權息／減資、三大法人與融資融券，重算分級、更新網站。資料齊全時不做事。電腦不用開機。

## 名單分級（規則 v2.0，2026/11/17 起）
- 今日發動：今天漲 ≥ 9%、前10個交易日都沒有漲 ≥ 9%（第一根漲停），今天成交額 ≥ 2,000 萬、前20日平均成交額 ≥ 300 萬
- 仍在動：活躍度前10%（上市、上櫃分開排名，20日均額 ≥ 1,000 萬）＋ 今天或前10個交易日內有單日漲 ≥ 9%

今日發動由 `research/early.py` 從5個事先寫好的候選中選出（E4）。沒有候選達到事先設的命中率門檻；
照事先寫的退路應選 E2，改選 E4 是看過結果後的判斷（挑選期與驗證期都比 E2 早、剩下空間大、命中率相近），詳見 `research/results/README.md`。
v1（舊版精選）的紀錄保留在 `data/archive/`。

名單是剛發動或還在動的股票，接下來容易大漲也容易大跌，不是買進訊號。研究整理，不構成投資建議。

## 檔案
| 檔案 | 用途 |
| --- | --- |
| `radar_core.py` | 共用計算：讀行情、除權息／減資調整、指標、分級 |
| `build_site.py` | 寫每日紀錄、算追蹤報酬、產生網站 |
| `site_template.html` | 網站畫面 |
| `fetch_prices.py` | 每日行情（含對參考價的漲跌，用來做除權息調整） |
| `fetch_disposal.py` | 處置股公告 |
| `fetch_adjust.py` | 除權息、減資參考價（價格調整用） |
| `fetch_chips.py` | 三大法人買賣超、融資融券餘額（`data/chips/YYYY.csv.gz`） |
| `fetch_history.py`, `merge_history.py` | 補 2020 年起的歷史行情 |
| `research/backtest.py` | 長期回測（檢驗方法事先寫在檔頭，不調門檻） |
| `research/detection.py` | 發現速度：大漲波段中第幾天抓到、抓到後還剩多少（v1） |
| `research/early.py` | 找最早的發動訊號：5個候選，2020–2023 挑、2024–2026 驗證 |
| `research/exits.py` | 出場方式：7種賣法，選出移動停損10% |
| `research/regime.py` | 大盤濾網：3種都沒幫助；「逆勢」為探索性標示 |
| `research/chips.py` | 籌碼條件：5種都未達門檻；「融資未增」為標示 |
| `data/prices.csv` | 今年日行情；更早的在 `data/history/YYYY.csv.gz` |
| `data/signals.csv` | 每天「今日發動」與「仍在動」完整名單與分級（只新增；改版時舊檔移到 `data/archive/`） |
| `data/days.csv` | 每天各級檔數（零檔也記） |
| `data/disposal.csv` | 處置股起迄日 |
| `data/fetch_log.csv` | 抓取失敗紀錄（有失敗才會出現） |

## 工作流程（Actions）
- `update-prices`：每日排程
- `backfill-history`：手動，補指定期間的歷史行情
- `disposal-history`：手動，補 2020 年起的處置股

## 事先定好的停止條件
累積約26週實際結果後，若今日發動的5日報酬中位數不高於「仍在動的其他股票」，或今日發動賠10%以上比例超過35%，就停用今日發動分級，只保留「仍在動」。在那之前不調整門檻。
