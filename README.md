# 強勢股雷達

網站：https://dingtingdong.github.io/surge-data/ （手機可加到主畫面）

每個平日台北 18:30，GitHub Actions 自動抓證交所＋櫃買中心收盤行情與處置股公告、重算分級、更新網站。電腦不用開機。

## 名單分級（規則 v1.2）
- 仍在動：活躍度前10%（上市、上櫃分開排名）＋ 今天或前10個交易日內有單日漲 ≥ 9%
- 觀察：仍在動 ＋ 當天上漲 ＋ 量能比（5日均額 ÷ 20日均額）1～2
- 精選：仍在動 ＋ 當天上漲 ＋ 量能比 ≥ 2

這是「波動雷達」：名單上的股票接下來容易大漲也容易大跌，不是買進訊號。研究整理，不構成投資建議。

## 檔案
| 檔案 | 用途 |
| --- | --- |
| `radar_core.py` | 共用計算：讀行情、除權息／減資調整、指標、分級 |
| `build_site.py` | 寫每日紀錄、算追蹤報酬、產生網站 |
| `site_template.html` | 網站畫面 |
| `fetch_prices.py` | 每日行情（含對參考價的漲跌，用來做除權息調整） |
| `fetch_disposal.py` | 處置股公告 |
| `fetch_history.py`, `merge_history.py` | 補 2020 年起的歷史行情 |
| `research/backtest.py` | 長期回測（檢驗方法事先寫在檔頭，不調門檻） |
| `data/prices.csv` | 今年日行情；更早的在 `data/history/YYYY.csv.gz` |
| `data/signals.csv` | 每天「仍在動」完整名單與分級（只新增） |
| `data/days.csv` | 每天各級檔數（零檔也記） |
| `data/disposal.csv` | 處置股起迄日 |
| `data/fetch_log.csv` | 抓取失敗紀錄（有失敗才會出現） |

## 工作流程（Actions）
- `update-prices`：每日排程
- `backfill-history`：手動，補指定期間的歷史行情
- `disposal-history`：手動，補 2020 年起的處置股

## 事先定好的停止條件
累積約26週實際結果後，若精選的5日報酬中位數不高於「仍在動的其他股票」，或精選賠10%以上比例超過35%，就停用精選分級，只保留「仍在動」。在那之前不調整門檻。
