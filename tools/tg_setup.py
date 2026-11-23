"""Telegram 設定：用 TELEGRAM_BOT_TOKEN 讀機器人收到的訊息，找出使用者的 chat id（不是機密），再發一則測試訊息。絕對不印 token。"""
import os, requests
tok = os.environ.get("TELEGRAM_BOT_TOKEN")
if not tok: print("沒有設定 TELEGRAM_BOT_TOKEN"); raise SystemExit(0)
try: j = requests.get(f"https://api.telegram.org/bot{tok}/getUpdates", timeout=20).json()
except Exception: print("連不上 Telegram"); raise SystemExit(0)
if not j.get("ok"): print("token 不對或機器人無法使用"); raise SystemExit(0)
ids = []
for u in j.get("result", []):
    m = u.get("message") or u.get("edited_message") or {}
    c = m.get("chat") or {}
    if c.get("type") == "private" and c.get("id") not in ids: ids.append(c.get("id"))
print("CHAT_IDS", ids if ids else "（找不到：請先在 Telegram 對機器人傳任何一句話，再重跑）")
for c in ids[-1:]:
    r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", data={"chat_id": c, "text": "強勢股雷達：通知設定成功 ✅"}, timeout=20)
    print("test message", "ok" if r.status_code == 200 else f"failed {r.status_code}")
