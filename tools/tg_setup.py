"""Telegram 設定：用 TELEGRAM_BOT_TOKEN 讀機器人收到的訊息，找出使用者的 chat id，直接用 Telegram 告訴使用者（repo 是公開的，log 不印 id、絕不印 token）。"""
import os, requests
import re
tok = re.sub(r"\s+", "", os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip('"').strip("'")  # token 不會有空白；貼上時夾到的空白、換行都拿掉
print("token 格式", "看起來正確" if re.fullmatch(r"\d{6,12}:[A-Za-z0-9_-]{30,}", tok) else f"不像 BotFather 給的格式（長度 {len(tok)}，有冒號：{':' in tok}，有空白：{' ' in tok}）")
if not tok: print("沒有設定 TELEGRAM_BOT_TOKEN"); raise SystemExit(0)
try: j = requests.get(f"https://api.telegram.org/bot{tok}/getUpdates", timeout=20).json()
except Exception: print("連不上 Telegram"); raise SystemExit(0)
if not j.get("ok"): print("Telegram 回覆錯誤：", j.get("error_code"), j.get("description")); raise SystemExit(0)
res = j.get("result", []); print("收到的更新", len(res), "種類", sorted({k for u in res for k in u if k != "update_id"}), "對話類型", sorted({((u.get("message") or u.get("edited_message") or u.get("my_chat_member") or {}).get("chat") or {}).get("type", "?") for u in res}))
try: me = requests.get(f"https://api.telegram.org/bot{tok}/getMe", timeout=20).json().get("result", {}); print("機器人帳號 @" + str(me.get("username")))
except Exception: pass
ids = []
for u in j.get("result", []):
    m = u.get("message") or u.get("edited_message") or u.get("my_chat_member") or {}
    c = m.get("chat") or {}
    if c.get("type") == "private" and c.get("id") not in ids: ids.append(c.get("id"))
print("找到", len(ids), "個對話", "（末 3 碼 " + str(ids[-1])[-3:] + "）" if ids else "（找不到：請先在 Telegram 對機器人傳任何一句話，再重跑）")  # repo 是公開的，log 不印完整 id
for c in ids[-1:]:
    r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", data={"chat_id": c, "text": f"強勢股雷達：機器人連線成功 ✅\n你的 chat id：{c}\n請到 GitHub repo → Settings → Secrets and variables → Actions，新增 secret：名稱 TELEGRAM_CHAT_ID，值填上面這串數字。"}, timeout=20)
    print("test message", "ok" if r.status_code == 200 else f"failed {r.status_code}")
