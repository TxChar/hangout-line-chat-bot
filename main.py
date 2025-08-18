from flask import Flask, request, abort
from linebot.v3.messaging import (
    MessagingApi,
    Configuration,
    ApiClient,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhook import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from dotenv import load_dotenv
from pyngrok import ngrok
import json
import os
from logical import chat_answer

load_dotenv()

app = Flask(__name__)

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
SECRET = os.getenv("SECRET")

if not ACCESS_TOKEN or not SECRET:
    print("⚠️  ACCESS_TOKEN หรือ SECRET ไม่ถูกต้อง กรุณาตรวจสอบไฟล์ .env")
    exit(1)

configuration = Configuration(access_token=ACCESS_TOKEN)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)
handler = WebhookHandler(SECRET)

print(f"✅ โหลด ACCESS_TOKEN และ SECRET เรียบร้อย")


@app.route("/", methods=["POST"])
def linebot():
    body = request.get_data(as_text=True)
    signature = request.headers.get("X-Line-Signature", "")

    # กรณี LINE Verify Webhook (ไม่มี signature)
    if not signature:
        print("📝 LINE Verify Webhook - ตอบกลับ 200 OK")
        return "OK", 200

    print(f"📨 รับข้อความ: signature={signature[:20]}...")

    try:
        # ตรวจสอบ signature
        handler.handle(body, signature)

        # จัดการข้อความ
        json_data = json.loads(body)
        events = json_data.get("events", [])

        if not events:
            print("⚠️  ไม่มี events ใน request")
            return "OK", 200

        event = events[0]

        # เช็คว่าเป็น text message หรือไม่
        if (
            event.get("type") != "message"
            or event.get("message", {}).get("type") != "text"
        ):
            print("ℹ️  ไม่ใช่ text message - ข้าม")
            return "OK", 200

        msg = event["message"]["text"]
        tk = event["replyToken"]

        print(f"💬 ข้อความ: {msg}")

        # ประมวลผลข้อความด้วย chatbot logic
        reply_msg = chat_answer(msg) if msg else "บอทน้อยไม่เข้าใจ"

        # ส่งข้อความตอบกลับ
        line_bot_api.reply_message(
            ReplyMessageRequest(replyToken=tk, messages=[TextMessage(text=reply_msg)])
        )

        print(f"✅ ตอบกลับ: {reply_msg[:50]}...")

    except InvalidSignatureError:
        print("❌ Invalid signature")
        abort(403)

    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Body: {body}")

    return "OK", 200


if __name__ == "__main__":
    try:
        print("🚀 เริ่มต้น LINE ChatBot...")

        # เปิด ngrok tunnel
        public_url = ngrok.connect(3000)
        print(f"🌐 Ngrok URL: {public_url}")
        print("📋 คัดลอก URL ข้างต้นไปใส่ใน LINE Developer Console (Webhook URL)")
        print("📍 กด Verify และ Use webhook ใน LINE Console")
        print("-" * 60)

        # รัน Flask app
        app.run(host="0.0.0.0", port=3000, debug=False)

    except KeyboardInterrupt:
        print("\n👋 หยุดทำงาน")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
