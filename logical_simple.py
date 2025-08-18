import pandas as pd
import os
import re
from difflib import SequenceMatcher

# Load data
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "hangout_info.csv")
df = pd.read_csv(csv_path)

# Corpus definitions
question_greeting_corpus = [
    "สวัสดี",
    "สวัสดีงับ",
    "ว่าไง",
    "หืมมม",
    "ดี",
    "ดีจ้า",
    "ไง",
    "โย่ว",
    "hello",
    "hi",
]
question_hangout_corpus = [
    "ร้านเหล้า",
    "แฮงค์เอาท์",
    "ร้านนั่งชิล",
    "ร้านดื่ม",
    "ร้านกลางคืน",
    "ร้านเหล้ากลางคืน",
    "ร้านเหล้าที่ไหนดี",
    "ที่ไหน",
    "hangout",
    "bar",
    "pub",
]
question_ranking_corpus = [
    "จัดอันดับ",
    "ร้านที่ดีที่สุด",
    "ร้านน่าไป",
    "อันดับ",
    "ranking",
    "best",
    "top",
]
question_location_corpus = [
    "อยู่ที่ไหน",
    "ไปยังไง",
    "สถานที่ของร้าน",
    "สถานที่",
    "ขอโลเคชั่น",
    "ปักหมุด",
    "location",
    "address",
    "where",
]
question_recommend_corpus = ["ช่วยแนะนำ", "แนะนำ", "recommend", "suggest"]


def simple_text_similarity(text1, text2):
    """คำนวณความคล้ายคลึงด้วย string matching"""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def find_best_match(user_input, corpus):
    """หาคำที่ตรงกันมากที่สุดใน corpus"""
    user_input = user_input.lower()
    best_score = 0
    best_match = None

    for item in corpus:
        # ตรวจสอบว่ามีคำนี้อยู่ในข้อความไหม
        if item.lower() in user_input:
            return item, 1.0

        # คำนวณความคล้ายคลึง
        score = simple_text_similarity(user_input, item)
        if score > best_score:
            best_score = score
            best_match = item

    return best_match, best_score


def classify_question(user_input):
    """จำแนกประเภทคำถาม"""
    user_input = user_input.lower()

    # เช็คคำที่สำคัญในแต่ละหมวดหมู่
    if any(word in user_input for word in ["สวัสดี", "ว่าไง", "ดี", "hello", "hi"]):
        return "greeting"
    elif any(
        word in user_input
        for word in ["อันดับ", "จัดอันดับ", "ดีที่สุด", "ranking", "best", "top"]
    ):
        return "ranking"
    elif any(
        word in user_input
        for word in ["ที่ไหน", "สถานที่", "โลเคชั่น", "location", "address", "where"]
    ):
        return "location"
    elif any(word in user_input for word in ["แนะนำ", "recommend", "suggest"]):
        return "recommend"
    elif any(
        word in user_input for word in ["ร้าน", "เหล้า", "ดื่ม", "hangout", "bar", "pub"]
    ):
        return "hangout"
    else:
        return "unknown"


def handle_greeting():
    """จัดการคำทักทาย"""
    return "สวัสดีครับ! ผมเป็นบอทแนะนำร้านแฮงค์เอาท์ 🍻\n\nคุณสามารถถามได้เช่น:\n• ร้านเหล้าที่ไหนดี\n• จัดอันดับร้านให้หน่อย\n• แนะนำร้านหน่อย"


def handle_hangout_question(user_input=None):
    """จัดการคำถามเกี่ยวกับร้านแฮงค์เอาท์"""
    if df.empty:
        return "ขออภัยครับ ไม่มีข้อมูลร้านในระบบ"

    # เลือกร้านสุ่ม 3 ร้าน
    sample_shops = df.sample(min(3, len(df)))

    response = "🍻 แนะนำร้านแฮงค์เอาท์:\n\n"
    for idx, row in sample_shops.iterrows():
        response += f"📍 {row.get('name', 'ไม่มีชื่อ')}\n"
        response += f"   📍 {row.get('location', 'ไม่มีที่อยู่')}\n"
        if "rating" in row and pd.notna(row["rating"]):
            response += f"   ⭐ {row['rating']}\n"
        response += "\n"

    return response


def handle_ranking_question():
    """จัดการคำถามเกี่ยวกับการจัดอันดับ"""
    if df.empty:
        return "ขออภัยครับ ไม่มีข้อมูลร้านในระบบ"

    # เรียงตาม rating ถ้ามี
    if "rating" in df.columns:
        top_shops = df.nlargest(5, "rating")
    else:
        top_shops = df.head(5)

    response = "🏆 Top ร้านแฮงค์เอาท์:\n\n"
    for idx, (_, row) in enumerate(top_shops.iterrows(), 1):
        response += f"{idx}. {row.get('name', 'ไม่มีชื่อ')}\n"
        response += f"   📍 {row.get('location', 'ไม่มีที่อยู่')}\n"
        if "rating" in row and pd.notna(row["rating"]):
            response += f"   ⭐ {row['rating']}\n"
        response += "\n"

    return response


def handle_location_question(user_input):
    """จัดการคำถามเกี่ยวกับสถานที่"""
    if df.empty:
        return "ขออภัยครับ ไม่มีข้อมูลร้านในระบบ"

    # ลองหาชื่อร้านในข้อความ
    shop_found = None
    for _, row in df.iterrows():
        shop_name = row.get("name", "")
        if shop_name and shop_name.lower() in user_input.lower():
            shop_found = row
            break

    if shop_found is not None:
        response = f"📍 {shop_found.get('name', 'ไม่มีชื่อ')}\n"
        response += f"ที่อยู่: {shop_found.get('location', 'ไม่มีที่อยู่')}\n"
        if "rating" in shop_found and pd.notna(shop_found["rating"]):
            response += f"⭐ {shop_found['rating']}\n"
        return response
    else:
        # ถ้าไม่เจอร้านเฉพาะ ให้แสดงร้านทั้งหมด
        response = "📍 รายการร้านทั้งหมด:\n\n"
        for _, row in df.head(5).iterrows():
            response += f"• {row.get('name', 'ไม่มีชื่อ')}\n"
            response += f"  📍 {row.get('location', 'ไม่มีที่อยู่')}\n\n"
        return response


def chat_answer(user_input):
    """ฟังก์ชันหลักสำหรับตอบคำถาม"""
    if not user_input or not user_input.strip():
        return "บอทน้อยไม่เข้าใจ 🤔"

    question_type = classify_question(user_input)

    if question_type == "greeting":
        return handle_greeting()
    elif question_type == "hangout":
        return handle_hangout_question(user_input)
    elif question_type == "ranking":
        return handle_ranking_question()
    elif question_type == "location":
        return handle_location_question(user_input)
    elif question_type == "recommend":
        return handle_hangout_question(user_input)
    else:
        return "บอทน้อยไม่เข้าใจคำถามนี้ 🤔\n\nลองถามแบบนี้ดู:\n• ร้านเหล้าที่ไหนดี\n• จัดอันดับร้านให้หน่อย\n• แนะนำร้านหน่อย"


# Test function
if __name__ == "__main__":
    test_questions = ["สวัสดี", "ร้านเหล้าที่ไหนดี", "จัดอันดับร้านให้หน่อย", "แนะนำร้านหน่อย"]

    for q in test_questions:
        print(f"Q: {q}")
        print(f"A: {chat_answer(q)}")
        print("-" * 50)
