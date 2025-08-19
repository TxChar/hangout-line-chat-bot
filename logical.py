import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

# บังคับใช้ CPU แทน MPS เพื่อหลีกเลี่ยง tensor conversion error
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

model = SentenceTransformer(
    "sentence-transformers/distiluse-base-multilingual-cased-v2", device=device
)

# Load data
df = pd.read_csv("hangout_info.csv")

# Corpus definitions
question_greeting_corpus = [
    "สวัสดี",
    "สวัสดีงับ",
    "ว่าไง",
    "หืมมม...",
    "ดี",
    "ดีจ้า",
    "ไง",
    "โย่ว",
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
]
question_ranking_corpus = ["จัดอันดับ", "ร้านที่ดีที่สุด", "ร้านน่าไป", "อันดับ", "จัดอันดับ"]
question_location_corpus = [
    "อยู่ที่ไหน",
    "ไปยังไง",
    "สถานที่ของร้าน",
    "สถานที่",
    "ขอโลเคชั่น",
    "ปักหมุด",
]
question_recommend_corpus = [
    "ช่วยแนะนำ",
    "แนะนำ",
    "ช่วยเหลือ",
    "ต้องการรู้",
    "ต้องการหาร้าน",
    "แนะนำร้าน",
    "ช่วยพาไป",
    "ช่วยเลือก",
    "แนะนำร้านเหล้า",
    "แนะนำร้าน",
    "ต้องการรู้จักร้าน",
    "ร้านที่ต้องการ",
    "นำเสนอ",
    "เสนอร้านเหล้า",
]
question_detail_corpus = [
    "รายละเอียด",
    "ละเอียด",
    "เนื้อหา",
    "รายละเอียดร้าน",
    "ละเอียดร้าน",
    "เนื้อหาร้าน",
]
question_stores_corpus = [
    "ร้านทั้งหมด",
    "ทุกร้าน",
    "เฉพาะชื่อร้าน",
    "ชื่อร้านเท่านั้น",
    "รายชื่อร้าน",
    "อันดับร้าน",
    "รายชื่อร้านเหล้า",
    "อันดับร้านเหล้า",
]
agree_term = ["ใช่", "ต้องการ", "ช่าย", "ต้อง"]
disagree_term = ["ไม่ใช่", "ไม่ต้องการ", "ม่าย", "ไม่"]
cancle_corpus = ["ยกเลิก", "ต้องการยกเลิก"]
question_thinking_corpus = agree_term + disagree_term
asking_thank_corpus = [
    "ขอบคุณ",
    "ขอบคุณจ้า",
    "ขอบคุณครับ",
    "ขอบคุณค้าบ",
    "ขอบคุณค้า",
    "ขอบคุณค่ะ",
    "แต้งจ้า",
    "แต้ง",
    "ขอบจ้า",
    "บายๆ",
    "เจอกันใหม่",
]
asking_hangout_corpus = [
    "ร้านแฮงค์เอาท์ใกล้จตุจักร",
    "ร้านเหล้าใกล้จตุจักร",
    "ร้านจตุจักร",
    "ใกล้จตุจักร",
    "ร้านจตุจกร",
    "ใกล้จตุจกร",
    "ร้านเหล้าจตุจักร",
]
asking_corpus = asking_hangout_corpus + question_detail_corpus + question_stores_corpus
combined_question_corpus = (
    cancle_corpus
    + asking_thank_corpus
    + question_greeting_corpus
    + question_hangout_corpus
    + question_ranking_corpus
    + question_location_corpus
    + question_recommend_corpus
    + question_thinking_corpus
    + question_detail_corpus
    + question_stores_corpus
)

user_input = []


def calculate_similarity_score(question, corpus):
    question_vec = model.encode(
        question, convert_to_tensor=True, normalize_embeddings=True
    )
    corpus_vec = model.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)
    cosine_scores = util.cos_sim(question_vec, corpus_vec)

    # แปลง tensor เป็น numpy array บน CPU
    if hasattr(cosine_scores, "cpu"):
        entity_vector = cosine_scores.cpu().numpy()
    else:
        entity_vector = np.array(cosine_scores)

    score = np.max(entity_vector)
    if score >= 0.6:
        match_entity = corpus[np.argmax(entity_vector)]
        return [match_entity, score]
    else:
        return [
            f"บอทน้อยไม่เข้าใจ (T_T) กรุณาถามบอทน้อยอีกครั้งเช่น ร้านเหล้าใกล้จตุจักร ร้านเหล้า",
            score,
        ]


def greeting_filtering(input):
    ans = ""
    if input in question_greeting_corpus:
        ans = (
            input + "ถามบอทน้อยเกี่ยวกับ (ร้านแฮงค์เอาท์ใกล้จตุจักร)หรือ(ร้านเหล้าแนะนำ) ได้เลยนะ !🍻"
        )
    elif input in question_hangout_corpus:
        ans = (
            input
            + "หากพูดถึงร้านแฮงค์เอาท์บริเวณนี้บอทน้อยขอแนะนำให้ค้นหาว่า ร้านเหล้าใกล้จตุจักร หรือ ร้านแนะนำ"
        )
    return ans


def store_ranking_filtering(input):
    answer_sentence = ""
    if input in asking_hangout_corpus:
        answer_sentence = "บอทน้อยสงสัยว่า คุณต้องการ(รายละเอียดร้าน)หรือ(รายชื่อร้าน)?"
    elif input in question_stores_corpus:
        hangout_ans_df = df[["ชื่อร้าน"]].to_dict(orient="records")
        answer_sentence += f"บอทน้อยขอแนะนำ นี้คือรายชื่อร้านที่ดีที่สุดทั้งหมด \n"
        answer_sentence += "\n"
        for i in range(len(hangout_ans_df)):
            for key, value in hangout_ans_df[i].items():
                answer_sentence += f"ร้านที่ {i+1} : {value}" + "\n"
        answer_sentence += "คุณสามารถถามรายละเอียดเพิ่มเติมได้เช่น ร้านแนะนำ ร้านเหล้าแนะนำ"
    elif input in question_detail_corpus:
        hangout_ans_df = df.to_dict(orient="records")
        answer_sentence += f"บอทน้อยขอแนะนำ นี้คือรายละเอียดและชื่อร้าน \n"
        answer_sentence += "\n"
        for i in range(len(hangout_ans_df)):
            for key, value in hangout_ans_df[i].items():
                answer_sentence += f"{key} : {value}" + "\n"
            answer_sentence += "\n"
        answer_sentence += "คุณสามารถถามรายละเอียดเพิ่มเติมได้เช่น ร้านแนะนำ ร้านเหล้าแนะนำ"
    return answer_sentence


def recommendation(input):
    message = "ไมทราบ"
    if input in question_recommend_corpus:
        user_input.clear()
        message = "🌃 ต้องการร้านเปิดหลังเที่ยงคืนไหม (ต้องการ, ไม่ต้องการ)"
    elif input in question_thinking_corpus:
        if input in agree_term:
            input = "ใช่"
        elif input in disagree_term:
            input = "ไม่ใช่"
        user_input.append(input)
        len_user_input = len(user_input)
        if len_user_input == 1:
            message = "🚗 ต้องการที่จอดรถไหม (ต้องการ, ไม่ต้องการ)"
        elif len_user_input == 2:
            message = "📞 ต้องการช่องทางการติดต่อไหม (ต้องการ, ไม่ต้องการ)"
        elif len_user_input == 3:
            message = store(user_input)
        else:
            message = "😫บอทน้อยพบว่าคุณใส่ความต้องการมากเกินไป กรุณาถามบอทน้อยอีกครั้งเช่น ร้านเหล้าใกล้จตุจักร ร้านเหล้า"
            user_input.clear()
    return message


def store(input):
    late_input = input[0]
    parking_input = input[1]
    contact_input = input[2]
    store_dataframe = df.copy()
    if late_input == "ใช่":
        store_dataframe = store_dataframe[store_dataframe["มีที่จอดรถ"] == "ใช่"]
    elif late_input == "ไม่ใช่":
        store_dataframe = store_dataframe[store_dataframe["มีที่จอดรถ"] == "ไม่ใช่"]
    if parking_input == "ใช่":
        store_dataframe = store_dataframe[store_dataframe["เปิดหลังเที่ยงคืน"] == "ใช่"]
    elif parking_input == "ไม่ใช่":
        store_dataframe = store_dataframe[store_dataframe["เปิดหลังเที่ยงคืน"] == "ไม่ใช่"]
    if contact_input == "ใช่":
        pass
    if contact_input == "ไม่ใช่":
        store_dataframe = store_dataframe.drop(
            columns=["ช่องทางติดต่อ", "เว็บไซต์"], errors="ignore"
        )
    store_dataframe = store_dataframe.drop(
        columns=["อันดับ", "มีที่จอดรถ", "เปิดหลังเที่ยงคืน"], errors="ignore"
    )
    answer_sentence = ""
    hangout_ans_df = store_dataframe.to_dict(orient="records")
    len_hangout_ans = len(hangout_ans_df)
    if len_hangout_ans > 0:
        answer_sentence += f"บอทน้อยขอแนะนำร้านแฮงค์เอาท์ใกล้จตุจักรที่คุณต้องการ (^_^)"
        answer_sentence += "\n\n"
        for i in range(len_hangout_ans):
            answer_sentence += f"ร้านที่ : {i+1}" + "\n"
            for key, value in hangout_ans_df[i].items():
                answer_sentence += f"{key} : {value}" + "\n"
            answer_sentence += "\n"
        answer_sentence += "ขอบคุณที่สอบถามกับบอทน้อย😙 คุณสามารถสอบถามเกี่ยวกับร้านเหล้าได้เพิ่มเติมนะแล้วไว้เจอกันใหม่สวัสดีจ้าา!"
    else:
        answer_sentence += f"บอทน้อยพบว่าร้านที่คุณต้องการไม่มีอยู่ในสมองอันชาญฉลาดของบอทน้อย"
        answer_sentence += "\n\n"
        answer_sentence += f"กรุณาค้นหา ร้านแนะนำ ใหม่อีกครั้ง"
    return answer_sentence


def chat_answer(input):
    corpus = combined_question_corpus
    output_corpus = calculate_similarity_score(input, corpus)
    if input in asking_corpus:
        answer = store_ranking_filtering(input)
    elif output_corpus[0] in question_greeting_corpus:
        answer = greeting_filtering(output_corpus[0])
    elif output_corpus[0] in cancle_corpus:
        user_input.clear()
        answer = f"[คุณยกเลิกการแนะนำร้านแล้ว!]บอทน้อยเข้าใจว่าคุณใจโลเลไม่รักจริง😳🔥 \n\nแต่คุณยังสามารถสอบถาม(ร้านเหล้าแนะนำ)หรือ(ร้านเหล้าใกล้จตุจักร)ได้น้าา!!"
    elif output_corpus[0] in asking_thank_corpus:
        answer = "ขอบคุณที่สอบถามกับบอทน้อย😙 คุณสามารถสอบถามเกี่ยวกับร้านเหล้าได้เพิ่มเติมนะแล้วไว้เจอกันใหม่สวัสดีจ้าา!"
    elif output_corpus[0] in question_hangout_corpus:
        answer = greeting_filtering(output_corpus[0])
    elif output_corpus[0] in question_recommend_corpus:
        answer = recommendation(output_corpus[0])
    elif output_corpus[0] in question_thinking_corpus:
        answer = recommendation(output_corpus[0])
    else:
        answer = f"{input} บอทน้อยไม่เข้าใจ😭กรุณาถามบอทน้อยอีกครั้งเช่น ร้านเหล้าใกล้จตุจักร ร้านเหล้า"
    return answer
