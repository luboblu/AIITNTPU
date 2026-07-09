import os
import base64
from openai import OpenAI
from io import BytesIO
import numpy as np
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import torch
import warnings
from dotenv import load_dotenv
from PIL import Image
import requests

client = OpenAI()

load_dotenv()
API_KEY = os.getenv("NV_IMAGE2TEXT_API_KEY")
API_URL = os.getenv("NV_IMAGE2TEXT_API_URL")
MAX_B64_SIZE = 180_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_and_partition_text(file_path, chunk_size=300, chunk_overlap=50):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    sections = content.split("####")
    partitioned = {}
    for sec in sections:
        if sec.strip():
            lines = sec.strip().split("\n")
            header = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n", ".", "\u3002"]
            )
            docs = [Document(page_content=body)]
            splits = splitter.split_documents(docs)
            segments = [f"{header}\n{d.page_content}" for d in splits]
            partitioned[header] = segments
    return partitioned

def initialize_rag(file_path):
    partitions = load_and_partition_text(file_path)
    model = SentenceTransformer("intfloat/multilingual-e5-base", device=device)
    indexes, segments_map = {}, {}
    for header, segs in partitions.items():
        embeddings = model.encode(segs, batch_size=8, show_progress_bar=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))
        indexes[header] = index
        segments_map[header] = segs
    return model, indexes, segments_map

model, indexes, segments_map = initialize_rag('AIITNTPU.txt')
model.to(device)

def query_rag(query):
    prefix = query[:5]
    target = None
    for name in ["中華民國腦性麻痺協會", "漸凍人協會", "陽光基金會", "AIITNTPU計畫內容介紹"]:
        if name in prefix:
            target = name
            break
    results = []
    def search_in(headers):
        q_emb = model.encode([query], batch_size=1, show_progress_bar=False)
        for hdr in headers:
            D, I = indexes[hdr].search(np.array(q_emb), k=3)
            for dist, idx in zip(D[0], I[0]):
                if idx != -1 and dist < 1.5:
                    results.append(segments_map[hdr][idx])
    if target and target in indexes:
        search_in([target])
    else:
        search_in(list(indexes.keys()))
    return results or ["很抱歉，找不到相關內容。"]

def generate_answer(query, contexts, history=None):
    context = "\n".join(contexts)[:2000]
    messages = [{
    "role": "system",
    "content": (
        "你是AIITNTPU計畫的客服助理，請用自然、口語、簡潔的方式回答使用者的問題。"
        "若需要使用先前的圖片說明或對話歷史，可自行整合內容。請避免重複提到“上下文”或“如前所述”等字眼。"
        "回答字數請控制在200字以內。"
    )
}]

    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"問題: {query}\n上下文: {context}"})
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=200
    )
    answer = resp.choices[0].message.content
    if history is not None:
        history.append({"role": "assistant", "content": answer})
    return answer

def chat_text(user_input, chat_history, message_state):
    segs = query_rag(user_input)
    message_state.append({"role": "user", "content": user_input})
    reply = generate_answer(user_input, segs, message_state)
    message_state.append({"role": "assistant", "content": reply})
    chat_history.append((user_input, reply))
    return chat_history, message_state, ""  # ⬅️ 清空輸入欄


def chat_audio(audio_file, chat_history, message_state):
    if not audio_file:
        return chat_history, message_state, None
    try:
        with open(audio_file, "rb") as af:
            resp = client.audio.transcriptions.create(model="whisper-1", file=af)
        transcription = resp.text
    except Exception as e:
        return chat_history + [("語音辨識失敗", str(e))], message_state, None
    segs = query_rag(transcription)
    message_state.append({"role": "user", "content": transcription})
    reply = generate_answer(transcription, segs, message_state)
    message_state.append({"role": "assistant", "content": reply})
    chat_history.append((transcription, reply))
    return chat_history, message_state, None

def encode_img(image: Image.Image, quality: int = 85) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()

def chat_image(image_path, chat_history, message_state):
    if not image_path:
        return chat_history, message_state, None
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        error_msg = f"圖片讀取失敗：{e}"
        chat_history.append(("圖片上傳失敗", error_msg))
        message_state.append({"role": "assistant", "content": error_msg})
        return chat_history, message_state, None

    b64 = encode_img(img)
    quality = 85
    while len(b64) >= MAX_B64_SIZE and quality >= 20:
        quality -= 15
        b64 = encode_img(img, quality)
    if len(b64) >= MAX_B64_SIZE:
        img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
        b64 = encode_img(img, quality)
    if len(b64) >= MAX_B64_SIZE:
        error_msg = "圖片過大，請使用更小的圖片"
        chat_history.append(("圖片過大", error_msg))
        message_state.append({"role": "assistant", "content": error_msg})
        return chat_history, message_state, None

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "google/gemma-3-27b-it",
        "messages": [
            {"role": "system", "content": "請用繁體中文描述以下圖片。"},
            {"role": "user", "content": f'<img src="data:image/jpeg;base64,{b64}" />'},
        ],
        "max_tokens": 512,
        "temperature": 0.2,
        "top_p": 0.7,
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        reply = f"圖片辨識失敗：{e}"

    img_html = f'<img src="data:image/jpeg;base64,{b64}" width="220"/>'
    chat_history.append((img_html, reply))
    # 加入完整標記讓後續能提問「這是給誰的證書？」這類問題
    message_state.append({
        "role": "assistant",
        "content": f"[圖片描述] {reply}"
    })

    return chat_history, message_state, None


with gr.Blocks() as demo:
    gr.Markdown("## AIITNTPU 計畫客服助理（支援上下文與圖片顯示）")
    chatbot_ui = gr.Chatbot(label="對話紀錄")
    history_state = gr.State([])

    with gr.Row():
        txt = gr.Textbox(label="輸入您的問題（文字）")
        mic = gr.Microphone(label="語音輸入（講完自動辨識）", type="filepath")
        img = gr.Image(label="圖片輸入", type="filepath")

    txt.submit(chat_text, [txt, chatbot_ui, history_state], [chatbot_ui, history_state, txt])
    mic.change(chat_audio, [mic, chatbot_ui, history_state], [chatbot_ui, history_state, mic])
    img.change(chat_image, [img, chatbot_ui, history_state], [chatbot_ui, history_state, img])


    gr.Markdown("---\n📌 支援多輪對話 + 語音輸入 + 圖片辨識 + 圖片顯示")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True, server_port=7860)
