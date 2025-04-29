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
# 新增影像處理相關套件
from PIL import Image
import requests
# 禁用不必要警告與平行處理
client = OpenAI()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)

# 設定 API 金鑰
load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
# NVIDIA Integrate 影像轉文字設定
API_KEY = os.getenv("NV_IMAGE2TEXT_API_KEY")
API_URL = os.getenv("NV_IMAGE2TEXT_API_URL")
MAX_B64_SIZE = 180_000
# 指定運算裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




# Step 1: 載入並切分文本
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
                separators=["\n", ".", "。"]
            )
            docs = [Document(page_content=body)]
            splits = splitter.split_documents(docs)
            segments = [f"{header}\n{d.page_content}" for d in splits]
            partitioned[header] = segments
    return partitioned

# Step 2: 初始化 RAG 系統
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

# Step 3: 查詢函數
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

# Step 4: 生成回答
def generate_answer(query, contexts):
    context = "\n".join(contexts)[:2000]
    messages = [
        {"role": "system", "content": "你是AIITNTPU計畫客服助理，請根據上下文回答問題，不超過200字。"},
        {"role": "user", "content": f"問題: {query}\n上下文: {context}"}
    ]
    resp = client.chat.completions.create(model="gpt-4.1",
    messages=messages,
    temperature=0.7,
    max_tokens=200)
    return resp.choices[0].message.content

# Gradio 處理文字
def chat_text(user_input):
    segs = query_rag(user_input)
    return generate_answer(user_input, segs)

# Base64 編碼與壓縮輔助
def encode_img(image: Image.Image, quality: int = 85) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


# Gradio 處理語音
def chat_audio(audio_file):
    try:
        if not audio_file:
            return ""
        # 讀取上傳的檔案
        with open(audio_file, "rb") as af:
            resp = client.audio.transcriptions.create(model="whisper-1", file=af)
        transcription = resp.text
    except Exception as e:
        return f"語音轉錄出錯：{e}"

    # RAG 
    segs = query_rag(transcription)
    return generate_answer(transcription, segs)

# Base64 編碼與壓縮輔助
def encode_img(image: Image.Image, quality: int = 85) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()

# 呼叫 NVIDIA API 進行圖片描述
def nvidia_image_to_text(image_path: str):
    # 檢查是否有檔案
    if not image_path:
        return ""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return f"無法讀取圖片：{e}"

    b64 = encode_img(img)
    quality = 85
    # 壓縮直到小於限制
    while len(b64) >= MAX_B64_SIZE and quality >= 20:
        quality -= 15
        b64 = encode_img(img, quality)
    if len(b64) >= MAX_B64_SIZE:
        img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)

        b64 = encode_img(img, quality)
    if len(b64) >= MAX_B64_SIZE:
        raise gr.Error("影像經過壓縮後仍超出大小限制，請使用更小的影像。")

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
    response = requests.post(API_URL, headers=headers, json=payload)
    try:
        response.raise_for_status()
    except requests.HTTPError:
        detail = response.json().get("detail", response.text)
        raise gr.Error(f"API Error {response.status_code}: {detail}")
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        raise gr.Error(f"Unexpected response format: {data}")


# Gradio 介面設置
with gr.Blocks() as demo:
    gr.Markdown("### AIITNTPU 客服助理")
    with gr.Row():
        txt = gr.Textbox(label="輸入您的問題（文字）")
        mic = gr.Microphone(label="語音輸入，講完即自動辨識", type="filepath")
        img = gr.Image(label="圖片輸入，拖進來我會描述", type="filepath")
    out = gr.Textbox(label="回覆")
    txt.submit(chat_text, txt, out)
    mic.change(chat_audio, mic, out)
    img.change(nvidia_image_to_text, img, out)
    gr.Markdown("---\nAIITNTPU 計畫客服助理")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True, server_port=7860)