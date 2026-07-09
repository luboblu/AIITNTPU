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
from dotenv import load_dotenv
from PIL import Image
import requests

# 台語 LLM 相關
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
import accelerate

# 初始化 OpenAI 客戶端\

client = OpenAI()

# 載入環境變數
load_dotenv()
API_KEY      = os.getenv("NV_IMAGE2TEXT_API_KEY")
API_URL      = os.getenv("NV_IMAGE2TEXT_API_URL")
MAX_B64_SIZE = 180_000

# 指定運算裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化台語 LLM 管線
model_dir_tg = "Bohanlu/Taigi-Llama-2-Chat-7B"
tokenizer_tg = AutoTokenizer.from_pretrained(model_dir_tg, use_fast=False)
accelerator = accelerate.Accelerator()

def get_taiwanese_pipeline(path: str, tokenizer: AutoTokenizer, accelerator: accelerate.Accelerator) -> TextGenerationPipeline:
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    terminators = [tokenizer.eos_token_id, tokenizer.pad_token_id]
    return TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        num_workers=accelerator.state.num_processes * 4,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=terminators,
    )

pipe_tg = get_taiwanese_pipeline(model_dir_tg, tokenizer_tg, accelerator)

# ── RAG 系統相關 ─────────────────────────────────────────────────────────────

def load_and_partition_text(file_path, chunk_size=300, chunk_overlap=50):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    sections = content.split("####")
    partitioned = {}
    for sec in sections:
        if sec.strip():
            lines = sec.strip().split("\n")
            header = lines[0].strip()
            body   = "\n".join(lines[1:]).strip()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n", ".", "。"]
            )
            docs   = [Document(page_content=body)]
            splits = splitter.split_documents(docs)
            segments = [f"{header}\n{d.page_content}" for d in splits]
            partitioned[header] = segments
    return partitioned


def initialize_rag(file_path):
    partitions   = load_and_partition_text(file_path)
    model        = SentenceTransformer("intfloat/multilingual-e5-base", device=device)
    indexes      = {}
    segments_map = {}
    for header, segs in partitions.items():
        embeddings = model.encode(segs, batch_size=8, show_progress_bar=True)
        dim   = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))
        indexes[header]      = index
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

# ── 回答生成，依語言切換 ─────────────────────────────────────────────────

def generate_answer(query, contexts, lang="中文"):
    context = "\n".join(contexts)[:2000]
    if lang == "台語":
        # 使用台語 LLM
        user_msg = f"問題: {query}\n上下文: {context}"
        out = pipe_tg([
            {"role": "user", "content": user_msg}
        ], return_full_text=False, repetition_penalty=1.05, do_sample=True)[0]['generated_text']
        return out
    else:
        # 中文或 English
        system_prompt = "你是AIITNTPU計畫客服助理，請根據上下文回答問題，不超過200字。"
        if lang == "English":
            system_prompt = ("You are the AIITNTPU project assistant. "
                             "Please answer the question based on the context in English, under 200 words.")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"問題: {query}\n上下文: {context}"}
        ]
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        return resp.choices[0].message.content

# ── 圖片壓縮輔助函式 ────────────────────────────────────────────────────────

def encode_img(image: Image.Image, quality: int = 85) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()

# ── 呼叫 NVIDIA API 取得圖片描述 ─────────────────────────────────────────────

def describe_image_with_nvidia(image_path: str) -> str:
    if not image_path:
        return ""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return f"無法讀取圖片：{e}"
    b64     = encode_img(img)
    quality = 85
    while len(b64) >= MAX_B64_SIZE and quality >= 20:
        quality -= 15
        b64 = encode_img(img, quality)
    if len(b64) >= MAX_B64_SIZE:
        img = img.resize((img.width//2, img.height//2), Image.LANCZOS)
        b64 = encode_img(img, quality)
    if len(b64) >= MAX_B64_SIZE:
        raise gr.Error("影像經過壓縮後仍超出大小限制，請使用更小的影像。")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept":        "application/json",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": "google/gemma-3-27b-it",
        "messages": [
            {"role": "system", "content": "請用繁體中文描述以下圖片。"},
            {"role": "user",   "content": f'<img src="data:image/jpeg;base64,{b64}" />'},
        ],
        "max_tokens": 512,
        "temperature": 0.2,
        "top_p": 0.7,
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()

# ── 三大 Handler：文字 / 語音 / 圖片，皆帶入 lang，並清空輸入欄位 ─────────────

def chat_text(user_input, messages, lang):
    segs   = query_rag(user_input)
    answer = generate_answer(user_input, segs, lang)
    new_messages = messages + [
        {"role": "user",      "content": user_input},
        {"role": "assistant", "content": answer}
    ]
    return new_messages, new_messages, ""


def chat_audio(audio_file, messages, lang):
    if not audio_file:
        return messages, messages
    with open(audio_file, "rb") as af:
        resp = client.audio.transcriptions.create(model="whisper-1", file=af)
    transcription = resp.text
    segs   = query_rag(transcription)
    answer = generate_answer(transcription, segs, lang)
    new_messages = messages + [
        {"role": "user",      "content": transcription},
        {"role": "assistant", "content": answer}
    ]
    return new_messages, new_messages


def chat_image(image_path, messages, lang):
    if not image_path:
        return messages, messages, None
    ext = os.path.splitext(image_path)[1].lower().strip('.')
    data_uri = f"data:image/{ext};base64,{base64.b64encode(open(image_path,'rb').read()).decode()}"
    img_md = f"![image]({data_uri})"
    desc    = describe_image_with_nvidia(image_path)
    # 台語模式時，用台語 LLM 進一步生成
    if lang == "台語":
        answer = generate_answer(desc, [], lang)
    else:
        answer = desc
    new_messages = messages + [
        {"role": "user",      "content": img_md},
        {"role": "assistant", "content": answer}
    ]
    return new_messages, new_messages, None

# ── Gradio 介面 ─────────────────────────────────────────────────────────────

with gr.Blocks() as demo:
    gr.Markdown("### AIITNTPU 客服助理")

    chatbot = gr.Chatbot(label="對話紀錄", type="messages")
    state   = gr.State([])

    with gr.Row():
        txt  = gr.Textbox(label="文字輸入")
        mic  = gr.Microphone(label="語音輸入（講完自動轉文字）", type="filepath")
        img  = gr.Image(label="圖片輸入（拖進來自動描述）", type="filepath")
        lang = gr.Dropdown(choices=["中文", "English", "台語"], value="中文", label="選擇語言")

    txt.submit(chat_text,  inputs=[txt, state, lang], outputs=[chatbot, state, txt])
    mic.change(chat_audio, inputs=[mic, state, lang], outputs=[chatbot, state])
    img.change(chat_image, inputs=[img, state, lang], outputs=[chatbot, state, img])

    gr.Markdown("---\nAIITNTPU 計畫客服助理")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True, server_port=7860)
