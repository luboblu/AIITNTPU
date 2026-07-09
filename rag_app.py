import os
import re
import faiss
import fitz
import torch
import gradio as gr
import numpy as np
import json
import base64
import requests
import mimetypes
import tempfile
from io import BytesIO
from PIL import Image
from pathlib import Path
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from docx import Document
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

# =========================
# 1. 基本設定與環境變數
# =========================
load_dotenv()
DATA_ROOT = Path("./data")
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
FIXED_TOP_K = 4
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MARGIN = 0.05
MIN_SIM = 0.20
MEMORY_TURNS = 5
MEMORY_CHARS_LIMIT = 1200

# NVIDIA Integrate (影像模型)
NIM_INVOKE_URL = os.getenv("NV_NIM_INVOKE_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
NIM_API_KEY = os.getenv("NV_NIM_API_KEY")
NIM_MODEL_MAIN = os.getenv("NV_NIM_MODEL_MAIN", "google/gemma-3-27b-it")
NIM_MODEL_FALL = os.getenv("NV_NIM_MODEL_FALL", "meta/llama-3.2-11b-vision-instruct")

MAX_B64_SIZE = 3_500_000 
MIN_EDGE_LIMIT = 640

# 組織註冊
ORG_REGISTRY = {
    "msm": {
        "label": "勵友協會（就業輔導）",
        "path": DATA_ROOT / "msm",
        "system_prompt": "你是基督教勵友中心的溫暖客服助理，請依據提供的資料回答問題，語氣親切、鼓勵且簡潔（120字內）。"
    },
    "tyad": {
        "label": "桃園輔具中心（輔具/補助）",
        "path": DATA_ROOT / "tyad",
        "system_prompt": "你是桃園市北區輔具資源中心的客服助理，請根據資料內容回覆使用者問題，語氣溫和、清楚，條列說明（150字內）。"
    }
}

LANG_LABELS = ["繁體中文", "English", "Vietnamese"]
LANG_CODE = {"繁體中文": "zh-Hant", "English": "en", "Vietnamese": "vi"}

def lang_instruction(code: str) -> str:
    return {"zh-Hant": "請全程使用繁體中文作答。", "en": "Please answer entirely in English.", "vi": "Vui lòng trả lời hoàn toàn bằng tiếng Việt."}[code]

# =========================
# 2. 文件解析與掃描
# =========================
_Q_PAT = re.compile(r"^(?:Q|問)\s*[:：．.]?\s*(.+)", flags=re.IGNORECASE)
_A_PAT = re.compile(r"^(?:A|答)\s*[:：．.]?\s*(.+)", flags=re.IGNORECASE)

def load_txt_segments(path: Path):
    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines = raw.splitlines()
    segs, buf = [], []
    def flush():
        nonlocal buf
        if buf:
            txt = "\n".join(buf).strip()
            if txt: segs.append(txt)
            buf = []
    for ln in lines:
        if ln.strip().startswith(("##", "###")):
            flush(); buf = [ln.strip()]
        else:
            if not buf: buf = [ln.strip()]
            else: buf.append(ln.strip())
    flush()
    return segs

def load_docx_segments(path: Path):
    doc = Document(str(path))
    segs, buf_q = [], None
    for p in doc.paragraphs:
        t = p.text.strip()
        if not t: continue
        mq, ma = _Q_PAT.match(t), _A_PAT.match(t)
        if mq:
            if buf_q: segs.append("問題：" + buf_q)
            buf_q = mq.group(1).strip(); continue
        if ma and buf_q:
            segs.append(f"問題：{buf_q}\n答案：{ma.group(1).strip()}")
            buf_q = None; continue
        segs.append(t)
    if buf_q: segs.append("問題：" + buf_q)
    return segs

def load_pdf_segments(path: Path):
    segs = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            text = page.get_text("text")
            if text.strip(): segs.append(text.strip())
    return segs

def scan_org_dir(org_path: Path):
    segs, tags = [], []
    org_path.mkdir(parents=True, exist_ok=True)
    for p in sorted(org_path.rglob("*")):
        if not p.is_file(): continue
        ext = p.suffix.lower()
        if ext == ".txt": s = load_txt_segments(p)
        elif ext == ".docx": s = load_docx_segments(p)
        elif ext == ".pdf": s = load_pdf_segments(p)
        else: continue
        segs.extend(s); tags.extend([p.name] * len(s))
    return segs, tags

# =========================
# 3. 向量檢索核心
# =========================
def make_embedder(name: str):
    return SentenceTransformer(name, device="cuda" if torch.cuda.is_available() else "cpu")

def build_index(segments, embedder):
    emb = embedder.encode(segments, convert_to_numpy=True)
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index, emb

# =========================
# 4. 影像與語音輔助功能
# =========================
def _encode_b64(img, fmt="JPEG"):
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def nvidia_image_to_text(client, image_path, lang_code):
    pil = Image.open(image_path)
    b64 = _encode_b64(pil)
    ask = {"zh-Hant": "用繁體中文描述這張圖片", "en": "Describe image", "vi": "Mô tả ảnh"}[lang_code]
    payload = {
        "model": NIM_MODEL_MAIN,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": ask},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]}]
    }
    headers = {"Authorization": f"Bearer {NIM_API_KEY}", "Content-Type": "application/json"}
    resp = requests.post(NIM_INVOKE_URL, headers=headers, json=payload)
    return resp.json()["choices"][0]["message"]["content"].strip()

# =========================
# 5. Agent 總機邏輯
# =========================
def get_agent_tools():
    return [{
        "type": "function",
        "function": {
            "name": "query_org_database",
            "description": "查詢機構專屬資料庫。msm 負責青少年/就業；tyad 負責輔具/補助。",
            "parameters": {
                "type": "object",
                "properties": {
                    "org_key": {"type": "string", "enum": ["msm", "tyad"]},
                    "query": {"type": "string", "description": "優化後的檢索詞"}
                },
                "required": ["org_key", "query"]
            }
        }
    }]

def build_corpora_and_clients():
    embedder = make_embedder(MODEL_NAME)
    corpora = {}
    for key, meta in ORG_REGISTRY.items():
        segs, tags = scan_org_dir(meta["path"])
        if not segs: segs, tags = ["尚無文件"], ["none"]
        idx, _ = build_index(segs, embedder)
        corpora[key] = {"segments": segs, "tags": tags, "index": idx}
    return embedder, corpora, OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# 6. Gradio UI 與 整合流程
# =========================
def build_app():
    embedder, corpora, client = build_corpora_and_clients()

    def agent_process(user_text, history, lang_label):
        lang_code = LANG_CODE[lang_label]
        msgs = [{"role": "system", "content": "你是一個自動導覽總機。請判斷要使用工具查詢 msm 或 tyad，或直接回答。"}]
        for u, a in (history[-3:] if history else []):
            if isinstance(u, str): msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": user_text})

        res = client.chat.completions.create(model=OPENAI_MODEL, messages=msgs, tools=get_agent_tools())
        call = res.choices[0].message.tool_calls

        if call:
            args = json.loads(call[0].function.arguments)
            org_key = args["org_key"]
            q = args["query"]
            # 檢索
            pool = corpora[org_key]
            q_vec = embedder.encode([q]); faiss.normalize_L2(q_vec)
            D, I = pool["index"].search(q_vec, FIXED_TOP_K)
            hits = [pool["segments"][i] for i in I[0] if i != -1]
            refs = "\n".join([f"[{pool['tags'][i]}] {pool['segments'][i][:200]}" for i in I[0] if i != -1])
            # 生成
            ans = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": f"{ORG_REGISTRY[org_key]['system_prompt']}\n{lang_instruction(lang_code)}"},
                          {"role": "user", "content": f"問題：{q}\n參考資料：{hits}"}]
            ).choices[0].message.content
            final_ans = f"【{ORG_REGISTRY[org_key]['label']}】\n{ans}"
        else:
            final_ans = res.choices[0].message.content; refs = ""

        history = list(history or []); history.append((user_text, final_ans))
        return history, refs, history

    def voice_agent_once(audio, history, lang):
        if not audio: return history, "", history
        with open(audio, "rb") as f:
            transcript = client.audio.transcriptions.create(model="whisper-1", file=f).text
        return agent_process(transcript, history, lang)

    def image_once(img_path, history, lang):
        if not img_path: return history, "", history
        caption = nvidia_image_to_text(client, img_path, LANG_CODE[lang])
        history = list(history or []); history.append(( (img_path,), caption ))
        return history, "", history

    css = ".chatbox { height: 500px; }"
    with gr.Blocks(title="全能 AI 總機", css=css) as demo:
        gr.Markdown("## 🎧 全能 AI 總機 (RAG + 語音 + 影像)")
        lang = gr.Dropdown(choices=LANG_LABELS, value="繁體中文", label="回覆語言")
        chat = gr.Chatbot(elem_classes="chatbox")
        state = gr.State([])
        with gr.Row():
            txt = gr.Textbox(label="文字輸入", placeholder="請問如何申請補助？")
            mic = gr.Microphone(label="語音輸入", type="filepath")
            img = gr.Image(label="影像輸入", type="filepath")
        btn = gr.Button("發送文字問題")
        ref_box = gr.Textbox(label="參考來源", lines=5)

        btn.click(agent_process, [txt, state, lang], [chat, ref_box, state])
        txt.submit(agent_process, [txt, state, lang], [chat, ref_box, state])
        mic.change(voice_agent_once, [mic, state, lang], [chat, ref_box, state])
        img.change(image_once, [img, state, lang], [chat, ref_box, state])

    return demo

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7861, share=True)