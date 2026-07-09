import os
import re
import faiss
import fitz
import torch
import gradio as gr
import numpy as np
from pathlib import Path
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from docx import Document
from dotenv import load_dotenv

# ==== 影像處理與HTTP ====
from io import BytesIO
from PIL import Image
import base64
import requests
import json
import mimetypes
import tempfile

# ==== 文字檢索強化 ====
from rank_bm25 import BM25Okapi  # pip install rank_bm25

# =========================
# 基本設定
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

# ==== NVIDIA Integrate chat/completions 設定 ====
NIM_INVOKE_URL = os.getenv("NV_NIM_INVOKE_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
NIM_API_KEY = os.getenv("NV_NIM_API_KEY")
NIM_MODEL_MAIN = os.getenv("NV_NIM_MODEL_MAIN", "google/gemma-3-27b-it")
NIM_MODEL_FALL = os.getenv("NV_NIM_MODEL_FALL", "meta/llama-3.2-11b-vision-instruct")

# Base64 限制
MAX_B64_SIZE = 3_500_000  # ~3.5MB
MIN_EDGE_LIMIT = 640

# =========================
# 組織資料
# =========================
ORG_REGISTRY = {
    "msm": {
        "label": "勵友協會（就業輔導）",
        "path": DATA_ROOT / "msm",
        "system_prompt": """你是基督教勵友中心的溫暖客服助理，
請依據提供的資料回答問題，語氣親切、鼓勵且簡潔（120字內）。"""
    },
    "tyad": {
        "label": "桃園輔具中心（輔具/補助）",
        "path": DATA_ROOT / "tyad",
        "system_prompt": """你是桃園市北區輔具資源中心的客服助理，
請根據資料內容回覆使用者問題，語氣溫和、清楚，條列說明（150字內）。"""
    }
}

# =========================
# 語言（以下拉式選單為準）
# =========================
LANG_LABELS = ["繁體中文", "English", "Vietnamese"]
LANG_CODE = {
    "繁體中文": "zh-Hant",
    "English": "en",
    "Vietnamese": "vi",
}
def lang_instruction(code: str) -> str:
    return {
        "zh-Hant": "請全程使用繁體中文作答。",
        "en": "Please answer entirely in English.",
        "vi": "Vui lòng trả lời hoàn toàn bằng tiếng Việt.",
    }[code]

# =========================
# 檔案載入
# =========================
_Q_PAT = re.compile(r"^(?:Q|問)\s*[:：．.]?\s*(.+)", flags=re.IGNORECASE)
_A_PAT = re.compile(r"^(?:A|答)\s*[:：．.]?\s*(.+)", flags=re.IGNORECASE)

def load_txt_segments(path: Path):
    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines = raw.splitlines()
    has_md = any(ln.strip().startswith(("##", "###")) for ln in lines)
    if not has_md:
        return [seg.strip() for seg in re.split(r"\n\s*\n", raw) if seg.strip()]
    segs, buf = [], []
    def flush():
        nonlocal buf
        if buf:
            txt = "\n".join(buf).strip()
            if txt:
                segs.append(txt)
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
        t = (p.text or "").strip()
        if not t:
            continue
        mq, ma = _Q_PAT.match(t), _A_PAT.match(t)
        if mq:
            if buf_q: segs.append("問題：" + buf_q)
            buf_q = mq.group(1).strip(); continue
        if ma and buf_q:
            ans = ma.group(1).strip()
            segs.append(f"問題：{buf_q}\n答案：{ans}")
            buf_q = None; continue
        segs.append(t)
    if buf_q: segs.append("問題：" + buf_q)
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if len(cells) == 2 and ("問" in cells[0] or "Q" in cells[0]):
                segs.append("問題：" + cells[0]); segs.append("答案：" + cells[1])
            elif cells:
                segs.append(" | ".join(cells))
    return segs

def load_pdf_segments(path: Path):
    segs = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            text = page.get_text("text")
            if not text.strip(): continue
            lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.split("\n")]
            buff = ""
            for ln in lines:
                if ln: buff += ln + "\n"
                if len(buff) > 900:
                    segs.append(buff.strip()); buff = ""
            if buff.strip(): segs.append(buff.strip())
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
# 向量化 + 檢索資源
# =========================
def make_embedder(name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return SentenceTransformer(name, device=device)

def build_index(segments, embedder):
    emb = embedder.encode(segments, convert_to_numpy=True, show_progress_bar=True, batch_size=128)
    faiss.normalize_L2(emb)
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)
    return index, emb

# =========================
# OpenAI / Whisper
# =========================
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API")
    if not api_key:
        raise RuntimeError("請設定 OPENAI_API_KEY 或 OPENAI_API 環境變數")
    return OpenAI(api_key=api_key)

# =========================
# 小聊偵測與回覆（遵守選單語言）
# =========================
def is_small_talk(client, text: str) -> bool:
    t = (text or "").strip()
    if any(x in t for x in ["？", "?", "怎麼", "如何", "為什麼", "怎樣"]) or len(t) >= 12:
        return False
    try:
        prompt = ("判斷下列訊息是否屬於一般聊天/寒暄/情緒支持/非知識型提問。"
                  "若是，輸出『1』；否則輸出『0』。只輸出數字。")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt},{"role": "user", "content": t}],
            temperature=0, max_tokens=2
        )
        return (resp.choices[0].message.content or "").strip() == "1"
    except Exception:
        return False

def generate_small_talk(client, org_key: str, text: str, lang_code: str):
    meta = ORG_REGISTRY.get(org_key, {})
    base_prompt = meta.get("system_prompt", "")
    sys_prompt = (
        base_prompt
        + "\n\n你現在面對的是一位使用者發出的『一般聊天或寒暄』訊息。"
        + "\n請根據該組織的語氣風格生成**簡短自然的回覆（50字以內）**，不要提及機構名稱，也不要引導對方發問。"
        + f"\n\n{lang_instruction(lang_code)}"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": sys_prompt},{"role": "user", "content": text.strip()}],
            temperature=0.7, max_tokens=80,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        fallback = {
            "zh-Hant": "謝謝你的分享，我在這裡陪你。",
            "en": "Thanks for sharing—I'm here for you.",
            "vi": "Cảm ơn bạn đã chia sẻ, mình luôn ở đây.",
        }
        return fallback[lang_code]

# =========================
# 記憶摘要與回答生成（遵守選單語言）
# =========================
def build_memory_summary(history):
    if not history: return ""
    recent = history[-MEMORY_TURNS:]
    parts = [f"【先前問題】{u}\n【先前回覆】{a}" for u, a in recent]
    memo = "\n\n".join(parts)
    return memo[-MEMORY_CHARS_LIMIT:] if len(memo) > MEMORY_CHARS_LIMIT else memo

def generate_answer(client, system_prompt, query, retrieved, memory_summary="", lang_code="zh-Hant"):
    ctx = " ".join(retrieved)[:4000]
    sys_prompt = (
        system_prompt
        + "\n\n你可以參考【短期記憶摘要】來維持上下文一致，但回覆必須以同組織的『參考段落』為主。"
        + "\n若參考內容不足以直接回答：允許給出『一般性背景說明或建議』，"
          "但請先用一句話溫和提示『以下為通用背景資訊，非出自本機構文件』，之後再給建議。"
        + f"\n\n{lang_instruction(lang_code)}"
    )
    user_content = f"【短期記憶摘要】\n{memory_summary}\n\n【目前問題】{query}\n\n【參考段落】\n{ctx}"
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}],
        temperature=0.2, max_tokens=450,
    )
    return resp.choices[0].message.content.strip()

# =========================
# 守門器（自適應）
# =========================
def _adaptive_thresholds(q_text: str):
    L = len(q_text or "")
    margin = MARGIN - (0.01 if L >= 15 else 0.0)
    min_sim = MIN_SIM - (0.03 if L >= 30 else (0.01 if L >= 15 else 0.0))
    return max(margin, 0.02), max(min_sim, 0.12)

# =========================
# 鄰近文塊拼接
# =========================
def neighbor_answer_augmentation(hits, segments, tags, window_size=2, top_k=3):
    results, seen = [], set()
    for idx, tag, seg in hits:
        if idx in seen: continue
        results.append((tag, seg)); seen.add(idx)
        for off in range(-window_size, window_size + 1):
            if off == 0: continue
            j = idx + off
            if 0 <= j < len(segments) and tags[j] == tag and j not in seen:
                results.append((tags[j], segments[j])); seen.add(j)
        if len(results) >= top_k: break
    return results[:top_k]

# =========================
# 基本／混合檢索
# =========================
def search_semantic(pool, question, k, embedder, window_size=2, widen=2):
    segs, tags, index = pool["segments"], pool["tags"], pool["index"]
    q_vec = embedder.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    broad_k = max(8, k * widen)
    D, I = index.search(q_vec, broad_k)
    raw_hits = [(int(ix), tags[int(ix)], segs[int(ix)]) for ix in I[0]]
    return neighbor_answer_augmentation(raw_hits, segs, tags, window_size=window_size, top_k=k)

def build_bm25_corpus(segments):
    tokenized = [re.findall(r"\w+", s.lower()) for s in segments]
    return BM25Okapi(tokenized), tokenized

def _query_rewrites(client, q: str, topn: int = 3, lang_code: str = "zh-Hant") -> list[str]:
    prompt_map = {
        "zh-Hant": f"請產生{topn}個中文查詢改寫或同義展開，每行一個，簡短，專有名詞保留：\n{q}",
        "en": f"Generate {topn} query rewrites or synonym expansions in English, one per line, concise, preserve proper nouns:\n{q}",
        "vi": f"Hãy tạo {topn} cách viết lại hoặc mở rộng truy vấn bằng tiếng Việt, mỗi dòng một câu, ngắn gọn, giữ nguyên danh từ riêng:\n{q}",
    }
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_map[lang_code]}],
            temperature=0.3, max_tokens=120
        )
        lines = [x.strip() for x in (resp.choices[0].message.content or "").splitlines()]
        out = [x for x in lines if x]
        return out[:topn] or [q]
    except Exception:
        return [q]

def hybrid_search(org_key, question, k, embedder, corpora, client, relax=0, lang_code="zh-Hant"):
    """ relax=0,1,2,3: 0) 純語義 1) 放寬語義 2) 混合排序（BM25+語義） 3) 改寫+混合 """
    pool = corpora[org_key]
    segs, tags, index, emb = pool["segments"], pool["tags"], pool["index"], pool["emb"]

    if relax in (0, 1):
        return search_semantic(pool, question, k, embedder, window_size=(3 if relax>=1 else 2), widen=(6 if relax>=1 else 2))

    if relax == 2:
        if "bm25" not in pool:
            bm25, tokenized = build_bm25_corpus(segs)
            pool["bm25"] = (bm25, tokenized)
        else:
            bm25, tokenized = pool["bm25"]
        q_tokens = re.findall(r"\w+", (question or "").lower())
        bm25_scores = bm25.get_scores(q_tokens)
        broad_k = max(24, k * 6)
        cand_ids = np.argsort(bm25_scores)[-broad_k:][::-1]
        q_emb = embedder.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        sem_scores = emb[cand_ids] @ q_emb.T
        sem_scores = sem_scores.reshape(-1)
        fused = 0.6 * sem_scores + 0.4 * bm25_scores[cand_ids]
        cand_sorted = cand_ids[np.argsort(fused)[::-1]]
        raw_hits = [(int(ix), tags[int(ix)], segs[int(ix)]) for ix in cand_sorted[:max(12, k*3)]]
        return neighbor_answer_augmentation(raw_hits, segs, tags, window_size=3, top_k=k)

    rewrites = _query_rewrites(client, question, topn=3, lang_code=lang_code)
    all_hits = []
    for rq in rewrites:
        q_enc = embedder.encode([rq], convert_to_numpy=True)
        faiss.normalize_L2(q_enc)
        D, I = index.search(q_enc, max(24, k*6))
        all_hits.extend([(int(ix), tags[int(ix)], segs[int(ix)]) for ix in I[0]])
    seen = set(); uniq = []
    for ix, tg, sg in all_hits:
        if ix in seen: continue
        uniq.append((ix, tg, sg)); seen.add(ix)
    return neighbor_answer_augmentation(uniq[:max(24, k*6)], segs, tags, window_size=3, top_k=k)

# =========================
# 建立語料庫
# =========================
def build_corpora():
    embedder = make_embedder(MODEL_NAME)
    corpora = {}
    for key, meta in ORG_REGISTRY.items():
        segs, tags = scan_org_dir(meta["path"])
        if not segs:
            segs = [f"{meta['label']}：尚未加入任何文件。"]
            tags = ["(no-data)"]
        index, emb = build_index(segs, embedder)
        centroid = emb.mean(axis=0, dtype=np.float32)
        centroid /= (np.linalg.norm(centroid) + 1e-12)
        corpora[key] = {"segments": segs, "tags": tags, "index": index, "emb": emb, "centroid": centroid}
    return embedder, corpora

# =========================
# Image→Text（遵守選單語言）
# =========================
def _encode_b64(img: Image.Image, fmt: str, quality: int = 90) -> str:
    buf = BytesIO()
    fmt_up = (fmt or "JPEG").upper()
    save_kwargs = {}
    if fmt_up in ("JPEG", "JPG", "WEBP"):
        if fmt_up == "JPEG":
            img = img.convert("RGB")
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
    img.save(buf, format=fmt_up, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode()

def _nim_chat(payload: dict) -> str:
    if not NIM_API_KEY:
        raise RuntimeError("未設定 NV_NIM_API_KEY")
    headers = {
        "Authorization": f"Bearer {NIM_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    resp = requests.post(NIM_INVOKE_URL, headers=headers, json=payload, timeout=90)
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        raise RuntimeError(f"{resp.status_code} {resp.reason}: {resp.text}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        raise RuntimeError(f"回應格式非預期：{data}")

def nvidia_image_to_text(image_input, lang_code: str = "zh-Hant") -> tuple[str, Image.Image]:
    if isinstance(image_input, Image.Image):
        pil = image_input.copy()
    else:
        pil = Image.open(image_input)

    target_fmt = "JPEG"
    target_mime = "image/jpeg"
    quality = 90
    scale = 1.0
    b64 = _encode_b64(pil, target_fmt, quality)

    tries = 0
    while len(b64) > MAX_B64_SIZE and tries < 12:
        tries += 1
        if quality > 50: quality -= 10
        else: scale *= 0.75
        new_w = max(MIN_EDGE_LIMIT, int(pil.width * scale))
        new_h = max(MIN_EDGE_LIMIT, int(pil.height * scale))
        if new_w == pil.width and new_h == pil.height: break
        pil = pil.resize((new_w, new_h), Image.LANCZOS)
        b64 = _encode_b64(pil, target_fmt, quality)

    if len(b64) > MAX_B64_SIZE:
        raise RuntimeError("影像仍過大，請選較小檔案。")

    ask = {
        "zh-Hant": "用繁體中文描述這張圖片",
        "en": "Describe this image in English.",
        "vi": "Mô tả bức ảnh này bằng tiếng Việt.",
    }[lang_code]

    payload = {
        "model": NIM_MODEL_MAIN,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": ask},
                {"type": "image_url", "image_url": {"url": f"data:{target_mime};base64,{b64}"}}
            ]
        }],
        "max_tokens": 512,
        "temperature": 0.2,
        "top_p": 0.7,
    }
    try:
        caption = _nim_chat(payload)
        return caption, pil
    except Exception as e1:
        payload["model"] = NIM_MODEL_FALL
        try:
            caption = _nim_chat(payload)
            return caption, pil
        except Exception as e2:
            raise RuntimeError(f"Image caption 連續失敗。\n[Gemma]: {e1}\n[Llama vision]: {e2}")

def _save_temp_image(pil_img: Image.Image, suffix: str = ".jpg") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    pil_img.save(path, format="JPEG", quality=90, optimize=True)
    return path

# =========================
# 主流程
# =========================
def build_corpora_and_clients():
    embedder, corpora = build_corpora()
    client = get_openai_client()
    return embedder, corpora, client

def build_app():
    embedder, corpora, client = build_corpora_and_clients()
    org_labels = [v["label"] for v in ORG_REGISTRY.values()]
    key_by_label = {v["label"]: k for k, v in ORG_REGISTRY.items()}

    def rag_core(org_label, question, history, lang_code: str):
        org_key = key_by_label[org_label]

        # 1) 小聊（遵守選單語言）
        if is_small_talk(client, question):
            ans = generate_small_talk(client, org_key, question, lang_code)
            history = list(history or []); history.append((question, ans))
            return ans, "", history

        # 2) 守門器
        q_vec = embedder.encode([question], convert_to_numpy=True); faiss.normalize_L2(q_vec)
        qv = q_vec[0]
        sims = {k: float(np.dot(qv, corpora[k]["centroid"])) for k in corpora}
        chosen, best = sims[org_key], max(sims.values())
        margin, min_sim = _adaptive_thresholds(question)
        gate_block = ((best - chosen) >= margin) or (best < min_sim)

        # 3) 檢索：基本 → 放寬 → 混合 → 改寫
        hits = hybrid_search(org_key, question, FIXED_TOP_K, embedder, corpora, client, relax=0, lang_code=lang_code)
        if not hits:
            for relax in (1, 2, 3):
                hits = hybrid_search(org_key, question, FIXED_TOP_K, embedder, corpora, client, relax=relax, lang_code=lang_code)
                if hits: break

        # 4) 有命中 → 生成（遵守選單語言）
        if hits:
            retrieved = [h[1] for h in hits]
            memo = build_memory_summary(history or [])
            ans = generate_answer(client, ORG_REGISTRY[org_key]["system_prompt"], question, retrieved, memo, lang_code=lang_code)
            refs = "\n\n".join([f"[{i+1}] {h[0]}: {h[1][:600]}" for i, h in enumerate(hits)])
            history = list(history or []); history.append((question, ans))
            return ans, refs, history

        # 5) 無命中
        if gate_block:
            sorry = {
                "zh-Hant": "不好意思，目前沒有找到相關資料",
                "en": "Sorry, I couldn't find relevant materials.",
                "vi": "Xin lỗi, hiện chưa tìm thấy nội dung liên quan.",
            }[lang_code]
            ans = sorry
            history = list(history or []); history.append((question, ans))
            return ans, "", history
        else:
            sys_prompt = (
                ORG_REGISTRY[org_key]["system_prompt"]
                + "\n\n參考資料未命中。請以通用背景知識給出安全、務實的建議，"
                  "但先提示『以下為通用背景資訊，非出自本機構文件』，字數以 150 字內為宜。"
                + f"\n\n{lang_instruction(lang_code)}"
            )
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": question}],
                temperature=0.3, max_tokens=160,
            )
            ans = resp.choices[0].message.content.strip()
            history = list(history or []); history.append((question, ans))
            return ans, "", history

    # --- 文字 ---
    def query_once(org_label, user_text, history, lang_label):
        if not user_text or not user_text.strip():
            return history, "請輸入問題。", history
        lang_code = LANG_CODE[lang_label]
        answer, refs, history = rag_core(org_label, user_text.strip(), history, lang_code)
        chat_history = list(history or [])
        return chat_history, refs, history

    # --- 語音 ---
    def voice_once(org_label, audio_path, history, lang_label):
        if not audio_path:
            return history, "未偵測到音訊檔案。", history
        try:
            with open(audio_path, "rb") as af:
                asr = client.audio.transcriptions.create(model="whisper-1", file=af)
            user_text = (asr.text or "").strip()
        except Exception as e:
            user_text = f"(語音轉文字失敗：{e})"
        if not user_text: user_text = "(未辨識到內容)"
        return query_once(org_label, user_text, history, lang_label)

    # --- 圖片 ---
    def image_once(org_label, image_path, history, lang_label):
        if not image_path:
            return history, "", history
        try:
            lang_code = LANG_CODE[lang_label]
            caption, _ = nvidia_image_to_text(image_path, lang_code=lang_code)
            right_user_msg = ("使用者上傳的圖片", image_path)
            history = list(history or []); history.append((right_user_msg, caption))
            return history, "", history
        except Exception as e:
            lang_code = LANG_CODE[lang_label]
            err = {
                "zh-Hant": "（圖片描述失敗）",
                "en": "(Image caption failed)",
                "vi": "(Mô tả ảnh thất bại)"
            }[lang_code]
            right_user_msg = ("使用者上傳的圖片", image_path)
            history = list(history or []); history.append((right_user_msg, f"{err} {e}"))
            return history, "", history

    # --- UI ---
    css = """
    #chatbox { resize: both; overflow: auto; min-height: 360px; }
    #chatbox img, #chatbox [data-testid="file"] img, #chatbox .file-preview img, #chatbox .message img {
        max-height: none !important; width: 100% !important; height: auto !important; object-fit: contain !important;
    }
    #chatbox .message, #chatbox .wrap, #chatbox .prose { max-width: 100% !important; }
    """

    with gr.Blocks(
        title="多組織在地 RAG + Whisper + GPT-4 + 影像轉文字（固定輸出語言）",
        css=css
    ) as demo:
        # gr.Markdown("### 🎧 多組織在地 RAG + Whisper + GPT-4 + 影像轉文字（固定輸出語言）")
        with gr.Row():
            org = gr.Dropdown(choices=org_labels, value=org_labels[0], label="選擇組織")
            lang_sel = gr.Dropdown(choices=LANG_LABELS, value="繁體中文", label="回覆語言（固定輸出）")
        chatbox = gr.Chatbot(label="對話", height=520, elem_id="chatbox")
        state = gr.State([])

        with gr.Row():
            question = gr.Textbox(label="輸入你的問題")
            mic = gr.Microphone(label="語音輸入（講完自動辨識）", type="filepath", format="wav")
            img = gr.Image(label="上傳圖片（右側顯示原圖，左側生成描述）", type="filepath")
        run = gr.Button("查詢")
        refs = gr.Textbox(label="參考段落（含來源檔名）", lines=10, visible=False)

        # 文字
        run.click(fn=query_once, inputs=[org, question, state, lang_sel], outputs=[chatbox, refs, state])
        question.submit(fn=query_once, inputs=[org, question, state, lang_sel], outputs=[chatbox, refs, state])

        # 語音
        mic.change(fn=voice_once, inputs=[org, mic, state, lang_sel], outputs=[chatbox, refs, state])

        # 圖片
        img.change(fn=image_once, inputs=[org, img, state, lang_sel], outputs=[chatbox, refs, state])

    return demo

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7861, share=True)
