# 新版嘗試_切分文檔_20241210
# 可以查詢針對某協會的內容

import os
import openai
import numpy as np
import faiss
from flask import Flask, request, abort
from linebot.models import MessageEvent, TextMessage, TextSendMessage, AudioMessage
from linebot import LineBotApi, WebhookHandler  # 核心 LINE Bot SDK
from linebot.exceptions import LineBotApiError, InvalidSignatureError  # LINE SDK 錯誤處理
from sentence_transformers import SentenceTransformer
import warnings
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# 禁用並行處理的警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 忽略將來不再使用的警告
warnings.filterwarnings("ignore", category=FutureWarning)

# 設定 LINE Bot 與 OpenAI 金鑰
line_bot_api = LineBotApi('...')
handler = WebhookHandler('...')
openai.api_key = '...'

# 初始化 Flask 應用
app = Flask(__name__)

# Step 1: 加載並切分文本
def load_and_partition_text(file_path, chunk_size=300, chunk_overlap=50):
    """讀取文本並分割成分區和段落"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    sections = content.split("####")  # 按分區標題分割
    partitioned_segments = {}

    for section in sections:
        if section.strip():  # 過濾空分區
            lines = section.strip().split("\n")
            header = lines[0].strip()  # 第一行作為標題
            body = "\n".join(lines[1:]).strip()  # 剩餘部分作為內容

            # 處理多行文本並進行分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n", ".", "。"]
            )
            documents = [Document(page_content=body)]
            docs_split = text_splitter.split_documents(documents)
            segments = [f"{header}\n{doc.page_content}" for doc in docs_split]

            partitioned_segments[header] = segments

    return partitioned_segments


# Step 2: 初始化模型與索引
def initialize_rag(file_path, chunk_size=300, chunk_overlap=50):
    """初始化 RAG 系統，建立索引與嵌入模型"""
    partitions = load_and_partition_text(file_path, chunk_size, chunk_overlap)
    model = SentenceTransformer('sentence-transformers/LaBSE', device="cuda" if torch.cuda.is_available() else "cpu")

    indexes = {}
    partition_segments = {}

    for header, segments in partitions.items():
        partition_segments[header] = segments

        embeddings = model.encode(segments, batch_size=8, show_progress_bar=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        indexes[header] = index
        print(f"分區 '{header}' 索引已建立，共 {index.ntotal} 條記錄。")

    return model, indexes, partition_segments

# Step 3: 改進的檢索系統
def query_rag_system(query, model, indexes, partition_segments):
    """檢索系統，從特定分區或全局中查詢"""
    results = []
    query_prefix = query[:5]  # 提取查詢的前五個字

    # 根據查詢前綴判斷目標分區
    if any(prefix in query_prefix for prefix in ["中華民國腦", "腦麻", "腦麻協會"]):
        target_section = "中華民國腦性麻痺協會"
    elif any(prefix in query_prefix for prefix in ["漸凍人協會", "漸凍人"]):
        target_section = "漸凍人協會"
    elif any(prefix in query_prefix for prefix in ["陽光基金會"]):
        target_section = "陽光基金會"
    elif any(prefix in query_prefix for prefix in ["AIITN"]):
        target_section = "AIITNTPU計畫內容介紹"
    else:
        target_section = None  # 全局檢索

    print(f"查詢目標分區: {target_section}")
    print(f"可用分區標題: {list(indexes.keys())}")

    # 檢查 target_section 是否匹配到分區
    if target_section and target_section not in indexes:
        # 嘗試模糊匹配找到最接近的分區
        from difflib import get_close_matches
        possible_matches = get_close_matches(target_section, indexes.keys(), n=1, cutoff=0.8)
        if possible_matches:
            target_section = possible_matches[0]
            print(f"模糊匹配找到的分區: {target_section}")
        else:
            print(f"未找到匹配的分區: {target_section}，執行全局檢索...")
            target_section = None  # 切換為全局檢索

    # 如果有目標分區，則僅檢索該分區
    if target_section and target_section in indexes:
        print(f"檢索目標分區: {target_section}")
        query_embedding = model.encode([query], batch_size=1, show_progress_bar=False)
        distances, indices = indexes[target_section].search(np.array(query_embedding), k=5)

        # 提取相關段落
        for i, dist in zip(indices[0], distances[0]):
            if i != -1 and dist < 1.5:  # 相似度閾值
                results.append(partition_segments[target_section][i])
    else:
        # 全局檢索，檢索所有分區
        print("執行全局檢索...")
        for header, index in indexes.items():
            query_embedding = model.encode([query], batch_size=1, show_progress_bar=False)
            distances, indices = index.search(np.array(query_embedding), k=3)

            # 提取相關段落
            for i, dist in zip(indices[0], distances[0]):
                if i != -1 and dist < 1.5:
                    results.append(partition_segments[header][i])

    # 如果檢索結果為空，返回提示
    if not results:
        results = ["很抱歉，我無法找到與您的問題相關的內容。"]

    print(f"檢索結果: {results}")
    return results


# Step 4: 生成回應
def generate_answer(query, retrieved_segments):
    """根據檢索結果生成回答"""
    context = "\n".join(retrieved_segments)
    if len(context) > 2000:
        context = context[:2000]  # 限制上下文長度

    input_text = f"問題: {query}\n上下文: {context}"
    try:
        # 使用新的 OpenAI Chat API 調用方式
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是AIITNTPU計畫的客服助理，需要回答有關中華民國腦性麻痺協會，漸凍人協會，陽光基金會，以及AIITNTPU包容科技計畫的內容，請根據提供的文本回答問題。回答盡量在200字以內"},
                {"role": "user", "content": input_text}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"生成回應時出錯: {str(e)}"


# 加載文本並初始化 RAG 系統
file_path = r"C:\Users\imntpu\AIIT\AIITNTPU\AIITNTPU.txt"
rag_model, rag_indexes, rag_segments = initialize_rag(file_path, chunk_size=300, chunk_overlap=50)

# Step 5: LINE Bot 處理文字訊息
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    try:
        # 確保是有效的 reply_token
        if not event.reply_token or event.reply_token == "00000000000000000000000000000000":
            print(f"收到無效的 reply_token: {event.reply_token}，跳過回應。")
            return

        # 檢查是否為 LINE 自動回覆的關鍵字
        keyword_responses = ['計畫簡介', '計畫團隊', '相關網站', '招募公告', '我要報名', '聯絡資訊']  # LINE 自動回覆設定的關鍵字

        # 如果包含自動回覆的關鍵字，直接跳過 OpenAI 回覆
        if any(keyword in event.message.text for keyword in keyword_responses):
            print(f"自動回覆關鍵字檢測到: {event.message.text}，跳過 OpenAI 回覆處理。")
            return  # 直接返回，讓 LINE 自動回覆處理

        # 若非關鍵字則進行 RAG 檢索與回答
        user_message = event.message.text
        print(f"收到的用戶訊息: {user_message}")

        # 使用 RAG 系統檢索相關段落
        retrieved_segments = query_rag_system(user_message, rag_model, rag_indexes, rag_segments)
        print(f"檢索到的相關段落: {retrieved_segments}")

        # 生成回應
        response_message = generate_answer(user_message, retrieved_segments)
        print(f"生成的回應: {response_message}")

        # 回應用戶訊息
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=response_message)
        )
    except LineBotApiError as e:
        print(f"LineBotApiError: {e}")
    except Exception as e:
        print(f"處理訊息時發生錯誤: {e}")


# Step 6: LINE Bot 處理音訊訊息
@handler.add(MessageEvent, message=AudioMessage)
def handle_audio_message(event):
    """處理音訊訊息"""
    message_content = line_bot_api.get_message_content(event.message.id)
    audio_path = './temp_audio.m4a'

    with open(audio_path, 'wb') as audio_file:
        for chunk in message_content.iter_content():
            audio_file.write(chunk)

    try:
        with open(audio_path, "rb") as audio_file:
            response = openai.Audio.transcribe(model="whisper-1", file=audio_file)
        transcribed_text = response['text']
    except Exception as e:
        transcribed_text = f"音訊轉錄出錯: {str(e)}"

    if transcribed_text:
        retrieved_segments = query_rag_system(transcribed_text, rag_model, rag_indexes, rag_segments)
        response_message = generate_answer(transcribed_text, retrieved_segments)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=response_message))

# 啟動 Flask 應用
@app.route("/callback", methods=['POST'])
def callback():
    try:
        signature = request.headers['X-Line-Signature']
        body = request.get_data(as_text=True)

        # 檢查 Webhook 事件
        print(f"收到 Webhook 請求: {body}")
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.error("簽名驗證失敗")
        return 'Invalid signature', 400
    except Exception as e:
        app.logger.error(f"處理請求時發生錯誤: {e}")
        return 'Internal Server Error', 500
    return 'OK', 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

