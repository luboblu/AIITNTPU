# 多語言多模態 RAG 問答系統

## 📖 專案簡介

這是一個基於檢索增強生成（RAG）技術的多語言、多模態智能問答系統，專為社福機構和公共服務組織設計。系統支援文字、語音、圖片等多種輸入方式，能夠自動檢索相關文件並生成準確的回答。

### 🎯 主要特色

- **🌐 多語言支援**：繁體中文、英文、越南文
- **🎤 多模態輸入**：文字、語音、圖片
- **📚 智能檢索**：混合檢索策略（向量檢索 + BM25 + 多查詢重寫）
- **🏢 多組織管理**：可同時服務多個機構，各自獨立的知識庫
- **💾 本地部署**：使用本地 LLM 模型（GGUF 格式），保護數據隱私
- **🧠 上下文記憶**：維持對話連貫性
- **🔍 來源追蹤**：提供答案的參考文件來源

## 🏗️ 系統架構

```
┌─────────────┐
│   使用者     │
│ (文字/語音/圖片)│
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Gradio 介面    │
└──────┬──────────┘
       │
       ▼
┌─────────────────────────────────┐
│         RAG 核心引擎             │
│  ┌──────────┐  ┌──────────┐    │
│  │ 小聊偵測  │  │ 守門機制  │    │
│  └──────────┘  └──────────┘    │
│  ┌──────────────────────────┐  │
│  │   混合檢索系統             │  │
│  │  - FAISS 向量檢索         │  │
│  │  - BM25 關鍵字檢索        │  │
│  │  - 多查詢重寫             │  │
│  └──────────────────────────┘  │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────┐
│  本地 LLM 模型   │
│ (llama.cpp GGUF)│
└─────────────────┘
```

## 🛠️ 技術棧

### 核心技術
- **LLM 框架**：llama-cpp-python（支援 GGUF 模型）
- **向量檢索**：FAISS + SentenceTransformers
- **文字檢索**：BM25Okapi
- **語音識別**：Omnilingual ASR
- **圖片理解**：NVIDIA NIM API
- **介面框架**：Gradio

### 主要依賴套件
```
gradio>=4.0.0
faiss-cpu
sentence-transformers
llama-cpp-python
rank-bm25
python-docx
PyMuPDF
Pillow
python-dotenv
requests
```

## 📋 環境需求

- **Python**：3.9+
- **記憶體**：建議 16GB 以上（視模型大小而定）
- **GPU**：可選（支援 GPU 加速）
- **磁碟空間**：20GB 以上（含模型檔案）

## 🚀 安裝步驟

### 1. 克隆專案
```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. 建立虛擬環境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安裝依賴套件
```bash
pip install -r requirements.txt
```

### 4. 下載並放置模型檔案

#### LLM 模型（GGUF 格式）
- 下載 GGUF 格式的 LLM 模型（例如：`gpt-oss-20b-Q6_K.gguf`）
- 放置於專案根目錄
- 修改程式碼中的 `MODEL_PATH` 變數指向你的模型檔案

#### Embedding 模型
系統會自動下載 `paraphrase-multilingual-MiniLM-L12-v2`，無需手動處理。

### 5. 準備知識庫文件

建立以下資料夾結構：
```
data/
├── msm/          # 勵友協會文件
│   ├── faq.txt
│   ├── manual.docx
│   └── guide.pdf
└── tyad/         # 桃園輔具中心文件
    ├── faq.txt
    └── services.pdf
```

支援的文件格式：
- `.txt`：純文字檔案
- `.docx`：Word 文件
- `.pdf`：PDF 文件

### 6. 設定環境變數

我們提供了 `.env.example` 範本檔案，請依照以下步驟設定：

```bash
# 複製範本檔案
cp .env.example .env

# 編輯 .env 檔案，填入您的 API Key
nano .env  # 或使用其他編輯器
```

`.env` 檔案內容參考：
```bash
# NVIDIA NIM API（圖片辨識功能）
NV_NIM_INVOKE_URL=https://integrate.api.nvidia.com/v1/chat/completions
NV_NIM_API_KEY=<請填入您的 NVIDIA API Key>
NV_NIM_MODEL_MAIN=google/gemma-3-27b-it
NV_NIM_MODEL_FALL=meta/llama-3.2-11b-vision-instruct

# OpenAI API（可選）
# OPENAI_API_KEY=<您的 OpenAI API Key>
```

**如何取得 API Key：**
- **NVIDIA NIM API**: https://build.nvidia.com/
- **OpenAI API**: https://platform.openai.com/api-keys

⚠️ **安全提醒**：
- `.env` 檔案已包含在 `.gitignore` 中，請勿手動移除
- 不要在公開場合分享您的 API Key
- 定期更換 API Key 以確保安全
- 只將 `.env.example` 上傳到版本控制，`.env` 請保留在本地

## 🎮 使用方法

### 啟動系統
```bash
python local_rag_gradio_multi_qa_patch.py
```

系統啟動後會顯示：
```
Running on local URL:  http://0.0.0.0:7861
Running on public URL: https://xxxxx.gradio.live
```

### 使用介面

1. **選擇組織**：從下拉選單選擇服務組織
2. **選擇語言**：選擇回覆語言（繁體中文/English/Vietnamese）
3. **輸入問題**：
   - **文字輸入**：直接在文字框輸入問題
   - **語音輸入**：點擊麥克風圖示錄音
   - **圖片輸入**：上傳圖片進行辨識
4. **查看回答**：系統會在對話框顯示答案，並在下方顯示參考來源

### 操作範例

#### 文字問答
```
Q: 勵友中心的聯絡電話是多少？
A: 勵友中心的聯絡電話是 02-1234-5678...
```

#### 語音問答
1. 點擊麥克風圖示
2. 錄製語音問題
3. 系統自動轉錄並回答

#### 圖片問答
1. 上傳圖片
2. 系統自動描述圖片內容（根據選擇的語言）

## ⚙️ 系統設定

### 核心參數調整

在程式碼中可調整的關鍵參數：

```python
# 檢索參數
FIXED_TOP_K = 4          # 每次檢索返回的段落數
MARGIN = 0.05            # 守門器相似度門檻差距
MIN_SIM = 0.20           # 最低相似度要求

# 記憶參數
MEMORY_TURNS = 5         # 記憶回合數
MEMORY_CHARS_LIMIT = 1200  # 記憶字數上限

# LLM 參數
n_gpu_layers = -1        # GPU 層數（-1 = 全部使用 GPU）
n_ctx = 32768           # 上下文長度
temperature = 0.7        # 生成溫度
```

### 添加新組織

在 `ORG_REGISTRY` 字典中添加新組織：

```python
ORG_REGISTRY = {
    "new_org": {
        "label": "新組織名稱",
        "path": DATA_ROOT / "new_org",
        "system_prompt": """你的組織專屬 prompt..."""
    }
}
```

## 📁 檔案結構

```
.
├── local_rag_gradio_multi_qa_patch.py  # 主程式
├── .env                                 # 環境變數（⚠️ 請勿上傳，本地使用）
├── .env.example                         # 環境變數範本（可上傳）
├── .gitignore                           # Git 忽略清單
├── requirements.txt                     # Python 依賴
├── gpt-oss-20b-Q6_K.gguf               # LLM 模型（需自行下載）
├── data/                                # 知識庫資料夾
│   ├── msm/                            # 勵友協會文件
│   └── tyad/                           # 桃園輔具中心文件
└── README.md                            # 本文件
```

### 建議的 .gitignore 設定

我們已提供 `.gitignore` 檔案以保護敏感資訊，包含以下內容：
```
# 環境變數（包含 API Keys）
.env
.env.local

# 模型檔案（通常很大）
*.gguf
*.bin
*.safetensors

# 快取和臨時檔案
__pycache__/
*.pyc
.cache/
*.log

# 資料檔案（可能包含敏感資訊）
data/
uploads/
temp/
```

## 🔧 進階功能

### 檢索策略

系統提供 4 種檢索模式（relax 參數）：

0. **基礎語義檢索**：純向量相似度搜尋
1. **寬鬆語義檢索**：增加鄰近段落和候選數量
2. **混合檢索**：結合 BM25 和向量檢索（60% 語義 + 40% 關鍵字）
3. **多查詢重寫**：生成多個查詢變體並合併結果

系統會自動從模式 0 開始，若無結果則逐步升級。

### 守門機制

防止跨組織回答：
- 計算問題與各組織知識庫的相似度
- 若目標組織相似度明顯低於其他組織，則拒絕回答
- 可調整 `MARGIN` 和 `MIN_SIM` 參數控制嚴格程度

### 小聊偵測

自動識別一般聊天內容：
- 長度判斷（< 12 字元）
- 內容分析（是否包含問句標記）
- LLM 分類判斷

識別為小聊後，會給出簡短自然的回應，不進行知識庫檢索。

## ⚠️ 注意事項

### 安全性
- **絕對不要**將 `.env` 檔案上傳到 Git 或公開平台
- **絕對不要**在程式碼中硬編碼 API Key
- 使用環境變數管理所有敏感資訊
- 定期檢查並更新 `.gitignore` 確保敏感檔案不被追蹤
- 若不慎洩漏 API Key，請立即撤銷並重新生成

### 模型選擇
- 建議使用量化模型（Q4_K_M 或 Q6_K）平衡性能與品質
- GPU 記憶體不足時可調整 `n_gpu_layers` 參數

### API 配置
- **圖片功能**需要 NVIDIA NIM API Key
- **語音功能**已改用本地 Omnilingual ASR，不需 OpenAI API

### 效能優化
- 首次啟動會建立向量索引，需要一些時間
- 大量文件時建議增加系統記憶體
- 可啟用 GPU 加速向量檢索

### 資料隱私
- 本系統使用本地模型，敏感資料不會上傳外部服務
- 圖片辨識功能使用 NVIDIA API，請注意隱私政策

## 🐛 常見問題

### Q1: 模型載入失敗
**A**: 確認模型檔案路徑正確，且有足夠記憶體。可嘗試使用較小的量化模型。

### Q2: GPU 記憶體不足
**A**: 調整 `n_gpu_layers` 參數，減少使用 GPU 的層數，或使用 CPU 模式（設為 0）。

### Q3: 回答語言不正確
**A**: 檢查 `lang_instruction` 函數是否正確對應語言代碼，並確保 prompt 中強調語言要求。

### Q4: 檢索結果不準確
**A**: 嘗試調整檢索參數（TOP_K、MARGIN）或升級檢索策略（relax 參數）。

### Q5: 語音辨識失敗
**A**: 確認音訊格式正確（WAV），且 Omnilingual ASR 模型已正確載入。

## 📊 效能指標

- **冷啟動時間**：約 30-60 秒（視模型大小）
- **單次問答**：2-5 秒（視檢索複雜度和生成長度）
- **記憶體使用**：8-16 GB（視模型大小）
- **並發支援**：單實例（可使用 Gradio 的 queue 功能）

## 🔄 更新日誌

### v1.0.0 (Current)
- ✅ 支援多語言（中/英/越）
- ✅ 多模態輸入（文字/語音/圖片）
- ✅ 混合檢索策略
- ✅ 本地 GGUF 模型支援
- ✅ Omnilingual ASR 語音識別
- ✅ NVIDIA NIM 圖片理解

## 📝 授權

[請根據實際情況填寫授權資訊]

## 👥 貢獻者

- 盧信廷 - 主要開發者

## 📧 聯絡方式

如有問題或建議，請聯繫：[your-email@example.com]

---

**注意**：本系統僅供研究和教育用途，實際部署前請確保符合相關法規和隱私政策。
