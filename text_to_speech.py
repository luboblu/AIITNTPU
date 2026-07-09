import os
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# =================設定區=================
# 我們使用社群量化的 GGUF 版本 (Q6_K 品質較好)
REPO_ID = "YC-Chen/Breeze-7B-Instruct-v1_0-GGUF"
FILENAME = "breeze-7b-instruct-v1_0-q6_k.gguf"
MODEL_PATH = f"./{FILENAME}"

# 測試句子 (包含專有名詞與日常用語)
TEST_SENTENCES = [
    "大家好，我是桃園輔具中心的助理。",
    "請問輔具要怎麼租借？",
    "這裡有提供輪椅和氣墊床的服務。",
]
# =======================================

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print(f"找不到模型檔案，正在從 Hugging Face 下載 {FILENAME}...")
        print("這可能需要一點時間 (約 5~6 GB)...")
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=FILENAME,
                local_dir=".",
                local_dir_use_symlinks=False
            )
            print("下載完成！")
        except Exception as e:
            print(f"下載失敗: {e}")
            print("請手動下載或是檢查網路連線。")
            exit(1)
    else:
        print(f"偵測到模型檔案：{MODEL_PATH}")

def run_test():
    # 1. 確保模型存在
    download_model_if_needed()

    # 2. 初始化模型
    print("正在載入 Breeze-7B 模型 (GPU 加速開啟)...")
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=-1,      # -1 代表全部丟給顯卡跑，如果顯存不足請改為 20 或 30
            n_ctx=4096,           # 上下文長度
            verbose=False         # 關閉雜訊
        )
    except Exception as e:
        print(f"載入失敗: {e}")
        print("如果是顯存不足 (OOM)，請嘗試將 n_gpu_layers 設小一點 (例如 10)。")
        return

    # 3. 定義 Prompt (關鍵！教導模型如何翻譯)
    system_prompt = (
        "你是一位精通台灣客家語（四縣腔）的語言專家。"
        "你的任務是將使用者輸入的繁體中文句子，翻譯並轉換為「客家語羅馬拼音」。"
        "請嚴格遵守以下規則：\n"
        "1. 直接輸出拼音，不要輸出任何漢字。\n"
        "2. 不要解釋，不要囉嗦，只給拼音結果。\n"
        "3. 拼音系統使用通用拼音或白話字皆可，以準確發音為主。"
    )

    print("\n========== 開始測試客家語拼音轉換 ==========\n")

    for text in TEST_SENTENCES:
        print(f"【輸入中文】：{text}")
        
        # 建立對話格式
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"請轉成客語拼音：{text}"}
        ]

        # 執行推論
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.1,  # 溫度設低一點，讓結果穩定
            max_tokens=256
        )

        result = response["choices"][0]["message"]["content"].strip()
        print(f"【Breeze 回答】：{result}")
        print("-" * 30)

if __name__ == "__main__":
    run_test()