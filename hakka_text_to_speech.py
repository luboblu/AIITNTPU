import torch
import gradio as gr
import numpy as np
from transformers import VitsModel, VitsTokenizer, set_seed

MODEL_ID = "facebook/mms-tts-hak"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) 載入模型與 tokenizer
tokenizer = VitsTokenizer.from_pretrained(MODEL_ID)
model = VitsModel.from_pretrained(MODEL_ID).to(device)
SR = model.config.sampling_rate  # 取樣率 (Hz)

# 驗證模型和 tokenizer 是否正常工作
print("正在驗證模型...")
try:
    # 改用客家語羅馬拼音測試
    test_text = "ngai ho"
    test_inputs = tokenizer(test_text, return_tensors="pt")
    print(f"測試文字: {test_text}")
    print(f"Tokenizer 輸出 shape: {test_inputs['input_ids'].shape}")
    print(f"Token IDs: {test_inputs['input_ids']}")
    print(f"Vocabulary size: {len(tokenizer.get_vocab())}")
    print(f"模型採樣率: {SR} Hz")
    
    if test_inputs['input_ids'].size(1) == 0:
        print("⚠️ 警告：測試文字沒有產生任何 tokens")
        print("這表示模型可能需要特定的羅馬拼音格式")
    else:
        print("✅ 模型驗證完成！")
except Exception as e:
    print(f"❌ 模型驗證失敗: {e}")
    print("請檢查模型是否正確載入")

print("\n" + "="*50)
print("客家語 TTS 系統啟動中...")
print("注意：此模型需要客家語羅馬拼音輸入")
print("="*50)

def tts_hakka(text, speaking_rate, noise_scale, noise_scale_duration, seed):
    """
    文字 -> 客家語音 (hak)
    注意：此模型需要羅馬拼音輸入，不支援中文字符
    - speaking_rate: 語速（數值越大 -> 越快）
    - noise_scale: 語音隨機度（音色/抖動）
    - noise_scale_duration: 時長隨機度（節奏）
    - seed: 隨機種子（固定重現用；-1 表示不固定）
    """
    if not text or text.strip() == "":
        return None

    # 清理和預處理文字
    text = text.strip().lower()  # 轉為小寫
    
    # 檢查是否包含中文字符（這會導致失敗）
    import re
    if re.search(r'[\u4e00-\u9fff]', text):
        print("警告：輸入包含中文字符，模型不支援中文輸入")
        print("請使用客家語羅馬拼音，例如：ngai ho, lia ho, do cia")
        return None
    
    # 只保留字母、數字、空格和常見標點
    text = re.sub(r'[^a-zA-Z0-9\s\'-]', '', text)
    
    if not text:
        print("錯誤：處理後的文字為空")
        return None

    if seed is not None and int(seed) >= 0:
        set_seed(int(seed))

    try:
        # tokenize
        inputs = tokenizer(text, return_tensors="pt")
        
        # 檢查 tokenizer 輸出
        print(f"輸入文字: '{text}'")
        print(f"Token IDs shape: {inputs['input_ids'].shape}")
        print(f"Token IDs: {inputs['input_ids']}")
        
        # 確保有有效的 tokens
        if inputs['input_ids'].size(1) == 0:
            print("錯誤：tokenizer 沒有產生任何 tokens")
            print("建議嘗試：")
            print("- ngai ho (我好)")
            print("- lia ho (你好)")  
            print("- do cia (謝謝)")
            print("- 檢查輸入是否為有效的客家語羅馬拼音")
            return None
            
        # 確保 input_ids 為整數型別
        input_ids = inputs["input_ids"].to(device, dtype=torch.long)
        
        # 動態調整推論用的 config
        original_speaking_rate = getattr(model.config, 'speaking_rate', 1.0)
        original_noise_scale = getattr(model.config, 'noise_scale', 0.667)
        original_noise_scale_duration = getattr(model.config, 'noise_scale_duration', 0.8)
        
        model.config.speaking_rate = float(speaking_rate)
        model.config.noise_scale = float(noise_scale)
        model.config.noise_scale_duration = float(noise_scale_duration)

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            wav = outputs.waveform.squeeze().cpu().float().numpy()

        # 恢復原始設定
        model.config.speaking_rate = original_speaking_rate
        model.config.noise_scale = original_noise_scale
        model.config.noise_scale_duration = original_noise_scale_duration

        # 轉成 int16 讓 Gradio 播放
        wav = np.clip(wav, -1.0, 1.0)
        wav_int16 = (wav * 32767.0).astype(np.int16)
        print("成功生成語音！")
        return (SR, wav_int16)
        
    except Exception as e:
        print(f"錯誤: {str(e)}")
        print(f"錯誤類型: {type(e)}")
        if "embedding" in str(e).lower():
            print("提示：這可能是因為輸入格式不正確")
            print("請確保使用客家語羅馬拼音而非中文字符")
        return None

with gr.Blocks(title="MMS-TTS Hakka (hak)") as demo:
    gr.Markdown("## **MMS-TTS：客家語（hak）文字轉語音**  \n模型：`facebook/mms-tts-hak`")
    
    gr.Markdown("""
    ### ⚠️ 重要提示：此模型需要羅馬拼音輸入
    - **不支援中文字符**：模型無法處理漢字輸入
    - **需要客家語羅馬拼音**：請使用台灣客家語拼音方案或其他羅馬拼音系統
    - **輸入格式**：例如英文字母和空格組成的拼音
    
    ### 客家語羅馬拼音範例：
    - **ngai** (我)
    - **lia** (你) 
    - **ho** (好)
    - **ngai ho** (我好)
    - **m ho** (不好)
    - **do cia** (多謝/謝謝)
    - **he moi** (係麼个/什麼)
    
    ### 建議測試文字（羅馬拼音）：
    - ngai ho (我好)
    - lia ho (你好)  
    - do cia (謝謝)
    - he moi (什麼)
    
    **注意**：由於缺乏官方的輸入格式說明，可能需要嘗試不同的拼音系統。
    """)
    
    # 添加快速選擇範例
    with gr.Row():
        gr.Markdown("### 快速選擇常用短語：")
    
    with gr.Row():
        example_buttons = []
        examples = [
            ("ngai ho", "我好"),
            ("lia ho", "你好"), 
            ("do cia", "謝謝"),
            ("he moi", "什麼"),
            ("m ho", "不好"),
            ("ngai ai", "我愛")
        ]
        
        for romanization, chinese in examples:
            btn = gr.Button(f"{romanization} ({chinese})", size="sm")
            example_buttons.append((btn, romanization))
    
    with gr.Row():
        with gr.Column(scale=2):
            text = gr.Textbox(
                label="輸入客家語羅馬拼音",
                placeholder="例如：ngai ho, lia ho, do cia",
                lines=4,
                value="ngai ho"  # 預設客家語拼音
            )
            btn = gr.Button("產生語音", variant="primary")
        with gr.Column(scale=1):
            speaking_rate = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="speaking_rate（越大越快）")
            noise_scale = gr.Slider(0.0, 1.5, value=0.667, step=0.01, label="noise_scale（語音隨機度）")
            noise_scale_duration = gr.Slider(0.0, 1.5, value=0.8, step=0.01, label="noise_scale_duration（節奏隨機度）")
            seed = gr.Number(value=0, precision=0, label="Seed（-1 表示不固定）")

    audio = gr.Audio(label="輸出語音", type="numpy")
    
    # 添加狀態顯示
    status = gr.Textbox(label="狀態", interactive=False)

    def tts_wrapper(*args):
        try:
            result = tts_hakka(*args)
            if result is None:
                return None, "錯誤：無法生成語音，請檢查輸入是否為有效的客家語羅馬拼音"
            return result, "成功生成語音"
        except Exception as e:
            return None, f"錯誤：{str(e)}"

    # 綁定範例按鈕
    for example_btn, romanization in example_buttons:
        example_btn.click(
            fn=lambda x=romanization: x,
            inputs=[],
            outputs=text
        )

    btn.click(
        fn=tts_wrapper,
        inputs=[text, speaking_rate, noise_scale, noise_scale_duration, seed],
        outputs=[audio, status],
    )

if __name__ == "__main__":
    demo.launch(share=True)