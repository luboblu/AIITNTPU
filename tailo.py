"""
使用 Facebook MMS-TTS-NAN 的台語語音合成系統
將台羅拼音轉換為台語語音
"""

import os
import tempfile
from pathlib import Path
import torch
import torchaudio
import numpy as np
from taibun import Converter
import gradio as gr
import time
import scipy.io.wavfile

class MMSTaiwaneseeTTS:
    """使用 Facebook MMS-TTS-NAN 的台語語音合成系統"""
    
    def __init__(self):
        self.init_tailo_converter()
        self.init_mms_taiwanese()
        
    def init_tailo_converter(self):
        """初始化台羅拼音轉換器"""
        try:
            self.tailo_converter = Converter(
                system='Tailo',
                dialect='south',
                format='mark',
                delimiter='-',
                sandhi='auto',
                punctuation='format'
            )
            self.tailo_status = "✅ 台羅轉換器已載入"
        except Exception as e:
            self.tailo_converter = None
            self.tailo_status = f"❌ 台羅轉換器載入失敗: {str(e)}"
    
    def init_mms_taiwanese(self):
        """初始化 MMS-TTS-NAN 台語模型"""
        try:
            from transformers import pipeline, VitsModel, AutoTokenizer
            
            print("📥 正在載入 Facebook MMS-TTS-NAN 模型...")
            
            # 使用 pipeline 方式（推薦）
            self.tts_pipeline = pipeline(
                "text-to-speech", 
                model="facebook/mms-tts-nan",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # 或者直接載入模型（備用方案）
            self.model = VitsModel.from_pretrained("facebook/mms-tts-nan")
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-nan")
            
            self.tts_available = True
            self.tts_status = "✅ Facebook MMS-TTS-NAN 模型已載入"
            
            print("✅ MMS-TTS-NAN 模型初始化成功")
            
        except ImportError as e:
            self.tts_available = False
            self.tts_status = "❌ 請安裝 transformers: pip install transformers"
            print(f"❌ transformers 未安裝: {e}")
            
        except Exception as e:
            self.tts_available = False
            self.tts_status = f"❌ 模型載入失敗: {str(e)}"
            print(f"❌ MMS-TTS-NAN 模型載入失敗: {e}")
    
    def chinese_to_tailo(self, chinese_text):
        """中文轉台羅拼音"""
        if not self.tailo_converter:
            return "", "❌ 台羅轉換器未載入"
        
        if not chinese_text.strip():
            return "", "請輸入中文文字"
        
        try:
            tailo_result = self.tailo_converter.get(chinese_text)
            char_count = len(chinese_text)
            syllable_count = len(tailo_result.split())
            status = f"✅ 轉換成功！{char_count}字 → {syllable_count}音節"
            return tailo_result, status
            
        except Exception as e:
            return "", f"❌ 轉換失敗: {str(e)}"
    
    def generate_taiwanese_speech_pipeline(self, text):
        """使用 pipeline 生成台語語音"""
        if not self.tts_available:
            return None, "❌ TTS 模型未就緒"
        
        try:
            print(f"🎵 正在使用 MMS-TTS-NAN 生成台語語音: {text}")
            
            # 使用 pipeline 生成語音
            result = self.tts_pipeline(text)
            
            # 獲取音頻數據和採樣率
            audio_data = result["audio"]
            sampling_rate = result["sampling_rate"]
            
            print(f"🔊 語音生成完成，採樣率: {sampling_rate}, 音頻長度: {len(audio_data)}")
            
            # 準備輸出檔案
            output_dir = "./results"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"taiwanese_mms_{int(time.time())}.wav")
            
            # 保存音頻檔案
            scipy.io.wavfile.write(output_file, sampling_rate, audio_data)
            print(f"💾 音頻已保存到: {output_file}")
            
            return output_file, "✅ 台語語音生成成功！"
            
        except Exception as e:
            print(f"❌ 語音生成錯誤: {e}")
            import traceback
            traceback.print_exc()
            return None, f"❌ 語音生成失敗: {str(e)}"
    
    def generate_taiwanese_speech_direct(self, text):
        """直接使用模型生成台語語音（備用方案）"""
        if not self.tts_available:
            return None, "❌ TTS 模型未就緒"
        
        try:
            print(f"🎵 正在使用直接模型生成台語語音: {text}")
            
            # 文字編碼
            inputs = self.tokenizer(text, return_tensors="pt")
            
            # 生成語音
            with torch.no_grad():
                output = self.model(**inputs).waveform
            
            # 獲取採樣率
            sampling_rate = self.model.config.sampling_rate
            
            print(f"🔊 語音生成完成，採樣率: {sampling_rate}, 音頻形狀: {output.shape}")
            
            # 準備輸出檔案
            output_dir = "./results"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"taiwanese_mms_direct_{int(time.time())}.wav")
            
            # 轉換為 numpy 陣列
            audio_numpy = output.squeeze().cpu().numpy()
            
            # 保存音頻檔案
            scipy.io.wavfile.write(output_file, sampling_rate, audio_numpy)
            print(f"💾 音頻已保存到: {output_file}")
            
            return output_file, "✅ 台語語音生成成功！"
            
        except Exception as e:
            print(f"❌ 語音生成錯誤: {e}")
            import traceback
            traceback.print_exc()
            return None, f"❌ 語音生成失敗: {str(e)}"
    
    def generate_taiwanese_speech(self, text):
        """生成台語語音（優先使用 pipeline）"""
        # 先嘗試 pipeline 方式
        result = self.generate_taiwanese_speech_pipeline(text)
        if result[0] is not None:
            return result
        
        print("🔄 Pipeline 失敗，嘗試直接模型方式...")
        # 如果 pipeline 失敗，嘗試直接模型方式
        return self.generate_taiwanese_speech_direct(text)
    
    def complete_taiwanese_pipeline(self, chinese_text):
        """完整台語語音合成流程：中文 → 台羅 → 台語語音"""
        print(f"🚀 開始台語語音合成流程：{chinese_text}")
        
        # 步驟1: 中文轉台羅
        tailo_text, tailo_status = self.chinese_to_tailo(chinese_text)
        print(f"📝 台羅轉換結果: {tailo_text}")
        
        if not tailo_text:
            return tailo_text, tailo_status, None, "❌ 台羅轉換失敗"
        
        # 步驟2: 台羅拼音直接轉台語語音
        audio_file, audio_status = self.generate_taiwanese_speech(tailo_text)
        
        return tailo_text, tailo_status, audio_file, audio_status
    
    def tailo_to_speech_direct(self, tailo_text):
        """直接將台羅拼音轉為語音（跳過中文轉換）"""
        if not tailo_text.strip():
            return None, "❌ 請輸入台羅拼音"
        
        return self.generate_taiwanese_speech(tailo_text)

# 安裝說明
MMS_INSTALL = """
## 📦 安裝 MMS-TTS-NAN 和相關依賴

```bash
# 基本安裝
pip install transformers
pip install torch torchaudio
pip install scipy
pip install taibun  # 台羅拼音轉換

# 如果要使用 GPU 加速
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🎯 Facebook MMS-TTS-NAN 台語模型

- ✅ **閩南語支援**：專門訓練的台語語音合成
- ✅ **簡單易用**：直接文字輸入，無需複雜轉換
- ✅ **高品質**：基於 VITS 架構的端到端合成
- ✅ **多語言**：支援超過 1000 種語言

## 📝 支援的輸入格式

1. **台羅拼音** (推薦): `Tsit-tah ū tsi̍t ê tuā siong-tiûnn`
2. **Pe̍h-ōe-jī**: `Chit-tah ū chı̍t ê tōa siông-tiûⁿ`
3. **漢羅混用**: `這搭 ū 一个大 siong-tiûnn`

## ⚠️ 使用注意事項

- 模型首次載入需要下載，請確保網路連線
- 建議使用台羅拼音輸入以獲得最佳效果
- 如果語音效果不佳，嘗試調整輸入文字格式
"""

# 初始化系統
print("🚀 正在初始化 MMS-TTS-NAN 台語 TTS 系統...")
try:
    mms_tts = MMSTaiwaneseeTTS()
    mms_ready = True
    print("✅ MMS-TTS-NAN 台語 TTS 系統初始化成功")
except Exception as e:
    print(f"❌ 系統初始化失敗: {e}")
    mms_ready = False

def mms_convert_only(chinese_text):
    """只轉換台羅拼音"""
    if not mms_ready:
        return "", "❌ 系統未就緒"
    return mms_tts.chinese_to_tailo(chinese_text)

def mms_full_pipeline(chinese_text):
    """完整台語語音合成流程"""
    if not mms_ready:
        return "", "❌ 系統未就緒", None, "❌ 系統未就緒"
    return mms_tts.complete_taiwanese_pipeline(chinese_text)

def mms_tailo_to_speech(tailo_text):
    """台羅拼音直接轉語音"""
    if not mms_ready:
        return None, "❌ 系統未就緒"
    return mms_tts.tailo_to_speech_direct(tailo_text)

def get_example_text(example_type):
    """獲取範例文字"""
    examples = {
        "基本問候": "你好",
        "日常對話": "今天天氣真好",
        "簡短句子": "謝謝",
        "台語常用": "食飽未",
        "數字": "一二三四五"
    }
    return examples.get(example_type, "")

def get_tailo_example(example_type):
    """獲取台羅拼音範例"""
    examples = {
        "問候": "lí-hó",
        "謝謝": "to-siā",
        "食飽未": "chia̍h-pá-buē",
        "今天": "kin-ji̍t",
        "數字": "tsi̍t jī sann sì gōo"
    }
    return examples.get(example_type, "")

# 創建 Gradio 介面
with gr.Blocks(title="MMS台語TTS", theme=gr.themes.Soft()) as mms_demo:
    
    gr.Markdown("""
    # 🎎 MMS-TTS-NAN 台語語音合成系統
    ### 使用 Facebook MMS-TTS-NAN 模型
    
    **專業台語語音合成** - 支援台羅拼音轉台語語音
    """)
    
    # 系統狀態
    if mms_ready:
        with gr.Row():
            mms_tailo_status = gr.Textbox(
                label="台羅轉換狀態", 
                value=mms_tts.tailo_status,
                interactive=False
            )
            mms_tts_status = gr.Textbox(
                label="MMS-TTS 狀態",
                value=mms_tts.tts_status,
                interactive=False
            )
    else:
        gr.Markdown("❌ **系統未就緒** - 請安裝 transformers 和相關依賴")
    
    # 主要功能 - 中文轉台語語音
    with gr.Tab("🔄 中文轉台語語音"):
        with gr.Row():
            with gr.Column(scale=1):
                mms_input = gr.Textbox(
                    label="中文輸入",
                    placeholder="輸入中文，生成台語語音...",
                    lines=4
                )
                
                with gr.Row():
                    mms_convert_btn = gr.Button("🔄 轉台羅", variant="secondary")
                    mms_generate_btn = gr.Button("🎵 台語語音", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                mms_tailo_output = gr.Textbox(
                    label="台羅拼音",
                    lines=3,
                    interactive=False,
                    show_copy_button=True
                )
                
                mms_audio_output = gr.Audio(
                    label="台語語音輸出",
                    type="filepath"
                )
        
        # 狀態顯示
        with gr.Row():
            mms_status1 = gr.Textbox(label="轉換狀態", interactive=False, scale=1)
            mms_status2 = gr.Textbox(label="語音狀態", interactive=False, scale=1)
        
        # 快速範例
        with gr.Row():
            mms_example_selector = gr.Dropdown(
                label="選擇範例",
                choices=["基本問候", "日常對話", "簡短句子", "台語常用", "數字"],
                value="基本問候",
                scale=2
            )
            mms_load_example_btn = gr.Button("📋 載入", scale=1)
    
    # 直接台羅拼音輸入
    with gr.Tab("📝 台羅拼音轉語音"):
        with gr.Row():
            with gr.Column(scale=1):
                tailo_input = gr.Textbox(
                    label="台羅拼音輸入",
                    placeholder="直接輸入台羅拼音，如：lí-hó",
                    lines=4
                )
                tailo_generate_btn = gr.Button("🎵 生成語音", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                tailo_audio_output = gr.Audio(
                    label="台語語音輸出",
                    type="filepath"
                )
                tailo_status = gr.Textbox(label="生成狀態", interactive=False)
        
        # 台羅拼音範例
        with gr.Row():
            tailo_example_selector = gr.Dropdown(
                label="台羅拼音範例",
                choices=["問候", "謝謝", "食飽未", "今天", "數字"],
                value="問候",
                scale=2
            )
            tailo_load_example_btn = gr.Button("📋 載入", scale=1)
    
    # 範例展示
    with gr.Tab("🎯 語音範例"):
        gr.Examples(
            examples=[
                ["你好"],
                ["謝謝"], 
                ["食飽未"],
                ["今天天氣很好"],
                ["一二三四五"]
            ],
            inputs=[mms_input],
            outputs=[mms_tailo_output, mms_status1, mms_audio_output, mms_status2],
            fn=mms_full_pipeline,
            cache_examples=False
        )
        
        gr.Examples(
            examples=[
                ["lí-hó"],
                ["to-siā"],
                ["chia̍h-pá-buē"],
                ["kin-ji̍t tinn-khì tsin hó"],
                ["tsi̍t jī sann sì gōo"]
            ],
            inputs=[tailo_input],
            outputs=[tailo_audio_output, tailo_status],
            fn=mms_tailo_to_speech,
            cache_examples=False
        )
    
    # 安裝說明
    with gr.Accordion("📖 安裝說明", open=False):
        gr.Markdown(MMS_INSTALL)
    
    # 事件綁定 - 中文轉台語
    mms_convert_btn.click(
        fn=mms_convert_only,
        inputs=[mms_input],
        outputs=[mms_tailo_output, mms_status1]
    )
    
    mms_generate_btn.click(
        fn=mms_full_pipeline,
        inputs=[mms_input],
        outputs=[mms_tailo_output, mms_status1, mms_audio_output, mms_status2]
    )
    
    mms_load_example_btn.click(
        fn=get_example_text,
        inputs=[mms_example_selector],
        outputs=[mms_input]
    )
    
    # 事件綁定 - 台羅拼音直接轉語音
    tailo_generate_btn.click(
        fn=mms_tailo_to_speech,
        inputs=[tailo_input],
        outputs=[tailo_audio_output, tailo_status]
    )
    
    tailo_load_example_btn.click(
        fn=get_tailo_example,
        inputs=[tailo_example_selector],
        outputs=[tailo_input]
    )
    
    gr.Markdown("""
    ---
    🎯 **Facebook MMS-TTS-NAN** - 專業台語語音合成  
    🌏 **多語言支援** - 超過 1000 種語言  
    📝 **台羅拼音** - 標準台語羅馬字輸入  
    🔊 **VITS 架構** - 端到端高品質語音合成
    """)

if __name__ == "__main__":
    mms_demo.launch(
        server_name="0.0.0.0",
        server_port=7864,
        share=True,
        show_error=True
    )