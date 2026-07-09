import numpy as np
import torch
import gradio as gr
from scipy.signal import resample_poly
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

MODEL_ID = "formospeech/whisper-large-v3-taiwanese-hakka"

# 裝置與精度
use_cuda = torch.cuda.is_available()
device_index = 0 if use_cuda else -1
dtype = torch.float16 if use_cuda else torch.float32

# 載入模型與處理器
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)
if use_cuda:
    model = model.to("cuda")

# 建立 ASR pipeline（等會直接餵 raw array）
asr = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device_index,
    torch_dtype=dtype,
)

# 模型卡列出的方言 ID（作為 prompt 使用）
DIALECT_IDS = {
    "（不指定）": "",
    "四縣": "htia_sixian",
    "海陸": "htia_hailu",
    "大埔": "htia_dapu",
    "饒平": "htia_raoping",
    "詔安": "htia_zhaoan",
    "南四縣": "htia_nansixian",
}

def _to_mono(audio: np.ndarray) -> np.ndarray:
    """立體聲 -> 單聲道"""
    if audio.ndim == 2:
        return audio.mean(axis=1)
    return audio

def _resample(wave: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """用 polyphase 法重采樣（高品質、效能好）"""
    if sr_in == sr_out:
        return wave
    # 以整數比例做上/下採樣
    # 使用 resample_poly(up, down) 避免 FFT resample 的邊界偽影
    # 參考：SciPy resample_poly 文件
    from math import gcd
    g = gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    return resample_poly(wave, up, down).astype(np.float32)

def transcribe(audio_tuple, dialect_label, temperature, num_beams):
    """
    audio_tuple: (sample_rate:int, audio:np.ndarray) 來自 Gradio Audio(type='numpy')
    """
    if audio_tuple is None:
        return "請先錄音或上傳音檔"

    # 取得 sr 與 waveform
    try:
        sr_in, audio = audio_tuple
    except Exception:
        return "音訊格式有誤，請重新錄音或上傳"

    # 單聲道 & 轉 float32
    audio = _to_mono(np.asarray(audio)).astype(np.float32)

    # Whisper 系列常見輸入取樣率為 16k（依 feature_extractor 為準）
    target_sr = processor.feature_extractor.sampling_rate
    audio = _resample(audio, int(sr_in), int(target_sr))

    # 設定 dialect prompt（模型在訓練時使用 dialect IDs 當 prompts）
    dialect_id = DIALECT_IDS.get(dialect_label or "（不指定）", "")
    generate_kwargs = {
        "task": "transcribe",
        "language": "Chinese",      # Whisper 的語言 token
        "temperature": float(temperature),
        "num_beams": int(num_beams),
        "max_new_tokens": 128,
    }
    if dialect_id:
        prompt_ids = processor.get_prompt_ids(dialect_id)
        prompt_ids = torch.from_numpy(prompt_ids)
        if use_cuda:
            prompt_ids = prompt_ids.to("cuda")
        generate_kwargs["prompt_ids"] = prompt_ids

    # 直接餵 raw array，完全不經檔名 => 無需 ffmpeg
    # HF ASR pipeline 支援輸入 {"array": np.array, "sampling_rate": sr}
    out = asr({"array": audio, "sampling_rate": target_sr},
              chunk_length_s=30,
              batch_size=8,
              return_timestamps=False,
              generate_kwargs=generate_kwargs)

    text = out["text"] if isinstance(out, dict) and "text" in out else str(out)
    if dialect_id:
        text = text.replace(f" {dialect_id}", "").replace(dialect_id, "")
    return text

with gr.Blocks(title="Hakka ASR (no-FFmpeg)") as demo:
    gr.Markdown(
        "## **台灣客家語 ASR：語音 → 文字（不需 FFmpeg）**  \n"
        f"模型：`{MODEL_ID}`（Whisper-large-v3 客家語微調）"
    )
    with gr.Row():
        with gr.Column(scale=2):
            audio = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",               # <— 關鍵：輸出 (sr, np.array)
                label="上傳或錄音（WAV/MP3 皆可；後端接收為 raw waveform）",
            )
            dialect = gr.Dropdown(
                choices=list(DIALECT_IDS.keys()),
                value="（不指定）",
                label="方言（可選，選對通常更準）",
            )
            temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="temperature（越高越隨機）")
            num_beams = gr.Slider(1, 5, value=1, step=1, label="beam size（>1 較穩但較慢）")
            btn = gr.Button("開始辨識", variant="primary")
        with gr.Column(scale=3):
            out_text = gr.Textbox(label="辨識結果", lines=12)

    btn.click(transcribe, [audio, dialect, temperature, num_beams], [out_text])

if __name__ == "__main__":
    demo.launch(share=True)
