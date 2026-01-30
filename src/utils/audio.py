import os
import subprocess
import whisper
import torch
import soundfile as sf
from .config import PROJECT_CONFIG

def whisper_check(model, audio_path: str, target_text: str) -> tuple[float, str]:
    """Transcribes audio and returns a similarity score."""
    try:
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1024:
            size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 'N/A'
            return 0.0, f"[ERROR: Invalid audio file '{audio_path}' (size: {size} bytes)]"

        # Whisper may have issues with MPS, fallback to CPU
        whisper_device = "cpu" if PROJECT_CONFIG['device'].startswith("mps") else PROJECT_CONFIG['device']

        result = model.transcribe(audio_path, language="de", fp16=torch.cuda.is_available())
        transcribed_text = result['text']

        from .text import calculate_similarity
        score = calculate_similarity(transcribed_text, target_text)
        return score, transcribed_text

    except Exception as e:
        return 0.0, f"[ERROR: Whisper check failed - {e}]"

def load_whisper_model():
    """Loads the Whisper model specified in the config."""
    model_name = PROJECT_CONFIG.get('whisper_model_name', 'base')
    whisper_device = "cpu" if PROJECT_CONFIG['device'].startswith("mps") else PROJECT_CONFIG['device']
    print(f"Loading Whisper model '{model_name}' on device '{whisper_device}'...")
    return whisper.load_model(model_name, device=whisper_device)

def get_audio_duration(file_path: str) -> float:
    """Returns the duration of an audio file in seconds."""
    try:
        info = sf.info(file_path)
        return info.duration
    except Exception:
        return 0.0

def run_loudness_normalization(input_path: str, output_path: str):
    """Applies EBU R128 loudness normalization using ffmpeg."""
    print("Applying EBU R128 loudness normalization with ffmpeg...")
    settings = PROJECT_CONFIG
    loudnorm_filter = (
        f"loudnorm=I={settings['ebu_target_loudness']}:"
        f"TP={settings['ebu_true_peak']}:"
        f"LRA={settings['ebu_loudness_range']}"
    )
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-af", loudnorm_filter,
        "-ar", "48000", # Qwen-TTS base model sample rate is 48k
        output_path
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return True, ""
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please ensure it is installed and in your system's PATH.")
    except subprocess.CalledProcessError as e:
        return False, e.stderr
