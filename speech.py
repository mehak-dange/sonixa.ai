import os
import uuid
import shutil
import subprocess
from pathlib import Path

import whisper

# Load model once when server starts
# "small" is better than "base" for multilingual accuracy
MODEL_NAME = "small"
model = whisper.load_model(MODEL_NAME)

UPLOAD_DIR = Path("temp_audio")
UPLOAD_DIR.mkdir(exist_ok=True)


def convert_to_wav(input_path: str, output_path: str):
    """
    Convert any uploaded audio to clean mono 16k WAV using ffmpeg.
    This improves ASR accuracy.
    """
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def transcribe_audio(file_path: str, forced_language: str | None = None):
    """
    Transcribe audio using Whisper.
    forced_language can be:
    - "kn" for Kannada
    - "hi" for Hindi
    - "en" for English
    - None for auto-detection
    """
    wav_path = str(UPLOAD_DIR / f"{uuid.uuid4().hex}.wav")

    try:
        convert_to_wav(file_path, wav_path)

        kwargs = {
            "fp16": False,
            "task": "transcribe"
        }

        if forced_language:
            kwargs["language"] = forced_language

        result = model.transcribe(wav_path, **kwargs)

        detected_language = result.get("language", "unknown")
        transcript = result.get("text", "").strip()

        return {
            "success": True,
            "transcript": transcript,
            "detected_language": detected_language
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "transcript": "",
            "detected_language": "unknown"
        }

    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def save_uploaded_file(upload_file):
    """
    Save FastAPI UploadFile temporarily.
    """
    suffix = Path(upload_file.filename).suffix if upload_file.filename else ".webm"
    temp_path = str(UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}")

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return temp_path