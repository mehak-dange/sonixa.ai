import json
import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from extract_llm import extract_data

# recording settings
fs = 16000
seconds = 6

print("speak now...")

recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype="float32")
sd.wait()

audio = np.int16(recording * 32767)
write("audio.wav", fs, audio)

model = whisper.load_model("base")

result = model.transcribe("audio.wav", fp16=False)

raw_text = result["text"].strip()
detected_language = result.get("language", "unknown")

print("\n=== raw whisper text ===")
print(raw_text)
print("\n=== detected language ===")
print(detected_language)

# transliterate only if Hindi script is present
if any('\u0900' <= ch <= '\u097F' for ch in raw_text):
    processed_text = transliterate(raw_text, sanscript.DEVANAGARI, sanscript.ITRANS).lower()
else:
    processed_text = raw_text.lower()

processed_text = " ".join(processed_text.split())

print("\n=== processed text ===")
print(processed_text)

final_result = extract_data(processed_text)

print("\n=== structured output ===")
print(json.dumps(final_result, indent=2, ensure_ascii=False))