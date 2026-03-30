import os
import whisper
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from extract_llm import extract_data
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# better model than base
model = whisper.load_model("small")


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    language: str = Form("auto")
):
    try:
        ext = file.filename.split(".")[-1] if "." in file.filename else "webm"
        file_path = f"temp_audio.{ext}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        # force language only if user selected one
        if language == "auto":
            result = model.transcribe(file_path, fp16=False, temperature=0)
        else:
            result = model.transcribe(
                file_path,
                fp16=False,
                temperature=0,
                language=language
            )

        raw_text = result["text"].strip()
        detected_language = result.get("language", "unknown")

        # transliterate only if Hindi script appears
        if any('\u0900' <= ch <= '\u097F' for ch in raw_text):
            processed_text = transliterate(
                raw_text, sanscript.DEVANAGARI, sanscript.ITRANS
            ).lower()
        else:
            processed_text = raw_text.lower()

        processed_text = " ".join(processed_text.split())

        structured = extract_data(processed_text)

        if os.path.exists(file_path):
            os.remove(file_path)

        return {
            "detected_language": detected_language,
            "raw_text": raw_text,
            "processed_text": processed_text,
            "structured_output": structured
        }

    except Exception as e:
        return {"error": str(e)}