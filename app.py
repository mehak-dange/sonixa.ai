import os
import whisper
from datetime import datetime, timezone
from bson import ObjectId
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from extract_llm import extract_data
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from db import users_collection, interactions_collection
from auth_utils import hash_password, verify_password, create_access_token, decode_access_token

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Load whisper once
model = whisper.load_model("small")


# -----------------------------
# Pydantic models
# -----------------------------
class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class SaveReportRequest(BaseModel):
    domain: str
    detected_language: str | None = None
    raw_text: str
    processed_text: str
    structured_output: dict
    final_report: dict


# -----------------------------
# Helper functions
# -----------------------------
def serialize_user(user):
    return {
        "id": str(user["_id"]),
        "name": user["name"],
        "email": user["email"]
    }


def get_current_user_from_token(authorization: str = None):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")

    token = authorization.split(" ")[1]
    payload = decode_access_token(token)

    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user


# -----------------------------
# Basic routes
# -----------------------------
@app.get("/")
def home():
    return FileResponse("static/login.html")


@app.get("/health")
def health():
    return {"message": "Sonixa backend is running"}


# -----------------------------
# Auth routes
# -----------------------------
@app.post("/register")
def register_user(data: RegisterRequest):
    existing_user = users_collection.find_one({"email": data.email.lower()})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = {
        "name": data.name.strip(),
        "email": data.email.lower(),
        "password": hash_password(data.password),
        "created_at": datetime.now(timezone.utc)
    }

    result = users_collection.insert_one(new_user)
    created_user = users_collection.find_one({"_id": result.inserted_id})

    token = create_access_token({"user_id": str(created_user["_id"])})

    return {
        "message": "Registration successful",
        "token": token,
        "user": serialize_user(created_user)
    }


@app.post("/login")
def login_user(data: LoginRequest):
    user = users_collection.find_one({"email": data.email.lower()})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"user_id": str(user["_id"])})

    return {
        "message": "Login successful",
        "token": token,
        "user": serialize_user(user)
    }


@app.get("/me")
def get_me(authorization: str = Header(None)):
    user = get_current_user_from_token(authorization)
    return {"user": serialize_user(user)}


# -----------------------------
# Audio processing route
# This returns AI draft only.
# It does NOT save to DB.
# -----------------------------
@app.post("/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    mode: str = Form("mixed"),
    authorization: str = Header(None)
):
    try:
        # user must be logged in
        _ = get_current_user_from_token(authorization)

        ext = file.filename.split(".")[-1] if "." in file.filename else "webm"
        file_path = f"temp_audio.{ext}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        result = model.transcribe(
            file_path,
            fp16=False,
            temperature=0,
            task="transcribe"
        )

        raw_text = result["text"].strip()
        detected_language = result.get("language", "unknown")

        # transliterate only if Devanagari appears
        if any('\u0900' <= ch <= '\u097F' for ch in raw_text):
            processed_text = transliterate(
                raw_text,
                sanscript.DEVANAGARI,
                sanscript.ITRANS
            ).lower()
        else:
            processed_text = raw_text.lower()

        processed_text = " ".join(processed_text.split())

        structured = extract_data(processed_text, mode)

        if os.path.exists(file_path):
            os.remove(file_path)

        return {
            "mode": mode,
            "detected_language": detected_language,
            "raw_text": raw_text,
            "processed_text": processed_text,
            "structured_output": structured
        }

    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# Save final manually edited report
# This is the real DB save route
# -----------------------------
@app.post("/save-report")
def save_report(data: SaveReportRequest, authorization: str = Header(None)):
    try:
        user = get_current_user_from_token(authorization)

        report_doc = {
            "user_id": str(user["_id"]),
            "domain": data.domain,
            "detected_language": data.detected_language,
            "raw_text": data.raw_text,
            "processed_text": data.processed_text,
            "structured_output": data.structured_output,  # AI draft
            "final_report": data.final_report,            # human-edited final report
            "created_at": datetime.now(timezone.utc)
        }

        result = interactions_collection.insert_one(report_doc)

        return {
            "message": "Report saved successfully",
            "report_id": str(result.inserted_id)
        }

    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# History route
# Returns saved final reports
# -----------------------------
@app.get("/history")
def get_history(authorization: str = Header(None)):
    try:
        user = get_current_user_from_token(authorization)

        history = list(
            interactions_collection.find(
                {"user_id": str(user["_id"])},
                {"user_id": 0}
            ).sort("created_at", -1)
        )

        formatted = []
        for item in history:
            formatted.append({
                "id": str(item["_id"]),
                "domain": item.get("domain"),
                "detected_language": item.get("detected_language"),
                "raw_text": item.get("raw_text"),
                "processed_text": item.get("processed_text"),
                "structured_output": item.get("structured_output"),
                "final_report": item.get("final_report"),
                "created_at": item.get("created_at")
            })

        return {"history": formatted}

    except Exception as e:
        return {"error": str(e)}