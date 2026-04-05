import os
import whisper
from datetime import datetime, timezone
from bson import ObjectId
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from db import users_collection, patients_collection, conversations_collection
from auth_utils import hash_password, verify_password, create_access_token, decode_access_token
from extract_llm import extract_data

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

model = whisper.load_model("small")


class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class SaveConversationRequest(BaseModel):
    domain: str

    # Healthcare fields
    patient_name: str | None = None
    patient_age: str | None = None
    patient_gender: str | None = None
    patient_phone: str | None = None
    patient_notes: str | None = None

    # Finance fields
    account_holder_name: str | None = None
    account_number: str | None = None
    contact_number: str | None = None
    finance_notes: str | None = None

    detected_language: str | None = None
    raw_text: str
    processed_text: str
    structured_output: dict
    final_report: dict


def serialize_user(user):
    return {
        "id": str(user["_id"]),
        "name": user["name"],
        "email": user["email"]
    }


def serialize_objectid(doc):
    doc["id"] = str(doc["_id"])
    del doc["_id"]
    return doc


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


@app.get("/")
def home():
    return FileResponse("static/login.html")


@app.get("/health")
def health():
    return {"message": "Sonixa backend is running"}


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


@app.post("/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    domain: str = Form("healthcare"),
    language_hint: str = Form("auto"),
    authorization: str = Header(None)
):
    try:
        _ = get_current_user_from_token(authorization)

        ext = file.filename.split(".")[-1] if "." in file.filename else "webm"
        file_path = f"temp_audio.{ext}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        if language_hint == "kannada":
            result = model.transcribe(
                file_path,
                fp16=False,
                temperature=0,
                task="transcribe",
                language="kn"
            )
        elif language_hint == "hindi":
            result = model.transcribe(
                file_path,
                fp16=False,
                temperature=0,
                task="transcribe",
                language="hi"
            )
        elif language_hint == "english":
            result = model.transcribe(
                file_path,
                fp16=False,
                temperature=0,
                task="transcribe",
                language="en"
            )
        else:
            result = model.transcribe(
                file_path,
                fp16=False,
                temperature=0,
                task="transcribe"
            )

        raw_text = result["text"].strip()
        detected_language = result.get("language", "unknown")

        if any('\u0900' <= ch <= '\u097F' for ch in raw_text):
            processed_text = transliterate(
                raw_text,
                sanscript.DEVANAGARI,
                sanscript.ITRANS
            ).lower()
        else:
            processed_text = raw_text.lower()

        processed_text = " ".join(processed_text.split())

        structured = extract_data(processed_text, domain)

        if os.path.exists(file_path):
            os.remove(file_path)

        return {
            "domain": domain,
            "language_hint": language_hint,
            "detected_language": detected_language,
            "raw_text": raw_text,
            "processed_text": processed_text,
            "structured_output": structured
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/save-conversation")
def save_conversation(data: SaveConversationRequest, authorization: str = Header(None)):
    try:
        user = get_current_user_from_token(authorization)
        doctor_id = str(user["_id"])

        # HEALTHCARE FLOW
        if data.domain == "healthcare":
            existing_patient = patients_collection.find_one({
                "doctor_id": doctor_id,
                "type": "healthcare",
                "name": (data.patient_name or "").strip().lower(),
                "phone": (data.patient_phone or "").strip()
            })

            if existing_patient:
                patient_id = existing_patient["_id"]
                patients_collection.update_one(
                    {"_id": patient_id},
                    {
                        "$set": {
                            "age": data.patient_age,
                            "gender": data.patient_gender,
                            "notes": data.patient_notes,
                            "updated_at": datetime.now(timezone.utc)
                        }
                    }
                )
            else:
                patient_doc = {
                    "doctor_id": doctor_id,
                    "type": "healthcare",
                    "name": (data.patient_name or "").strip().lower(),
                    "display_name": (data.patient_name or "").strip(),
                    "age": data.patient_age,
                    "gender": data.patient_gender,
                    "phone": (data.patient_phone or "").strip(),
                    "notes": data.patient_notes,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc)
                }
                result = patients_collection.insert_one(patient_doc)
                patient_id = result.inserted_id

            conversation_doc = {
                "doctor_id": doctor_id,
                "record_type": "healthcare",
                "record_id": str(patient_id),
                "domain": data.domain,
                "detected_language": data.detected_language,
                "raw_text": data.raw_text,
                "processed_text": data.processed_text,
                "structured_output": data.structured_output,
                "final_report": data.final_report,
                "created_at": datetime.now(timezone.utc)
            }

            conv_result = conversations_collection.insert_one(conversation_doc)

            return {
                "message": "Patient and conversation saved successfully",
                "record_id": str(patient_id),
                "conversation_id": str(conv_result.inserted_id)
            }

        # FINANCE FLOW
        existing_account = patients_collection.find_one({
            "doctor_id": doctor_id,
            "type": "finance",
            "account_holder_name": (data.account_holder_name or "").strip().lower(),
            "account_number": (data.account_number or "").strip()
        })

        if existing_account:
            account_id = existing_account["_id"]
            patients_collection.update_one(
                {"_id": account_id},
                {
                    "$set": {
                        "contact_number": data.contact_number,
                        "notes": data.finance_notes,
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )
        else:
            account_doc = {
                "doctor_id": doctor_id,
                "type": "finance",
                "account_holder_name": (data.account_holder_name or "").strip().lower(),
                "display_name": (data.account_holder_name or "").strip(),
                "account_number": (data.account_number or "").strip(),
                "contact_number": (data.contact_number or "").strip(),
                "notes": data.finance_notes,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            result = patients_collection.insert_one(account_doc)
            account_id = result.inserted_id

        conversation_doc = {
            "doctor_id": doctor_id,
            "record_type": "finance",
            "record_id": str(account_id),
            "domain": data.domain,
            "detected_language": data.detected_language,
            "raw_text": data.raw_text,
            "processed_text": data.processed_text,
            "structured_output": data.structured_output,
            "final_report": data.final_report,
            "created_at": datetime.now(timezone.utc)
        }

        conv_result = conversations_collection.insert_one(conversation_doc)

        return {
            "message": "Finance record and conversation saved successfully",
            "record_id": str(account_id),
            "conversation_id": str(conv_result.inserted_id)
        }

    except Exception as e:
        return {"error": str(e)}


# HEALTHCARE LISTING
@app.get("/patients")
def get_patients(authorization: str = Header(None), q: str = ""):
    try:
        user = get_current_user_from_token(authorization)
        doctor_id = str(user["_id"])

        patients = list(
            patients_collection.find({"doctor_id": doctor_id, "type": "healthcare"}).sort("updated_at", -1)
        )

        formatted = []
        for patient in patients:
            patient = serialize_objectid(patient)
            if q and q.lower() not in str(patient).lower():
                continue
            formatted.append(patient)

        return {"patients": formatted}

    except Exception as e:
        return {"error": str(e)}


@app.get("/patients/by-phone/{phone}")
def get_patient_by_phone(phone: str, authorization: str = Header(None)):
    try:
        user = get_current_user_from_token(authorization)
        doctor_id = str(user["_id"])

        patient = patients_collection.find_one({
            "doctor_id": doctor_id,
            "type": "healthcare",
            "phone": phone.strip()
        })

        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        return {"patient": serialize_objectid(patient)}

    except Exception as e:
        return {"error": str(e)}


@app.get("/patients/{patient_id}")
def get_patient(patient_id: str, authorization: str = Header(None)):
    try:
        user = get_current_user_from_token(authorization)
        doctor_id = str(user["_id"])

        patient = patients_collection.find_one({
            "_id": ObjectId(patient_id),
            "doctor_id": doctor_id,
            "type": "healthcare"
        })

        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        return {"patient": serialize_objectid(patient)}

    except Exception as e:
        return {"error": str(e)}


@app.get("/patients/{patient_id}/conversations")
def get_patient_conversations(patient_id: str, authorization: str = Header(None)):
    try:
        user = get_current_user_from_token(authorization)
        doctor_id = str(user["_id"])

        conversations = list(
            conversations_collection.find({
                "doctor_id": doctor_id,
                "record_type": "healthcare",
                "record_id": patient_id
            }).sort("created_at", -1)
        )

        formatted = []
        for convo in conversations:
            convo = serialize_objectid(convo)
            formatted.append(convo)

        return {"conversations": formatted}

    except Exception as e:
        return {"error": str(e)}


# FINANCE LISTING
@app.get("/finance-records")
def get_finance_records(authorization: str = Header(None), q: str = ""):
    try:
        user = get_current_user_from_token(authorization)
        doctor_id = str(user["_id"])

        records = list(
            patients_collection.find({"doctor_id": doctor_id, "type": "finance"}).sort("updated_at", -1)
        )

        formatted = []
        for record in records:
            record = serialize_objectid(record)
            if q and q.lower() not in str(record).lower():
                continue
            formatted.append(record)

        return {"records": formatted}

    except Exception as e:
        return {"error": str(e)}


@app.get("/finance-records/by-contact/{contact}")
def get_finance_record_by_contact(contact: str, authorization: str = Header(None)):
    try:
        user = get_current_user_from_token(authorization)
        doctor_id = str(user["_id"])

        record = patients_collection.find_one({
            "doctor_id": doctor_id,
            "type": "finance",
            "contact_number": contact.strip()
        })

        if not record:
            raise HTTPException(status_code=404, detail="Finance record not found")

        return {"record": serialize_objectid(record)}

    except Exception as e:
        return {"error": str(e)}


@app.get("/finance-records/{record_id}")
def get_finance_record(record_id: str, authorization: str = Header(None)):
    try:
        user = get_current_user_from_token(authorization)
        doctor_id = str(user["_id"])

        record = patients_collection.find_one({
            "_id": ObjectId(record_id),
            "doctor_id": doctor_id,
            "type": "finance"
        })

        if not record:
            raise HTTPException(status_code=404, detail="Finance record not found")

        return {"record": serialize_objectid(record)}

    except Exception as e:
        return {"error": str(e)}


@app.get("/finance-records/{record_id}/conversations")
def get_finance_record_conversations(record_id: str, authorization: str = Header(None)):
    try:
        user = get_current_user_from_token(authorization)
        doctor_id = str(user["_id"])

        conversations = list(
            conversations_collection.find({
                "doctor_id": doctor_id,
                "record_type": "finance",
                "record_id": record_id
            }).sort("created_at", -1)
        )

        formatted = []
        for convo in conversations:
            convo = serialize_objectid(convo)
            formatted.append(convo)

        return {"conversations": formatted}

    except Exception as e:
        return {"error": str(e)}


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, authorization: str = Header(None)):
    try:
        user = get_current_user_from_token(authorization)
        doctor_id = str(user["_id"])

        result = conversations_collection.delete_one({
            "_id": ObjectId(conversation_id),
            "doctor_id": doctor_id
        })

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {"message": "Deleted successfully"}

    except Exception as e:
        return {"error": str(e)}