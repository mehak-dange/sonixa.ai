import json
import re
from google import genai

client = genai.Client()


def _safe_json_parse(raw_output: str):
    raw_output = raw_output.strip()
    raw_output = raw_output.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return None


def extract_data(input_text, mode="healthcare"):
    clean_text = re.sub(r"\s+", " ", input_text.strip().lower())

    if mode == "finance":
        prompt = f"""
You are a strict financial conversation information extraction engine.

Your task is to extract structured information from multilingual Indian financial/payment speech.
The input may be in Kannada, Hindi, English, or code-mixed text.

Input:
"{clean_text}"

Return ONLY valid JSON.
Do NOT use markdown.
Do NOT use ```json.
Do NOT add explanation.

Use exactly this schema:
{{
  "payer_name": null,
  "payment_status": null,
  "amount": null,
  "payment_mode": null,
  "payment_date": null,
  "reason_for_payment": null,
  "account_or_loan_confirmation": null,
  "executive_notes": ""
}}

Normalization examples:
- payment done / payment ho gaya / payment aytu -> completed
- not paid / payment nahi hua -> pending
- cash / nagadu -> cash
- online / upi / gpay / phonepe -> online
- loan / khata / account -> account_or_loan_confirmation
- 500 rupees / 500 rs -> 500

Rules:
- Keep values short and normalized.
- If a field is not clearly present, return null.
- Do not invent values.
"""
        fallback_data = {
            "payer_name": None,
            "payment_status": None,
            "amount": None,
            "payment_mode": None,
            "payment_date": None,
            "reason_for_payment": None,
            "account_or_loan_confirmation": None,
            "executive_notes": ""
        }
    else:
        prompt = f"""
You are a strict healthcare information extraction engine.

Your task is to extract structured symptom/report information from multilingual Indian patient speech.
The input may be in Kannada, Hindi, English, or code-mixed text.

Input:
"{clean_text}"

Return ONLY valid JSON.
Do NOT use markdown.
Do NOT use ```json.
Do NOT add explanation.

Use exactly this schema:
{{
  "symptoms": [],
  "duration": null,
  "body_part": [],
  "severity": null,
  "past_history": null,
  "observations": null,
  "probable_status": null,
  "treatment_advice": null,
  "notes": ""
}}

Normalization rules:
- fever, bukhar, bukhaar, jwara, jvara, ಜ್ವರ -> fever
- leg pain, pair dard, kaalu novu, ಕಾಲು ನೋವು -> leg pain
- stomach pain, pet dard, hotte novu, ಹೊಟ್ಟೆ ನೋವು, abdominal pain -> stomach pain
- headache, sar dard, tale novu, ತಲೆ ನೋವು, head pain -> headache
- cough, khansi, kemmnu, ಕೆಮ್ಮು -> cough
- cold, sardi, sheet, running nose, ಜಲದು -> cold
- body pain, sharir dard, mai dard, deha novu, ದೇಹ ನೋವು -> body pain
- weakness, kamzori, bala illa, ದುರ್ಬಲತೆ -> weakness
- vomiting, ulti, vanti, ವಾಂತಿ -> vomiting
- throat pain, gala dard, gantalu novu, ಗಂಟಲು ನೋವು -> throat pain
- chest pain, seene me dard, ede novu, ಎದೆ ನೋವು -> chest pain
- breathing problem, saans ki dikkat, usirata tondare, ಉಸಿರಾಟ ತೊಂದರೆ, difficulty breathing -> breathing problem

Duration normalization:
- 1 day, ek din, ondu dina, ಒಂದು ದಿನ -> 1 day
- 2 days, do din, yeradu dina, ಎರಡು ದಿನ -> 2 days
- 3 days, teen din, mooru dina, ಮೂರು ದಿನ -> 3 days
- since yesterday, kal se, ninne inda, ನಿನ್ನೆ ಇಂದ -> since yesterday
- 1 week, ek hafte se, ondu vaara, ಒಂದು ವಾರ -> 1 week

Body part normalization:
- leg, pair, kaalu, ಕಾಲು -> leg
- stomach, pet, hotte, ಹೊಟ್ಟೆ -> stomach
- head, sar, tale, ತಲೆ -> head
- throat, gala, gantalu, ಗಂಟಲು -> throat
- chest, seena, ede, ಎದೆ -> chest

Severity normalization:
- mild, halka, swalpa -> mild
- severe, zyada, tumba, bahut -> severe

Rules:
- Always extract symptoms if any symptom-like phrase is present.
- If multiple symptoms are present, include all.
- If something is not clearly present, return null or [].
- Do not guess diseases aggressively.
- probable_status should be short.
"""
        fallback_data = {
            "symptoms": [],
            "duration": None,
            "body_part": [],
            "severity": None,
            "past_history": None,
            "observations": None,
            "probable_status": None,
            "treatment_advice": None,
            "notes": ""
        }

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    parsed = _safe_json_parse(response.text or "")
    if parsed is None:
        parsed = fallback_data.copy()

    if mode == "finance":
        payment_status_map = {
            "completed": ["payment done", "payment ho gaya", "payment aytu", "paid"],
            "pending": ["not paid", "payment nahi hua", "pending", "baaki"]
        }
        payment_mode_map = {
            "cash": ["cash", "nagadu"],
            "online": ["online", "upi", "gpay", "phonepe", "paytm", "bank transfer"]
        }

        if not parsed.get("payment_status"):
            for std, variants in payment_status_map.items():
                if any(v in clean_text for v in variants):
                    parsed["payment_status"] = std
                    break

        if not parsed.get("payment_mode"):
            for std, variants in payment_mode_map.items():
                if any(v in clean_text for v in variants):
                    parsed["payment_mode"] = std
                    break

        if not parsed.get("account_or_loan_confirmation"):
            if any(word in clean_text for word in ["loan", "account", "khata"]):
                parsed["account_or_loan_confirmation"] = "mentioned"

        if not parsed.get("amount"):
            amount_match = re.search(r"\b(\d{2,6})\b", clean_text)
            if amount_match:
                parsed["amount"] = amount_match.group(1)

        return {
            "payer_name": parsed.get("payer_name"),
            "payment_status": parsed.get("payment_status"),
            "amount": parsed.get("amount"),
            "payment_mode": parsed.get("payment_mode"),
            "payment_date": parsed.get("payment_date"),
            "reason_for_payment": parsed.get("reason_for_payment"),
            "account_or_loan_confirmation": parsed.get("account_or_loan_confirmation"),
            "executive_notes": parsed.get("executive_notes", "")
        }

    symptom_map = {
        "fever": ["fever", "bukhar", "bukhaar", "jwara", "jvara", "ಜ್ವರ"],
        "leg pain": ["leg pain", "pair dard", "kaalu novu", "ಕಾಲು ನೋವು"],
        "stomach pain": ["stomach pain", "pet dard", "hotte novu", "ಹೊಟ್ಟೆ ನೋವು", "abdominal pain"],
        "headache": ["headache", "sar dard", "tale novu", "ತಲೆ ನೋವು", "head pain"],
        "cough": ["cough", "khansi", "kemmnu", "ಕೆಮ್ಮು"],
        "cold": ["cold", "sardi", "sheet", "running nose", "ಜಲದು"],
        "body pain": ["body pain", "sharir dard", "mai dard", "deha novu", "ದೇಹ ನೋವು"],
        "weakness": ["weakness", "kamzori", "bala illa", "ದುರ್ಬಲತೆ"],
        "vomiting": ["vomiting", "ulti", "vanti", "ವಾಂತಿ"],
        "throat pain": ["throat pain", "gala dard", "gantalu novu", "ಗಂಟಲು ನೋವು"],
        "chest pain": ["chest pain", "seene me dard", "ede novu", "ಎದೆ ನೋವು"],
        "breathing problem": [
            "breathing problem",
            "saans ki dikkat",
            "usirata tondare",
            "ಉಸಿರಾಟ ತೊಂದರೆ",
            "difficulty breathing"
        ]
    }

    body_part_map = {
        "leg": ["leg", "pair", "kaalu", "ಕಾಲು"],
        "stomach": ["stomach", "pet", "hotte", "ಹೊಟ್ಟೆ"],
        "head": ["head", "sar", "tale", "ತಲೆ"],
        "throat": ["throat", "gala", "gantalu", "ಗಂಟಲು"],
        "chest": ["chest", "seena", "ede", "ಎದೆ"]
    }

    duration_map = {
        "1 day": ["1 day", "ek din", "ondu dina", "ಒಂದು ದಿನ"],
        "2 days": ["2 days", "do din", "yeradu dina", "ಎರಡು ದಿನ"],
        "3 days": ["3 days", "teen din", "mooru dina", "ಮೂರು ದಿನ"],
        "since yesterday": ["since yesterday", "kal se", "ninne inda", "ನಿನ್ನೆ ಇಂದ"],
        "1 week": ["1 week", "ek hafte se", "ondu vaara", "ಒಂದು ವಾರ"]
    }

    severity_map = {
        "mild": ["mild", "halka", "swalpa"],
        "severe": ["severe", "zyada", "tumba", "bahut"]
    }

    detected_symptoms = set(parsed.get("symptoms", []))
    detected_body_parts = set(parsed.get("body_part", []))
    detected_duration = parsed.get("duration", None)
    detected_severity = parsed.get("severity", None)

    for standard, variants in symptom_map.items():
        if any(word in clean_text for word in variants):
            detected_symptoms.add(standard)

    for standard, variants in body_part_map.items():
        if any(word in clean_text for word in variants):
            detected_body_parts.add(standard)

    if not detected_duration:
        for standard, variants in duration_map.items():
            if any(word in clean_text for word in variants):
                detected_duration = standard
                break

    if not detected_severity:
        for standard, variants in severity_map.items():
            if any(word in clean_text for word in variants):
                detected_severity = standard
                break

    return {
        "symptoms": sorted(list(detected_symptoms)),
        "duration": detected_duration,
        "body_part": sorted(list(detected_body_parts)),
        "severity": detected_severity,
        "past_history": parsed.get("past_history"),
        "observations": parsed.get("observations"),
        "probable_status": parsed.get("probable_status"),
        "treatment_advice": parsed.get("treatment_advice"),
        "notes": parsed.get("notes", "")
    }