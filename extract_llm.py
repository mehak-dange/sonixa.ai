import json
import re
from google import genai

client = genai.Client()


def _healthcare_prompt(clean_text: str) -> str:
    return f"""
You are a strict healthcare information extraction engine.

Extract structured healthcare information from multilingual Indian patient speech.
Input may be in Kannada, Hindi, English, or code-mixed text.

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
  "past_history": "",
  "observations": "",
  "probable_status": "",
  "treatment_advice": "",
  "notes": ""
}}

Normalization rules:
- fever, bukhar, bukhaar, jwara, jvara -> fever
- leg pain, pair dard, kaalu novu -> leg pain
- stomach pain, pet dard, hotte novu, abdominal pain -> stomach pain
- headache, sar dard, tale novu, head pain -> headache
- cough, khansi, kemmnu -> cough
- cold, sardi, sheet, running nose -> cold
- body pain, sharir dard, mai dard, deha novu -> body pain
- weakness, kamzori, bala illa -> weakness
- vomiting, ulti, vanti -> vomiting
- throat pain, gala dard, gantalu novu -> throat pain
- chest pain, seene me dard, ede novu -> chest pain
- breathing problem, saans ki dikkat, usirata tondare, difficulty breathing -> breathing problem

Duration normalization:
- 1 day, ek din, ondu dina -> 1 day
- 2 days, do din, yeradu dina -> 2 days
- 3 days, teen din, mooru dina -> 3 days
- since yesterday, kal se, ninne inda -> since yesterday
- 1 week, ek hafte se, ondu vaara -> 1 week

Body part normalization:
- leg, pair, kaalu -> leg
- stomach, pet, hotte -> stomach
- head, sar, tale -> head
- throat, gala, gantalu -> throat
- chest, seena, ede -> chest

Severity normalization:
- mild, halka, swalpa -> mild
- severe, zyada, tumba, bahut -> severe

Rules:
- Extract only information present in the text.
- If missing, use null for scalar fields, [] for arrays, "" for text fields.
- Do not guess diseases aggressively.
- probable_status can be a soft label like "possible fever complaint" if strongly supported.
"""


def _finance_prompt(clean_text: str) -> str:
    return f"""
You are a strict finance/survey information extraction engine.

Extract structured finance and verification information from multilingual Indian speech.
Input may be in Kannada, Hindi, English, or code-mixed text.

Input:
"{clean_text}"

Return ONLY valid JSON.
Do NOT use markdown.
Do NOT use ```json.
Do NOT add explanation.

Use exactly this schema:
{{
  "payer_name": "",
  "payment_status": "",
  "amount": "",
  "payment_mode": "",
  "payment_date": "",
  "reason_for_payment": "",
  "account_or_loan_confirmation": "",
  "identity_verification": "",
  "executive_notes": ""
}}

Normalization hints:
- paid, payment done, jama kiya -> paid
- pending, not paid, baki -> pending
- cash, by cash -> cash
- online, upi, gpay, phonepe -> online/upi
- loan, account, emi -> preserve as spoken meaning
- If amount is spoken, preserve it as clean text like "500 rupees"

Rules:
- Extract only what is present.
- If unknown, use empty string.
- Do not invent values.
"""


def _safe_json_response(prompt: str) -> dict:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    raw_output = response.text.strip()
    raw_output = raw_output.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {}


def extract_data(input_text: str, mode: str = "healthcare") -> dict:
    clean_text = re.sub(r"\s+", " ", input_text.strip().lower())

    if mode == "finance":
        data = _safe_json_response(_finance_prompt(clean_text))
        return {
            "payer_name": data.get("payer_name", ""),
            "payment_status": data.get("payment_status", ""),
            "amount": data.get("amount", ""),
            "payment_mode": data.get("payment_mode", ""),
            "payment_date": data.get("payment_date", ""),
            "reason_for_payment": data.get("reason_for_payment", ""),
            "account_or_loan_confirmation": data.get("account_or_loan_confirmation", ""),
            "identity_verification": data.get("identity_verification", ""),
            "executive_notes": data.get("executive_notes", "")
        }

    # default healthcare
    data = _safe_json_response(_healthcare_prompt(clean_text))

    symptom_map = {
        "fever": ["fever", "bukhar", "bukhaar", "jwara", "jvara"],
        "leg pain": ["leg pain", "pair dard", "kaalu novu"],
        "stomach pain": ["stomach pain", "pet dard", "hotte novu", "abdominal pain"],
        "headache": ["headache", "sar dard", "tale novu", "head pain"],
        "cough": ["cough", "khansi", "kemmnu"],
        "cold": ["cold", "sardi", "sheet", "running nose"],
        "body pain": ["body pain", "sharir dard", "mai dard", "deha novu"],
        "weakness": ["weakness", "kamzori", "bala illa"],
        "vomiting": ["vomiting", "ulti", "vanti"],
        "throat pain": ["throat pain", "gala dard", "gantalu novu"],
        "chest pain": ["chest pain", "seene me dard", "ede novu"],
        "breathing problem": ["breathing problem", "saans ki dikkat", "usirata tondare", "difficulty breathing"]
    }

    body_part_map = {
        "leg": ["leg", "pair", "kaalu"],
        "stomach": ["stomach", "pet", "hotte"],
        "head": ["head", "sar", "tale"],
        "throat": ["throat", "gala", "gantalu"],
        "chest": ["chest", "seena", "ede"]
    }

    duration_map = {
        "1 day": ["1 day", "ek din", "ondu dina"],
        "2 days": ["2 days", "do din", "yeradu dina"],
        "3 days": ["3 days", "teen din", "mooru dina"],
        "since yesterday": ["since yesterday", "kal se", "ninne inda"],
        "1 week": ["1 week", "ek hafte se", "ondu vaara"]
    }

    severity_map = {
        "mild": ["mild", "halka", "swalpa"],
        "severe": ["severe", "zyada", "tumba", "bahut"]
    }

    detected_symptoms = set(data.get("symptoms", []))
    detected_body_parts = set(data.get("body_part", []))
    detected_duration = data.get("duration", None)
    detected_severity = data.get("severity", None)

    for standard, variants in symptom_map.items():
        for word in variants:
            if word in clean_text:
                detected_symptoms.add(standard)

    for standard, variants in body_part_map.items():
        for word in variants:
            if word in clean_text:
                detected_body_parts.add(standard)

    if not detected_duration:
        for standard, variants in duration_map.items():
            for word in variants:
                if word in clean_text:
                    detected_duration = standard
                    break
            if detected_duration:
                break

    if not detected_severity:
        for standard, variants in severity_map.items():
            for word in variants:
                if word in clean_text:
                    detected_severity = standard
                    break
            if detected_severity:
                break

    return {
        "symptoms": sorted(list(detected_symptoms)),
        "duration": detected_duration,
        "body_part": sorted(list(detected_body_parts)),
        "severity": detected_severity,
        "past_history": data.get("past_history", ""),
        "observations": data.get("observations", ""),
        "probable_status": data.get("probable_status", ""),
        "treatment_advice": data.get("treatment_advice", ""),
        "notes": data.get("notes", "")
    }