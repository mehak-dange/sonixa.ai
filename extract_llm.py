import json
import re
from google import genai

client = genai.Client()


def extract_data(input_text):
    clean_text = re.sub(r"\s+", " ", input_text.strip().lower())

    prompt = f"""
You are a strict healthcare information extraction engine.

Your task is to extract structured symptom information from multilingual Indian patient speech.
The input may be in romanized Hindi, English, Kannada-English mix, or code-mixed text.

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
  "severity": null
}}

Important normalization rules:
- bukhar, bukhaar, fever, jwara, jvara -> fever
- pair dard, leg pain, kaalu novu -> leg pain
- pet dard, stomach pain, abdominal pain -> stomach pain
- sar dard, headache, head pain -> headache
- khansi, cough -> cough
- sardi, cold, running nose -> cold
- body pain, sharir dard, mai dard -> body pain
- weakness, kamzori -> weakness
- ulti, vomiting -> vomiting
- gala dard, throat pain -> throat pain
- chest pain, seene me dard -> chest pain
- saans ki dikkat, breathing issue, difficulty breathing -> breathing problem

Duration normalization:
- do din, 2 din, 2 days, yeradu dina -> 2 days
- ek din, 1 din, 1 day, ondu dina -> 1 day
- teen din, 3 din, 3 days, mooru dina -> 3 days
- kal se, since yesterday -> since yesterday
- ek hafte se, 1 week, one week -> 1 week

Body part normalization:
- pair, leg, kaalu -> leg
- pet, stomach -> stomach
- sar, head -> head
- gala, throat -> throat
- seena, chest -> chest

Severity normalization:
- halka, mild -> mild
- zyada, severe, bahut -> severe

Rules:
- Always extract symptoms if any symptom-like phrase is present.
- Always normalize symptom names into simple English labels.
- If multiple symptoms are present, include all of them.
- If duration is clearly mentioned, normalize it.
- If body part is clearly mentioned, include it.
- If severity is not mentioned, return null.
- If nothing is found, return empty list [].
- Do not guess diseases.
- Focus only on symptoms, duration, body_part, severity.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    raw_output = response.text.strip()
    raw_output = raw_output.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        data = {
            "symptoms": [],
            "duration": None,
            "body_part": [],
            "severity": None
        }

    # -------------------------
    # Fallback normalization map
    # -------------------------
    symptom_map = {
        "fever": ["bukhar", "bukhaar", "fever", "jwara", "jvara"],
        "leg pain": ["pair dard", "leg pain", "kaalu novu"],
        "stomach pain": ["pet dard", "stomach pain", "abdominal pain"],
        "headache": ["sar dard", "headache", "head pain"],
        "cough": ["khansi", "cough"],
        "cold": ["sardi", "cold", "running nose"],
        "body pain": ["body pain", "sharir dard", "mai dard"],
        "weakness": ["weakness", "kamzori"],
        "vomiting": ["ulti", "vomiting"],
        "throat pain": ["gala dard", "throat pain"],
        "chest pain": ["chest pain", "seene me dard"],
        "breathing problem": ["saans ki dikkat", "breathing issue", "difficulty breathing"]
    }

    body_part_map = {
        "leg": ["pair", "leg", "kaalu"],
        "stomach": ["pet", "stomach"],
        "head": ["sar", "head"],
        "throat": ["gala", "throat"],
        "chest": ["seena", "chest"]
    }

    duration_map = {
        "1 day": ["ek din", "1 din", "1 day", "ondu dina"],
        "2 days": ["do din", "2 din", "2 days", "yeradu dina"],
        "3 days": ["teen din", "3 din", "3 days", "mooru dina"],
        "since yesterday": ["kal se", "since yesterday"],
        "1 week": ["ek hafte se", "1 week", "one week"]
    }

    severity_map = {
        "mild": ["halka", "mild"],
        "severe": ["zyada", "severe", "bahut"]
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

    final_data = {
        "symptoms": sorted(list(detected_symptoms)),
        "duration": detected_duration,
        "body_part": sorted(list(detected_body_parts)),
        "severity": detected_severity
    }

    return final_data