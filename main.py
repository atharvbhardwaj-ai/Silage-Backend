import os
import io
import json
import base64
import hmac
import hashlib
from datetime import datetime, timezone
from typing import Optional, Union

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from pypdf import PdfReader
import qrcode

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not configured in environment variables.")

ai_client = genai.Client(api_key=api_key)

# Cryptographic signing secret
SIGNING_SECRET = os.getenv("DHARINI_SIGNING_SECRET", "DHARINI_NATIONAL_DAIRY_SECRET_KEY_2026")

app = FastAPI(
    title="DHARINI Silage & Cattle Feed Quality Inspection Engine",
    version="2.3.1",
    description="Multimodal Vision, Optical Litmus pH, Sensory Observation, NIR Parser, and Cryptographic QR Engine for SIH26111"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BATCH_REGISTRY = {}


def generate_signature(data: dict) -> str:
    """Generates an HMAC-SHA256 signature for offline cryptographic verification."""
    serialized_str = json.dumps(data, sort_keys=True)
    return hmac.new(
        SIGNING_SECRET.encode("utf-8"),
        serialized_str.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()[:16]


def verify_signature(data: dict, sig: str) -> bool:
    """Verifies HMAC signature offline without database lookup."""
    expected = generate_signature(data)
    return hmac.compare_digest(expected, sig)


def generate_qr_passport(payload: dict) -> str:
    """Serializes batch assessment summary, attaches HMAC signature, and emits Base64 QR."""
    sig = generate_signature(payload)
    passport_payload = {
        "data": payload,
        "sig": f"DHARINI-HMAC-{sig}"
    }

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=2,
    )
    qr.add_data(json.dumps(passport_payload))
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extracts raw text from an uploaded PDF file."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return full_text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")


def safe_float(val: Optional[Union[float, str]]) -> Optional[float]:
    """Gracefully handles nulls, strings with whitespace, or dashes without raising 500s."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip()
    if not val_str or val_str.lower() in ["null", "none", "undefined", "e.g.", "na", "n/a"]:
        return None
    # Handle ranges like "3.8 - 4.2" by taking the midpoint
    if "-" in val_str:
        parts = val_str.split("-")
        try:
            return round((float(parts[0].strip()) + float(parts[1].strip())) / 2.0, 2)
        except Exception:
            pass
    try:
        return float(val_str)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# OPTICAL LITMUS pH PARSER (Helper)
# ---------------------------------------------------------------------------
async def evaluate_litmus_image(litmus_file: UploadFile) -> dict:
    if litmus_file.content_type not in ["image/jpeg", "image/png", "image/webp", "image/jpg"]:
        return {
            "is_valid_strip": False,
            "error_type": "INVALID_FORMAT",
            "message": "Litmus test image must be a JPEG, PNG, or WEBP file."
        }

    try:
        contents = await litmus_file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return {
            "is_valid_strip": False,
            "error_type": "DECODE_ERROR",
            "message": f"Image decode failed: {str(e)}"
        }

    prompt = """
You are an expert chemical colorimetric sensor analyzing a photo of a pH litmus strip dipped in agricultural silage juice.

VALIDATION RULES:
1. Is this an actual paper test strip / indicator paper?
   If it is a person, food, animal, outdoor field, or random household object:
   Set is_valid_test_strip to false and error_reason to "NOT_A_TEST_STRIP".
2. Is it a full-range Universal pH indicator strip (showing graded tones from yellow/orange to olive/green/blue across a 1-14 scale)?
   If it is a simple binary red-to-blue or blue-to-red litmus paper:
   Set is_valid_test_strip to false and error_reason to "BINARY_LITMUS_DETECTED".

COLOR SPECTRUM CALIBRATION FOR SILAGE (Universal Indicator Paper):
- Deep Red / Magenta: pH ~ 2.0 - 3.0
- Orange / Warm Amber: pH ~ 3.8 - 4.2 (Optimal Lactic Silage)
- Yellowish-Brown / Dull Yellow: pH ~ 4.5 - 4.8
- Olive Green: pH ~ 5.2 - 5.8 (Spoiled / High Butyric Acid)
- Dark Green / Cyan / Blue: pH >= 6.5

Return ONLY valid JSON matching this schema:
{
    "is_valid_test_strip": true,
    "detected_ph": float,
    "observed_color": "Orange" | "Warm Amber" | "Dull Yellow" | "Olive Green" | "Blue",
    "confidence_score": float,
    "notes": "1-sentence description of strip appearance."
}

IF INVALID:
{
    "is_valid_test_strip": false,
    "detected_ph": null,
    "observed_color": "None",
    "error_reason": "NOT_A_TEST_STRIP" | "BINARY_LITMUS_DETECTED",
    "notes": "Clear instruction for the user."
}
"""

    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[pil_image, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception as e:
        return {
            "is_valid_strip": False,
            "error_type": "INFERENCE_ERROR",
            "message": f"Litmus AI analysis failed: {str(e)}"
        }


# ---------------------------------------------------------------------------
# 1. STANDALONE VISUAL AI SCAN (Tab 2)
# ---------------------------------------------------------------------------
@app.post("/api/v1/visual-scan")
async def visual_scan(
    file: UploadFile = File(...),
    silage_type: str = Form("Maize Silage")
):
    if file.content_type not in ["image/jpeg", "image/png", "image/webp", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file format. Upload JPEG, PNG, or WEBP.")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {str(e)}")

    prompt = f"""
You are an expert ICAR-NDRI veterinary dairy nutritionist inspecting a sample of: {silage_type}.

VALIDATION GATE:
Verify if this image depicts REAL, RAW agricultural fodder, silage bunker face, or cattle feed.
If the image shows cooked food, people, vehicles, animals, or unrelated household objects:
Set is_valid_feed to false.

IF VALID, EVALUATE VISUAL PARAMETERS:
1. Mould Growth (hyphae, Aspergillus sporulation).
2. Colour & Fermentation (bright olive-green vs caramelized dark brown/black rot).
3. Chop Length & Compaction (optimal 1.0 - 2.0 cm, too fine < 0.8 cm, coarse > 2.5 cm).
4. Grain Processing (cracked kernels vs intact whole grains).
5. Visible Spoilage & Foreign Material (slimy decay, stones, soil, weeds, plastic wrap).

Return ONLY valid JSON matching this schema:
{{
    "is_valid_feed": true,
    "silage_type": "{silage_type}",
    "mould_growth": {{
        "detected": true | false,
        "coverage_pct": float,
        "severity": "None" | "Low" | "Moderate" | "Severe"
    }},
    "colour_texture": {{
        "index_name": "Olive-Green" | "Golden-Brown" | "Dull Brown" | "Black/Rotten",
        "caramelization_risk": "None" | "Low" | "High"
    }},
    "chop_compaction": {{
        "verdict": "Optimal (1.0 - 2.0 cm)" | "Too Coarse (> 2.5 cm)" | "Too Fine (< 0.8 cm)",
        "rumen_fiber_adequacy": "Optimal" | "Sub-optimal" | "Acidosis Risk"
    }},
    "grain_processing": "Adequately Cracked" | "Intact/Uncracked" | "Not Applicable",
    "foreign_material": {{
        "detected": true | false,
        "details": "None" | "Soil/Clay" | "Plastic wrap traces" | "Weeds"
    }},
    "summary_observation": "1-2 sentence veterinary field observation."
}}

IF INVALID:
{{
    "is_valid_feed": false,
    "silage_type": "{silage_type}",
    "summary_observation": "The uploaded image does not depict raw silage or cattle feed."
}}
"""

    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[pil_image, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        result = json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visual inference failed: {str(e)}")

    if not result.get("is_valid_feed", False):
        return {
            "is_valid": False,
            "status": "Invalid Image",
            "message": result.get("summary_observation"),
            "recommendation": "Please upload a clear, focused photo of real silage or chopped fodder."
        }

    return {"is_valid": True, "analysis": result}


# ---------------------------------------------------------------------------
# 2. STANDALONE NIR REPORT PARSER (Tab 3)
# ---------------------------------------------------------------------------
@app.post("/api/v1/nir-report")
async def parse_nir_report(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Upload a PDF.")

    pdf_bytes = await file.read()
    raw_text = extract_text_from_pdf(pdf_bytes)

    if len(raw_text) < 30:
        raise HTTPException(status_code=400, detail="Uploaded PDF contains unreadable or empty text layers.")

    prompt = f"""
You are an expert dairy chemist parsing an official laboratory NIR Spectroscopy or forage analytical report.

DOCUMENT TEXT:
\"\"\"
{raw_text[:4000]}
\"\"\"

VALIDATION GATE:
Check if this document is actually an agricultural fodder, feed, or silage lab test report.
If it is a resume, invoice, bank statement, or unrelated PDF:
Set is_valid_nir_report to false.

IF VALID, EXTRACT KEY VALUES:
- Crude Protein (%)
- Moisture / Dry Matter (%)
- Crude Fiber (%)
- NDF (%)
- ADF (%)
- Energy Value (TDN % or ME MJ/kg)
- Aflatoxin (ppb)
- Fermentation pH

Return ONLY valid JSON matching this schema:
{{
    "is_valid_nir_report": true,
    "lab_name_or_header": "Detected lab name or 'Unknown'",
    "crude_protein_pct": float or null,
    "moisture_pct": float or null,
    "dry_matter_pct": float or null,
    "fiber_pct": float or null,
    "ndf_pct": float or null,
    "adf_pct": float or null,
    "energy_tdn_pct": float or null,
    "aflatoxin_ppb": float or null,
    "ph": float or null,
    "brief_lab_summary": "2-sentence plain English summary for a dairy farmer."
}}

IF INVALID:
{{
    "is_valid_nir_report": false,
    "brief_lab_summary": "Uploaded document is not a recognized agricultural feed or NIR report."
}}
"""

    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        parsed_nir = json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NIR parsing failed: {str(e)}")

    if not parsed_nir.get("is_valid_nir_report", False):
        return {
            "is_valid": False,
            "status": "Invalid Report",
            "message": parsed_nir.get("brief_lab_summary"),
            "recommendation": "Please upload an official laboratory NIR spectroscopy report PDF."
        }

    return {"is_valid": True, "data": parsed_nir}


# ---------------------------------------------------------------------------
# 3. TAB 1: NEW SILAGE ANALYSIS (Matches Updated Frontend UI)
# ---------------------------------------------------------------------------
@app.post("/api/v1/new-analysis")
async def new_silage_analysis(
    silage_image: UploadFile = File(...),
    litmus_image: Optional[UploadFile] = File(None),
    nir_pdf: Optional[UploadFile] = File(None),
    silage_type: str = Form("Maize Silage"),
    batch_id: Optional[str] = Form(None),
    # Radio options from frontend
    squeeze_test_observation: Optional[str] = Form("No drops, only wet palms"),
    # Additional Data inputs from UI
    sensor_moisture: Optional[str] = Form(None),
    sensor_ph: Optional[str] = Form(None),
    ph_reading: Optional[str] = Form(None),  # Alias for pH / Litmus Test input box
    sensor_temp_c: Optional[str] = Form(None),
    additional_observation: Optional[str] = Form(None),
    sample_date: Optional[str] = Form(None)
):
    # 1. Sanitize numeric inputs
    clean_moisture_override = safe_float(sensor_moisture)
    clean_ph = safe_float(sensor_ph) if sensor_ph else safe_float(ph_reading)
    clean_temp = safe_float(sensor_temp_c)

    # 2. Visual AI Evaluation on Silage Image
    visual_res = await visual_scan(file=silage_image, silage_type=silage_type)
    if not visual_res.get("is_valid", False):
        return visual_res

    v_data = visual_res["analysis"]

    # 3. NIR Report Parsing if PDF provided
    nir_data = None
    if nir_pdf and nir_pdf.filename:
        nir_res = await parse_nir_report(file=nir_pdf)
        if nir_res.get("is_valid", False):
            nir_data = nir_res["data"]

    # 4. Resolve Moisture % (Direct Sensor > Squeeze Radio > NIR)
    moisture_source = "squeeze_test"
    if clean_moisture_override is not None:
        final_moisture = clean_moisture_override
        moisture_source = "sensor_direct"
    elif nir_data and nir_data.get("moisture_pct") is not None:
        final_moisture = float(nir_data["moisture_pct"])
        moisture_source = "nir_lab"
    else:
        obs = (squeeze_test_observation or "").lower().strip()
        if "seeping" in obs:
            final_moisture = 72.0
        elif "crumble" in obs or "minimal" in obs:
            final_moisture = 56.0
        else:
            final_moisture = 66.5

    # 5. Resolve pH (Priority: Sensor/Manual Input > Litmus Image Scan > NIR Lab > AI Baseline)
    litmus_analysis_result = None
    ph_source = "visual_estimate"

    if clean_ph is not None:
        final_ph = clean_ph
        ph_source = "sensor_or_manual_input"
    elif litmus_image and litmus_image.filename:
        litmus_res = await evaluate_litmus_image(litmus_file=litmus_image)
        litmus_analysis_result = litmus_res

        if not litmus_res.get("is_valid_test_strip", False):
            reason = litmus_res.get("error_reason", "INVALID")
            if reason == "BINARY_LITMUS_DETECTED":
                return {
                    "is_valid": False,
                    "status": "Incompatible Litmus Paper",
                    "message": "Binary red/blue litmus paper detected. It cannot provide numeric pH precision.",
                    "recommendation": "Please use ₹2 universal multi-color indicator paper (pH 1–14) for accurate silage testing."
                }
            else:
                return {
                    "is_valid": False,
                    "status": "Invalid Litmus Image",
                    "message": litmus_res.get("notes", "Uploaded photo does not contain a recognizable pH indicator strip."),
                    "recommendation": "Upload a close-up photo of the universal pH paper strip dipped in silage runoff."
                }
        else:
            final_ph = float(litmus_res["detected_ph"])
            ph_source = "optical_universal_litmus"
    elif nir_data and nir_data.get("ph") is not None:
        final_ph = float(nir_data["ph"])
        ph_source = "nir_lab"
    else:
        final_ph = 5.8 if v_data["mould_growth"]["detected"] else 4.1

    # 6. Resolve Protein & Fiber
    if nir_data and nir_data.get("crude_protein_pct") is not None:
        final_protein = float(nir_data["crude_protein_pct"])
    else:
        final_protein = 8.0 if v_data["mould_growth"]["detected"] else 9.2

    if nir_data and nir_data.get("fiber_pct") is not None:
        final_fiber = float(nir_data["fiber_pct"])
    else:
        final_fiber = 28.0 if "Too Coarse" in v_data["chop_compaction"]["verdict"] else 24.0

    # 7. Aflatoxin ppb Metric
    mould_detected = v_data["mould_growth"]["detected"]
    mould_pct = float(v_data["mould_growth"]["coverage_pct"])
    if nir_data and nir_data.get("aflatoxin_ppb") is not None:
        final_aflatoxin = float(nir_data["aflatoxin_ppb"])
    else:
        final_aflatoxin = round(12.0 + (mould_pct * 0.8), 1) if mould_detected else 5.0

    # 8. Check Additional Sensory Observations
    sensory_notes = (additional_observation or "").strip()

    # 9. Quality Classification
    if final_aflatoxin >= 20.0 or v_data["foreign_material"]["detected"]:
        quality_status = "Unsafe"
    elif final_ph >= 5.2 or final_moisture > 71.0 or (clean_temp is not None and clean_temp > 35.0):
        quality_status = "Poor"
    elif mould_detected or final_ph > 4.4 or final_moisture < 60.0:
        quality_status = "Needs Attention"
    else:
        quality_status = "Good"

    # 10. Auto-assign Batch ID if not typed
    current_date_str = sample_date or datetime.now(timezone.utc).strftime("%d %b %Y")
    assigned_batch_id = batch_id.strip() if (batch_id and batch_id.strip() and batch_id.strip() != "SIL-2026-0043") else f"SIL-2026-{datetime.now(timezone.utc).strftime('%H%M%S')}"

    # Advisories
    if quality_status == "Good":
        feed_advisory = "Optimal lactic preservation. Excellent energy balance for lactating cattle."
        storage_advisory = "Maintain uniform face cutting at silage bunker to limit secondary aerobic exposure."
    elif quality_status == "Needs Attention":
        feed_advisory = "Discard surface mouldy crust before feeding. Blend with dry fodder buffer."
        storage_advisory = "Check edge plastic seal for punctures and reseal tightly to limit clostridial growth."
    elif quality_status == "Poor":
        feed_advisory = "Elevated pH or moisture imbalance detected. Risk of reduced intake; add buffer."
        storage_advisory = "Aerobic heating / clostridial risk detected. Accelerate bunker feed-out rate."
    else:
        feed_advisory = "DO NOT FEED. Severe toxin/adulteration hazard. Risk of milk aflatoxin contamination."
        storage_advisory = "Quarantine contaminated section. Core sample before considering bulk disposal."

    # 11. Generate Signed QR Passport
    qr_payload = {
        "bid": assigned_batch_id,
        "type": silage_type,
        "date": current_date_str,
        "status": quality_status,
        "moist_pct": final_moisture,
        "protein_pct": final_protein,
        "fiber_pct": final_fiber,
        "ph": final_ph,
        "afla_ppb": final_aflatoxin,
        "mould": mould_detected
    }
    qr_base64 = generate_qr_passport(qr_payload)

    # Save to memory registry
    BATCH_REGISTRY[assigned_batch_id] = qr_payload

    return {
        "is_valid": True,
        "batch_id": assigned_batch_id,
        "silage_type": silage_type,
        "sample_date": current_date_str,
        "quality_status": quality_status,
        "field_telemetry": {
            "temperature_c": clean_temp,
            "sensory_notes": sensory_notes or "None provided"
        },
        "source_attribution": {
            "moisture_source": moisture_source,
            "ph_source": ph_source
        },
        "metrics": {
            "moisture_pct": final_moisture,
            "protein_pct": final_protein,
            "fiber_pct": final_fiber,
            "aflatoxin_ppb": final_aflatoxin,
            "ph": final_ph
        },
        "visual_summary": v_data,
        "litmus_summary": litmus_analysis_result,
        "nir_summary": nir_data,
        "advisory": {
            "feeding": feed_advisory,
            "storage": storage_advisory
        },
        "qr_passport_base64": qr_base64
    }


# ---------------------------------------------------------------------------
# 4. TAB 4: QR SCAN & BATCH AUTHENTICATION
# ---------------------------------------------------------------------------
class QRVerifyRequest(BaseModel):
    qr_string: Optional[str] = None
    batch_id: Optional[str] = None


@app.post("/api/v1/verify-qr")
async def verify_qr(payload: QRVerifyRequest):
    if payload.qr_string:
        try:
            parsed = json.loads(payload.qr_string)
            data = parsed.get("data", {})
            sig_raw = parsed.get("sig", "")

            prefix = "DHARINI-HMAC-"
            if not sig_raw.startswith(prefix):
                return {
                    "verified": False,
                    "status": "Invalid Signature Format / Non-DHARINI QR",
                    "batch_data": None
                }

            sig = sig_raw[len(prefix):]
            is_authentic = verify_signature(data, sig)

            return {
                "verified": is_authentic,
                "status": "Authentic Certified Batch (Cryptographically Verified)" if is_authentic else "Warning: Tampered Batch / Forged Signature",
                "batch_data": data if is_authentic else None
            }
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid DHARINI QR payload format.")

    if payload.batch_id:
        batch_clean = payload.batch_id.strip()
        if batch_clean in BATCH_REGISTRY:
            return {
                "verified": True,
                "status": "Authentic Certified Batch (Found in Registry)",
                "batch_data": BATCH_REGISTRY[batch_clean]
            }
        if batch_clean == "SIL-2026-0043":
            demo_data = {
                "bid": "SIL-2026-0043",
                "type": "Maize Silage",
                "date": "30 Sep 2026",
                "status": "Good",
                "moist_pct": 66.5,
                "protein_pct": 9.4,
                "fiber_pct": 24.0,
                "ph": 4.05,
                "afla_ppb": 4.5,
                "mould": False
            }
            return {
                "verified": True,
                "status": "Authentic Certified Batch (Verified Demo Batch)",
                "batch_data": demo_data
            }
        return {
            "verified": False,
            "status": "Unrecognized Batch ID",
            "batch_data": None
        }

    raise HTTPException(status_code=400, detail="Must provide either qr_string or batch_id.")