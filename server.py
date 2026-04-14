"""
FGAI — Abuse Detection Server
Run: python server.py
Open: http://localhost:5000
"""

import os
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    print("  INFO: deep-translator not installed (fallback unavailable)")

app = Flask(__name__, static_folder=".")
CORS(app, origins="*")

# ── Config ────────────────────────────────────────────────────────────────────
SARVAM_API_KEY       = os.environ.get("SARVAM_API_KEY", "")
SARVAM_STT_URL       = "https://api.sarvam.ai/speech-to-text"
SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"

LABELS = {0: "Non-Offensive", 1: "Offensive"}

# ── Language codes ─────────────────────────────────────────────────────────────
SARVAM_LANG_CODES = {
    "hindi":     "hi-IN",
    "tamil":     "ta-IN",
    "kannada":   "kn-IN",
    "malayalam": "ml-IN",
}
GOOGLE_LANG_CODES = {
    "hindi":     "hi",
    "tamil":     "ta",
    "kannada":   "kn",
    "malayalam": "ml",
}

# ── Auto-detect model folder paths ────────────────────────────────────────────
def find_model_path(lang):
    candidates = [
        f"model/{lang}_en",
        f"model/{lang}",
        f"models/{lang}_en",
        f"models/{lang}",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

MODEL_PATHS = {}
for _lang in ["hindi", "tamil", "kannada", "malayalam"]:
    _path = find_model_path(_lang)
    if _path:
        MODEL_PATHS[_lang] = _path

# ── Load all found models at startup ──────────────────────────────────────────
print("\n" + "="*55)
print("  FGAI — Loading BERT models...")
print("="*55)

loaded_models = {}

for lang, path in MODEL_PATHS.items():
    try:
        print(f"  Loading {lang:12s} <- {path}")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model     = AutoModelForSequenceClassification.from_pretrained(path)
        model.eval()
        loaded_models[lang] = {"tokenizer": tokenizer, "model": model}
        print(f"  OK: {lang} loaded")
    except Exception as e:
        print(f"  FAIL: {lang}: {e}")

if not MODEL_PATHS:
    print("  WARNING: No model folders found!")

print("="*55)
print(f"  {len(loaded_models)}/4 models loaded")
print(f"  Paths found: {MODEL_PATHS}")
print("="*55 + "\n")


# ── Translation ────────────────────────────────────────────────────────────────
def translate_to_english(text: str, lang: str) -> str:
    """Translate to English — Sarvam API first, deep-translator as fallback."""

    # ── 1. Sarvam Translate API ───────────────────────────────────────────────
    try:
        src_code = SARVAM_LANG_CODES.get(lang, "hi-IN")
        resp = requests.post(
            SARVAM_TRANSLATE_URL,
            headers={
                "api-subscription-key": SARVAM_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "input":                text,
                "source_language_code": src_code,
                "target_language_code": "en-IN",
                "speaker_gender":       "Male",
                "mode":                 "formal",
                "model":                "mayura:v1",
                "enable_preprocessing": True,
            },
            timeout=15,
        )
        data = resp.json()
        print(f"  [translate] Sarvam response ({resp.status_code}): {data}")
        translated = data.get("translated_text") or data.get("translation") or ""
        if translated:
            return translated
    except Exception as e:
        print(f"  [translate] Sarvam error: {e}")

    # ── 2. Fallback: deep-translator ──────────────────────────────────────────
    if TRANSLATOR_AVAILABLE:
        try:
            src_code   = GOOGLE_LANG_CODES.get(lang, "auto")
            translated = GoogleTranslator(source=src_code, target="en").translate(text)
            print(f"  [translate] deep-translator result: {translated}")
            return translated or ""
        except Exception as e:
            print(f"  [translate] deep-translator error: {e}")

    return ""


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(text: str, lang: str) -> dict:
    if lang not in loaded_models:
        available = list(loaded_models.keys())
        return {"error": f"Model for '{lang}' not loaded. Available: {available}"}

    tokenizer = loaded_models[lang]["tokenizer"]
    model     = loaded_models[lang]["model"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)[0]
        pred_id = torch.argmax(probs).item()

    confidence = round(probs[pred_id].item() * 100, 2)
    label      = LABELS.get(pred_id, str(pred_id))

    return {
        "label":      label,
        "label_id":   pred_id,
        "confidence": confidence,
        "language":   lang,
        "text":       text,
        "probs": {
            "non_offensive": round(probs[0].item() * 100, 2),
            "offensive":     round(probs[1].item() * 100, 2),
        }
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/style.css")
def styles():
    return send_from_directory(".", "style.css")

@app.route("/health")
def health():
    return jsonify({
        "status":         "ok",
        "models_loaded":  list(loaded_models.keys()),
        "model_paths":    MODEL_PATHS,
        "sarvam_key_set": bool(SARVAM_API_KEY),
    })

@app.route("/analyze-text", methods=["POST"])
def analyze_text():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body sent"}), 400

    text = data.get("text", "").strip()
    lang = data.get("language", "").lower()

    if not text:
        return jsonify({"error": "Text is empty"}), 400
    if lang not in loaded_models:
        return jsonify({"error": f"Model for '{lang}' not loaded. Available: {list(loaded_models.keys())}"}), 400

    result = predict(text, lang)
    result["translation"] = translate_to_english(text, lang)
    return jsonify(result)

@app.route("/analyze-speech", methods=["POST"])
def analyze_speech():
    if "file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio    = request.files["file"]
    lang     = request.form.get("language", "").lower()
    stt_lang = request.form.get("stt_language", "hi-IN")

    # Normalise stt_lang — accept both "hi-IN" and legacy "HI-EN" style values
    stt_lang = stt_lang.lower().replace("-en", "-in")   # hi-en → hi-in
    parts = stt_lang.split("-")
    if len(parts) == 2:
        stt_lang = parts[0].lower() + "-" + parts[1].upper()  # correct case: hi-IN

    VALID_STT_CODES = {
        "hi-IN","ta-IN","kn-IN","ml-IN","bn-IN","mr-IN","od-IN","pa-IN",
        "te-IN","en-IN","gu-IN","as-IN","ur-IN","ne-IN","kok-IN","ks-IN",
        "sd-IN","sa-IN","sat-IN","mni-IN","brx-IN","mai-IN","doi-IN","unknown"
    }
    if stt_lang not in VALID_STT_CODES:
        stt_lang = SARVAM_LANG_CODES.get(lang, "hi-IN")
        print(f"  [STT] Invalid stt_language received, fell back to: {stt_lang}")

    if lang not in loaded_models:
        return jsonify({"error": f"Model for '{lang}' not loaded. Available: {list(loaded_models.keys())}"}), 400

    try:
        sarvam_resp = requests.post(
            SARVAM_STT_URL,
            headers={"api-subscription-key": SARVAM_API_KEY},
            files={"file": (audio.filename, audio.read(), audio.mimetype)},
            data={
                "model":             "saarika:v2.5",
                "language_code":     stt_lang,
                "with_timestamps":   "false",
                "with_disfluencies": "false",
            },
            timeout=60,
        )
        sarvam_data = sarvam_resp.json()
        if not sarvam_resp.ok:
            msg = sarvam_data.get("message") or sarvam_data.get("detail") or str(sarvam_data)
            return jsonify({"error": f"Sarvam STT error: {msg}"}), 502

        transcript = sarvam_data.get("transcript") or sarvam_data.get("text") or ""
        if not transcript:
            return jsonify({"error": "No speech detected in audio"}), 422

    except requests.exceptions.Timeout:
        return jsonify({"error": "Sarvam API timed out. Try a shorter clip."}), 504
    except Exception as e:
        return jsonify({"error": f"STT error: {str(e)}"}), 500

    result = predict(transcript, lang)
    result["transcript"]  = transcript
    result["translation"] = translate_to_english(transcript, lang)
    return jsonify(result)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    from waitress import serve
    print(f"  Production server starting on http://0.0.0.0:{port}")
    print(f"  Open in browser -> http://localhost:{port}\n")
    serve(app, host="0.0.0.0", port=port)
