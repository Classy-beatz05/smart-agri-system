"""
app.py — Multilingual Smart Agriculture Recommendation System
Flask backend: crop + fertilizer prediction, multilingual API, TTS support.

Usage:
    python app.py
    Open: http://localhost:5000
"""

from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import os
import json
import warnings
import time
warnings.filterwarnings("ignore")


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load(name):
    p = os.path.join(MODEL_DIR, f"{name}.pkl")
    with open(p, "rb") as f:
        return pickle.load(f)


crop_model = load("crop_model")
crop_ms = load("crop_ms")
crop_sc = load("crop_sc")
fert_model = load("fert_model")
fert_ms = load("fert_ms")
fert_sc = load("fert_sc")
le_city = load("le_city")
le_crop = load("le_crop")

with open(
    os.path.join(MODEL_DIR, "meta.json"),
    "r",
    encoding="utf-8"
) as f:
    META = json.load(f)

# ── Static data ───────────────────────────────────────────────────────────────
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

CROP_INFO = {
    "apple":       {"emoji": "🍎", "color": "#e74c3c", "season": "Rabi",   "yield": "15–20 t/ha"},
    "banana":      {"emoji": "🍌", "color": "#f39c12", "season": "Annual", "yield": "25–40 t/ha"},
    "blackgram":   {"emoji": "🫘", "color": "#5d4037", "season": "Kharif", "yield": "0.8–1.2 t/ha"},
    "chickpea":    {"emoji": "🫘", "color": "#e67e22", "season": "Rabi",   "yield": "1.0–1.5 t/ha"},
    "coconut":     {"emoji": "🥥", "color": "#27ae60", "season": "Annual", "yield": "80–100 nuts/tree"},
    "coffee":      {"emoji": "☕", "color": "#6f4e37", "season": "Annual", "yield": "0.8–1.5 t/ha"},
    "cotton":      {"emoji": "🌿", "color": "#7f8c8d", "season": "Kharif", "yield": "1.5–3.0 t/ha"},
    "grapes":      {"emoji": "🍇", "color": "#8e44ad", "season": "Annual", "yield": "15–25 t/ha"},
    "jute":        {"emoji": "🌿", "color": "#2ecc71", "season": "Kharif", "yield": "2.0–3.0 t/ha"},
    "kidneybeans": {"emoji": "🫘", "color": "#c0392b", "season": "Rabi",   "yield": "1.0–1.8 t/ha"},
    "lentil":      {"emoji": "🫘", "color": "#d4ac0d", "season": "Rabi",   "yield": "0.8–1.5 t/ha"},
    "maize":       {"emoji": "🌽", "color": "#f1c40f", "season": "Kharif", "yield": "3.5–6.0 t/ha"},
    "mango":       {"emoji": "🥭", "color": "#e67e22", "season": "Annual", "yield": "10–20 t/ha"},
    "mothbeans":   {"emoji": "🫘", "color": "#a04000", "season": "Kharif", "yield": "0.5–1.0 t/ha"},
    "mungbean":    {"emoji": "🫘", "color": "#1e8449", "season": "Kharif", "yield": "0.8–1.2 t/ha"},
    "muskmelon":   {"emoji": "🍈", "color": "#f0b27a", "season": "Zaid",   "yield": "10–20 t/ha"},
    "orange":      {"emoji": "🍊", "color": "#e67e22", "season": "Annual", "yield": "10–20 t/ha"},
    "papaya":      {"emoji": "🍈", "color": "#f39c12", "season": "Annual", "yield": "40–60 t/ha"},
    "pigeonpeas":  {"emoji": "🫘", "color": "#ca6f1e", "season": "Kharif", "yield": "1.0–2.0 t/ha"},
    "pomegranate": {"emoji": "🍎", "color": "#cb4335", "season": "Annual", "yield": "10–15 t/ha"},
    "rice":        {"emoji": "🌾", "color": "#27ae60", "season": "Kharif", "yield": "3.0–5.0 t/ha"},
    "watermelon":  {"emoji": "🍉", "color": "#e74c3c", "season": "Zaid",   "yield": "20–40 t/ha"},
}

FERT_INFO = {
    "Urea":         {"full": "Urea (46% N)", "npk": "46-0-0", "icon": "🟢",
                     "desc": "High-nitrogen; promotes rapid leaf/stem growth.", "apply": "Split 2–3 times during growth"},
    "DAP":          {"full": "Diammonium Phosphate", "npk": "18-46-0", "icon": "🔵",
                     "desc": "High N+P; boosts root & seedling establishment.", "apply": "At sowing, mix into soil"},
    "MOP":          {"full": "Muriate of Potash", "npk": "0-0-60", "icon": "🔴",
                     "desc": "Rich potassium; improves fruit quality & immunity.", "apply": "Basal before planting"},
    "SSP":          {"full": "Single Super Phosphate", "npk": "0-16-0+12%S", "icon": "🟠",
                     "desc": "P+S together; great for legumes & oilseeds.", "apply": "Incorporate before sowing"},
    "NPK 10-26-26": {"full": "NPK Complex 10-26-26", "npk": "10-26-26", "icon": "🟣",
                     "desc": "High P+K; perfect for fruiting & tubers.", "apply": "Basal, ideal for transplanted crops"},
    "NPK 12-32-16": {"full": "NPK Complex 12-32-16", "npk": "12-32-16", "icon": "⚫",
                     "desc": "Very high P; excellent for germination in P-deficient soils.", "apply": "Seed-row placement at sowing"},
}

# Advice rules (simple deterministic)


def generate_advice(N, P, K, ph, temp, humidity, rainfall, crop):
    tips = []
    if N < 30:
        tips.append("low_N")
    if P < 20:
        tips.append("low_P")
    if K < 20:
        tips.append("low_K")
    if ph < 5.5:
        tips.append("low_ph")
    if ph > 7.8:
        tips.append("high_ph")
    if humidity > 85:
        tips.append("high_humidity")
    if rainfall < 30:
        tips.append("low_rain")
    return tips


# ── Translations ──────────────────────────────────────────────────────────────
with open(
    os.path.join(BASE_DIR, "translations", "all.json"),
    "r",
    encoding="utf-8"
) as f:
    TRANSLATIONS = json.load(f)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html",
                           cities=META["cities"],
                           months=MONTHS,
                           meta=META,
                           translations=json.dumps(TRANSLATIONS),
                           )


@app.route("/predict", methods=["POST"])
def predict():
    t_start = time.time()
    try:
        def flt(k):
            v = request.json.get(k, "")
            if v == "" or v is None:
                raise ValueError(f"'{k}' is required.")
            return float(v)

        N = flt("N")
        P = flt("P")
        K = flt("K")
        temp = flt("temperature")
        humidity = flt("humidity")
        ph = flt("ph")
        rainfall = flt("rainfall")
        ec = flt("ec")
        month = int(flt("month"))
        city = str(request.json.get("city", "Mumbai")).strip().title()
        lang = str(request.json.get("lang", "en"))

        if city not in META["cities"]:
            city = "Mumbai"
        if not (1 <= month <= 12):
            raise ValueError("Month must be 1–12.")

        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        city_enc = int(le_city.transform([city])[0])

        # ── Crop prediction ────────────────────────────────────────────────
        crop_feat = np.array(
            [[N, P, K, temp, humidity, ph, rainfall, ec, month_sin, month_cos, city_enc]])
        crop_scaled = crop_sc.transform(crop_ms.transform(crop_feat))
        probs = crop_model.predict_proba(crop_scaled)[0]
        best_idx = int(np.argmax(probs))
        crop_name = str(crop_model.classes_[best_idx])
        crop_conf = round(float(probs[best_idx]) * 100, 1)

        info = CROP_INFO.get(
            crop_name, {"emoji": "🌱", "color": "#27ae60", "season": "—", "yield": "—"})

        # ── Fertilizer prediction ──────────────────────────────────────────
        crop_enc_val = int(le_crop.transform([crop_name])[0])
        fert_feat = np.array(
            [[N, P, K, temp, humidity, ph, rainfall, crop_enc_val]])
        fert_scaled = fert_sc.transform(fert_ms.transform(fert_feat))
        fert_probs = fert_model.predict_proba(fert_scaled)[0]
        fert_idx = int(np.argmax(fert_probs))
        fert_name = str(fert_model.classes_[fert_idx])
        fert_conf = round(float(fert_probs[fert_idx]) * 100, 1)
        fert_info = FERT_INFO.get(fert_name, {
                                  "full": fert_name, "npk": "—", "icon": "🟤", "desc": "Balanced nutrition.", "apply": "—"})

        # ── Advice ────────────────────────────────────────────────────────
        advice_flags = generate_advice(
            N, P, K, ph, temp, humidity, rainfall, crop_name)

        elapsed = round((time.time() - t_start) * 1000, 1)

        return jsonify({
            "success": True,
            "prediction_ms": elapsed,
            "crop": {
                "name":       crop_name,
                "display":    crop_name.capitalize(),
                "emoji":      info["emoji"],
                "color":      info["color"],
                "confidence": crop_conf,
                "season":     info["season"],
                "yield":      info["yield"],
            },
            "fertilizer": {
                "name":       fert_name,
                "full":       fert_info["full"],
                "npk":        fert_info["npk"],
                "icon":       fert_info["icon"],
                "desc":       fert_info["desc"],
                "apply":      fert_info["apply"],
                "confidence": fert_conf,
            },
            "advice_flags": advice_flags,
            "month_name":   MONTHS[month - 1],
            "city":         city,
        })

    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/translations")
def get_translations():
    return jsonify(TRANSLATIONS)


@app.route("/api/meta")
def get_meta():
    return jsonify(META)


if __name__ == "__main__":
    print("🌾 Multilingual Smart Agriculture System")
    print("   Open: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
