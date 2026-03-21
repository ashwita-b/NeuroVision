"""
app.py  --  NeuroVision: Unified Detection + Chatbot System
==========================================================
All routing lives here. Heavy logic is delegated to:
  • detection.py       -- model inference
  • gradcam.py         -- Grad-CAM + analysis JSON
  • chatbot_engine.py  -- NLP / RAG chatbot logic
"""

from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, send_from_directory)
import os, json, re, hashlib
import numpy as np

app = Flask(__name__)
app.secret_key = "neurovision_secret_key_2024_prod_12345"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS']  = ['.jpg', '.jpeg', '.png', '.gif']
app.config['SESSION_PERMANENT']  = False

# ── User Database ──────────────────────────────────────────────────────────────
USERS_FILE = 'users.json'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def hash_password(p):  return hashlib.sha256(p.encode()).hexdigest()
def validate_email(e): return re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', e) is not None
def validate_password(p):
    if len(p) < 6: return False, "Password must be at least 6 characters"
    return True, ""

# ── Lazy-import modules ────────────────────────────────────────────────────────
def _get_detection():
    import detection as det
    return det

def _get_gradcam():
    import gradcam as gc
    return gc

def _get_chatbot():
    import chatbot_engine as cb
    return cb

# ── Auth helpers ───────────────────────────────────────────────────────────────
def _require_login():
    return 'user' not in session

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES -- Auth
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if 'user' in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        if 'email' in request.form:          # ── Register ──
            name  = request.form.get("fullname", "").strip()
            email = request.form.get("email", "").strip().lower()
            pw    = request.form.get("password", "").strip()
            cpw   = request.form.get("confirm_password", "").strip()
            terms = request.form.get("terms")

            if not all([name, email, pw, cpw]):
                return render_template("login.html", register_error="All fields are required")
            if not validate_email(email):
                return render_template("login.html", register_error="Invalid email address")
            ok, msg = validate_password(pw)
            if not ok:
                return render_template("login.html", register_error=msg)
            if pw != cpw:
                return render_template("login.html", register_error="Passwords do not match")
            if not terms:
                return render_template("login.html", register_error="You must agree to the Terms & Conditions")

            users = load_users()
            if email in users:
                return render_template("login.html", register_error="Email already registered")
            users[email] = {'name': name, 'email': email,
                            'password_hash': hash_password(pw),
                            'created_at': str(np.datetime64('now'))}
            save_users(users)
            return render_template("login.html",
                                   success_message="Registration successful! Please login.")

        else:                                # ── Login ──
            email = request.form.get("username", "").strip().lower()
            pw    = request.form.get("password", "").strip()
            if not email or not pw:
                return render_template("login.html", error="Please enter both email and password")
            if not validate_email(email):
                return render_template("login.html", error="Invalid email address")
            users = load_users()
            if email in users and users[email]['password_hash'] == hash_password(pw):
                session['user'] = email
                session['name'] = users[email]['name']
                return redirect(url_for("home"))
            return render_template("login.html", error="Invalid email or password")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES -- Pages
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/home")
def home():
    if _require_login(): return redirect(url_for("login"))
    return render_template("home.html")

@app.route("/detection")
def detection():
    if _require_login(): return redirect(url_for("login"))
    return render_template("detection.html")

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES -- Detection API
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/predict", methods=["POST"])
def predict():
    if _require_login():
        return jsonify({"error": "Not authenticated"}), 401

    det = _get_detection()
    gc  = _get_gradcam()

    if det.get_model() is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data   = request.get_json()
        images = (data or {}).get("image", [])
        if not images:
            return jsonify({"error": "No image data provided"}), 400

        results = []
        for img_str in images[:5]:
            img_array  = det.decode_base64_image(img_str)
            det_result = det.predict(img_array)

            if "error" in det_result:
                return jsonify({"error": det_result["error"]}), 500

            tumor_type = det_result["tumor_type"]
            class_idx  = det_result["class_index"]
            confidence = float(det_result["highest_confidence"].replace("%", ""))

            # ── Grad-CAM ──────────────────────────────────────────────────────
            gradcam_data = None
            if tumor_type != "No Tumor":
                analysis = gc.generate_gradcam_analysis(
                    img_array, tumor_type, class_idx, confidence
                )
                if analysis.get("generated"):
                    gradcam_data = {
                        "overlay":             analysis.get("overlay_base64"),
                        "heatmap":             analysis.get("heatmap_base64"),
                        "activation_analysis": analysis.get("activation_analysis"),
                        "conv_layer_used":     analysis.get("conv_layer_used"),
                        "generated":           True,
                    }

            # ── Persist JSON for chatbot ───────────────────────────────────────
            det.save_result_json(det_result, user=session.get("user"))
            if gradcam_data:
                gc.save_analysis_json(analysis)

            # ── Store latest scan timestamp in session ─────────────────────────
            # chatbot_engine uses this to match the correct entry in the JSON files
            scan_timestamp = det_result.get("timestamp", "")
            if scan_timestamp:
                session["current_scan_timestamp"] = scan_timestamp
                session["current_tumor_type"]     = tumor_type

            results.append({
                "tumor_type":         tumor_type,
                "confidence":         det_result["confidence"],
                "highest_confidence": det_result["highest_confidence"],
                "has_tumor":          det_result["has_tumor"],
                "gradcam":            gradcam_data,
                "timestamp":          scan_timestamp,   # ← passed to frontend too
            })

        return jsonify({"success": True, "predictions": results, "count": len(results)})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {e}"}), 500

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES -- Chatbot API
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/chat", methods=["POST"])
def chat():
    if _require_login():
        return jsonify({"reply": "Please login first."}), 401

    body    = request.get_json() or {}
    message = body.get("message", "").strip()
    mode    = body.get("mode", "general")

    if not message:
        return jsonify({"reply": "Please ask a question."})

    # ── Build context ──────────────────────────────────────────────────────────
    context = body.get("context", {})

    # Case 1 – general: inject conversation memory from session
    context["last_question"] = session.get("last_question", "")
    context["chat_history"]  = session.get("chat_history",  [])

    # Case 2 & 3 – result_analysis / comparative:
    # inject the timestamp of the most recently analysed scan so chatbot_engine
    # can look up the exact matching entry in detection_result.json and
    # gradcam_analysis.json rather than blindly using the last entry
    context["timestamp"] = session.get("current_scan_timestamp", "")

    # ── Get reply ──────────────────────────────────────────────────────────────
    cb    = _get_chatbot()
    reply = cb.get_reply(message, mode=mode, context=context)

    # ── Update session memory (general mode only) ──────────────────────────────
    if mode == "general":
        history = session.get("chat_history", [])
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant", "content": reply})
        session["last_question"] = message
        session["chat_history"]  = history[-10:]   # keep last 5 turns (10 entries)

    return jsonify({"reply": reply})


@app.route("/chat/reset", methods=["POST"])
def chat_reset():
    """Clear conversation memory. Call this when the chatbot panel is closed/reopened."""
    session.pop("last_question", None)
    session.pop("chat_history",  None)
    return jsonify({"status": "memory cleared"})

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES -- Static helpers
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory('static', filename)

# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    for d in ['static', 'templates', 'model', 'data']:
        os.makedirs(d, exist_ok=True)
    if not os.path.exists(USERS_FILE):
        save_users({})

    print("\n" + "="*55)
    print("NeuroVision : Detection and Chatbot System")
    print("="*55)
    print("http://localhost:5000")
    print("="*55 + "\n")

    app.run(debug=True, port=5000, host='0.0.0.0')