"""
chatbot_engine.py  (v4 – RAG + real JSON data)
-----------------------------------------------
Architecture:
  Case 1 – General/Educational  →  SBERT + FAISS + Groq LLM (RAG)
  Case 2 – Result Analysis      →  reads gradcam_analysis.json   (latest entry)
                                    reads detection_result.json   (latest entry)
  Case 3 – Comparative          →  reads detection_result.json   (all entries,
                                    filters by same tumor_type)
                                    cross-refs gradcam_analysis.json for Grad-CAM details

Both JSON files grow with every scan run — they are NEVER cached here.
Every Case 2 / Case 3 call reads fresh from disk so it always reflects
the most recent scan.

Setup:
  pip install groq faiss-cpu sentence-transformers torch
  export GROQ_API_KEY="your_key_from_console.groq.com"

File layout expected:
  data/tumorinfo.txt
  data/detection_result.json
  data/gradcam_analysis.json
"""

import os
import json

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
KB_PATH         = os.path.join(DATA_DIR, "tumorinfo.txt")
DETECTION_PATH  = os.path.join(BASE_DIR, "detection_result.json")
GRADCAM_PATH    = os.path.join(BASE_DIR, "gradcam_analysis.json")

# ── Config ─────────────────────────────────────────────────────────────────────
DEBUG      = True
USE_OLLAMA = False

TOP_K                = 4
SIMILARITY_THRESHOLD = 0.35
MEMORY_THRESHOLD     = 5

GROQ_MODEL   = "llama-3.1-8b-instant"
OLLAMA_MODEL = "llama3.2"
MAX_TOKENS   = 350
TEMPERATURE  = 0.1

# ── Lazy NLP components (Case 1 only) ─────────────────────────────────────────
_embedder    = None
_index       = None
_documents   = None
_queries     = None
_groq_client = None


# ═════════════════════════════════════════════════════════════════════════════
#  JSON HELPERS  –  always read fresh, never cached
# ═════════════════════════════════════════════════════════════════════════════

def _read_json(path: str) -> list:
    """Read a JSON file and return its contents as a list. Returns [] on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"[JSON READ ERROR] {path}: {e}")
        return []


def _latest_entry(data: list, tumor_type: str = None) -> dict:
    """
    Return the last entry in the list.
    If tumor_type is given, return the last entry matching that type.
    """
    if not data:
        return {}
    if tumor_type:
        matches = [d for d in data if d.get("tumor_type") == tumor_type]
        return matches[-1] if matches else {}
    return data[-1]


def _entries_for_type(data: list, tumor_type: str, exclude_last: bool = True) -> list:
    """
    Return all entries in data that match tumor_type.
    If exclude_last=True, skip the very last entry (that's the current scan).
    """
    matches = [d for d in data if d.get("tumor_type") == tumor_type]
    if exclude_last and len(matches) > 1:
        return matches[:-1]   # exclude current scan from comparison pool
    return matches


# ═════════════════════════════════════════════════════════════════════════════
#  CASE 1 – GENERAL / EDUCATIONAL  (RAG)
# ═════════════════════════════════════════════════════════════════════════════

def _load_nlp():
    global _embedder, _index, _documents, _queries
    if _embedder is not None:
        return

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    _embedder = SentenceTransformer("all-MiniLM-L6-v2")

    with open(KB_PATH, "r", encoding="utf-8-sig") as f:
        raw = f.read().replace("\r\n", "\n").strip()

    questions, answers = [], []
    for block in raw.split("\n\n"):
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        if not lines:
            continue
        questions.append(lines[0])
        answers.append(" ".join(lines[1:]) if len(lines) > 1 else lines[0])

    _queries   = questions
    _documents = answers

    vecs   = _embedder.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
    dim    = vecs.shape[1]
    _index = faiss.IndexFlatIP(dim)
    _index.add(vecs.astype("float32"))

    print(f"✅ chatbot_engine v4: loaded {len(_documents)} Q-A pairs")


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com"
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _retrieve(query: str) -> list:
    import numpy as np
    vec = _embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = _index.search(vec.astype("float32"), TOP_K)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or float(score) < SIMILARITY_THRESHOLD:
            continue
        results.append({
            "question": _queries[idx],
            "answer":   _documents[idx],
            "score":    float(score),
        })
    if DEBUG:
        for r in results:
            print(f"[RETRIEVE] {r['score']:.3f} | {r['question'][:70]}")
    return results


_SYSTEM_PROMPT = """You are NeuroBot, a helpful brain tumor education assistant.
Answer questions about brain tumors, MRI analysis, and the NeuroVision AI system.

Rules:
- Answer ONLY using the provided context passages.
- Be concise and clear (2-4 sentences unless more detail is needed).
- If context is insufficient, say: "I don't have specific information about that.
  For medical concerns, please consult a qualified healthcare professional."
- Never invent medical facts or give personal medical advice.
- Do not repeat the question back."""


def _generate(question: str, chunks: list, history: list = None) -> str:
    if not chunks:
        return (
            "I don't have specific information about that in my knowledge base. "
            "For medical concerns, please consult a qualified healthcare professional."
        )
    context_text = "\n".join(f"[{i}] {c['answer']}" for i, c in enumerate(chunks, 1))
    user_msg = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-4:])
    messages.append({"role": "user", "content": user_msg})

    return _generate_groq(messages) if not USE_OLLAMA else _generate_ollama(messages)


def _generate_groq(messages: list) -> str:
    try:
        resp = _get_groq_client().chat.completions.create(
            model=GROQ_MODEL, messages=messages,
            max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[GROQ ERROR] {e}")
        return f"Error reaching the language model. Please try again. ({e})"


def _generate_ollama(messages: list) -> str:
    import urllib.request
    payload = json.dumps({
        "model": OLLAMA_MODEL, "messages": messages, "stream": False,
        "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
    }).encode()
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/chat", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())["message"]["content"].strip()
    except Exception as e:
        print(f"[OLLAMA ERROR] {e}")
        return f"Error reaching Ollama. Is it running? ({e})"


GENERIC = {
    "hi":            "Hello! I'm NeuroBot. Ask me about brain tumors or your scan results.",
    "hello":         "Hi! How can I help you with brain tumor information today?",
    "hey":           "Hey! I'm here to help. Ask me anything about brain tumors.",
    "good morning":  "Good morning! How may I assist you?",
    "good afternoon":"Good afternoon! Feel free to ask your question.",
    "good evening":  "Good evening! How can I help?",
    "thank you":     "You're welcome! Let me know if you have more questions.",
    "thanks":        "You're welcome!",
    "thankyou":      "You're welcome!",
    "bye":           "Goodbye! Take care.",
    "who are you":   "I'm NeuroBot, an AI assistant specialising in brain tumor information and MRI analysis explanations.",
}


def _reply_general(message: str, context: dict) -> str:
    _load_nlp()
    norm = message.lower().strip()
    if norm in GENERIC:
        return GENERIC[norm]

    last_q  = context.get("last_question", "")
    history = context.get("chat_history", [])
    search_query = (last_q + " " + message) if (last_q and len(norm.split()) <= MEMORY_THRESHOLD) else message

    if DEBUG:
        print(f"[SEARCH QUERY] {search_query!r}")

    return _generate(message, _retrieve(search_query), history)


# ═════════════════════════════════════════════════════════════════════════════
#  CASE 2 – RESULT ANALYSIS
#  Reads the latest entries from both JSON files.
#  Answers questions about: what region the model focused on, why this class
#  was predicted, confidence breakdown across all classes, Grad-CAM heatmap.
# ═════════════════════════════════════════════════════════════════════════════

def _reply_result_analysis(message: str, context: dict) -> str:
    # ── Load fresh from disk every call ───────────────────────────────────────
    detection_data = _read_json(DETECTION_PATH)
    gradcam_data   = _read_json(GRADCAM_PATH)

    if not detection_data:
        return "No scan results found yet. Please upload and analyse an MRI scan first."

    ts = context.get("timestamp")
    if ts:
        det  = next((d for d in reversed(detection_data) if d.get("timestamp") == ts), detection_data[-1])
        gcam = next((d for d in reversed(gradcam_data)   if d.get("timestamp") == ts), {})
    else:
        det  = detection_data[-1]
        gcam = _latest_entry(gradcam_data, det.get("tumor_type", ""))

    # ── Build a structured data summary to pass to the LLM ───────────────────
    tumor_type = det.get("tumor_type", "Unknown")
    all_conf   = det.get("confidence", {})
    raw_probs  = det.get("raw_probabilities", {})
    analysis   = gcam.get("activation_analysis", {})

    sorted_probs = sorted(raw_probs.items(), key=lambda x: x[1], reverse=True)
    second       = sorted_probs[1] if len(sorted_probs) > 1 else None

    scan_summary = f"""
Scan result:
  Predicted class: {tumor_type}
  Confidence: {det.get("highest_confidence", "N/A")}
  Has tumor: {det.get("has_tumor", False)}
  Model: {det.get("model_version", "VGG16")}
  All class probabilities: {", ".join(f"{k}: {v}" for k, v in all_conf.items())}
  Runner-up class: {f"{second[0]} at {second[1]*100:.2f}%" if second else "N/A"}

Grad-CAM analysis:
  Conv layer: {gcam.get("conv_layer_used", "conv2d_12")}
  Peak activation region: {analysis.get("peak_region", "N/A")}
  Activation percentage: {analysis.get("activation_percentage", "N/A")}%
  High-intensity area: {analysis.get("high_area_percentage", "N/A")}% of image
  Focus description: {analysis.get("focus_description", "N/A")}
  Clinical context: {analysis.get("clinical_context", "N/A")}
  Match reason: {analysis.get("match_reason", "N/A")}
  Heatmap interpretation: {analysis.get("heatmap_interpretation", "N/A")}
""".strip()

    system = """You are NeuroBot, a brain tumor AI assistant.
You are given structured data from an MRI scan analysis and a Grad-CAM explainability report.
Answer the user's question in 2-3 sentences using only the data provided.
Be conversational and clear. Do not dump all the data — only answer what was asked.
Always end with a one-line reminder to consult a medical professional."""

    user_msg = f"Scan data:\n{scan_summary}\n\nUser question: {message}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_msg},
    ]
    return _generate_groq(messages)


# ═════════════════════════════════════════════════════════════════════════════
#  CASE 3 – COMPARATIVE ANALYSIS
#  Reads ALL entries from detection_result.json + gradcam_analysis.json,
#  filters by same tumor_type, builds a real comparison from actual scan history.
# ═════════════════════════════════════════════════════════════════════════════

def _reply_comparative(message: str, context: dict) -> str:
    # ── Load fresh from disk ──────────────────────────────────────────────────
    detection_data = _read_json(DETECTION_PATH)
    gradcam_data   = _read_json(GRADCAM_PATH)

    if not detection_data:
        return "No scan history found. Please upload and analyse at least one MRI scan first."

    ts  = context.get("timestamp")
    cur = next((d for d in reversed(detection_data) if d.get("timestamp") == ts), detection_data[-1]) if ts else detection_data[-1]

    tumor_type = cur.get("tumor_type", "Unknown")
    confidence = cur.get("highest_confidence", "N/A")
    cur_ts     = cur.get("timestamp", "")

    similar_det = [
        d for d in detection_data
        if d.get("tumor_type") == tumor_type and d.get("timestamp") != cur_ts
    ]

    def get_gcam(entry):
        t = entry.get("timestamp", "")
        match = next((g for g in gradcam_data if g.get("timestamp") == t), {})
        return match.get("activation_analysis", {})

    cur_gcam    = get_gcam(cur)
    cur_region  = cur_gcam.get("peak_region", "N/A")
    cur_act_pct = cur_gcam.get("activation_percentage", "N/A")

    # ── Build structured data summary for LLM ────────────────────────────────
    current_summary = f"""Current scan:
  Tumor type: {tumor_type}
  Confidence: {confidence}
  Grad-CAM peak region: {cur_region}
  Activation percentage: {cur_act_pct}%
  All class scores: {", ".join(f"{k}: {v}" for k, v in cur.get("confidence", {}).items())}"""

    if not similar_det:
        similar_summary = "No similar cases found in history yet."
    else:
        similar_lines = []
        confs = []
        regions = []
        for i, case in enumerate(similar_det, 1):
            gcam   = get_gcam(case)
            region = gcam.get("peak_region", "N/A")
            act    = gcam.get("activation_percentage", "N/A")
            c_conf = case.get("highest_confidence", "N/A")
            c_ts   = case.get("timestamp", "N/A")
            similar_lines.append(
                f"  Case {i} ({c_ts}): confidence={c_conf}, "
                f"Grad-CAM region={region}, activation={act}%, "
                f"scores={', '.join(f'{k}: {v}' for k, v in case.get('confidence', {}).items())}"
            )
            try:
                confs.append(float(c_conf.replace("%", "")))
            except (ValueError, AttributeError):
                pass
            if region != "N/A":
                regions.append(region)

        from collections import Counter
        avg_conf      = f"{sum(confs)/len(confs):.1f}%" if confs else "N/A"
        conf_range    = f"{min(confs):.1f}% to {max(confs):.1f}%" if confs else "N/A"
        common_region = Counter(regions).most_common(1)[0][0] if regions else "N/A"

        similar_summary = (
            f"{len(similar_det)} similar {tumor_type} case(s) in history:\n"
            + "\n".join(similar_lines)
            + f"\n\nPattern stats: confidence range={conf_range}, "
            f"avg={avg_conf}, most common Grad-CAM region={common_region}"
        )

    full_data = f"{current_summary}\n\n{similar_summary}"

    system = """You are NeuroBot, a brain tumor AI assistant.
You are given data about the current MRI scan and similar historical cases from the system.
Answer the user's question in 3-4 sentences using only this data.
Be conversational — do not list every number, just highlight what is relevant to the question.
End with a brief reminder that these are AI results and a radiologist should be consulted."""

    user_msg = f"Scan history data:\n{full_data}\n\nUser question: {message}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_msg},
    ]
    return _generate_groq(messages)


# ═════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def get_reply(message: str, mode: str = "general", context: dict = None) -> str:
    """
    Route to the correct case handler.

    Parameters
    ----------
    message : str  –  user's current message
    mode    : str  –  "general" | "result_analysis" | "comparative"
    context : dict –  payload from app.py

        For "general":
            last_question  : str   – previous user message (memory)
            chat_history   : list  – [{role, content}, ...] last N turns

        For "result_analysis" and "comparative":
            timestamp : str (optional) – ISO timestamp of the specific scan to use.
                        If omitted, the latest entry in the JSON file is used.

        After calling, update context in app.py:
            session["last_question"] = message
            session["chat_history"].append({"role": "user",      "content": message})
            session["chat_history"].append({"role": "assistant", "content": reply})

    Returns
    -------
    str – the bot's reply
    """
    if context is None:
        context = {}

    try:
        if mode == "result_analysis":
            return _reply_result_analysis(message, context)
        elif mode == "comparative":
            return _reply_comparative(message, context)
        else:
            return _reply_general(message, context)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"I encountered an error processing your question. Please try again. ({e})"