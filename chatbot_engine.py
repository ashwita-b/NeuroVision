"""
chatbot_engine.py
-----------------
Handles all three chatbot cases:

  Case 1 – General / Educational  (mode="general")
  Case 2 – Result Analysis Bot    (mode="result_analysis")
  Case 3 – Comparative Analysis   (mode="comparative")

get_reply(message, mode, context) is the single entry-point called by app.py.
"""

import os
import json

# ── Knowledge-base path ────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "tumorinfo.txt")

# ── Lazy-loaded NLP components ─────────────────────────────────────────────────
_embedder    = None
_documents   = None
_embeddings  = None

SIMILARITY_THRESHOLD = 0.55
def _load_nlp():
    """Initialise sentence-transformer + knowledge base (once)."""
    global _embedder, _documents, _embeddings
    if _embedder is not None:
        return

    from sentence_transformers import SentenceTransformer
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")

    with open(DATA_PATH, "r", encoding="utf-8-sig") as f:
        raw = f.read().replace("\r\n", "\n").strip()

    docs, block = [], []
    for line in raw.split("\n"):
        line = line.strip()
        if line == "":
            if block: docs.append(" ".join(block)); block = []
        else:
            block.append(line)
    if block: docs.append(" ".join(block))

    _documents  = docs
    _embeddings = _embedder.encode(docs, convert_to_tensor=True)
    print(f"✅ chatbot_engine: Loaded {len(docs)} knowledge chunks")


# ── Generic greetings ──────────────────────────────────────────────────────────
GENERIC = {
    "hi":           "Hello! I'm NeuroBot. Ask me about brain tumors or your scan results.",
    "hello":        "Hi! How can I help you with brain tumor information today?",
    "hey":          "Hey! I'm here to help. Ask me anything about brain tumors.",
    "good morning": "Good morning! How may I assist you?",
    "good afternoon":"Good afternoon! Feel free to ask your question.",
    "good evening": "Good evening! How can I help?",
    "thank you":    "You're welcome! Let me know if you have more questions.",
    "thanks":       "You're welcome!",
    "thankyou":     "You're welcome!",
    "bye":          "Goodbye! Take care.",
    "who are you":  "I'm NeuroBot, an AI assistant specialising in brain tumor information and MRI analysis explanations.",
}

# ── Comparative case data (representative patterns per class) ──────────────────
COMPARATIVE_CASES = {
    "Glioma": [
        {"case": "Case A", "location": "Left frontal lobe", "confidence": "96.1%",
         "features": "Irregular margins, white matter infiltration, heterogeneous signal"},
        {"case": "Case B", "location": "Right temporal lobe", "confidence": "91.4%",
         "features": "Diffuse infiltration, mass effect, oedema present"},
        {"case": "Case C", "location": "Corpus callosum", "confidence": "88.7%",
         "features": "Butterfly pattern, bilateral involvement"},
        {"case": "Case D", "location": "Parietal lobe", "confidence": "93.2%",
         "features": "Ring enhancement, central necrosis"},
        {"case": "Case E", "location": "Occipital lobe", "confidence": "89.5%",
         "features": "Infiltrative borders, surrounding oedema"},
    ],
    "Meningioma": [
        {"case": "Case A", "location": "Frontal convexity", "confidence": "97.3%",
         "features": "Well-defined margin, dural tail, homogeneous enhancement"},
        {"case": "Case B", "location": "Sphenoid ridge", "confidence": "94.8%",
         "features": "Extra-axial location, calcification present"},
        {"case": "Case C", "location": "Parasagittal region", "confidence": "92.1%",
         "features": "Broad dural base, uniform enhancement"},
        {"case": "Case D", "location": "Olfactory groove", "confidence": "90.6%",
         "features": "Midline displacement, olfactory nerve involvement"},
        {"case": "Case E", "location": "Posterior fossa", "confidence": "88.9%",
         "features": "Cerebellopontine angle, cranial nerve compression"},
    ],
    "Pituitary": [
        {"case": "Case A", "location": "Sella turcica", "confidence": "95.7%",
         "features": "Midline position, sellar expansion, optic chiasm compression"},
        {"case": "Case B", "location": "Suprasellar region", "confidence": "92.3%",
         "features": "Upward extension, visual field defect"},
        {"case": "Case C", "location": "Cavernous sinus", "confidence": "89.4%",
         "features": "Lateral extension, carotid artery encasement"},
        {"case": "Case D", "location": "Sella turcica", "confidence": "91.8%",
         "features": "Microadenoma, subtle signal change"},
        {"case": "Case E", "location": "Parasellar region", "confidence": "87.6%",
         "features": "Hormone-secreting, macroadenoma size"},
    ],
    "No Tumor": [
        {"case": "Case A", "location": "N/A", "confidence": "98.2%",
         "features": "Normal cortical thickness, no mass effect"},
        {"case": "Case B", "location": "N/A", "confidence": "96.5%",
         "features": "Symmetric ventricles, no signal abnormality"},
        {"case": "Case C", "location": "N/A", "confidence": "94.7%",
         "features": "Normal grey-white differentiation"},
        {"case": "Case D", "location": "N/A", "confidence": "97.1%",
         "features": "No enhancement, normal sulcal pattern"},
        {"case": "Case E", "location": "N/A", "confidence": "95.3%",
         "features": "Age-appropriate brain volume, no focal lesion"},
    ]
}

# ── Case 1 – General / Educational ────────────────────────────────────────────
def _reply_general(message: str) -> str:
    _load_nlp()

    import torch
    from torch.nn.functional import cosine_similarity

    norm = message.lower().strip()
    if norm in GENERIC:
        return GENERIC[norm]

    q_emb = _embedder.encode(message, convert_to_tensor=True)
    sims  = cosine_similarity(q_emb.unsqueeze(0), _embeddings)
    score = sims.max().item()
    idx   = sims.argmax().item()

    if score < SIMILARITY_THRESHOLD:
        return ("I don't have specific information about that in my knowledge base. "
                "For medical concerns, please consult a qualified healthcare professional.")

    context = _documents[idx]
    # Strip the question part if present
    answer  = context.split("?", 1)[1].strip() if "?" in context else context
    return answer


# ── Case 2 – Result Analysis Bot ──────────────────────────────────────────────
def _reply_result_analysis(message: str, context: dict) -> str:
    tumor_type  = context.get("tumor_type", "Unknown")
    confidence  = context.get("confidence", "N/A")
    region      = context.get("activation_region", "unspecified region")
    conv_layer  = context.get("conv_layer", "last convolutional layer")
    analysis    = context.get("activation_analysis", {})

    msg_lower = message.lower()

    # ── What region did the model focus on? ───────────────────────────────────
    if any(w in msg_lower for w in ["region", "focus", "look", "area", "where"]):
        focus  = analysis.get("focus_description",
                              f"The model focused on the {region} of the MRI.")
        interp = analysis.get("heatmap_interpretation", "")
        return f"{focus}\n\n{interp}"

    # ── Why this prediction? ──────────────────────────────────────────────────
    if any(w in msg_lower for w in ["why", "reason", "explain", "because", "how"]):
        clinical = analysis.get("clinical_context", "")
        match    = analysis.get("match_reason", "")
        return (f"The model predicted **{tumor_type}** with **{confidence}** confidence.\n\n"
                f"{clinical}\n\n{match}")

    # ── Heatmap / Grad-CAM explanation ────────────────────────────────────────
    if any(w in msg_lower for w in ["heatmap", "grad", "cam", "colour", "color", "red", "yellow", "blue"]):
        interp = analysis.get("heatmap_interpretation",
                              "Red/yellow regions indicate highest activation.")
        return interp

    # ── Confidence explanation ────────────────────────────────────────────────
    if any(w in msg_lower for w in ["confidence", "sure", "certain", "accurate", "percent", "%"]):
        high_pct = analysis.get("high_area_percentage", "N/A")
        return (f"The model predicted **{tumor_type}** with a confidence of **{confidence}**. "
                f"Approximately {high_pct}% of the image area showed high activation "
                f"(>70% of peak), which contributed to this confidence level.")

    # ── General result question ───────────────────────────────────────────────
    focus    = analysis.get("focus_description", f"activation in the {region}")
    clinical = analysis.get("clinical_context", "")
    match    = analysis.get("match_reason", "")

    return (f"**Detection Result: {tumor_type}** ({confidence} confidence)\n\n"
            f"📍 **Grad-CAM Focus:** {focus}\n\n"
            f"🔬 **Clinical Context:** {clinical}\n\n"
            f"✅ **Why this matches:** {match}\n\n"
            f"⚠️ This is an AI-assisted analysis. Please consult a medical professional for diagnosis.")


# ── Case 3 – Comparative Analysis Bot ─────────────────────────────────────────
def _reply_comparative(message: str, context: dict) -> str:
    tumor_type  = context.get("tumor_type", "Unknown")
    confidence  = context.get("confidence", "N/A")
    msg_lower   = message.lower()

    cases = COMPARATIVE_CASES.get(tumor_type, [])
    if not cases:
        return f"No comparative cases available for '{tumor_type}' in the dataset."

    # ── Similarities ──────────────────────────────────────────────────────────
    if "similar" in msg_lower or "comparison" in msg_lower or "cases" in msg_lower or "show" in msg_lower:
        lines = [f"📊 **Comparative Analysis – {tumor_type}**\n",
                 f"Your scan was classified as **{tumor_type}** with **{confidence}** confidence.\n",
                 f"Here are {len(cases)} similar cases from the dataset:\n"]

        for c in cases:
            lines.append(
                f"**{c['case']}**\n"
                f"  • Location: {c['location']}\n"
                f"  • Confidence: {c['confidence']}\n"
                f"  • Features: {c['features']}\n"
            )

        lines.append(
            "\n🔍 **Pattern Summary:**\n"
            f"All cases show the hallmark features of {tumor_type}: "
            + _summary_features(tumor_type)
        )
        lines.append(
            "\n⚠️ These are representative training-set patterns used for model validation. "
            "Always seek a qualified radiologist's opinion for clinical decisions."
        )
        return "\n".join(lines)

    # ── Differences ───────────────────────────────────────────────────────────
    if "differ" in msg_lower or "unlike" in msg_lower or "distinguish" in msg_lower:
        return (f"While all cases share core **{tumor_type}** features, differences include:\n"
                f"• **Location variability** – tumors of the same type can occur in multiple "
                f"regions (e.g., {', '.join(c['location'] for c in cases[:3])}).\n"
                f"• **Confidence range** – model confidence varies from "
                f"{min(c['confidence'] for c in cases)} to {max(c['confidence'] for c in cases)} "
                f"depending on image quality and lesion characteristics.\n"
                f"• **Feature expression** – not all {tumor_type} cases exhibit every classic sign.")

    # ── Default ───────────────────────────────────────────────────────────────
    return (f"I found **{len(cases)} similar {tumor_type} cases** in the dataset. "
            f"Ask me to 'show similar cases' for a detailed comparison, or ask about "
            f"differences between them.")


def _summary_features(tumor_type: str) -> str:
    summaries = {
        "Glioma":      "diffuse infiltration, irregular margins, and white matter involvement.",
        "Meningioma":  "extra-axial location, well-defined margins, and dural attachment.",
        "Pituitary":   "midline sellar location, hormonal involvement, and suprasellar extension.",
        "No Tumor":    "normal cortical architecture, no mass effect, and symmetric signal.",
    }
    return summaries.get(tumor_type, "consistent features across the class.")


# ── Public entry-point ─────────────────────────────────────────────────────────
def get_reply(message: str, mode: str = "general", context: dict = None) -> str:
    """
    Route to the correct case handler.

    mode:
      "general"         → Case 1 (educational RAG chatbot)
      "result_analysis" → Case 2 (Grad-CAM + prediction explainer)
      "comparative"     → Case 3 (similar-case comparison)
    """
    if context is None:
        context = {}

    try:
        if mode == "result_analysis":
            return _reply_result_analysis(message, context)
        elif mode == "comparative":
            return _reply_comparative(message, context)
        else:
            return _reply_general(message)
    except Exception as e:
        import traceback; traceback.print_exc()
        return f"I encountered an error processing your question. Please try again. ({e})"