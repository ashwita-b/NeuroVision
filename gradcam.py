"""
gradcam.py  --  NeuroVision
Grad-CAM using last conv layer of the flat VGG16 model.
Input: 224x224. Reuses detection.py model singleton.
"""

import os, base64, json
import numpy as np
import cv2

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
IMG_SIZE    = 224
OUTPUT_FILE = "gradcam_analysis.json"

TUMOR_CONTEXT = {
    "Glioma":     {"region":"cerebral cortex or white matter",  "features":"irregular margins and infiltrative pattern",   "note":"Gliomas show diffuse infiltration. High cortical/subcortical activation expected."},
    "Meningioma": {"region":"brain surface or falx cerebri",    "features":"well-defined margins and dural tail sign",     "note":"Meningiomas arise from meninges. Peripheral convexity activation is characteristic."},
    "Pituitary":  {"region":"sella turcica or parasellar area", "features":"midline location and suprasellar extension",   "note":"Pituitary tumors are midline. Central inferior activation near the sella expected."},
    "No Tumor":   {"region":"N/A",                              "features":"no abnormal mass",                            "note":"No significant focal activation expected when no tumor is present."},
}
QUADRANT_MAP = {(0,0):"upper-left",(0,1):"upper-right",
                (1,0):"lower-left",(1,1):"lower-right"}

def _analyse_heatmap(hm):
    h,w   = hm.shape
    mh,mw = h//2, w//2
    q  = {(0,0):hm[:mh,:mw].sum(),(0,1):hm[:mh,mw:].sum(),
          (1,0):hm[mh:,:mw].sum(),(1,1):hm[mh:,mw:].sum()}
    pk = max(q, key=q.get)
    t  = hm.sum()
    return {"peak_quadrant":            QUADRANT_MAP[pk],
            "peak_activation_percent":  round(float(q[pk]/t*100),1) if t>0 else 0.0,
            "high_activation_area_pct": round(float((hm>0.7*hm.max()).mean()*100),1)}

def _build_text(tumor_type, spatial, confidence):
    ctx = TUMOR_CONTEXT.get(tumor_type, TUMOR_CONTEXT["No Tumor"])
    r,a,h = (spatial["peak_quadrant"],
              spatial["peak_activation_percent"],
              spatial["high_activation_area_pct"])
    return {
        "peak_region": r, "activation_percentage": a, "high_area_percentage": h,
        "focus_description":
            f"Model focused on the {r} ({a:.1f}% of total activation). "
            f"~{h:.1f}% of image shows high activation (>70% of peak).",
        "clinical_context":
            f"In {tumor_type} cases, activation expected in {ctx['region']}. "
            f"Typical features: {ctx['features']}.",
        "match_reason": ctx["note"],
        "heatmap_interpretation":
            f"Red/yellow = highest activation (most influential for '{tumor_type}' "
            f"at {confidence:.1f}% confidence). Blue/dark = minimal contribution.",
    }

def generate_gradcam_analysis(img_array, tumor_type, pred_index, confidence):
    import tensorflow as tf
    from detection import get_model, get_last_conv_layer_name

    base = {"tumor_type": tumor_type, "class_index": pred_index, "generated": False}

    model = get_model()
    if model is None:
        base["error"] = "Model not loaded"; return base

    conv_layer_name, _ = get_last_conv_layer_name()
    if conv_layer_name is None:
        base["error"] = "Could not find last conv layer"; return base

    print(f"  Grad-CAM: layer='{conv_layer_name}'")

    try:
        img_t = tf.cast(np.expand_dims(img_array, 0), tf.float32)

        # Step 1: get conv output without tape (no gradient needed up to here)
        x = img_t
        for layer in model.layers:
            x = layer(x, training=False)
            if layer.name == conv_layer_name:
                conv_out = tf.identity(x)
                break

        # Step 2: watch conv_out and run remaining layers to get predictions
        with tf.GradientTape() as tape:
            tape.watch(conv_out)
            x = conv_out
            found = False
            for layer in model.layers:
                if found:
                    x = layer(x, training=False)
                if layer.name == conv_layer_name:
                    found = True
            preds_tf = x
            score = preds_tf[:, pred_index]

        grads   = tape.gradient(score, conv_out)
        pooled  = tf.reduce_mean(grads, axis=(0,1,2))
        conv_np = conv_out[0].numpy().copy()
        pool_np = pooled.numpy()

        for i in range(pool_np.shape[0]):
            conv_np[:,:,i] *= pool_np[i]

        heatmap = np.mean(conv_np, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

    except Exception as e:
        import traceback; traceback.print_exc()
        base["error"] = f"Gradient computation failed: {e}"; return base

    hm_r    = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    hm_col  = cv2.applyColorMap(np.uint8(255*hm_r), cv2.COLORMAP_JET)
    orig_bgr= cv2.cvtColor(np.uint8(img_array*255), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(orig_bgr, 0.5, hm_col, 0.5, 0)

    _, b1 = cv2.imencode('.png', overlay)
    _, b2 = cv2.imencode('.png', hm_col)

    spatial  = _analyse_heatmap(hm_r)
    analysis = _build_text(tumor_type, spatial, confidence)

    print(f"  Grad-CAM generated successfully")
    return {
        "tumor_type":          tumor_type,
        "class_index":         pred_index,
        "conv_layer_used":     conv_layer_name,
        "confidence":          round(confidence, 2),
        "activation_analysis": analysis,
        "overlay_base64":      "data:image/png;base64," + base64.b64encode(b1).decode(),
        "heatmap_base64":      "data:image/png;base64," + base64.b64encode(b2).decode(),
        "generated":           True,
    }



def save_analysis_json(analysis, path=OUTPUT_FILE, user=None):
    import datetime
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                log = json.load(f)
                if not isinstance(log, list):
                    log = [log]
            except:
                log = []
    else:
        log = []

    light = {k:v for k,v in analysis.items()
             if k not in ("overlay_base64","heatmap_base64")}
    light["generated"]  = True
    light["user"]       = user or "anonymous"
    light["timestamp"]  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log.append(light)

    with open(path, "w") as f:
        json.dump(log, f, indent=2)

if __name__ == "__main__":
    import argparse
    from detection import load_image_from_path, predict
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()
    img = load_image_from_path(args.image)
    det = predict(img)
    ci  = det["class_index"]
    cf  = float(det["highest_confidence"].replace("%",""))
    res = generate_gradcam_analysis(img, CLASS_NAMES[ci], ci, cf)
    save_analysis_json(res, args.output)
    print(json.dumps({k:v for k,v in res.items() if "base64" not in k}, indent=2))