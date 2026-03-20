"""
detection.py  --  NeuroVision  (FIXED)

Architecture confirmed by inspecting vgg16_weights.weights.h5:
  - Flat model (no VGG16 submodel wrapper)
  - Dense = 256 units (not 128)
  - Layer names: conv2d, conv2d_1 ... conv2d_12, dense, dense_1

Class order (alphabetical folder names from dataset):
  0: glioma      -> Glioma
  1: meningioma  -> Meningioma
  2: notumor     -> No Tumor
  3: pituitary   -> Pituitary

Preprocessing: /255.0  (matches ImageDataGenerator rescale=1./255 at training)
IMG_SIZE: 224
"""

import os
import base64
import json

import numpy as np
from io import BytesIO
from PIL import Image

CLASS_NAMES   = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
IMG_SIZE      = 224
MODEL_WEIGHTS = os.path.join("model", "vgg16_weights.weights.h5")
OUTPUT_FILE   = "detection_result.json"

_model = None


def _load_model():
    """
    Builds the exact flat architecture confirmed from inspecting the weights file.
    No VGG16 submodel wrapper. Dense=256. All conv layers flat.
    """
    from keras import Sequential, layers, Input

    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(f"Weights file not found: {MODEL_WEIGHTS}")

    print(f"  Rebuilding flat VGG16 architecture ({IMG_SIZE}x{IMG_SIZE})...")

    loaded_model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        # Block 1
        layers.Conv2D(64,  (3,3), activation='relu', padding='same'),
        layers.Conv2D(64,  (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2), strides=(2,2)),
        # Block 2
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2), strides=(2,2)),
        # Block 3
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2), strides=(2,2)),
        # Block 4
        layers.Conv2D(512, (3,3), activation='relu', padding='same'),
        layers.Conv2D(512, (3,3), activation='relu', padding='same'),
        layers.Conv2D(512, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2), strides=(2,2)),
        # Block 5
        layers.Conv2D(512, (3,3), activation='relu', padding='same'),
        layers.Conv2D(512, (3,3), activation='relu', padding='same'),
        layers.Conv2D(512, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2), strides=(2,2)),
        # Head confirmed from weights: dense=(25088,256), dense_1=(256,4)
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dense(4,   activation='softmax'),
    ])

    print(f"  Loading weights from: {MODEL_WEIGHTS}")
    loaded_model.load_weights(MODEL_WEIGHTS)
    loaded_model.trainable = False
    print("  Model loaded successfully")
    return loaded_model


def get_model():
    global _model
    if _model is not None:
        return _model
    try:
        print("  Building model...")
        _model = _load_model()

        print(f"\n  CLASS MAPPING:")
        for i, c in enumerate(CLASS_NAMES):
            print(f"    [{i}] -> {c}")

        # Sanity check
        b = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        g = np.full((1, IMG_SIZE, IMG_SIZE, 3), 0.5, dtype=np.float32)
        ob = _model.predict(b, verbose=0)[0]
        og = _model.predict(g, verbose=0)[0]
        print(f"\n  blank: {[f'{p:.3f}' for p in ob]}")
        print(f"  grey:  {[f'{p:.3f}' for p in og]}")
        if np.allclose(ob, og, atol=0.02):
            print("  WARNING: blank==grey, weights may not have loaded correctly")
        else:
            print("  Model responds correctly")
        print("Model ready\n")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  {e}")
        _model = None
    return _model


def get_last_conv_layer_name():
    """Return last conv layer name for Grad-CAM."""
    model = get_model()
    if model is None:
        return None, None
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            return layer.name, None
    return None, None


def preprocess(pil_img):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.array(img, dtype=np.float32) / 255.0


def decode_base64_image(b64str):
    if ',' in b64str:
        b64str = b64str.split(',')[1]
    return preprocess(Image.open(BytesIO(base64.b64decode(b64str))))


def load_image_from_path(path):
    return preprocess(Image.open(path))


def predict(img_array):
    model = get_model()
    if model is None:
        return {"error": "Model not loaded"}

    preds      = model.predict(np.expand_dims(img_array, 0), verbose=0)[0]
    class_idx  = int(np.argmax(preds))
    tumor_type = CLASS_NAMES[class_idx]

    print(f"  {[f'{CLASS_NAMES[i]}={preds[i]*100:.1f}%' for i in range(4)]}")
    print(f"  -> {tumor_type} ({preds[class_idx]*100:.2f}%)")

    return {
        "tumor_type":         tumor_type,
        "highest_confidence": f"{preds[class_idx]*100:.2f}%",
        "confidence":         {c: f"{v*100:.2f}%" for c, v in zip(CLASS_NAMES, preds)},
        "class_index":        class_idx,
        "image_shape":        list(img_array.shape),
        "has_tumor":          tumor_type != "No Tumor",
        "model_version":      "VGG16",
        "raw_probabilities":  {c: round(float(v), 6) for c, v in zip(CLASS_NAMES, preds)},
    }


def predict_from_base64(b64str):
    try:
        return predict(decode_base64_image(b64str))
    except Exception as e:
        return {"error": str(e)}


def save_result_json(result, path=OUTPUT_FILE, user=None):
    import datetime
    # Load existing log
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                log = json.load(f)
                if not isinstance(log, list):
                    log = [log]   # migrate old single-object format
            except:
                log = []
    else:
        log = []

    # Add metadata
    entry = dict(result)
    entry["user"]      = user or "anonymous"
    entry["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log.append(entry)

    with open(path, "w") as f:
        json.dump(log, f, indent=2)

        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  required=True)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()
    img = load_image_from_path(args.image)
    res = predict(img)
    save_result_json(res, args.output)
    print(json.dumps(res, indent=2))