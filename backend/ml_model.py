"""
YOLO object detection → EfficientNet crop classification pipeline.
Falls back to whole-image EfficientNet for scene categories (Place/Landmark, Environment).
"""

import gc
import logging
import os
import ssl
import certifi
import numpy as np
from PIL import Image as PILImage


ssl.create_default_context = lambda *a, **kw: ssl.create_default_context(
    *a, cafile=certifi.where(), **kw
)
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

logger = logging.getLogger(__name__)

_yolo = None
_effnet = None
_decode_preds = None

IMG_SIZE = (224, 224)

# Direct YOLO COCO class → category mapping (fast path, skips EfficientNet when confident)
YOLO_CATEGORY_MAP = {
    "person":     "Individual Person",
    "bicycle":    "Vehicle",  "car":       "Vehicle",  "motorcycle": "Vehicle",
    "airplane":   "Vehicle",  "bus":       "Vehicle",  "train":      "Vehicle",
    "truck":      "Vehicle",  "boat":      "Vehicle",
    "bird":       "Pet",      "cat":       "Pet",      "dog":        "Pet",
    "horse":      "Pet",      "sheep":     "Pet",      "cow":        "Pet",
    "bear":       "Pet",      "zebra":     "Pet",      "giraffe":    "Pet",
    "banana":     "Food",     "apple":     "Food",     "sandwich":   "Food",
    "orange":     "Food",     "broccoli":  "Food",     "carrot":     "Food",
    "hot dog":    "Food",     "pizza":     "Food",     "donut":      "Food",
    "cake":       "Food",
}

# EfficientNet ImageNet label keywords → category (used for crops and whole-image fallback)
CATEGORY_KEYWORDS = {
    "Vehicle": [
        "car", "truck", "bus", "train", "bicycle", "motorcycle", "airplane",
        "ship", "boat", "vehicle", "wagon", "jeep", "minivan", "convertible",
        "trailer", "locomotive", "tractor", "tank", "ambulance", "cab",
        "limousine", "sports_car", "pickup", "forklift", "garbage_truck",
        "fire_engine", "moped", "scooter", "canoe", "yacht", "speedboat",
        "airliner", "go-kart", "snowmobile",
    ],
    "Food": [
        "pizza", "hotdog", "hamburger", "ice_cream", "bagel", "pretzel",
        "burrito", "sandwich", "soup", "salad", "cake", "bread", "pancake",
        "waffle", "sushi", "taco", "noodle", "pasta", "cheese", "egg",
        "banana", "apple", "orange", "strawberry", "pineapple", "lemon",
        "custard", "trifle", "guacamole", "mashed_potato", "broccoli",
        "cauliflower", "mushroom", "corn", "artichoke", "cucumber",
        "bell_pepper", "carbonara", "espresso", "wine", "beer", "cocktail",
        "meatloaf", "potpie", "burger", "dough", "plate", "consomme",
    ],
    "Pet": [
        "dog", "cat", "puppy", "kitten", "retriever", "terrier", "poodle",
        "bulldog", "shepherd", "spaniel", "hound", "tabby", "persian_cat",
        "siamese_cat", "egyptian_cat", "collie", "corgi", "pug", "chihuahua",
        "dachshund", "labrador", "rabbit", "hamster", "parrot", "macaw",
        "canary", "guinea_pig", "lhasa", "shih-tzu", "maltese", "bird",
    ],
    "Fish": [
        "goldfish", "tench", "shark", "great_white_shark", "tiger_shark",
        "hammerhead", "stingray", "electric_ray", "eel", "electric_eel",
        "sturgeon", "salmon", "coho", "barracouta", "puffer", "lionfish",
        "seahorse", "starfish", "swordfish", "marlin", "piranha", "barracuda",
        "catfish", "carp", "bass", "trout", "clownfish", "angelfish",
        "guppy", "betta", "grouper", "halibut", "cod", "mackerel",
        "herring", "anchovy", "flounder", "perch", "pike", "manta_ray",
    ],
    "Flowers": [
        "daisy", "sunflower", "tulip", "orchid", "rose", "hibiscus", "lotus",
        "dahlia", "lily", "peony", "marigold", "chrysanthemum", "carnation",
        "poppy", "pansy", "jasmine", "lavender", "magnolia", "daffodil",
        "violet", "geranium", "zinnia", "azalea", "camellia", "snapdragon",
        "wildflower", "blossom", "petal", "bouquet", "foxglove", "bluebell",
        "primrose", "aster", "begonia", "freesia", "gardenia", "hyacinth",
    ],
    "Place/Landmark": [
        "castle", "palace", "church", "mosque", "temple", "bridge", "tower",
        "dome", "monastery", "lighthouse", "stadium", "obelisk", "pyramid",
        "triumphal_arch", "bell_cote", "library", "prison", "fountain",
        "planetarium", "dam", "boathouse", "barn", "monument",
    ],
    "Environment": [
        "alp", "valley", "volcano", "cliff", "seashore", "coral_reef",
        "lakeside", "geyser", "sandbar", "mountain", "forest", "desert",
        "beach", "glacier", "canyon", "reef", "promontory", "field",
        "meadow", "sky", "rainforest",
    ],
}


def _load_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        logger.info("Loading YOLOv8n...")
        _yolo = YOLO("yolov8n.pt")
        logger.info("YOLOv8n loaded.")
    return _yolo


def _load_effnet():
    global _effnet, _decode_preds
    if _effnet is None:
        from tensorflow.keras.applications.efficientnet import EfficientNetB0
        from tensorflow.keras.applications.imagenet_utils import decode_predictions
        logger.info("Loading EfficientNetB0...")
        _effnet = EfficientNetB0(weights="imagenet", include_top=True)
        _decode_preds = decode_predictions
        logger.info("EfficientNetB0 loaded.")
    return _effnet


def load_model():
    _load_yolo()
    _load_effnet()


def _effnet_classify(img_arr):
    """Run EfficientNetB0 on an HxWx3 array. Returns top-5 decoded predictions."""
    from tensorflow.keras.applications.efficientnet import preprocess_input
    model = _load_effnet()
    arr = np.expand_dims(np.array(img_arr, dtype="float32"), axis=0)
    arr = preprocess_input(arr)
    preds = model.predict(arr, verbose=0)
    return _decode_preds(preds, top=5)[0]


def _label_to_category(decoded_preds):
    """Map EfficientNet top-5 predictions to one of our 6 categories."""
    for _, label, _ in decoded_preds:
        label_lower = label.lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in label_lower:
                    return category
    return None


def predict_category(image_path):
    yolo = _load_yolo()
    _load_effnet()

    # ── Stage 1: YOLO object detection ───────────────────────────────
    yolo_results = yolo(str(image_path), verbose=False)[0]

    best_category = None
    best_conf = 0.0
    best_box = None

    if yolo_results.boxes is not None and len(yolo_results.boxes):
        for box in yolo_results.boxes:
            cname = yolo_results.names[int(box.cls[0])].lower()
            conf = float(box.conf[0])
            cat = YOLO_CATEGORY_MAP.get(cname)
            if cat and conf > best_conf:
                best_category = cat
                best_conf = conf
                best_box = box

    # ── Stage 2: EfficientNet on cropped object ───────────────────────
    img = PILImage.open(image_path).convert("RGB")

    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        crop = img.crop((x1, y1, x2, y2)).resize(IMG_SIZE)
        try:
            crop_preds = _effnet_classify(np.array(crop))
            crop_category = _label_to_category(crop_preds)
            if crop_category:
                return {
                    "category": crop_category,
                    "confidence": round(float(crop_preds[0][2]) * 100, 2),
                }
        except Exception as e:
            logger.warning(f"EfficientNet crop prediction failed: {e}")

        # EfficientNet on crop didn't match keywords; use YOLO's direct label
        if best_category:
            return {"category": best_category, "confidence": round(best_conf * 100, 2)}

    # ── Stage 3: Whole-image EfficientNet fallback (scenes) ──────────
    try:
        whole_preds = _effnet_classify(np.array(img.resize(IMG_SIZE)))
        scene_category = _label_to_category(whole_preds)
        if scene_category:
            return {
                "category": scene_category,
                "confidence": round(float(whole_preds[0][2]) * 100, 2),
            }
    except Exception as e:
        logger.error(f"Whole-image EfficientNet failed for {image_path}: {e}")

    return {"category": "Uncategorized", "confidence": 0.0}


def batch_predict(image_paths, batch_size=32):
    load_model()
    results = []
    for i in range(0, len(image_paths), batch_size):
        chunk = image_paths[i : i + batch_size]
        for path in chunk:
            try:
                prediction = predict_category(path)
                results.append({"filepath": path, **prediction})
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                results.append({"filepath": path, "category": "Uncategorized", "confidence": 0.0})
        gc.collect()
        logger.info(f"Batch progress: {min(i + batch_size, len(image_paths))}/{len(image_paths)}")
    return results
