import hashlib
import io

DEFAULT_GRAMS = 150.0
CONFIDENCE_THRESHOLD = 0.70

# Household unit → approximate grams
UNIT_CONVERSIONS = {
    "slice": 30.0,
    "cup": 240.0,
    "tbsp": 15.0,
    "tsp": 5.0,
    "piece": 100.0,
    "handful": 30.0,
    "bowl": 300.0,
    "plate": 400.0,
    "glass": 250.0,
    "oz": 28.35,
}

UNIT_OPTIONS = ["grams"] + sorted(UNIT_CONVERSIONS.keys())


def compute_nutrition(nutrition_per_100g: dict, grams: float) -> dict:
    factor = grams / 100.0
    return {
        "calories": round(nutrition_per_100g["kcal"] * factor, 1),
        "protein": round(nutrition_per_100g["protein"] * factor, 1),
        "fat": round(nutrition_per_100g["fat"] * factor, 1),
        "carbs": round(nutrition_per_100g["carbs"] * factor, 1),
    }


def safe_grams(estimated: float | None) -> float:
    if estimated and estimated > 0:
        return float(estimated)
    return DEFAULT_GRAMS


def unit_to_grams(unit: str, quantity: float) -> float:
    if unit == "grams" or unit not in UNIT_CONVERSIONS:
        return quantity
    return UNIT_CONVERSIONS[unit] * quantity


def image_hash(image_bytes: bytes) -> str:
    return hashlib.md5(image_bytes).hexdigest()


def compress_image(image_bytes: bytes, max_size: int = 1024, quality: int = 80) -> bytes:
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    except Exception:
        return image_bytes


def classify_meal_time(hour: int) -> str:
    if hour < 10:
        return "Breakfast"
    elif hour < 14:
        return "Lunch"
    elif hour < 17:
        return "Snack"
    elif hour < 21:
        return "Dinner"
    else:
        return "Late snack"
