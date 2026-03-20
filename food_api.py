import time
import functools
from difflib import SequenceMatcher

import requests

SEARCH_URL = "https://world.openfoodfacts.org/cgi/search.pl"

FALLBACK_NUTRITION = {
    "kcal": 150.0,
    "protein": 8.0,
    "fat": 5.0,
    "carbs": 18.0,
}


def _normalize(name: str) -> str:
    return " ".join(name.lower().strip().split())


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _score_product(product: dict, query: str) -> float:
    n = product.get("nutriments", {})
    name = product.get("product_name", "")
    sim = _similarity(query, name) if name else 0.0
    fields = ["energy-kcal_100g", "proteins_100g", "fat_100g", "carbohydrates_100g"]
    present = sum(1 for f in fields if n.get(f) is not None)
    completeness = present / len(fields)
    has_kcal = 1.0 if (n.get("energy-kcal_100g") or n.get("energy_100g")) else 0.0
    return sim * 0.4 + completeness * 0.4 + has_kcal * 0.2


@functools.lru_cache(maxsize=512)
def search_nutrition(food_name: str) -> tuple:
    """Query OpenFoodFacts with improved matching and caching.

    Returns (nutrition_dict, matched_product_name | None, source_string).
    nutrition_dict has keys: kcal, protein, fat, carbs (per 100 g).
    source_string is 'openfoodfacts' or 'fallback'.
    """
    normalized = _normalize(food_name)
    params = {
        "search_terms": normalized,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": 10,
        "fields": "product_name,nutriments",
    }

    data = None
    for attempt in range(3):
        try:
            resp = requests.get(SEARCH_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            break
        except (requests.RequestException, ValueError):
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))

    if data is None:
        return (dict(FALLBACK_NUTRITION), None, "fallback")

    products = data.get("products", [])
    if not products:
        return (dict(FALLBACK_NUTRITION), None, "fallback")

    scored = [(p, _score_product(p, normalized)) for p in products]
    scored.sort(key=lambda x: x[1], reverse=True)

    for product, _score in scored:
        n = product.get("nutriments", {})
        kcal = n.get("energy-kcal_100g") or n.get("energy_100g")
        if kcal is None:
            continue
        return (
            {
                "kcal": float(kcal),
                "protein": float(n.get("proteins_100g") or 0),
                "fat": float(n.get("fat_100g") or 0),
                "carbs": float(n.get("carbohydrates_100g") or 0),
            },
            product.get("product_name", ""),
            "openfoodfacts",
        )

    return (dict(FALLBACK_NUTRITION), None, "fallback")
