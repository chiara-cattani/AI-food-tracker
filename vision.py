import os
import base64
import json
import re

from openai import OpenAI

VISION_PROMPT = """You are a food recognition AI. Analyze this image and identify ALL foods visible.

For each food item, estimate the portion size in grams based on visual cues.

Return ONLY valid JSON in this exact format, with no extra text:
{
  "foods": [
    {
      "name": "food name",
      "confidence": 0.9,
      "estimated_grams": 200
    }
  ]
}

Rules:
- "name" should be a short, common name (e.g. "grilled chicken breast", "white rice", "mixed salad")
- "confidence" is a float between 0 and 1
- "estimated_grams" is your best visual estimate of the portion weight in grams
- If you cannot identify any food, return: {"foods": []}
- Return ONLY the JSON object, no markdown fences, no explanation
"""


def recognize_food(image_bytes: bytes) -> tuple[dict, str]:
    """Send image to OpenAI vision model.

    Returns (parsed_result_dict, raw_response_text).
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
        max_tokens=1024,
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()

    cleaned = re.sub(r"^```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                result = {"foods": []}
        else:
            result = {"foods": []}

    if "foods" not in result:
        result = {"foods": []}

    return result, raw
