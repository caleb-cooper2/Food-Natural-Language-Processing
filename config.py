UNIT_GRAMS = {
    "g": 1.0,        "gram": 1.0,        "grams": 1.0,
    "kg": 1000.0,    "kilogram": 1000.0, "kilograms": 1000.0,
    "ml": 1.0,       "millilitre": 1.0,  "millilitres": 1.0,
    "milliliter": 1.0, "milliliters": 1.0,
    "l": 1000.0,     "litre": 1000.0,    "litres": 1000.0,
    "liter": 1000.0, "liters": 1000.0,
    "cup": 250.0,    "cups": 250.0,
    "tbsp": 15.0,    "tablespoon": 15.0, "tablespoons": 15.0,
    "tsp": 5.0,      "teaspoon": 5.0,    "teaspoons": 5.0,
}

CSM_UNITS = {
    "slice", "slices", "piece", "pieces", "serving", "servings",
    "handful", "handfuls", "can", "cans", "bottle", "bottles",
    "bar", "bars", "sachet", "sachets", "packet", "packets",
    "container", "containers", "tub", "tubs",
}

ALL_UNITS = set(UNIT_GRAMS.keys()) | CSM_UNITS

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b-instruct-q4_K_M"
OLLAMA_TIMEOUT = 20.0 # how many secs before giving up and going to spacy if needed

LLM_SYSTEM_PROMPT = """\
Extract all food items. Return JSON only. No explanation.

Critical rules:
- The "food" field MUST be an exact substring of the input text.
- Do NOT replace brand names with generic terms.
- Do NOT simplify or generalise food names.
- Preserve original wording exactly as written.

Return format: {"items": [...]}
Each item: {"food": string, "quantity": number, "unit": string|null}
If no quantity stated, use 1. If no unit, use null. Keep brand names as-is.

Input: "spaghetti with a slice of bread topped with butter"
Output: [{"food":"spaghetti","quantity":1,"unit":"serving"},{"food":"bread","quantity":1,"unit":"slice"},{"food":"butter","quantity":1,"unit":"serving"}]

Input: "200g chicken breast and rice"
Output: [{"food":"chicken breast","quantity":200,"unit":"gram"},{"food":"rice","quantity":1,"unit":"serving"}]
"""

LLM_RERANK_PROMPT = """\
A user logged the food: "{query}"

Here are candidate database matches:
{candidates}

Which candidate best matches the user's food? 
Identify synonyms for similar foods, e.g. toast and bread
Do not just have a preference to select the first choice each time
Do not assume that the candidate list is already sorted in likely order
Reply with ONLY the number (1-{n}).\
"""

LLM_FEW_SHOT = [
    {"role": "user", "content": "two apples and a banana"},
    {"role": "assistant", "content": '{"items":[{"food":"apple","quantity":2,"unit":null},{"food":"banana","quantity":1,"unit":null}]}'},
    {"role": "user", "content": "spaghetti bolognaise with garlic bread"},
    {"role": "assistant", "content": '{"items":[{"food":"spaghetti bolognaise","quantity":1,"unit":"serving"},{"food":"garlic bread","quantity":1,"unit":"serving"}]}'},
    {"role": "user", "content": "porridge topped with honey and a cup of coffee"},
    {"role": "assistant", "content": '{"items":[{"food":"porridge","quantity":1,"unit":"serving"},{"food":"honey","quantity":1,"unit":"serving"},{"food":"coffee","quantity":1,"unit":"cup"}]}'},
]

DATA_BASE = "data/New Zealand FOODfiles 2024"
PRINCIPAL_XLSX = f"{DATA_BASE}/Principal files/Excel files"
SUPPORTING_XLSX = f"{DATA_BASE}/Supporting files/Excel files"
