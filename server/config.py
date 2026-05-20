"""
Shared constants for the food NLP pipeline

Covers unit mappings, Ollama connection settings, and LLM prompt templates
"""

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

LLM_FEW_SHOT = [
    {"role": "user", "content": "two apples and a banana"},
    {"role": "assistant", "content": '{"items":[{"food":"apple","quantity":2,"unit":null},{"food":"banana","quantity":1,"unit":null}]}'},
    {"role": "user", "content": "spaghetti bolognaise with garlic bread"},
    {"role": "assistant", "content": '{"items":[{"food":"spaghetti bolognaise","quantity":1,"unit":"serving"},{"food":"garlic bread","quantity":1,"unit":"serving"}]}'},
    {"role": "user", "content": "porridge topped with honey and a cup of coffee"},
    {"role": "assistant", "content": '{"items":[{"food":"porridge","quantity":1,"unit":"serving"},{"food":"honey","quantity":1,"unit":"serving"},{"food":"coffee","quantity":1,"unit":"cup"}]}'},
]

RAG_RERANK_ENABLED = True # Change to false if cross-encoding reranking intended
RAG_TOP_N = 6 # candidates passed to LLM
RAG_SCORE_GAP_THRESH = 25 # skip RAG if top candidate leads by this much
RAG_SYSTEM_PROMPT = """\
You are a food database expert for NZ FOODfiles 2024.
Given a user's food description and a numbered list of database candidates, \
respond with ONLY the integer index (1-based) of the best matching candidate.
If none are a reasonable match, respond with 0.
Do not explain your choice."""

NZ_DATA_BASE = "data/New Zealand FOODfiles 2024"
PRINCIPAL_XLSX = f"{NZ_DATA_BASE}/Principal files/Excel files"
SUPPORTING_XLSX = f"{NZ_DATA_BASE}/Supporting files/Excel files"
AUS_DATA_DIR = "data/AUSNUT 2023"
OFF_DATA_CSV = "data/Open Food Facts/openfoodfacts-data.csv"
