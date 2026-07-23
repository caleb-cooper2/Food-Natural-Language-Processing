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
Extract all food items from the input. Return JSON only. No explanation.

Rules:
- "food" MUST be an exact substring of the input text — copy it verbatim.
- ALWAYS include brand names as part of the food string. "vogels toast" not "toast", "anchor butter" not "butter".
- If a dish is described AND its components are listed, extract ONLY the components, not the dish name.
  Example: "pie floater — a mince pie in a bowl of pea soup" -> extract "mince pie" and "pea soup", NOT "pie floater".
- Do NOT generalise, simplify, or rephrase food names.
- Do NOT split a branded product into brand + generic (e.g. "weet-bix" stays as "weet-bix").
- Coffee drinks are single items — extract the full compound name: "flat white", "long black", "trim latte".
- Wine and spirits are food items — extract the full variety name: "sauvignon blanc", "pinot noir", "ginger beer".
- Multi-word dish names stay together: "chicken satay skewers", "garlic naan", "butter chicken", "pad thai".
- If a food's physical form or preparation is stated, add prep=<that word>, copied verbatim from the
  input (e.g. "carrot sticks" -> prep "sticks"; "grated cheese" -> prep "grated"; "whole apple" ->
  prep "whole"; "mashed potato" -> prep "mashed"). If no form is stated, omit prep. Do NOT guess a form.
- Tea: always extract the full variety name as the food. For "English Breakfast tea" or bare "cup of tea",
  add normalised="black tea". For other teas (green, peppermint, herbal) leave normalised absent.
- - Informal slang, abbreviations, and contractions (single or multi-word): keep the user's exact
  wording in "food" and put the expansion in "normalised". NEVER replace "food" with the expansion.
  "OJ"→"orange juice", "sav"→"sauvignon blanc", "spag bol"→"spaghetti bolognaise".
- If the input contains no identifiable food (e.g. "had dinner", "had a drink", "ate something"), return {"items":[]}.

Return format: {"items": [...]}
Each item: {"food": string, "quantity": number, "unit": string|null, "normalised": string|null (optional), "prep": string|null (optional)}
If no quantity stated, use 1. If no unit, use null.
"""

LLM_FEW_SHOT = [
    {"role": "user", "content": "two apples and a banana"},
    {"role": "assistant", "content": '{"items":[{"food":"apples","quantity":2,"unit":null},{"food":"banana","quantity":1,"unit":null}]}'},

    {"role": "user", "content": "spaghetti bolognaise with garlic bread"},
    {"role": "assistant", "content": '{"items":[{"food":"spaghetti bolognaise","quantity":1,"unit":"serving"},{"food":"garlic bread","quantity":1,"unit":"serving"}]}'},

    {"role": "user", "content": "porridge topped with honey and a cup of coffee"},
    {"role": "assistant", "content": '{"items":[{"food":"porridge","quantity":1,"unit":"serving"},{"food":"honey","quantity":1,"unit":"serving"},{"food":"coffee","quantity":1,"unit":"cup"}]}'},

    # Brand preservation
    {"role": "user", "content": "I had a slice of vogels toast with anchor butter"},
    {"role": "assistant", "content": '{"items":[{"food":"vogels toast","quantity":1,"unit":"slice"},{"food":"anchor butter","quantity":1,"unit":"serving"}]}'},

    # Composite dish deduplication
    {"role": "user", "content": "a pie floater which is a mince pie in a bowl of pea soup"},
    {"role": "assistant", "content": '{"items":[{"food":"mince pie","quantity":1,"unit":"serving"},{"food":"pea soup","quantity":1,"unit":"serving"}]}'},

    # Brand without explicit product type
    {"role": "user", "content": "bowl of weet-bix with trim milk"},
    {"role": "assistant", "content": '{"items":[{"food":"weet-bix","quantity":1,"unit":"serving"},{"food":"trim milk","quantity":1,"unit":"serving"}]}'},

    # Coffee drinks - full compound name
    {"role": "user", "content": "drank a flat white and had a long black after"},
    {"role": "assistant", "content": '{"items":[{"food":"flat white","quantity":1,"unit":null},{"food":"long black","quantity":1,"unit":null}]}'},

    # Wine varieties
    {"role": "user", "content": "two glasses of sauvignon blanc with camembert"},
    {"role": "assistant", "content": '{"items":[{"food":"sauvignon blanc","quantity":2,"unit":"glass"},{"food":"camembert","quantity":1,"unit":"serving"}]}'},

    # Multi-word dish names
    {"role": "user", "content": "chicken satay skewers with steamed jasmine rice"},
    {"role": "assistant", "content": '{"items":[{"food":"chicken satay skewers","quantity":1,"unit":"serving"},{"food":"jasmine rice","quantity":1,"unit":"serving"}]}'},

    # Abbreviation normalization
    {"role": "user", "content": "had some OJ and a sav before dinner"},
    {"role": "assistant", "content": '{"items":[{"food":"OJ","quantity":1,"unit":"glass","normalised":"orange juice"},{"food":"sav","quantity":1,"unit":"glass","normalised":"sauvignon blanc"}]}'},

    # Tea normalisation
    {"role": "user", "content": "a cup of English Breakfast tea and a biscuit"},
    {"role": "assistant", "content": '{"items":[{"food":"English Breakfast tea","quantity":1,"unit":"cup","normalised":"black tea"},{"food":"biscuit","quantity":1,"unit":null}]}'},

    # Slang / informal - bubbles = sparkling wine
    {"role": "user", "content": "pavlova and a glass of bubbles for dessert"},
    {"role": "assistant", "content": '{"items":[{"food":"pavlova","quantity":1,"unit":"serving"},{"food":"bubbles","quantity":1,"unit":"glass","normalised":"sparkling wine"}]}'},

    # Multi-word contraction - verbatim in food, expansion in normalised
    {"role": "user", "content": "spag bol for dinner"},
    {"role": "assistant", "content": '{"items":[{"food":"spag bol","quantity":1,"unit":"serving","normalised":"spaghetti bolognaise"}]}'},

    # Vague input - return empty
    {"role": "user", "content": "had dinner"},
    {"role": "assistant", "content": '{"items":[]}'},

    {"role": "user", "content": "ate something from the fridge"},
    {"role": "assistant", "content": '{"items":[]}'},

    # Preparation / form extraction
    {"role": "user", "content": "carrot sticks and a handful of grated cheese"},
    {"role": "assistant", "content": '{"items":[{"food":"carrot sticks","quantity":1,"unit":null,"prep":"sticks"},{"food":"grated cheese","quantity":1,"unit":"handful","prep":"grated"}]}'},

    # No form stated -> no prep field
    {"role": "user", "content": "an apple and a banana"},
    {"role": "assistant", "content": '{"items":[{"food":"apple","quantity":1,"unit":null},{"food":"banana","quantity":1,"unit":null}]}'}
]

ENTITY_MIN_CONFIDENCE = 0.35
RAG_RERANK_ENABLED = True # Change to false if cross-encoding reranking intended
RAG_TOP_N = 6 # candidates passed to LLM
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
