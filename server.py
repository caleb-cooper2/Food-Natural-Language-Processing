import json
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
from spacy.matcher import PhraseMatcher
from spacy.language import Language
from spacy.tokens import Span
import pandas as pd
import re
import math
from text_to_num import alpha2digit
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

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

def simple_plural(word):
    if word.endswith('y') and len(word) > 2 and word[-2] not in 'aeiou':
        return word[:-1] + 'ies'
    if word.endswith(('s', 'x', 'z', 'ch', 'sh')):
        return word + 'es'
    if word.endswith('f') and not word.endswith('ff'):
        return word[:-1] + 'ves'
    if word.endswith('fe'):
        return word[:-2] + 'ves'
    return word + 's'


def simple_singular(word):
    if word.endswith('ies') and len(word) > 3:
        return word[:-3] + 'y'
    if word.endswith('ves') and len(word) > 3:
        return word[:-3] + 'f'
    if word.endswith('es') and len(word) > 4 and word[-3] in 'sxz':
        return word[:-2]
    if word.endswith('s') and not word.endswith('ss') and len(word) > 3:
        return word[:-1]
    return word


def term_variants(term):
    words = term.split()
    if not words:
        return [term]
    plural = ' '.join(words[:-1] + [simple_plural(words[-1])])
    return [term] if plural == term else [term, plural]


def extract_quantity(doc, span_start, prev_end=0):
    window = doc[max(prev_end, span_start - 6): span_start]
    if not window:
        return 1.0, None, None, None, None

    original_tokens = list(window)
    window_text = window.text
    converted = alpha2digit(window_text, "en")

    for token in reversed(original_tokens):
        if token.text.lower() in ("a", "an"):
            return 1.0, token.idx, token.idx + len(token.text), None, None

    quantity = 1.0
    quantity_char_start = quantity_char_end = None
    unit = None
    unit_char_start = None

    original_words = window_text.split()
    converted_words = converted.split()
    original_index = 0
    num_words_consumed = 1

    for converted_word in converted_words:
        try:
            quantity = float(converted_word)
            num_words_consumed = len(original_words) - len(converted_words) + 1
            start_token = original_tokens[original_index]
            end_token = original_tokens[min(original_index + num_words_consumed - 1, len(original_tokens) - 1)]
            quantity_char_start = start_token.idx
            quantity_char_end = end_token.idx + len(end_token.text)
            break
        except ValueError:
            original_index += 1

    next_index = original_index + num_words_consumed
    while next_index < len(original_tokens):
        token = original_tokens[next_index]
        token_lower = token.text.lower()
        if token_lower in ALL_UNITS:
            unit = token_lower
            unit_char_start = token.idx
            break
        if token_lower in ("of", "the"):
            next_index += 1
            continue
        break

    return quantity, quantity_char_start, quantity_char_end, unit, unit_char_start


def extract_keywords(food_name_string):
    cleaned = re.sub(r'[™®©]', '', food_name_string)
    paren_terms = re.findall(r'\((.*?)\)', cleaned)
    cleaned = re.sub(r'\(.*?\)', '', cleaned)
    keywords = [p.strip().lower() for p in cleaned.split(',') if p.strip()]
    keywords += [t.strip().lower() for t in paren_terms if t.strip()]
    return keywords

def extract_brand_keywords(food_name, sampling_details = "", nlp = None):
    # Try to find brand names through a number of checks
    brands = []
    segments = [s.strip() for s in food_name.split(',')]

    trademark_matches = re.findall(r'\b([A-Za-z][\w\-]*)[™®©]', food_name)
    brands.extend([brand.lower() for brand in trademark_matches])

    if nlp is not None:
        doc = nlp(food_name)
        for entity in doc.ents:
            if entity.label_ in ("ORG", "PRODUCT"):
                candidate = entity.text.strip().lower()
                if len(candidate) > 2 and candidate not in ALL_UNITS:
                    brands.append(candidate)

    for segment in segments[2:]:
        cleaned_segment = re.sub(r'[™®©]', '', segment).strip()
        tokens = cleaned_segment.split()

        is_proper = all(token[0].isupper() for token in tokens if token and token[0].isalpha())
        if is_proper and 1 <= len(tokens) <= 3:
            brands.append(cleaned_segment.lower())

    if sampling_details:
        pattern = re.search(
            r'brands?[:\s]+([A-Za-z0-9\-\s(),]+?)(?:\.|mixed|total|sampled|respectively)',
            sampling_details, re.IGNORECASE
        )
        if pattern:
            brand_text = pattern.group(1)
            for chunk in re.split(r'\band\b|,', brand_text):
                chunk = re.sub(r'\(.*?\)', '', chunk).strip().lower()
                if chunk and len(chunk) > 1:
                    brands.append(chunk)

    return list(dict.fromkeys(brand for brand in brands if brand))

def find_item_char_positions(item, original_text):
    # Used for llm extraction of char positions
    food = item["food"]
    text_lower = original_text.lower()
    char_start = text_lower.find(food.lower())
    if char_start == -1:
        return None, None, None, None, None
    char_end = char_start + len(food)

    prefix_lower = original_text[:char_start].lower()
    quantity = item.get("quantity", 1.0)
    unit = item.get("unit")

    unit_char_start = None
    quantity_char_start = None
    quantity_char_end = None

    if unit:
        pos = prefix_lower.rfind(unit.lower())
        if pos != -1:
            unit_char_start = pos

    if quantity != 1.0:
        qty_str = str(int(quantity)) if quantity == int(quantity) else str(quantity)
        pos = prefix_lower.rfind(qty_str)
        if pos != -1:
            quantity_char_start = pos
            quantity_char_end = pos + len(qty_str)

    return char_start, char_end, quantity_char_start, quantity_char_end, unit_char_start

def build_food_matcher(nlp, food_index):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    term_candidates = {}

    for food_id, entry in food_index.items():
        terms = list(dict.fromkeys([entry["key_term"]] + entry["keywords"]))
        for term in terms:
            if len(term) <= 3:
                continue
            for variant in term_variants(term):
                term_candidates.setdefault(variant, []).append(food_id)

    for term, food_ids in term_candidates.items():
        ranked = sorted(list(dict.fromkeys(food_ids)), key=lambda food_id: len(food_index[food_id]["keywords"]))
        term_candidates[term] = ranked
        try:
            matcher.add(ranked[0], [nlp.make_doc(term)])
        except Exception:
            pass

    return matcher, term_candidates


@Language.factory("food_ner")
def create_food_ner(nlp, name, food_index): # spaCy passes this automatically, have to leave name unused as a result
    matcher, term_candidates = build_food_matcher(nlp, food_index)
    return FoodNERComponent(nlp, food_index, matcher, term_candidates)


class FoodNERComponent:
    def __init__(self, nlp, food_index, matcher, term_candidates):
        self.food_index = food_index
        self.matcher = matcher
        self.term_candidates = term_candidates

    def __call__(self, doc):
        matches = self.matcher(doc)
        spans = []
        for match_id, start, end in matches:
            food_id = doc.vocab.strings[match_id]
            span = Span(doc, start, end, label="FOOD")
            span._.food_id = food_id
            term = doc[start:end].text.lower()
            candidates = (
                self.term_candidates.get(term)
                or self.term_candidates.get(simple_singular(term))
                or [food_id]
            )
            span._.candidates = candidates
            spans.append(span)

        doc.ents = spacy.util.filter_spans(list(doc.ents) + spans)
        return doc


def rank_candidates(span_text, candidate_ids, food_index, limit=5):
    span_lower = span_text.lower()
    scored = []
    for food_id in candidate_ids:
        entry = food_index.get(food_id)
        if not entry:
            continue

        key_term = entry["key_term"]

        score = candidate_scorer(span_lower, key_term)

        semantic_hits = semantic_search(span_text, faiss_index, faiss_ids, embedding_model, k=5)
        semantic_map = {fid: s for fid, s in semantic_hits}

        if food_id in semantic_map:
            score += semantic_map[food_id] * 100

        token_count = len(key_term.split())
        query_tokens = len(span_lower.split())

        brands = entry.get("brands", [])
        if any(brand in span_lower for brand in brands):
            score += 25

        if query_tokens == 1:
            score -= (token_count - 1) * 5

        part = entry.get("part")
        if query_tokens == 1 and part:
            score += 5

        scored.append((food_id, score))

    scored.sort(key=lambda x: -x[1])
    return scored[:limit]

def candidate_scorer(query, candidate_key_term):
    query_lower = query.lower().strip()
    candidate_lower = candidate_key_term.lower().strip()

    if query_lower == candidate_lower:
        return 100.0

    query_tokens = set(query_lower.split())
    candidate_tokens = set(candidate_lower.split())

    base_score = fuzz.token_sort_ratio(query_lower, candidate_lower)

    extra_tokens = candidate_tokens - query_tokens
    specificity_penalty = (len(extra_tokens) / max(len(candidate_tokens), 1)) * 40

    length_ratio = min(len(query_lower), len(candidate_lower)) / max(len(query_lower), len(candidate_lower))
    length_bonus = length_ratio * 10

    return max(0.0, base_score - specificity_penalty + length_bonus)

def fuzzy_search(span_text, food_index, threshold=60, limit=5):
    span_lower = span_text.lower().strip()
    scored = []

    for food_id, entry in food_index.items():
        key_term = entry["key_term"]
        score = candidate_scorer(span_lower, key_term)

        if score >= threshold:
            scored.append((food_id, score))

    scored.sort(key=lambda x: -x[1])
    return scored[:limit]

def semantic_search(query, faiss_index, faiss_ids, model, k=10):
    vec = model.encode([query])
    vec = vec / np.linalg.norm(vec)

    scores, indices = faiss_index.search(vec.astype(np.float32), k)

    return [(faiss_ids[i], float(scores[0][j])) for j, i in enumerate(indices[0])]

def retrieve_candidates(food_description, food_index, limit=20):
    doc = nlp(food_description)
    food_entities = [entity for entity in doc.ents if entity.label_ == "FOOD"]

    candidates = []
    seen = set()

    for entity in food_entities:
        for candidate_id in entity._.candidates:
            if candidate_id not in seen:
                seen.add(candidate_id)
                candidates.append(candidate_id)

    if len(candidates) < limit:
        for food_id, entry in food_index.items():
            if food_id in seen:
                continue
            score = fuzz.token_sort_ratio(food_description.lower(), entry["key_term"])
            if score > 50:
                seen.add(food_id)
                candidates.append(food_id)

    semantic_hits = semantic_search(food_description, faiss_index, faiss_ids, embedding_model)

    for food_id, score in semantic_hits:
        if food_id not in seen:
            seen.add(food_id)
            candidates.append(food_id)

    return candidates[:limit]

def compute_confidence(ranked_scores):
    if not ranked_scores:
        return 0.0
    if len(ranked_scores) == 1:
        return ranked_scores[0] / 100.0

    gap = ranked_scores[0] - ranked_scores[1]
    return min(1.0, (ranked_scores[0] / 100.0) * (1 + gap / 100.0))

def resolve_grams(food_id, quantity, unit, food_index):
    if unit in UNIT_GRAMS:
        return quantity * UNIT_GRAMS[unit]

    servings = food_index.get(food_id, {}).get("serving_measure", [])

    if unit and servings:
        for serving in servings:
            if unit.lower() in str(serving.get("CSM", "")).lower():
                try:
                    return quantity * float(serving["Measure"])
                except (TypeError, ValueError):
                    pass

    if servings:
        try:
            return quantity * float(servings[0]["Measure"])
        except (TypeError, ValueError):
            pass

    return quantity * 100.0


def calculate_nutrients(food_id, grams, food_index):
    entry = food_index.get(food_id)
    if not entry:
        return {}
    factor = grams / 100.0
    return {
        nutrient: round(value * factor, 2) if value is not None else None
        for nutrient, value in entry["nutrients"].items()
    }


def resolve_recipe_nutrients(food_id, grams, food_index, recipe_index):
    if food_id not in recipe_index:
        return calculate_nutrients(food_id, grams, food_index)

    total = {}
    for component in recipe_index[food_id]:
        ingredient_grams = grams * component["weight_fraction"]
        for nutrient, value in calculate_nutrients(component["ingredient_id"], ingredient_grams, food_index).items():
            if value is not None:
                total[nutrient] = round(total.get(nutrient, 0.0) + value, 2)

    return total or calculate_nutrients(food_id, grams, food_index)


def build_index(food_df, csm_df, name_df, nlp):
    csm_lookup = {}
    if "FoodID" in csm_df.columns:
        csm_df = csm_df.copy()
        csm_df["FoodID"] = csm_df["FoodID"].astype(str).str.strip()
        for food_id, group in csm_df.groupby("FoodID"):
            csm_lookup[str(food_id)] = group[["CSM", "Measure"]].to_dict("records")

    name_lookup = {}
    if name_df is not None:
        for _, row in name_df.iterrows():
            food_id = str(row.get("FoodID", "")).strip()
            if food_id:
                name_lookup[food_id] = row

    index = {}

    for _, row in food_df.iterrows():
        food_id = str(row.get("FoodID", "")).strip()
        name = str(row.get("Food Name", "")).strip()
        if not food_id or not name:
            continue

        keywords = extract_keywords(name)

        name_row = name_lookup.get(food_id)
        if name_row is not None:
            short_name = str(name_row.get("Short Food Name") or "").strip()
            if short_name and short_name.lower() != "nan":
                keywords += [t.strip().lower() for t in re.split(r'[;,]', short_name) if t.strip()]

            alt_names = str(name_row.get("AlternativeNames") or "").strip()
            if alt_names and alt_names.lower() != "nan":
                keywords += [t.strip().lower() for t in re.split(r'[;,]', alt_names) if t.strip()]
            generic = str(name_row.get("Generic Name") or "").strip()
            kind = str(name_row.get("Kind") or "").strip()
            if kind and kind.lower() != "nan" and generic and generic.lower() != "nan":
                keywords.append(f"{kind} {generic}".strip().lower())
            elif generic and generic.lower() != "nan":
                keywords.append(generic.strip().lower())
            part = str(name_row.get("Part") or "").strip()
            sampling_details = str(name_row.get("Sampling Details") or "").strip()

        seen = set()
        deduped = []
        for keyword in keywords:
            keyword = keyword.strip()
            if keyword and keyword not in seen:
                seen.add(keyword)
                deduped.append(keyword)
        brand_keywords = extract_brand_keywords(name, sampling_details, nlp)
        keywords = brand_keywords + deduped

        def clean(value):
            try:
                as_float = float(value)
                return None if math.isnan(as_float) else as_float
            except (TypeError, ValueError):
                return None

        index[food_id] = {
            "name": name,
            "keywords": keywords,
            "key_term": keywords[0] if keywords else name.split(",")[0].lower(),
            "part": part if part and part.lower() != "nan" else None,
            "brands": brand_keywords,
            "serving_measure": csm_lookup.get(food_id, []),
            "is_recipe": food_id.startswith("R"),
            "nutrients": {
                "energy_kj": clean(row.get("Energy, total metabolisable (kJ)")),
                "protein_g": clean(row.get("Protein, total; calculated from total nitrogen")),
                "fat_g": clean(row.get("Fat, total")),
                "carbs_g": clean(row.get("Available carbohydrate, FSANZ")),
                "fibre_g": clean(row.get("Fibre, total dietary")),
                "sodium_mg": clean(row.get("Sodium")),
            }
        }
    return index


def build_recipe_index(ingredient_df, food_index):
    recipe_index = {}
    if ingredient_df is None:
        return recipe_index

    weight_col = "Weight Fraction(%)" if "Weight Fraction(%)" in ingredient_df.columns else "Weight Fraction (%)"

    for _, row in ingredient_df.iterrows():
        recipe_id = str(row.get("Recipe FoodID", "")).strip()
        ingredient_id = str(row.get("Ingredient FoodID", "")).strip()
        try:
            fraction = float(row.get(weight_col, 0)) / 100.0
        except (TypeError, ValueError):
            fraction = 0.0

        if recipe_id and ingredient_id and fraction > 0 and ingredient_id in food_index:
            recipe_index.setdefault(recipe_id, []).append({
                "ingredient_id": ingredient_id,
                "ingredient_name": food_index[ingredient_id]["name"],
                "weight_fraction": fraction,
            })

    return recipe_index

def build_embedding_index(food_index, model):
    ids = list(food_index.keys())

    texts = [
        f"{entry['key_term']} {' '.join(entry['keywords'][:3])}"
        for entry in food_index.values()
    ]

    embeddings = model.encode(texts, batch_size=128, show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))

    return index, ids


BASE = "data/New Zealand FOODfiles 2024"

csm_df = pd.read_excel(f"{BASE}/Principal files/Excel files/CSM.FT.XLSX", skiprows=1)
csm_df.columns = csm_df.columns.str.strip()

food_df = pd.read_excel(f"{BASE}/Principal files/Excel files/Unabridged/Unabridged DATA.AP.xlsx", skiprows=1)
food_df = food_df[food_df["FoodID"] != "FoodID"]  # drop units header row
food_df.columns = food_df.columns.str.strip()

name_df = pd.read_excel(f"{BASE}/Supporting files/Excel files/NAME.FT.XLSX", skiprows=1)
name_df.columns = name_df.columns.str.strip()

ingredient_df = pd.read_excel(f"{BASE}/Principal files/Excel files/INGREDIENT.FT.XLSX", skiprows=1)
ingredient_df.columns = ingredient_df.columns.str.strip()

nlp_ner = spacy.load("en_core_web_md")
food_index = build_index(food_df, csm_df, name_df, nlp_ner)
recipe_index = build_recipe_index(ingredient_df, food_index)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index, faiss_ids = build_embedding_index(food_index, embedding_model)

print(f"Food index: {len(food_index)} entries | Recipes: {len(recipe_index)}")
print(f"FAISS index build: {len(faiss_ids)} vectors")

if not Span.has_extension("food_id"):
    Span.set_extension("food_id", default=None)
if not Span.has_extension("candidates"):
    Span.set_extension("candidates", default=[])

nlp = spacy.load("en_core_web_md", disable=["ner"])
nlp.add_pipe("food_ner", last=True, config={"food_index": food_index})

def validate_llm_items(llm_items, food_index, nlp):
    enriched = []

    for item in llm_items:
        food_name = item["food"]

        # First, normal linking
        candidate_ids = retrieve_candidates(food_name, food_index)
        ranked = rank_candidates(food_name, candidate_ids, food_index)

        if ranked:
            ranked_scores = [score for _, score in ranked]
            item["link_confidence"] = compute_confidence(ranked_scores)
        else:
            item["link_confidence"] = 0.0

        # Next, fall back to brand
        if item["link_confidence"] < 0.4:
            brands = extract_brand_keywords(food_name, nlp=nlp)

            for brand in brands:
                generic_attempt = food_name.replace(brand, "").strip()

                if not generic_attempt:
                    continue

                candidate_ids = retrieve_candidates(generic_attempt, food_index)
                ranked = rank_candidates(generic_attempt, candidate_ids, food_index)

                if ranked:
                    score = ranked[0][1] / 100.0
                    if score > item["link_confidence"]:
                        item["food_generic"] = generic_attempt
                        item["link_confidence"] = score

        enriched.append(item)

    return enriched

def validate_llm_grounding(items, original_text):
    text_lower = original_text.lower()
    validated = []

    for item in items:
        food = item["food"].lower()

        if food not in text_lower:
            continue

        validated.append(item)

    return validated if validated else None

async def llm_extract(text):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            # Few-shot examples as conversation turns
            {"role": "user", "content": "two apples and a banana"},
            {"role": "assistant", "content": '{"items":[{"food":"apple","quantity":2,"unit":null},{"food":"banana","quantity":1,"unit":null}]}'},
            {"role": "user", "content": "spaghetti bolognaise with garlic bread"},
            {"role": "assistant", "content": '{"items":[{"food":"spaghetti bolognaise","quantity":1,"unit":"serving"},{"food":"garlic bread","quantity":1,"unit":"serving"}]}'},
            {"role": "user", "content": "porridge topped with honey and a cup of coffee"},
            {"role": "assistant", "content": '{"items":[{"food":"porridge","quantity":1,"unit":"serving"},{"food":"honey","quantity":1,"unit":"serving"},{"food":"coffee","quantity":1,"unit":"cup"}]}'},
            # Actual request
            {"role": "user", "content": text},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0, # must be deterministic for a parser
            "num_predict": 512,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            resp.raise_for_status()

        print(resp.json())

        raw = resp.json()["message"]["content"]
        parsed = json.loads(raw)

        print("Raw LLM output: ", raw)

        if isinstance(parsed, list):
            pass

        if isinstance(parsed, dict) and "items" in parsed:
            parsed = parsed["items"]

        elif isinstance(parsed, dict) and all(k in parsed for k in ("food", "quantity", "unit")):
            foods = parsed.get("food")
            quantities = parsed.get("quantity")
            units = parsed.get("unit")

            # If single values → wrap into lists
            if not isinstance(foods, list):
                foods = [foods]
            if not isinstance(quantities, list):
                quantities = [quantities]
            if not isinstance(units, list):
                units = [units]

            max_len = max(len(foods), len(quantities), len(units))

            def expand(lst):
                if len(lst) == max_len:
                    return lst
                if len(lst) == 1:
                    return lst * max_len
                return lst[:max_len]  # fallback (rare edge case)

            foods = expand(foods)
            quantities = expand(quantities)
            units = expand(units)

            parsed = [
                {
                    "food": str(foods[i]).strip().lower(),
                    "quantity": float(quantities[i]) if quantities[i] is not None else 1.0,
                    "unit": None if units[i] in (None, "null") else str(units[i]).lower()
                }
                for i in range(max_len)
            ]

        else:
            return None

        validated: list[dict] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            food = str(item.get("food", "")).strip().lower()
            if not food:
                continue
            validated.append({
                "food": food,
                "quantity": float(item.get("quantity") or 1.0),
                "unit": str(item["unit"]).lower() if item.get("unit") else None,
            })

        validated = validated if validated else None

        if validated:
            validated = validate_llm_grounding(validated, text)

        return validated

    except (httpx.ConnectError, httpx.TimeoutException):
        print("Ollama unavailable - falling back to spaCy pipeline")
        return None
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        print(f"LLM Parse error ({exc}) - falling back to spaCy pipeline")
        return None

def _build_candidate_list(ranked, grams):
    candidates = []
    for food_id, score in ranked[:5]:
        entry = food_index.get(food_id)
        if not entry:
            continue
        candidates.append({
            "food_id": food_id,
            "name": entry["name"],
            "score": round(score, 2),
            "is_recipe": entry.get("is_recipe", False),
            "recipe_ingredients": [
                {
                    "food_id": c["ingredient_id"],
                    "name": c["ingredient_name"],
                    "weight_fraction": c["weight_fraction"],
                }
                for c in recipe_index.get(food_id, [])
            ],
            "nutrients": resolve_recipe_nutrients(
                food_id, grams, food_index, recipe_index
            ),
        })
    return candidates

app = FastAPI(title="Food NLP API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ExtractRequest(BaseModel):
    text: str


@app.post("/extract")
async def extract(req: ExtractRequest):
    llm_items = await llm_extract(req.text)
    if llm_items is not None:
        llm_items = validate_llm_items(llm_items, food_index, nlp)
        avg_conf = (
            sum(item["link_confidence"] for item in llm_items) / len(llm_items)
            if llm_items else 0
        )
        if avg_conf < 0.5:
            llm_items = None

        results = []
        if llm_items is not None:
            for item in llm_items:
                food_description = item.get("food_generic") or item["food"]
                quantity = item["quantity"]
                unit = item["unit"]

                candidate_ids = retrieve_candidates(food_description, food_index)
                ranked = rank_candidates(food_description, candidate_ids, food_index)

                if not ranked:
                    continue

                ranked_ids = [cid for cid, _ in ranked]
                ranked_scores = [score for _, score in ranked]

                confidence = compute_confidence(ranked_scores)
                best_id = ranked_ids[0]

                grams = resolve_grams(best_id, quantity, unit, food_index)
                candidates = _build_candidate_list(ranked, grams)

                char_start, char_end, quantity_char_start, quantity_char_end, unit_char_start = find_item_char_positions(item, req.text)

                results.append({
                    "text": item["food"],
                    "resolved_text": food_description,
                    "char_start": char_start,
                    "char_end": char_end,
                    "quantity": quantity,
                    "quantity_char_start": quantity_char_start,
                    "quantity_char_end": quantity_char_end,
                    "unit": unit,
                    "unit_char_start": unit_char_start,
                    "grams": grams,
                    "confidence": confidence,
                    "match": candidates[0] if candidates else None,
                    "candidates": candidates,
                    "source": "llm",
                })
            print("LLM used")
            return {"entities": results, "text": req.text, "source": "llm"}

    doc = nlp(req.text)
    results = []
    prev_end = 0

    for entity in doc.ents:
        if entity.label_ != "FOOD":
            continue

        candidate_ids = retrieve_candidates(entity.text, food_index)
        ranked = rank_candidates(entity.text, candidate_ids, food_index)

        if not ranked:
            ranked = fuzzy_search(entity.text, food_index)

        if not ranked:
            continue

        ranked_ids = [candidate_id for candidate_id, _ in ranked]
        ranked_scores = [score for _, score in ranked]

        confidence = compute_confidence(ranked_scores)

        quantity, quantity_char_start, quantity_char_end, unit, unit_char_start = extract_quantity(doc, entity.start, prev_end)
        prev_end = entity.end

        best_id = ranked_ids[0]
        grams = resolve_grams(best_id, quantity, unit, food_index)

        candidates = _build_candidate_list(ranked, grams)

        results.append({
            "text": entity.text,
            "char_start": entity.start_char,
            "char_end": entity.end_char,
            "quantity": quantity,
            "quantity_char_start": quantity_char_start,
            "quantity_char_end": quantity_char_end,
            "unit": unit,
            "unit_char_start": unit_char_start,
            "grams": grams,
            "confidence": confidence,
            "match": candidates[0] if candidates else None,
            "candidates": candidates,
            "source": "spacy",
        })
    print("spaCy used")
    return {"entities": results, "text": req.text, "source": "spacy"}
