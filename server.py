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
from rapidfuzz import fuzz, process as fuzz_process

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
        all_names = [entry["name"], entry["key_term"]] + entry["keywords"]
        best = max((fuzz.token_sort_ratio(span_lower, n) for n in all_names if n), default=0)
        scored.append((food_id, best))
    scored.sort(key=lambda x: -x[1])
    return [food_id for food_id, _ in scored[:limit]]


def fuzzy_search(span_text, food_index, threshold=60, limit=5):
    choices = {food_id: entry["key_term"] for food_id, entry in food_index.items()}
    results = fuzz_process.extract(
        span_text.lower(), choices,
        scorer=fuzz.token_sort_ratio,
        limit=limit,
        score_cutoff=threshold,
    )
    return [food_id for _, _, food_id in results]


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


def build_index(food_df, csm_df, name_df):
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

        seen = set()
        deduped = []
        for keyword in keywords:
            keyword = keyword.strip()
            if keyword and keyword not in seen:
                seen.add(keyword)
                deduped.append(keyword)
        keywords = deduped

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

food_index = build_index(food_df, csm_df, name_df)
recipe_index = build_recipe_index(ingredient_df, food_index)

print(f"Food index: {len(food_index)} entries | Recipes: {len(recipe_index)}")

if not Span.has_extension("food_id"):
    Span.set_extension("food_id", default=None)
if not Span.has_extension("candidates"):
    Span.set_extension("candidates", default=[])

nlp = spacy.load("en_core_web_md", disable=["ner"])
nlp.add_pipe("food_ner", last=True, config={"food_index": food_index})

app = FastAPI(title="Food NLP API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ExtractRequest(BaseModel):
    text: str


@app.post("/extract")
def extract(req: ExtractRequest):
    doc = nlp(req.text)
    results = []
    prev_end = 0

    for entity in doc.ents:
        if entity.label_ != "FOOD":
            continue

        ranked_ids = rank_candidates(entity.text, entity._.candidates, food_index)

        if not ranked_ids:
            ranked_ids = fuzzy_search(entity.text, food_index)

        quantity, quantity_char_start, quantity_char_end, unit, unit_char_start = extract_quantity(doc, entity.start, prev_end)
        prev_end = entity.end

        best_id = ranked_ids[0] if ranked_ids else entity._.food_id
        grams = resolve_grams(best_id, quantity, unit, food_index)

        candidates = []
        for food_id in ranked_ids[:5]:
            entry = food_index.get(food_id)
            if not entry:
                continue
            candidates.append({
                "food_id": food_id,
                "name": entry["name"],
                "is_recipe": entry.get("is_recipe", False),
                "recipe_ingredients": [
                    {
                        "food_id": c["ingredient_id"],
                        "name": c["ingredient_name"],
                        "weight_fraction": c["weight_fraction"],
                    }
                    for c in recipe_index.get(food_id, [])
                ],
                "nutrients": resolve_recipe_nutrients(food_id, grams, food_index, recipe_index),
            })

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
            "match": candidates[0] if candidates else None,
            "candidates": candidates,
        })

    return {"entities": results, "text": req.text}
