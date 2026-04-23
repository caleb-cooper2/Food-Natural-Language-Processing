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

def extract_quantity(doc, span_start, prev_end = 0):
    window = doc[max(prev_end, span_start - 5): span_start]  # don't cross previous entity
    if not window:
        return 1.0, None, None

    window_text = window.text
    converted = alpha2digit(window_text, "en")

    quantity = 1.0
    quantity_start = quantity_end = None

    # Check for "a" or "an" first
    for token in reversed(list(window)):
        if token.text.lower() in ("a", "an"):
            quantity = 1.0
            quantity_start = token.idx
            quantity_end = token.idx + len(token.text)
            return quantity, quantity_start, quantity_end

    # Split both original and converted into token lists
    original_tokens = list(window)
    original_words = window_text.split()
    converted_words = converted.split()

    # Find which converted word is numeric, and which original words it replaced
    original_index = 0  # original word index
    for _, converted_word in enumerate(converted_words):
        try:
            quantity = float(converted_word)

            number_original_words_consumed = len(original_words) - len(converted_words) + 1

            start_token = original_tokens[original_index]
            end_token = original_tokens[min(original_index + number_original_words_consumed - 1, len(original_tokens) - 1)]
            quantity_start = start_token.idx
            quantity_end = end_token.idx + len(end_token.text)
            break
        except ValueError: # thrown when float(converted_word) is attempted on a normal word (not a number)
            original_index += 1

    return quantity, quantity_start, quantity_end

def build_food_matcher(nlp, food_index):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    term_candidates = {} # dict of str: list(strs)

    for food_id, entry in food_index.items():
        terms = list(dict.fromkeys([entry["key_term"]] + entry["keywords"]))
        for term in terms:
            if len(term) > 3:
                term_candidates.setdefault(term, []).append(food_id)

    for term, food_ids in term_candidates.items():
        # assume that the most generic food item is the one with the least amount of keywords
        ranked = sorted(food_ids, key=lambda fid: len(food_index[fid]["keywords"]))
        term_candidates[term] = ranked
        matcher.add(ranked[0], [nlp.make_doc(term)])

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
            span._.candidates = self.term_candidates.get(term, [food_id])
            spans.append(span)

        doc.ents = spacy.util.filter_spans(list(doc.ents) + spans)
        return doc

def extract_keywords(food_name_string):
    cleaned_name = re.sub(r'[™®©]', '', food_name_string)
    paren_terms = re.findall(r'\((.*?)\)', cleaned_name)
    cleaned_name = re.sub(r'\(.*?\)', '', cleaned_name)

    keywords = [p.strip().lower() for p in cleaned_name.split(',') if p.strip()]

    for term in paren_terms:
        keywords.append(term.strip().lower())

    return keywords

def resolve_grams(food_id, quantity_consumed, food_index):
    servings = food_index.get(food_id, {}).get("serving_measure", [])

    if not servings:
        return None

    try:
        return quantity_consumed * float(servings[0]["Measure"])
    except (TypeError, ValueError):
        return None

def calculate_nutrients(food_id, grams, food_index):
    entry = food_index.get(food_id)
    if not entry:
        return {}
    factor = grams / 100.0

    return {nutrient_per_gram: round(nutrient_per_gram * factor, 2) if nutrient_per_gram else None
            for nutrient, nutrient_per_gram in entry["nutrients"].items()}

def build_index():
    index = {}
    for _, row in food_df.iterrows():
        food_id = row["FoodID"]
        name = row["Food Name"]
        keywords = extract_keywords(name)

        food_serving_measure = csm_df[csm_df["FoodID"] == food_id][["CSM", "Measure"]].to_dict("records") if "FoodID" in csm_df.columns else []

        def clean(value):
            converted_float = float(value)
            return None if math.isnan(converted_float) else converted_float

        index[food_id] = {
            "name": name,
            "keywords": keywords,
            "key_term": keywords[0] if keywords else name.split(",")[0].lower(),
            "serving_measure": food_serving_measure,
            "nutrients": { # for now, just a summary of a few key nutrients
                "energy_kj": clean(row.get("Energy, total metabolisable (kJ)")),
                "protein_g": clean(row.get("Protein, total; calculated from total nitrogen")),
                "fat_g": clean(row.get("Fat, total")),
                "carbs_g": clean(row.get("Available carbohydrate, FSANZ")),
                "fibre_g": clean(row.get("Fibre, total dietary")),
                "sodium_mg": clean(row.get("Sodium")),
            }
        }
    return index

csm_df = pd.read_excel("data/New Zealand FOODfiles 2024/Principal files/Excel files/CSM.FT.XLSX", skiprows=1) # the NZ common serving measure descriptions
csm_df.columns = csm_df.columns.str.strip()

food_df = pd.read_excel("data/New Zealand FOODfiles 2024/Principal files/Excel files/Unabridged/Unabridged DATA.AP.xlsx", skiprows=1)
food_df = food_df[food_df["FoodID"] != "FoodID"]  # drop the units header row
food_df.columns = food_df.columns.str.strip()

food_index = build_index()

nlp = spacy.load("en_core_web_sm", disable=["ner"])
matcher = build_food_matcher(nlp, food_index)
nlp.add_pipe("food_ner", last=True, config={"food_index": food_index})
if not Span.has_extension("food_id"):
    Span.set_extension("food_id", default=None)
if not Span.has_extension("candidates"):
    Span.set_extension("candidates", default=[])

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

        food_id = entity._.food_id
        candidate_ids = entity._.candidates[:5]
        quantity, quantity_start, quantity_end = extract_quantity(doc, entity.start, prev_end)
        prev_end = entity.end
        grams = resolve_grams(food_id, quantity, food_index) if food_id else quantity * 100

        candidates = [
            {
                "food_id": fid,
                "name": food_index[fid]["name"],
                "nutrients": calculate_nutrients(fid, grams, food_index),
            }
            for fid in candidate_ids
            if fid in food_index
        ]

        results.append({
            "text": entity.text,
            "char_start": entity.start_char,
            "char_end": entity.end_char,
            "quantity": quantity,
            "qty_char_start": quantity_start,
            "qty_char_end": quantity_end,
            "grams": grams,
            "match": candidates[0] if candidates else None,
            "candidates": candidates,
        })
    print(results)
    return {"entities": results, "text": req.text}