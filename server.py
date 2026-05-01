import json
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
from spacy.matcher import PhraseMatcher
from spacy.language import Language
from spacy.tokens import Span
from text_to_num import alpha2digit
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
import numpy as np

from config import (
    ALL_UNITS, LLM_FEW_SHOT, LLM_RERANK_PROMPT, LLM_SYSTEM_PROMPT, OLLAMA_BASE_URL,
    OLLAMA_MODEL, OLLAMA_TIMEOUT, UNIT_GRAMS
)
from index import (
    extract_brand_keywords, load_indexes, simple_singular, term_variants,
)

food_index, recipe_index, faiss_index, faiss_ids = load_indexes()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print(f"Food index: {len(food_index)} entries | Recipes: {len(recipe_index)}")
print(f"FAISS index: {len(faiss_ids)} vectors")

def extract_quantity(doc, span_start, prev_end=0):
    window = doc[max(prev_end, span_start - 6): span_start]
    if not window:
        return 1.0, None, None, None, None

    original_tokens = list(window)
    converted_words = alpha2digit(window.text, "en").split()
    original_words  = window.text.split()

    for token in reversed(original_tokens):
        if token.text.lower() in ("a", "an"):
            return 1.0, token.idx, token.idx + len(token.text), None, None

    quantity = 1.0
    quantity_char_start = quantity_char_end = None
    unit = None
    original_index = 0

    for ci, converted_word in enumerate(converted_words):
        try:
            quantity = float(converted_word)
            consumed = len(original_words) - len(converted_words) + 1
            start_tok = original_tokens[original_index]
            end_tok   = original_tokens[min(original_index + consumed - 1, len(original_tokens) - 1)]
            quantity_char_start = start_tok.idx
            quantity_char_end   = end_tok.idx + len(end_tok.text)
            original_index     += consumed
            break
        except ValueError:
            original_index += 1

    unit_char_start = None
    i = original_index
    while i < len(original_tokens):
        tok_lower = original_tokens[i].text.lower()
        if tok_lower in ALL_UNITS:
            unit = tok_lower
            unit_char_start = original_tokens[i].idx
            break
        if tok_lower in ("of", "the"):
            i += 1
            continue
        break

    return quantity, quantity_char_start, quantity_char_end, unit, unit_char_start

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
        spans = []
        for match_id, start, end in self.matcher(doc):
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

if not Span.has_extension("food_id"):
    Span.set_extension("food_id", default=None)
if not Span.has_extension("candidates"):
    Span.set_extension("candidates", default=[])

nlp = spacy.load("en_core_web_md", disable=["ner"])
nlp.add_pipe("food_ner", last=True, config={"food_index": food_index})


def candidate_scorer(query, candidate_key_term):
    query_lower = query.lower().strip()
    candidate_lower = candidate_key_term.lower().strip()

    if query_lower == candidate_lower:
        return 100.0

    query_tokens = set(query_lower.split())
    candidate_tokens = set(candidate_lower.split())

    precision_score = fuzz.token_sort_ratio(query_lower, candidate_lower)
    partial_score = fuzz.token_sort_ratio(query_lower, candidate_lower)

    base_score = 0.65 * precision_score + 0.35 * partial_score

    extra_tokens = candidate_tokens - query_tokens
    specificity_penalty = (len(extra_tokens) / max(len(candidate_tokens), 1)) * 15

    length_ratio = min(len(query_lower), len(candidate_lower)) / max(len(query_lower), len(candidate_lower))
    length_bonus = length_ratio * 8

    return max(0.0, base_score - specificity_penalty + length_bonus)

def semantic_search(query, k=10):
    vec = embedding_model.encode([query])
    vec = vec / np.linalg.norm(vec)
    scores, indices = faiss_index.search(vec.astype(np.float32), k)
    return [(faiss_ids[i], float(scores[0][j])) for j, i in enumerate(indices[0])]

def fuzzy_search(span_text, threshold=60, limit=10):
    span_lower = span_text.lower().strip()
    scored = []

    for food_id, entry in food_index.items():
        key_term = entry["key_term"]
        score = candidate_scorer(span_lower, key_term)

        if score >= threshold:
            scored.append((food_id, score))

    scored.sort(key=lambda x: -x[1])
    return scored[:limit]

def rank_candidates(span_text, candidate_ids, limit = 10):
    span_lower   = span_text.lower()
    semantic_map = dict(semantic_search(span_text, k=10))
    scored = []

    for food_id in candidate_ids:
        entry = food_index.get(food_id)
        if not entry:
            continue
        key_term = entry["key_term"]
        lexical  = candidate_scorer(span_lower, key_term) / 100.0
        semantic = semantic_map.get(food_id, 0.0)

        semantic_weight = 0.4 * min(1.0, max(0.0, (semantic - 0.3) / 0.4))
        score = (1.0 - semantic_weight) * lexical + semantic_weight * semantic

        q_toks = len(span_lower.split())
        score += 0.25 * any(b in span_lower for b in entry.get("brands", []))
        score -= (len(key_term.split()) - 1) * 0.05 * (q_toks == 1)
        score += 0.05 * (q_toks == 1 and bool(entry.get("part")))
        score += 0.05 * (q_toks >= 2 and food_id in semantic_map)
        scored.append((food_id, max(0.0, min(1.5, score)) * 100))

    return sorted(scored, key=lambda x: -x[1])[:limit]

def retrieve_candidates(food_description, limit=20):
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

    semantic_hits = semantic_search(food_description)

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

def resolve_grams(food_id, quantity, unit):
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


def calculate_nutrients(food_id, grams):
    entry = food_index.get(food_id)
    if not entry:
        return {}
    factor = grams / 100.0
    return {
        nutrient: round(value * factor, 2) if value is not None else None
        for nutrient, value in entry["nutrients"].items()
    }


def resolve_recipe_nutrients(food_id, grams):
    if food_id not in recipe_index:
        return calculate_nutrients(food_id, grams)

    total = {}
    for component in recipe_index[food_id]:
        ingredient_grams = grams * component["weight_fraction"]
        for nutrient, value in calculate_nutrients(component["ingredient_id"], ingredient_grams).items():
            if value is not None:
                total[nutrient] = round(total.get(nutrient, 0.0) + value, 2)

    return total or calculate_nutrients(food_id, grams)

def parse_llm_output(raw):
    parsed = json.loads(raw)

    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict) and "items" in parsed:
        items = parsed["items"]
    else:
        return None

    validated = [
        {
            "food": str(item.get("food", "")).strip().lower(),
            "quantity": float(item.get("quantity") or 1.0),
            "unit": str(item["unit"]).lower() if item.get("unit") else None,
        }
        for item in items
        if isinstance(item, dict) and str(item.get("food", "")).strip()
    ]
    return validated or None

def validate_llm_grounding(items, original_text):
    text_lower = original_text.lower()
    validated = []

    for item in items:
        food = item["food"].lower()

        if food not in text_lower:
            continue

        validated.append(item)

    return validated if validated else None

def validate_llm_items(llm_items):
    enriched = []

    for item in llm_items:
        food_name = item["food"]

        # First, normal linking
        candidate_ids = retrieve_candidates(food_name)
        ranked = rank_candidates(food_name, candidate_ids)

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

                candidate_ids = retrieve_candidates(generic_attempt)
                ranked = rank_candidates(generic_attempt, candidate_ids)

                if ranked:
                    score = ranked[0][1] / 100.0
                    if score > item["link_confidence"]:
                        item["food_generic"] = generic_attempt
                        item["link_confidence"] = score

        enriched.append(item)

    return enriched

async def llm_extract(text):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": LLM_SYSTEM_PROMPT}, *LLM_FEW_SHOT, {"role": "user", "content": text}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 512},
    }
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            resp.raise_for_status()
        raw = resp.json()["message"]["content"]
        print("Raw LLM output:", raw)
        items = parse_llm_output(raw)
        if items:
            items = validate_llm_grounding(items, text)
        return items
    except (httpx.ConnectError, httpx.TimeoutException):
        print("Ollama unavailable — falling back to spaCy")
        return None
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        print(f"LLM parse error ({exc}) — falling back to spaCy")
        return None


async def llm_rerank(query, candidates):
    if len(candidates) <= 1:
        return 0
    candidate_text = "\n".join(f"{i+1}. {c['name']}" for i, c in enumerate(candidates))
    prompt = LLM_RERANK_PROMPT.format(query=query, candidates=candidate_text, n=len(candidates))
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 5},
    }
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            resp.raise_for_status()
        choice = int(resp.json()["message"]["content"].strip()) - 1
        return choice if 0 <= choice < len(candidates) else 0
    except Exception as e:
        print(f"Rerank error: {e}")
        return 0


def build_candidate_list(ranked, grams):
    candidates = []
    for food_id, score in ranked[:10]:
        entry = food_index.get(food_id)
        if not entry:
            continue
        candidates.append({
            "food_id": food_id,
            "name": entry["name"],
            "score": round(score, 2),
            "is_recipe": entry.get("is_recipe", False),
            "recipe_ingredients": [
                {"food_id": c["ingredient_id"], "name": c["ingredient_name"], "weight_fraction": c["weight_fraction"]}
                for c in recipe_index.get(food_id, [])
            ],
            "nutrients": resolve_recipe_nutrients(food_id, grams),
        })
    return candidates


async def process_llm_item(item, original_text):
    food_description = item.get("food_generic") or item["food"]
    quantity, unit = item["quantity"], item["unit"]

    ranked = rank_candidates(food_description, retrieve_candidates(food_description))
    if not ranked:
        return None

    confidence = compute_confidence([s for _, s in ranked])
    provisional = ranked[0][0]
    grams = resolve_grams(provisional, quantity, unit)
    candidates = build_candidate_list(ranked, grams)

    best_idx = (
        await llm_rerank(food_description, candidates)
        if len(candidates) > 1 and abs(ranked[0][1] - ranked[1][1]) < 10
        else 0
    )

    grams = resolve_grams(candidates[best_idx]["food_id"], quantity, unit)
    candidates = build_candidate_list(ranked, grams)

    char_start, char_end, qty_cs, qty_ce, unit_cs = find_item_char_positions(item, original_text)
    return {
        "text": item["food"],
        "resolved_text": food_description,
        "char_start": char_start,
        "char_end": char_end,
        "quantity": quantity,
        "quantity_char_start": qty_cs,
        "quantity_char_end": qty_ce,
        "unit": unit,
        "unit_char_start": unit_cs,
        "grams": grams,
        "confidence": confidence,
        "match": candidates[0] if candidates else None,
        "candidates": candidates,
        "source": "llm",
    }


def process_spacy_entity(entity, doc, prev_end):
    ranked = rank_candidates(entity.text, retrieve_candidates(entity.text))
    if not ranked:
        ranked = fuzzy_search(entity.text)
    if not ranked:
        return None

    confidence = compute_confidence([s for _, s in ranked])
    quantity, qty_cs, qty_ce, unit, unit_cs = extract_quantity(doc, entity.start, prev_end)
    best_id = ranked[0][0]
    grams = resolve_grams(best_id, quantity, unit)
    candidates = build_candidate_list(ranked, grams)

    return {
        "text": entity.text,
        "char_start": entity.start_char,
        "char_end": entity.end_char,
        "quantity": quantity,
        "quantity_char_start": qty_cs,
        "quantity_char_end": qty_ce,
        "unit": unit,
        "unit_char_start": unit_cs,
        "grams": grams,
        "confidence": confidence,
        "match": candidates[0] if candidates else None,
        "candidates": candidates,
        "source": "spacy",
    }


app = FastAPI(title="Food NLP API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ExtractRequest(BaseModel):
    text: str

@app.post("/extract")
async def extract(req: ExtractRequest):
    llm_items = await llm_extract(req.text)

    if llm_items is not None:
        llm_items = validate_llm_items(llm_items)
        avg_conf  = sum(i["link_confidence"] for i in llm_items) / len(llm_items) if llm_items else 0
        if avg_conf >= 0.5:
            results = [r for item in llm_items if (r := await process_llm_item(item, req.text))]
            print("LLM used")
            return {"entities": results, "text": req.text, "source": "llm"}

    # spaCy fallback
    doc, results, prev_end = nlp(req.text), [], 0
    for ent in (e for e in doc.ents if e.label_ == "FOOD"):
        if result := process_spacy_entity(ent, doc, prev_end):
            results.append(result)
        prev_end = ent.end

    print("spaCy used")
    return {"entities": results, "text": req.text, "source": "spacy"}
