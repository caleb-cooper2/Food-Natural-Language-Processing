"""
FastAPI food NLP extraction server

Exposes a single POST /extract endpoint that accepts free-text food descriptions and returns structured
entities with matched FOODfiles entries and nutrient data

Pipeline:

1. LLM extraction (Ollama/Qwen) -> identifies food items, quantities, and units
2. Grounding check -> verifies extracted foods appear in the original text
3. NEL (Named Entity Linking) -> retrieves and ranks candidates via RRF fusion of PhraseMatcher, FAISS semantic search,
   and BM25 lexical search
4. Cross-encoder reranking -> finds close matches using a bi-directional scorer
5. spaCy fallback -> used if LLM is unavailable or average link confidence < 0.5
"""

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
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from .logging_config import get_logger
import re
import os
from dotenv import load_dotenv

from .config import (
    ALL_UNITS, LLM_FEW_SHOT, LLM_SYSTEM_PROMPT, OLLAMA_BASE_URL,
    OLLAMA_MODEL, OLLAMA_TIMEOUT, UNIT_GRAMS, RAG_RERANK_ENABLED,
    RAG_TOP_N, RAG_SYSTEM_PROMPT, ENTITY_MIN_CONFIDENCE
)
from .index import extract_brand_keywords, load_indexes, simple_singular, term_variants

load_dotenv()

food_index, recipe_index, faiss_index, faiss_ids, bm25, bm25_ids = load_indexes()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

logger = get_logger("food-nlp")

logger.info(f"Food index: {len(food_index)} entries | Recipes: {len(recipe_index)}")
logger.info(f"FAISS index: {len(faiss_ids)} vectors")

from .known_densities import compute_global_median_density, derive_density
global_median_density = compute_global_median_density(food_index)

def extract_quantity(doc, span_start, prev_end=0):
    """
    Extracts quantity, unit and their char position from the tokens prior to a identified food in text
    :param doc: spaCy Doc object for the full input text
    :param span_start: Token index of the start of the food entity
    :param prev_end: Token index of the end of the previous entity, used to bound the search window
    :return: Tuple of (quantity, qty_char_start, qty_char_end, unit, unit_char_start)
    """
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
            end_tok = original_tokens[min(original_index + consumed - 1, len(original_tokens) - 1)]
            quantity_char_start = start_tok.idx
            quantity_char_end = end_tok.idx + len(end_tok.text)
            original_index += consumed
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
    """
    Locates the character positions of a food item's name, quantity, and unit within the original text.
    Used by LLM path where spaCy token indices are unavailable
    :param item: LLM-extracted item dict with keys: food, quantity, unit
    :param original_text: The original input string passed to /extract
    :return: Tuple of (char_start, char_end, qty_char_start, qty_char_end, unit_char_start)
    """
    food = item["food"]
    text_lower = original_text.lower()
    food_lower = food.lower()

    char_start = text_lower.find(food.lower())

    if char_start == -1:
        food_len = len(food_lower)
        best_score, best_start = 0, 0
        for i in range(max(1, len(text_lower) - food_len + 1)):
            window = text_lower[i:i + food_len]
            score = fuzz.ratio(food_lower, window)
            if score > best_score:
                best_score, best_start = score, i
        if best_score >= 70:
            char_start = best_start
        else:
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
    """
    Builds a PhaseMatcher and term-to-candidate mapping from the food index
    :param nlp: spaCy language model used to create phrase docs
    :param food_index: Primary food index dict
    :return: Tuple of (matcher, term_candidates) where term_candidates maps term -> list of FoodIDs
    """
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    term_candidates = {}

    for food_id, entry in food_index.items():
        if food_id.startswith("OFF:"):
            terms = list(dict.fromkeys(
                [entry["key_term"]] + entry.get("brands", [])
            ))
        else:
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
def create_food_ner(nlp, name): # spaCy passes this automatically, have to leave name unused as a result
    """
    spaCy factory that instantiates FoodNERComponent for the pipeline
    :param nlp: spaCy language model (injected by spaCy)
    :param name: Component name string (injected by spaCy)
    :param food_index: Primary food index dict passed via pipe config
    :return: FoodNERComponent instance
    """
    matcher, term_candidates = build_food_matcher(nlp, food_index)
    return FoodNERComponent(nlp, food_index, matcher, term_candidates)


class FoodNERComponent:
    """Custom spaCy pipeline component that tags FOOD entities using a PhraseMatcher"""

    def __init__(self, nlp, food_index, matcher, term_candidates):
        self.food_index = food_index
        self.matcher = matcher
        self.term_candidates = term_candidates

    def __call__(self, doc):
        """
        Tags FOOD spans and attach candidate FoodID lists as span extensions
        :param doc: spaCy Doc to annotate
        :return: Annotated Doc with FOOD entities and _.candidates extension set
        """
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
nlp.add_pipe("food_ner", last=True)


def candidate_scorer(query, candidate_key_term):
    """
    Scores a query string against a single candidate term using fuzzy matching with a specificity penalty
    :param query: User's food description string
    :param candidate_key_term: Key term from a food index entry
    :return: Float score in range [0, 108] (exact match returning 100.0)
    """
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
    """
    Search the FAISS index for the k most semantically similar food entries
    :param query: Food description string to encode and search
    :param k: Number of nearest neighbours to return
    :return: List of (food_id, cosine_similarity_score) tuples
    """
    vec = embedding_model.encode([query])
    vec = vec / np.linalg.norm(vec)
    scores, indices = faiss_index.search(vec.astype(np.float32), k)
    return [(faiss_ids[i], float(scores[0][j])) for j, i in enumerate(indices[0])]

def fuzzy_search(span_text, threshold=60, limit=10):
    """
    Search the food index by fuzzy key_term matching, used as a retrieval fallback
    :param span_text: Food description string to match against
    :param threshold: Minimum candidate_scorer score to include a result
    :param limit: Maximum number of results to return
    :return: List of (food_id, score) tuples sorted by decending score
    """
    span_lower = span_text.lower().strip()
    scored = []

    for food_id, entry in food_index.items():
        key_term = entry["key_term"]
        score = candidate_scorer(span_lower, key_term)

        if score >= threshold:
            scored.append((food_id, score))

    scored.sort(key=lambda x: -x[1])
    return scored[:limit]

def bm25_search(query, bm25, bm25_ids, k=20):
    """
    Searches the BM25 index for the k highest-scoring food entries
    :param query: Food description string to tokenise and score
    :param bm25: BM25 index object
    :param bm25_ids: List of FoodIDs corresponding to BM25 document positions
    :param k: Number of top results to return
    :return: List of (food_id, bm25_score) tuples for entries with score > 0
    """
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:k]
    return [(bm25_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]

def reciprocal_rank_fusion(*ranked_lists, k=60):
    """
    Join multiple ranked candidate lists into a single ranking using Reciprocal Rank Fusion.
    Score for each candidate is sum(1/(k+rank)) across all lists
    :param ranked_lists: Variable number of lists of (food_id, score) tuples
    :param k: RRF smoothing constant
    :return: List of (food_id, rrf_score) tuples sorted by descending score
    """
    scores = {}
    for ranked in ranked_lists:
        for rank, (food_id, _) in enumerate(ranked):
            scores[food_id] = scores.get(food_id, 0.0) + 1.0 / (k + rank+1)
    return sorted(scores.items(), key=lambda x: -x[1])


def rank_candidates(span_text, candidate_ids, limit = 10):
    """
    Score and rank a list of candidate FoodIDs against a query using lexical and semantic signals
    :param span_text: Food description string to score against
    :param candidate_ids: List of FoodIDs to rank
    :param limit: Maximum number of ranked results to return
    :return: List of (food_id, score) tuples sorted by descending score, scaled to [0, 150]
    """
    span_lower   = span_text.lower()
    semantic_map = dict(semantic_search(span_text, k=10))
    scored = []

    for food_id in candidate_ids:
        entry = food_index.get(food_id)
        if not entry:
            continue

        candidate_terms = list(dict.fromkeys(
            [entry["key_term"]] + entry.get("keywords", [])[:5]
        ))
        lexical = max(
            candidate_scorer(span_lower, term) for term in candidate_terms
        ) / 100.0

        semantic = semantic_map.get(food_id, 0.0)

        semantic_weight = 0.4 * min(1.0, max(0.0, (semantic - 0.3) / 0.4))
        score = (1.0 - semantic_weight) * lexical + semantic_weight * semantic

        q_toks = len(span_lower.split())
        score += 0.25 * any(b in span_lower for b in entry.get("brands", [])) # bonus for brands

        if q_toks <= 3: # penalty for short multi-token queries
            key_toks = len(entry["key_term"].split())
            extra = max(0, key_toks - q_toks)
            score -= extra * 0.04

        score += 0.05 * (q_toks >= 2 and food_id in semantic_map) # bonus for long tokens that are semantically matching

        candidate_key_lower = entry["key_term"].lower()
        if span_lower in candidate_key_lower or candidate_key_lower in span_lower:
            score += 0.15 # bonus for substring matches

        scored.append((food_id, max(0.0, min(1.5, score)) * 100))

    return sorted(scored, key=lambda x: -x[1])[:limit]

def retrieve_candidates(food_description, limit=50):
    """
    Retrieve a ranked list of candidate FoodIDs for a food description using RRF over three signals.
    Joining PhraseMatcher (spaCy NER), FAISS semantic search and BM25 lexical search.
    Fuzzy search fills the remaining slots if the fused list is shorter than the limit.
    :param food_description: Food description string to retrieve candidates for
    :param limit: Maximum number of candidate FoodIDs to return
    :return: List of FoodID strings
    """
    doc = nlp(food_description)
    phrase_ids = []
    seen = set()

    for entity in doc.ents:
        if entity.label_ == "FOOD":
            for cid in entity._.candidates:
                if cid not in seen:
                    seen.add(cid)
                    phrase_ids.append(cid)
    phrase_ranked = [(fid, 1.0) for fid in phrase_ids]

    semantic_ranked = semantic_search(food_description, k=20)

    bm25_ranked = bm25_search(food_description, bm25, bm25_ids, k=20)

    fused = reciprocal_rank_fusion(phrase_ranked, semantic_ranked, bm25_ranked)

    fused_ids = {fid for fid, _ in fused}
    if len(fused) < limit:
        fuzzy_hits = fuzzy_search(food_description, threshold=55, limit=20)
        for fid, _ in fuzzy_hits:
            if fid not in fused_ids:
                fused.append((fid, 0.0))
                fused_ids.add(fid)

    return [fid for fid, _ in fused[:limit]]


def compute_confidence(ranked_scores):
    """
    Compute a confidence score for the top-ranked match based on its score and gap to second place
    :param ranked_scores: List of float scores in descending order
    :return: Float confidence in range [0.0, 1.0]
    """
    if not ranked_scores:
        return 0.0
    if len(ranked_scores) == 1:
        return ranked_scores[0] / 100.0

    gap = ranked_scores[0] - ranked_scores[1]
    return min(1.0, (ranked_scores[0] / 100.0) * (1 + gap / 100.0))

def resolve_grams(food_id, quantity, unit):
    """
    Converts a quantity and unit to grams for a given food entry
    :param food_id: FoodID to look up serving measures for
    :param quantity: Numeric quantity value
    :param unit: Unit string (e.g. "cup", "slice") or None
    :return: Float gram weight
    """
    if unit in UNIT_GRAMS:
        return quantity * UNIT_GRAMS[unit]

    servings = food_index.get(food_id, {}).get("serving_measure", [])

    if unit and servings:
        for serving in servings:
            label = str(serving.get("CSM") or serving.get("name") or "").lower()
            if unit.lower() in label:
                weight = (
                        serving.get("Measure")
                        or serving.get("grams")
                        or serving.get("ml")
                )
                try:
                    return quantity * float(weight)
                except (TypeError, ValueError):
                    pass

    if servings:
        weight = (
                servings[0].get("Measure")
                or servings[0].get("grams")
                or servings[0].get("ml")
        )
        try:
            return quantity * float(weight)
        except (TypeError, ValueError):
            pass

    return quantity * 100.0


def calculate_nutrients(food_id, grams):
    """
    Calculate absolute nutrient values for a food entry scaled to a given gram weight
    :param food_id: FoodID to look up
    :param grams: Gram weight to scale per-100g nutrients by
    :return: Dict of nutrient_name -> rounded float value (or None if source value is None)
    """
    entry = food_index.get(food_id)
    if not entry:
        return {}
    factor = grams / 100.0
    return {
        nutrient: round(value * factor, 2) if value is not None else None
        for nutrient, value in entry["nutrients"].items()
    }


def resolve_recipe_nutrients(food_id, grams):
    """
    Calculates nutrients for a food, blending ingredient contributions if it is a recipe
    :param food_id: FoodID to resolve (could be a recipe or direct food entry)
    :param grams: Total gram weight to calculate nutrients for
    :return: Dict of nutrient_name -> rounded float value
    """
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
    """
    Parse and validate raw LLM JSON output into a list of food item dicts
    :param raw: Raw JSON string from the LLM response
    :return: List of {food, quantity, unit} dicts, or None if parsing fails or yields no items
    """
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
            "normalised": str(item["normalised"]).strip().lower() if item.get("normalised") else None,
            "prep": str(item["prep"]).strip().lower() if item.get("prep") else None
        }
        for item in items
        if isinstance(item, dict) and str(item.get("food", "")).strip()
    ]
    return validated or None

def validate_llm_grounding(items, original_text):
    """
    Filters LLM-extracted items to thsoe that can be grounded in the original input text. Handles exact substring matches
    in addition to fuzzy matches (where partial_ratio >= 80) to allow for LLM spelling corrections
    :param items: List of LLM-extracted item dicts
    :param original_text: The original input string passed to /extract
    :return: Filtered list of grounded items, or None if all items are rejected
    """
    text_lower = original_text.lower()
    validated = []

    for item in items:
        food = item["food"].lower()

        if food in text_lower:
            validated.append(item)
            continue

        score = fuzz.partial_ratio(food, text_lower)
        if score >= 80:
            logger.debug(f"[grounding] fuzzy accepted '{food}' (score={score}) against input text")
            validated.append(item)
            continue

        logger.debug(f"[grounding] REJECTED '{food}' (score={score}) - not found in input text")

    return validated if validated else None

def validate_llm_items(llm_items):
    """
    Enrich LLM-extracted items with link confidence scores, falling back to brand stripping if needed
    :param llm_items: list of grounded LLM item dicts
    :return: List of item dicts with link_confidence and optional food_generic fields added
    """
    enriched = []

    for item in llm_items:
        food_name = item["food"]

        candidate_ids = retrieve_candidates(food_name)
        ranked = rank_candidates(food_name, candidate_ids)

        if ranked:
            ranked_scores = [score for _, score in ranked]
            item["link_confidence"] = compute_confidence(ranked_scores)
        else:
            item["link_confidence"] = 0.0

        # If confidence is low, try stripping brand names and re-linking
        if item["link_confidence"] < 0.4:
            top_id = ranked[0][0] if ranked else ""
            entry = food_index.get(top_id, {})
            brands = (
                entry.get("brands", [])
                if top_id.startswith("OFF:")
                else extract_brand_keywords(food_name, nlp=nlp)
            )

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

async def openrouter_extract(text):
    """
    Processes text input using a OpenRouter AI model
    :param text: The text input to be processed by the OpenRouter model
    :return: The response object received from the OpenRouter API call
    """
    openrouter_model = os.getenv("OPENROUTER_MODEL")

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUT_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": openrouter_model,
        "messages": [{"role": "system", "content": LLM_SYSTEM_PROMPT}, *LLM_FEW_SHOT, {"role": "user", "content": text}],
        "temperature": 0
    }

    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        resp = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)

    logger.debug(f"Raw openrouter output: {resp.json()}")

    return resp

async def llm_extract(text, use_local=True):
    """
    Send text to the Ollama LLM for food entity extraction and return parsed items
    :param use_local: Whether the output should be generated by local LLM or openrouter
    :param text: Raw input text to extract food items from
    :return: List of grounded {food, quantity, unit} dicts, or None if LLM is unavailable or fails
    """
    logger.info(f"Running LLM extraction on {text}")
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": LLM_SYSTEM_PROMPT}, *LLM_FEW_SHOT, {"role": "user", "content": text}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 512}
    }
    try:
        logger.info("Using local LLM") if use_local else logger.info("Using OpenRouter")
        if use_local:
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
                resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
                resp.raise_for_status()

                raw = resp.json()["message"]["content"]
        else:
            resp = await openrouter_extract(text)
            resp.raise_for_status()

            raw = resp.json()["choices"][0]["message"]["content"]

        logger.debug(f"[LLM] raw_output={raw}")

        items = parse_llm_output(raw)
        if items:
            items = validate_llm_grounding(items, text)
        return items
    except (httpx.ConnectError, httpx.TimeoutException):
        logger.warning("LLM unavailable — falling back to spaCy")
        return None
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.warning(f"LLM parse error ({exc}) — falling back to spaCy")
        return None

def cross_encoder_rerank(query, candidates):
    """
    Rerank a candidate list using a cross-encoder relevance model. More reliable than generative LLM reranking
    as the cross-encoder looks at both query and candidate together, eliminating potential bias.
    :param query: Food description string used as the query
    :param candidates: List of candidate dicts with a 'name' key
    :return: Index of the highest-scoring candidate
    """
    logger.debug("Cross-encoder reranking triggered")

    if len(candidates) <= 1:
        return 0

    pairs = [(query, candidate["name"]) for candidate in candidates]
    scores = cross_encoder.predict(pairs)
    return int(np.argmax(scores))

async def llm_rag_rerank(food_description, candidates, original_text=None, use_local=True):
    """
    Using local LLM to select best matching candidate
    :param use_local: Whether the output should be generated by local LLM or openrouter
    :param food_description: Users food description string
    :param candidates: Ordered list of candidate dicts (name, food_id, key_term, keywords)
    :param original_text: The entire input text for better semantic understanding
    :return: 0-based index of best candidate
    """
    top = candidates[:RAG_TOP_N]

    # Build a context block for each candidate
    lines = []
    for i, c in enumerate(top, 1):
        entry = food_index.get(c["food_id"], {})
        kws = ", ".join(entry.get("keywords", []))
        lines.append(f"{i}. {c['name']}" + (f' ({kws})' if kws else ''))

    context_line = f'Full sentence context: "{original_text}"\n' if original_text else ""
    user_prompt = (
        f'{context_line}'
        f'Food described: "{food_description}"\n\n'
        f'Candidates:\n' + "\n".join(lines) +
        f'\n\nRespond with only the number of the best match (1-{len(top)}), or 0 if none fit.'
    )

    payload = {
        "model": OLLAMA_MODEL if use_local else os.getenv("OPENROUTER_MODEL"),
        "messages": [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt}
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 8}  # we only need a single digit
    }

    try:
        if use_local:
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
                resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
                resp.raise_for_status()
            raw = resp.json()["message"]["content"].strip()
        else:
            resp = await openrouter_rag_rerank(payload)
            raw = resp.json()["choices"][0]["message"]["content"].strip()

        idx = int(re.search(r'\d+', raw).group())
        if 1 <= idx <= len(top):
            return idx - 1 # convert to 0-based
    except Exception as exc:
        logger.warning(f"[RAG rerank] failed ({exc}), falling back to cross-encoder")
    return None # None signals fallback

async def openrouter_rag_rerank(payload):
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUT_API_KEY')}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        resp = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
    logger.info(f"rerank openrouter output: {resp.json()}")
    return resp

def build_candidate_list(ranked, grams):
    """
    Builds the full candidate response list from ranked (food_id, score) pairs
    :param ranked: List of (food_id, score) tuples in descending score order
    :param grams: Gram weight used to calculate nutrient values
    :return: List of candidate dicts with food_id, name, score, is_recipe, recipe_ingredients, nutrients
    """
    candidates = []
    for food_id, score in ranked[:10]:
        entry = food_index.get(food_id)
        if not entry:
            continue

        display_name = entry["name"]
        brands = entry.get("brands", [])
        if food_id.startswith("OFF:") and brands:
            primary_brand = brands[0]
            if primary_brand.lower() not in display_name.lower():
                display_name = f"{primary_brand.title()} – {display_name}"

        candidates.append({
            "food_id": food_id,
            "name": display_name,
            "score": round(score, 2),
            "is_recipe": entry.get("is_recipe", False),
            "recipe_ingredients": [
                {"food_id": c["ingredient_id"], "name": c["ingredient_name"], "weight_fraction": c["weight_fraction"]}
                for c in recipe_index.get(food_id, [])
            ],
            "nutrients": resolve_recipe_nutrients(food_id, grams)
        })
    return candidates


def attach_density(match, prep):
    """
    Resolves a density for a matched candidate and attaches it in place
    :param match: The chosen candidate dict (has food_id); mutated to gain a "density" block
    :param prep: Preparation word found by the LLM (e.g. "sticks", "grated", "whole"), or None
    """
    if not match:
        return
    entry = food_index.get(match["food_id"], {})
    density, log_sigma, source = derive_density(entry, global_median_density, prep)
    match["density"] = {
        "density_g_per_ml": round(density, 4),
        "density_log_sigma": round(log_sigma, 4),
        "density_source": source
    }


async def process_llm_item(item, original_text, use_local_llm):
    """
    Resolves a single LLM-extracted food item to a matched database entry with nutrients
    :param use_local_llm: Whether or not to use local LLM
    :param item: LLM item dict with keys: food, quantity, unit, and optionally food_generic
    :param original_text: Original input string, used for character position resolution
    :return: Structured entity result dict, or None if no candidates are found
    """
    food_description = item.get("food_generic") or item.get("normalised") or item["food"]
    quantity, unit = item["quantity"], item["unit"]

    logger.debug(f"[NEL] Retrieving candidates for: '{food_description}'")

    ranked = rank_candidates(food_description, retrieve_candidates(food_description))
    if not ranked:
        return None

    logger.debug(f"[NEL] Top candidates: {[fid for fid, _ in ranked[:5]]}")

    confidence = compute_confidence([s for _, s in ranked])

    RERANK_THRESHOLD_GAP = 20 # skip reranking if first candidate leads by this margin
    score_gap = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 999

    provisional_grams = resolve_grams(ranked[0][0], quantity, unit)
    candidates = build_candidate_list(ranked, provisional_grams)

    rag_idx = None
    best_idx = 0

    if score_gap < RERANK_THRESHOLD_GAP:
        if RAG_RERANK_ENABLED:
            rag_idx = await llm_rag_rerank(food_description, candidates, original_text, use_local_llm)
            best_idx = rag_idx if rag_idx is not None else cross_encoder_rerank(food_description, candidates)
        else:
            best_idx = cross_encoder_rerank(food_description, candidates)

    logger.debug(
        f"[rerank] before='{candidates[0]['name']}' "
        f"after='{candidates[best_idx]['name']}'"
    )

    # Reorder so best candidate is always at index 0
    if best_idx != 0:
        candidates.insert(0, candidates.pop(best_idx))

    grams = resolve_grams(candidates[0]["food_id"], quantity, unit)
    candidates[0]["nutrients"] = resolve_recipe_nutrients(candidates[0]["food_id"], grams)
    attach_density(candidates[0] if candidates else None, item.get("prep"))

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
        "rerank_source": "rag" if rag_idx is not None else "cross_encoder",
        "source": "llm"
    }


def process_spacy_entity(entity, doc, prev_end):
    """
    Resolves a single spaCy FOOD entity to a matched database entry with nutrients
    :param entity: spaCy Span with label FOOD
    :param doc: Full spaCy doc, required for quantity extraction context
    :param prev_end: Token index of the end of the previous entity to bound the quantity search window
    :return: Structured entity result dict or None if no candidates are found
    """
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
    attach_density(candidates[0] if candidates else None, None)

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
        "source": "spacy"
    }


app = FastAPI(title="Food NLP API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ExtractRequest(BaseModel):
    text: str

@app.post("/extract")
async def extract(req: ExtractRequest):
    """
    Extracts food entities from free text entries and returns matched FOODfiles entries with nutrients

    Attmepts LLM extraction first, falling back to spaCy if the LLM is unavailable or avg confidence scores is below 0.5

    :param req: Request body containing input text string
    :return: JSON with keys: entities (list), text (str), source ("llm" | "spacy")
    """
    logger.info("---- PIPELINE START ----")
    logger.info(f"Input: {req.text}")

    use_local_llm = os.getenv("USE_LOCAL_LLM", 'True').lower() == "true"

    logger.info(use_local_llm)

    logger.info("[Step 1] LLM Extraction")
    llm_items = await llm_extract(req.text, use_local_llm)

    if llm_items is not None:
        logger.info(f"[Step 2] Grounding + Validation")

        llm_items = validate_llm_items(llm_items)
        avg_conf  = sum(i["link_confidence"] for i in llm_items) / len(llm_items) if llm_items else 0
        logger.info(f"LLM avg confidence: {avg_conf:.2f}")

        if avg_conf >= 0.5:
            logger.info(f"[Step 3/4] NEL + Reranking")

            results = [r for item in llm_items if (r := await process_llm_item(item, req.text, use_local_llm)) and r["confidence"] >= ENTITY_MIN_CONFIDENCE]
            logger.info("LLM used")
            logger.info("---- PIPELINE END ----")
            return {"entities": results, "text": req.text, "source": "llm"}
        else:
            logger.warning("Low confidence -> spaCy fallback")

    # spaCy fallback
    doc, results, prev_end = nlp(req.text), [], 0
    for ent in (e for e in doc.ents if e.label_ == "FOOD"):
        if result := process_spacy_entity(ent, doc, prev_end):
            if result["confidence"] >= ENTITY_MIN_CONFIDENCE:
                results.append(result)
        prev_end = ent.end

    logger.info("spaCy used")
    logger.info("---- PIPELINE END ----")
    return {"entities": results, "text": req.text, "source": "spacy"}
