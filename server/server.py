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

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import ENTITY_MIN_CONFIDENCE, RAG_RERANK_ENABLED, RERANK_THRESHOLD_GAP
from .extraction import extract_quantity, find_item_char_positions, llm_extract, unload_ollama_model
from .logging_config import get_logger
from .nutrition import attach_density, build_candidate_list, resolve_grams, resolve_recipe_nutrients
from .rerank import cross_encoder_rerank, llm_rag_rerank
from .retrieval import compute_confidence, fuzzy_search, nlp, rank_candidates, retrieve_candidates, validate_llm_items

logger = get_logger("food-nlp")

app = FastAPI(title="Food NLP API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ExtractRequest(BaseModel):
    text: str


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

    ranked = rank_candidates(food_description, retrieve_candidates(food_description), limit=15)
    if not ranked:
        return None

    logger.debug(f"[NEL] Top candidates: {[fid for fid, _ in ranked]}")

    confidence = compute_confidence([s for _, s in ranked])

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


@app.post("/extract")
async def extract(req: ExtractRequest):
    """
    Extracts food entities from free text entries and returns matched FOODfiles entries with nutrients

    Attempts LLM extraction first, falling back to spaCy if the LLM is unavailable or avg confidence score is below 0.5

    :param req: Request body containing input text string
    :return: JSON with keys: entities (list), text (str), source ("llm" | "spacy")
    """
    logger.info("---- PIPELINE START ----")
    logger.info(f"Input: {req.text}")

    use_local_llm = os.getenv("USE_LOCAL_LLM", 'True').lower() == "true"

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

            if use_local_llm:
                await unload_ollama_model()

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

    if use_local_llm:
        await unload_ollama_model()

    return {"entities": results, "text": req.text, "source": "spacy"}
