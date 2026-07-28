"""
Stage 1-2 of the pipeline: pulls food items out of the raw text with an LLM (whether that be Ollama locally, or OpenRouter), then check
every item it returned actually appears in what the user typed

Also holds the character-position helpers both this path and the spaCy fallback need for highlighting the input
"""

import json
import os

import httpx
from rapidfuzz import fuzz
from text_to_num import alpha2digit

from .config import (
    ALL_UNITS, LLM_FEW_SHOT, LLM_SYSTEM_PROMPT, OLLAMA_BASE_URL,
    OLLAMA_MODEL, OLLAMA_TIMEOUT, OPENROUTER_CHAT_URL
)
from .logging_config import get_logger

logger = get_logger("food-nlp")


# Quantity and character positions

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

    char_start = text_lower.find(food_lower)

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


# LLM extraction

def openrouter_headers():
    """Auth headers for OpenRouter, shared by extraction and RAG reranking"""
    return {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }


async def openrouter_extract(text):
    """
    Processes text input using a OpenRouter AI model
    :param text: The text input to be processed by the OpenRouter model
    :return: The response object received from the OpenRouter API call
    """
    payload = {
        "model": os.getenv("OPENROUTER_MODEL"),
        "messages": [{"role": "system", "content": LLM_SYSTEM_PROMPT}, *LLM_FEW_SHOT, {"role": "user", "content": text}],
        "temperature": 0
    }

    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        resp = await client.post(OPENROUTER_CHAT_URL, headers=openrouter_headers(), json=payload)

    logger.debug(f"Raw openrouter output: {resp.json()}")

    return resp


async def llm_extract(text, use_local=True):
    """
    Send text to the LLM for food entity extraction and return parsed items
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
    Filters LLM-extracted items to those that can be grounded in the original input text. Handles exact substring matches
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


async def unload_ollama_model():
    """Asks Ollama to drop the model from VRAM immediately"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(f"{OLLAMA_BASE_URL}/api/generate", json={"model": OLLAMA_MODEL, "keep_alive": 0})
    except httpx.HTTPError as exc:
        logger.warning(f"Could not unload Ollama model: {exc}")
