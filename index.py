"""
Builds and perists the food matching indexes

We read the NZ FOODfiles 2024 excel db sources and produces four indexes:
- food_index.json -> per-food metadata, keywords, nutrients, serving sizes
- recipe_index.json -> recipe to ingredient composition mapping
- faiss.index -> dense vector index for semantic search
- bm25.pkl -> sparse BM25 index for lexical search
"""

import json
import math
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer

from config import ALL_UNITS, PRINCIPAL_XLSX, SUPPORTING_XLSX

INDEX_DIR = Path("data/indexes")

def simple_plural(word):
    """
    Return a naive English plural for a single word
    :param word: Singular word to turn plural
    :return: Plural word string
    """
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
    """
    Return a naive English singular for a single word
    :param word: Plural word to turn singular
    :return: Singular word string
    """
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
    """
    Return singular and plural forms of a term or just [term] if identical
    :param term: Space-separated term to expand
    :return: List of one or two variant strings
    """
    words = term.split()
    if not words:
        return [term]
    plural = ' '.join(words[:-1] + [simple_plural(words[-1])])
    return [term] if plural == term else [term, plural]

def extract_keywords(food_name_string):
    """
    Derives a ranked keyword list from a raw FOODfiles food name string
    :param food_name_string: Raw food name e.g. "Strawberry, raw, New Zealand"
    :return: List of lowercase keyword strings, comma segments first then parentheticals
    """
    cleaned = re.sub(r'[™®©]', '', food_name_string)
    paren_terms = re.findall(r'\((.*?)\)', cleaned)
    cleaned = re.sub(r'\(.*?\)', '', cleaned)
    keywords = [p.strip().lower() for p in cleaned.split(',') if p.strip()]
    keywords += [t.strip().lower() for t in paren_terms if t.strip()]
    return keywords

def extract_brand_keywords(food_name, sampling_details = "", nlp = None):
    """
    Identify brand names within a food name string using trademark symbols, NER, capitalisation and sampling details
    :param food_name: Raw FOODfiles food name string
    :param sampling_details: Optional sampling details field from NAME.FT, used to parse explicit strings
    :param nlp: Optional spaCy model for ORG/PRODUCT NER, skipped if None
    :return: Deduplicated list of lowercase brand strings
    """

    brands = []
    segments = [s.strip() for s in food_name.split(',')]

    # 1. trademark symbols
    trademark_matches = re.findall(r'\b([A-Za-z][\w\-]*)[™®©]', food_name)
    brands.extend([brand.lower() for brand in trademark_matches])

    # 2. spaCy NER for ORG/PRODUCT entities
    if nlp is not None:
        doc = nlp(food_name)
        for entity in doc.ents:
            if entity.label_ in ("ORG", "PRODUCT"):
                candidate = entity.text.strip().lower()
                if len(candidate) > 2 and candidate not in ALL_UNITS:
                    brands.append(candidate)

    # 3. proper-cased segments after the second comma
    for segment in segments[2:]:
        cleaned_segment = re.sub(r'[™®©]', '', segment).strip()
        tokens = cleaned_segment.split()

        is_proper = all(token[0].isupper() for token in tokens if token and token[0].isalpha())
        if is_proper and 1 <= len(tokens) <= 3:
            brands.append(cleaned_segment.lower())

    # 4. explicit brand mentions in the sampling details field
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

def extract_name_metadata(name_row):
    """
    Extracts structured metadata fields from a NAME.FT row
    :param name_row: Dict like row from the NAME.FT dataframe
    :return: Dict with keys: short_name, alt_names, generic, kind, part, sampling_details
    """
    def safe(col):
        clean = str(name_row.get(col) or "").strip()
        return "" if clean.lower() == "nan" else clean

    return {
        "short_name": safe("Short Food Name"),
        "alt_names": safe("AlternativeNames"),
        "generic": safe("Generic Name"),
        "kind": safe("Kind"),
        "part": safe("Part"),
        "sampling_details": safe("Sampling Details"),
    }

def build_index(food_df, csm_df, name_df, nlp):
    """
    Builds the primary food index dict keyed by FoodID's
    :param food_df: DataFrame from Unbridged Data.AP.xlsx
    :param csm_df: DataFrame from CSM.FT.XLSX (common serving measures)
    :param name_df: DataFrame from NAME.FT.XLSX (curated name metadata)
    :param nlp: spaCy model used for brand NER during keyword extraction
    :return: Dict mapping FoodID -> {name, key_term, keywords, brands, part, serving_measure, is_recipe, nutrients}
    """
    csm_df = csm_df.copy()
    csm_df["FoodID"] = csm_df["FoodID"].astype(str).str.strip()
    csm_lookup = {
        fid: grp[["CSM", "Measure"]].to_dict("records")
        for fid, grp in csm_df.groupby("FoodID")
    } if "FoodID" in csm_df.columns else {}

    name_lookup = (
        {str(r.get("FoodID", "")).strip(): r for _, r in name_df.iterrows()}
        if name_df is not None else {}
    )

    def clean_num(value):
        try:
            f = float(value)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    index = {}
    for _, row in food_df.iterrows():
        food_id = str(row.get("FoodID", "")).strip()
        name = str(row.get("Food Name", "")).strip()
        if not food_id or not name:
            continue

        keywords = extract_keywords(name)
        meta = extract_name_metadata(name_lookup[food_id]) if food_id in name_lookup else {}

        # Append curated name varients from NAME.FT in decreasing specificity
        if meta.get("short_name"):
            keywords += [t.strip().lower() for t in re.split(r'[;,]', meta["short_name"]) if t.strip()]
        if meta.get("alt_names"):
            keywords += [t.strip().lower() for t in re.split(r'[;,]', meta["alt_names"]) if t.strip()]
        if meta.get("generic"):
            prefix = f"{meta['kind']} " if meta.get("kind") else ""
            keywords.append(f"{prefix}{meta['generic']}".strip().lower())

        short = meta.get("short_name", "")
        short_terms = [t.strip().lower() for t in re.split(r'[;,]', short) if t.strip() and len(t.strip()) > 2]

        seen = set()
        deduped = [k for k in keywords if k and not (k in seen or seen.add(k))]
        brands = extract_brand_keywords(name, meta.get("sampling_details", ""), nlp)

        # Prefer the curated short_name as key_term
        if short_terms:
            key_term = short_terms[0]
        else:
            key_term = (deduped[0] if deduped else name.split(",")[0].lower())

        index[food_id] = {
            "name": name,
            "key_term": key_term,
            "keywords": brands + deduped,
            "part": meta.get("part") or None,
            "brands": brands,
            "serving_measure": csm_lookup.get(food_id, []),
            "is_recipe": food_id.startswith("R"),
            "nutrients": {
                "energy_kj": clean_num(row.get("Energy, total metabolisable (kJ)")),
                "protein_g": clean_num(row.get("Protein, total; calculated from total nitrogen")),
                "fat_g":     clean_num(row.get("Fat, total")),
                "carbs_g":   clean_num(row.get("Available carbohydrate, FSANZ")),
                "fibre_g":   clean_num(row.get("Fibre, total dietary")),
                "sodium_mg": clean_num(row.get("Sodium")),
            },
        }
    return index


def build_recipe_index(ingredient_df, food_index):
    """
    Building a recipe composition index mapping recipe FoodID to a list of weighted ingredients
    :param ingredient_df: DataFrame from INGREDIENT.FT.XLSX
    :param food_index: Primary food index, used to validate ingredient IDs and resolve names
    :return: Dict mapping recipe FoodID -> list of {ingredient_id, ingredient_name, weight_fraction}
    """
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
    """
    Builds a FAISS inner product index of L2 normalised sentence embeddings.

    Each food encoded from name, key_term and up to 10 keywords seperated with ' | ' to ensure encoder treats them as seperate

    :param food_index: Primary food index dict
    :param model: SentanceTransformer model used to encode food text
    :return: Tuple of (faiss_index, ids) where ids[i] is the FoodID for vector i
    """
    ids = list(food_index.keys())
    texts = []

    for entry in food_index.values():
        all_tems = list(dict.fromkeys(
            [entry["name"]] + [entry["key_term"]] + entry["keywords"]
        ))
        text = " | ".join(all_tems[:10])
        texts.append(text.lower())

    embeddings = model.encode(texts, batch_size=128, show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    return index, ids

def build_bm25_index(food_index):
    """
    Builds a BM25 index over full food names and all keywords
    :param food_index: Primary food index dict
    :return: Tuple of (bm25, ids) where ids[i] is the FoodID for document i
    """
    ids = list(food_index.keys())
    corpus = []
    for entry in food_index.values():
        text = entry["name"] + " " + " ".join(entry["keywords"])
        corpus.append(text.lower().split())
    bm25 = BM25Okapi(corpus)
    return bm25, ids

def save_indexes(food_index, recipe_index, fi, faiss_ids, bm25, bm25_ids):
    """
    Persist all indexes to INDEX_DIR.
    :param food_index: Primary food index dict
    :param recipe_index: Recipe composition index dict
    :param fi: Built FAISS index object
    :param faiss_ids: List of FoodIDs corresponding to FAISS vectors
    :param bm25: Built BM25Okapi object
    :param bm25_ids: List of FoodIDs corresponding to BM25 documents
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (INDEX_DIR / "food_index.json").write_text(json.dumps(food_index))
    (INDEX_DIR / "recipe_index.json").write_text(json.dumps(recipe_index))
    (INDEX_DIR / "faiss_ids.json").write_text(json.dumps(faiss_ids))
    faiss.write_index(fi, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump((bm25, bm25_ids), f)
    print(f"Indexes saved to {INDEX_DIR}/")


def load_indexes():
    """
    Load all pre-built indexes from INDEX_DIR.

    :raises FileNotFoundError: If any index is missing (i.e. index.py has not been run).
    :return: Tuple of (food_index, recipe_index, faiss_index, faiss_ids, bm25, bm25_ids).
    """
    missing = [
        p for p in ("food_index.json", "recipe_index.json", "faiss.index", "faiss_ids.json")
        if not (INDEX_DIR / p).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing index files: {missing}. Run `python index.py` to build them."
        )
    food_index = json.loads((INDEX_DIR / "food_index.json").read_text())
    recipe_index = json.loads((INDEX_DIR / "recipe_index.json").read_text())
    fi = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    faiss_ids = json.loads((INDEX_DIR / "faiss_ids.json").read_text())
    with open(INDEX_DIR / "bm25.pkl", "rb") as f:
        bm25, bm25_ids = pickle.load(f)
    return food_index, recipe_index, fi, faiss_ids, bm25, bm25_ids

if __name__ == "__main__":
    print("Loading source data...")
    csm_df = pd.read_excel(f"{PRINCIPAL_XLSX}/CSM.FT.XLSX", skiprows=1)
    csm_df.columns = csm_df.columns.str.strip()

    food_df = pd.read_excel(f"{PRINCIPAL_XLSX}/Unabridged/Unabridged DATA.AP.xlsx", skiprows=1)
    food_df = food_df[food_df["FoodID"] != "FoodID"]
    food_df.columns = food_df.columns.str.strip()

    name_df = pd.read_excel(f"{SUPPORTING_XLSX}/NAME.FT.XLSX", skiprows=1)
    name_df.columns = name_df.columns.str.strip()

    ingredient_df = pd.read_excel(f"{PRINCIPAL_XLSX}/INGREDIENT.FT.XLSX", skiprows=1)
    ingredient_df.columns = ingredient_df.columns.str.strip()

    print("Building food index...")
    nlp_ner = spacy.load("en_core_web_md")
    food_index = build_index(food_df, csm_df, name_df, nlp_ner)
    recipe_index = build_recipe_index(ingredient_df, food_index)
    print(f"{len(food_index)} foods | {len(recipe_index)} recipes")

    print("Building embedding index...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    faiss_index, faiss_ids = build_embedding_index(food_index, embedding_model)
    print(f"{len(faiss_ids)} vectors")

    print("Building BM25 index")
    bm25, bm25_ids = build_bm25_index(food_index)
    print(f"{len(bm25_ids)} BM25 entries")

    save_indexes(food_index, recipe_index, faiss_index, faiss_ids, bm25, bm25_ids)