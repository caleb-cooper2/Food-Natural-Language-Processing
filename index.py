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

def extract_name_metadata(name_row):
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
    ids = list(food_index.keys())
    corpus = []
    for entry in food_index.values():
        text = entry["name"] + " " + " ".join(entry["keywords"])
        corpus.append(text.lower().split())
    bm25 = BM25Okapi(corpus)
    return bm25, ids

def save_indexes(food_index, recipe_index, fi, faiss_ids, bm25, bm25_ids):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (INDEX_DIR / "food_index.json").write_text(json.dumps(food_index))
    (INDEX_DIR / "recipe_index.json").write_text(json.dumps(recipe_index))
    (INDEX_DIR / "faiss_ids.json").write_text(json.dumps(faiss_ids))
    faiss.write_index(fi, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump((bm25, bm25_ids), f)
    print(f"Indexes saved to {INDEX_DIR}/")


def load_indexes():
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