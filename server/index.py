"""
Builds and persists the food matching indexes

We read the NZ FOODfiles 2024 and AUSNUT 2023 excel database sources and produce five indexes:
- food_index.json -> per-food metadata, keywords, nutrients, and serving sizes (NZ + AU entries)
- recipe_index.json -> recipe to ingredient composition mapping (NZ + AU recipes)
- faiss.index -> dense vector index for semantic search
- bm25.pkl -> sparse BM25 index for lexical search
- faiss_ids.json -> ordered list of FoodIDs corresponding to FAISS vectors
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

try:
    from .config import ALL_UNITS, PRINCIPAL_XLSX, SUPPORTING_XLSX, AUS_DATA_DIR, OFF_DATA_CSV
except ImportError:
    from config import ALL_UNITS, PRINCIPAL_XLSX, SUPPORTING_XLSX, AUS_DATA_DIR, OFF_DATA_CSV

INDEX_DIR = Path("data/indexes")

AUS_NUTRIENT_MAP = {
    "Energy with dietary fibre (kJ)": "energy_kj",
    "Protein (g)": "protein_g",
    "Total fat (g)": "fat_g",
    "Available carbohydrate, without sugar alcohols (g)": "carbs_g",
    "Dietary fibre (g)": "fibre_g",
    "Sodium (Na) (mg)": "sodium_mg",
}

OFF_NUTRIENT_MAP ={
    "energy-kj_100g": "energy_kj",
    "proteins_100g": "protein_g",
    "fat_100g": "fat_g",
    "carbohydrates_100g": "carbs_g",
    "fiber_100g": "fibre_g",
    "sodium_100g": "sodium_mg",  # g/100g -> needs ×1000
}

def parse_off_serving(serving_size_str, serving_quantity=None):
    """
    Attempt to parse a OFF serving_size. It starts as a free-text string so we try to standardise it.
    :param serving_size_str: Raw OFF serving_size string
    :param serving_quantity: Numeric serving_quantity field as fallback gram weight
    :return: List with one serving measure dict, or [] if unparseable
    """
    raw = str(serving_size_str).strip() if serving_size_str else ""
    if raw.lower() in ("", "nan", "not indicated.", "not indicated"):
        try:
            grams = float(serving_quantity)
            if math.isnan(grams):
                return []
            return [{"name": "1 serving", "grams": grams, "ml": None}]
        except (TypeError, ValueError):
            return []

    label = raw
    s = raw.replace(",", ".") # replace European decimals

    grams = ml = None

    match = re.search(r'\((\d+\.?\d*)\s*g\)', s, re.IGNORECASE)
    if match:
        grams = float(match.group(1))

    if grams is None:
        match = re.match(r'^(\d+\.?\d*)\s*g\b', s, re.IGNORECASE)
        if match:
            grams = float(match.group(1))

    if grams is None:
        match = re.search(r'(\d+\.?\d*)\s*ml\b', s, re.IGNORECASE)
        if match:
            ml = float(match.group(1))

    if grams is None and ml is None:
        try:
            grams = float(serving_quantity)
        except (TypeError, ValueError):
            return []

    return [{"name": label, "grams": grams, "ml": ml}]

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

def clean_num(value):
    try:
        f = float(value)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None

def find_first_measured_density(serving_measures):
    """
    Returns the first measured density in a list of serving measures, or None if none are measured
    """
    for measure in serving_measures:
        density = measure.get("density_g_per_ml")
        if density is not None:
            return density
    return None

def build_nz_food_index(food_df, csm_df, name_df, nlp):
    """
    Builds the primary food index dict keyed by FoodID's
    :param food_df: DataFrame from Unbridged Data.AP.xlsx
    :param csm_df: DataFrame from CSM.FT.XLSX (common serving measures)
    :param name_df: DataFrame from NAME.FT.XLSX (curated name metadata)
    :param nlp: spaCy model used for brand NER during keyword extraction
    :return: Dict mapping NZ:FoodID -> {source, name, key_term, description, keywords, brands, serving_measure, is_recipe, nutrients}
    """
    index = {}

    csm_columns = ["CSM", "Measure"]
    if "Density (g/cm3)" in csm_df.columns:
        csm_columns.append("Density (g/cm3)")

    csm_lookup = {
        str(food_id).strip(): [
            {
                "CSM": record["CSM"],
                "Measure": record["Measure"],
                "density_g_per_ml": clean_num(record["Density (g/cm3)"])
            }
            for record in group[csm_columns].to_dict("records")
        ]
        for food_id, group in csm_df.groupby("FoodID")
    }

    name_lookup = {str(r.get("FoodID", "")).strip(): r for _, r in name_df.iterrows()}

    for _, row in food_df.iterrows():
        raw_id = str(row.get("FoodID", "")).strip()
        if not raw_id: continue

        prefixed_id = f"NZ:{raw_id}"
        name = str(row.get("Food Name", "")).strip()

        keywords = extract_keywords(name)
        meta = extract_name_metadata(name_lookup.get(raw_id, {}))

        # Append curated name varients from NAME.FT in decreasing specificity
        if meta.get("short_name"):
            keywords += [t.strip().lower() for t in re.split(r'[;,]', meta["short_name"]) if t.strip()]
        if meta.get("alt_names"):
            keywords += [t.strip().lower() for t in re.split(r'[;,]', meta["alt_names"]) if t.strip()]
        if meta.get("generic"):
            prefix = f"{meta['kind']} " if meta.get("kind") else ""
            keywords.append(f"{prefix}{meta['generic']}".strip().lower())

        brands = extract_brand_keywords(name, meta.get("sampling_details", ""), nlp)

        index[prefixed_id] = {
            "source": "nz_foodfiles_2024",
            "name": name,
            "key_term": keywords[0] if keywords else name.lower(),
            "description": meta.get("generic", ""),
            "keywords": list(dict.fromkeys(brands + keywords)),
            "brands": brands,
            "serving_measure": csm_lookup.get(raw_id, []),
            "measured_density_g_per_ml": find_first_measured_density(csm_lookup.get(raw_id, [])),
            "is_recipe": raw_id.startswith("R"),
            "nutrients": {
                "energy_kj": clean_num(row.get("Energy, total metabolisable (kJ)")),
                "protein_g": clean_num(row.get("Protein, total; calculated from total nitrogen")),
                "fat_g":     clean_num(row.get("Fat, total")),
                "carbs_g":   clean_num(row.get("Available carbohydrate, FSANZ")),
                "fibre_g":   clean_num(row.get("Fibre, total dietary")),
                "sodium_mg": clean_num(row.get("Sodium")),
            }
        }
    return index

def build_aus_food_index(nutrient_df, detail_df, measure_df, nlp):
    """
    Builds the AU food index dict keyed by prefixed FoodIDs
    :param nutrient_df: DataFrame from AUSNUT 2023 - Food nutrient profiles.xlsx (per-100g nutrients)
    :param detail_df: DataFrame from AUSNUT 2023 - Food details xlsx (names, descriptions, derivation)
    :param measure_df: DataFrame from AUSNUT 2023 - Food measures.xlsx (serving sizes with gram weights)
    :param nlp: spaCy model used for brand NER during keyword extraction
    :return: Dict mapping AU:FoodID -> {source, name, key_term, description, keywords, brands, serving_measure, is_recipe, nutrients}
    """
    index = {}

    measure_lookup = {}
    aus_density_lookup = {}
    for _, row in measure_df.iterrows():
        fid = str(row.get("Public food key", "")).strip()
        if not fid:
            continue

        descriptors = [
            str(row.get(f"Descriptor {i}", "") or "").strip()
            for i in range(1, 5)
        ]
        descriptor_str = " ".join(d for d in descriptors if d and d.lower() != "nan")

        if "density" in descriptor_str.lower():
            density = clean_num(row.get("Gram amount"))
            if density is not None:
                aus_density_lookup[fid] = density
            continue

        qty = str(row.get("Quantity", "") or "").strip()
        measure_lookup.setdefault(fid, []).append({
            "name": f"{qty} {descriptor_str}".strip(),
            "grams": clean_num(row.get("Gram amount")),
            "ml": clean_num(row.get("Volume")),
        })

    nutrient_lookup = {
        str(r.get("Public food key")).strip(): r
        for _, r in nutrient_df.iterrows()
    }

    for _, row in detail_df.iterrows():
        raw_id = str(row.get("Public food key", "")).strip()
        prefixed_id = f"AU:{raw_id}"
        name = str(row.get("Food name", "")).strip()
        description = str(row.get("Food description", "")).strip()
        derivation = str(row.get("Derivation", "") or "").strip()

        nut_row = nutrient_lookup.get(raw_id, {})

        brands = extract_brand_keywords(name, str(row.get("Sampling details", "")), nlp)
        keywords = extract_keywords(name)
        if row.get("Food group name"):
            keywords.append(str(row.get("Food group name")).lower())

        index[prefixed_id] = {
            "source": "ausnut_2023",
            "name": name,
            "description": description,
            "key_term": keywords[0] if keywords else name.lower(),
            "keywords": list(dict.fromkeys(brands + keywords)),
            "brands": brands,
            "serving_measure": measure_lookup.get(raw_id, []),
            "measured_density_g_per_ml": aus_density_lookup.get(raw_id),
            "is_recipe": derivation.lower() == "recipe",
            "nutrients": {
                mapped_key: clean_num(nut_row.get(orig_col))
                for orig_col, mapped_key in AUS_NUTRIENT_MAP.items()
            }
        }
    return index

def build_off_food_index(df):
    """
    Builds the OFF food index dict
    :param df: DataFrame from the openfoodfacts CSV export
    :return: Dict mapping OFF:barcode -> food index entry
    """
    index = {}

    for _, row in df.iterrows():
        code = str(row.get("code", "") or "").strip()
        if not code:
            continue

        name = str(row.get("product_name", "") or "").strip()
        if not name or name.lower() == "nan":
            continue

        prefixed_id = f"OFF:{code}"

        generic = str(row.get("generic_name", "") or "").strip()
        generic = "" if generic.lower() == "nan" else generic

        categories_raw = str(row.get("categories_en", "") or "")
        keywords = [k.strip().lower() for k in categories_raw.split(",") if k.strip()]
        keywords = list(dict.fromkeys(keywords))

        brands_raw = str(row.get("brands", "") or "")
        brands = [
            b.strip().lower() for b in brands_raw.split(",")
            if b.strip() and b.strip().lower() not in ("nan", "")
        ]

        serving_measure = parse_off_serving(row.get("serving_size"), row.get("serving_quantity"))

        # sodium: OFF stores g/100g, pipeline expects mg/100g
        sodium_g = clean_num(row.get("sodium_100g"))
        sodium_mg = round(sodium_g * 1000, 3) if sodium_g is not None else None

        index[prefixed_id] = {
            "source": "openfoodfacts",
            "name": name,
            "key_term": keywords[0] if keywords else name.lower(),
            "description": generic,
            "keywords": list(dict.fromkeys(brands + keywords)),
            "brands": brands,
            "serving_measure": serving_measure,
            "is_recipe": False,
            "nutrients": {
                "energy_kj": clean_num(row.get("energy-kj_100g")),
                "protein_g": clean_num(row.get("proteins_100g")),
                "fat_g": clean_num(row.get("fat_100g")),
                "carbs_g": clean_num(row.get("carbohydrates_100g")),
                "fibre_g": clean_num(row.get("fiber_100g")),
                "sodium_mg": sodium_mg,
            },
        }

    return index


def build_nz_recipe_index(ingredient_df, food_index):
    """
    Builds the NZ recipe composition index from INGREDIENT.FT.XLSX.
    Ingredient weights are stored as percentage fractions in the source and converted to decimals.
    :param ingredient_df: DataFrame from INGREDIENT.FT.XLSX
    :param food_index: Merged food index, used to validate ingredient IDs and resolve names
    :return: Dict mapping NZ:FoodID -> list of {ingredient_id, ingredient_name, weight_fraction}
    """
    recipe_index = {}
    if ingredient_df is None:
        return recipe_index

    weight_col = "Weight Fraction(%)" if "Weight Fraction(%)" in ingredient_df.columns else "Weight Fraction (%)"

    for _, row in ingredient_df.iterrows():
        recipe_id = f"NZ:{str(row.get('Recipe FoodID', '')).strip()}"
        ingredient_id = f"NZ:{str(row.get('Ingredient FoodID', '')).strip()}"
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

def build_aus_recipe_index(recipe_df, food_index):
    """
    Builds the AU recipe composition index from the AUSNUT recipe xlsx.
    Ingredient weights are absolute grams per recipe, normalised to weight fractions.
    :param recipe_df: DataFrame from AUSNUT 2023 - Food nutrient recipes.xlsx
    :param food_index: Merged food index, used to validate ingredient IDs and resolve names
    :return: Dict mapping AU:FoodID -> list of {ingredient_id, ingredient_name, weight_fraction}
    """
    recipe_index = {}
    if recipe_df is None:
        return recipe_index

    raw = {}
    for _, row in recipe_df.iterrows():
        recipe_id = f"AU:{str(row.get('Public food key', '')).strip()}"
        ingredient_id = f"AU:{str(row.get('Ingredient public food key', '')).strip()}"
        try:
            grams = float(row.get("Ingredient Weight (g)", 0) or 0)
        except (TypeError, ValueError):
            grams = 0.0

        if recipe_id and ingredient_id and grams > 0 and ingredient_id in food_index:
            raw.setdefault(recipe_id, []).append({
                "ingredient_id": ingredient_id,
                "ingredient_name": food_index[ingredient_id]["name"],
                "grams": grams,
            })

        for recipe_id, ingredients in raw.items():
            total = sum(i["grams"] for i in ingredients)
            if total <= 0:
                continue
            recipe_index[recipe_id] = [
                {
                    "ingredient_id": i["ingredient_id"],
                    "ingredient_name": i["ingredient_name"],
                    "weight_fraction": round(i["grams"] / total, 6),
                }
                for i in ingredients
            ]

    return recipe_index


def build_embedding_index(food_index, model):
    """
    Builds a FAISS inner product index of L2-normalised sentence embeddings.

    Each food is encoded from its name, description, and up to 10 keywords separated with ' | ' so the encoder treats them
    as distinct fields rather than one long string.

    :param food_index: Merged food index dict (NZ + AU entries)
    :param model: SentenceTransformer model used to encode food text
    :return: Tuple of (faiss_index, ids) where ids[i] is the FoodID for vector i
    """
    ids = list(food_index.keys())
    texts = []

    for entry in food_index.values():
        parts = [
            entry["name"],
            entry.get("description") or "",
            " ".join(entry["keywords"][:10])
        ]
        text = " | ".join([p.lower() for p in parts if p])
        texts.append(text.lower())

    embeddings = model.encode(texts, batch_size=128, show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    return index, ids

def build_bm25_index(food_index):
    """
    Builds a BM25 index over full food names and all keywords
    :param food_index: Merged food index dict (NZ + AU entries)
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

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and collapse internal whitespace from all column names."""
    df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
    return df

if __name__ == "__main__":
    nlp_ner = spacy.load("en_core_web_md")

    print("Loading NZ source data...")
    nz_csm_df = pd.read_excel(f"{PRINCIPAL_XLSX}/CSM.FT.XLSX", skiprows=1)
    nz_food_df = pd.read_excel(f"{PRINCIPAL_XLSX}/Unabridged/Unabridged DATA.AP.xlsx", skiprows=1)
    nz_name_df = pd.read_excel(f"{SUPPORTING_XLSX}/NAME.FT.xlsx", skiprows=1)

    nz_index = build_nz_food_index(nz_food_df, nz_csm_df, nz_name_df, nlp_ner)

    print("Loading AUSNUT source data...")
    aus_nutrition_df = normalise_columns(pd.read_excel(f"{AUS_DATA_DIR}/AUSNUT 2023 - Food nutrient profiles.xlsx", sheet_name=1, skiprows=2))
    aus_details_df = normalise_columns(pd.read_excel(f"{AUS_DATA_DIR}/AUSNUT-2023-Food-details-4.xlsx", sheet_name=1, skiprows=2))
    aus_measures_df = normalise_columns(pd.read_excel(f"{AUS_DATA_DIR}/AUSNUT 2023 - Food measures.xlsx", sheet_name=1, skiprows=2))

    aus_index = build_aus_food_index(aus_nutrition_df, aus_details_df, aus_measures_df, nlp_ner)

    print("Loading OpenFoodFacts data...")
    off_df = pd.read_csv(OFF_DATA_CSV, low_memory=False, dtype={"code": str})
    off_index = build_off_food_index(off_df)

    print("Merging Indexes...")
    merged_food_index = {**off_index, **nz_index, **aus_index}

    print("Building recipe indexes...")
    nz_ingredient_df = pd.read_excel(f"{PRINCIPAL_XLSX}/INGREDIENT.FT.XLSX", skiprows=1)

    aus_recipe_df = normalise_columns(pd.read_excel(f"{AUS_DATA_DIR}/AUSNUT 2023 - Food nutrient recipes.xlsx", sheet_name=1, skiprows=2))

    nz_recipe_index = build_nz_recipe_index(nz_ingredient_df, merged_food_index)
    aus_recipe_index = build_aus_recipe_index(aus_recipe_df, merged_food_index)
    recipe_index = {**nz_recipe_index, **aus_recipe_index}

    print(f"{len(merged_food_index)} foods | {len(nz_recipe_index)} NZ recipes | {len(aus_recipe_index)} AU recipes | {len(off_index)} branded OFF products")

    print("Building embedding index...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    faiss_index, faiss_ids = build_embedding_index(merged_food_index, embedding_model)
    print(f"{len(faiss_ids)} vectors")

    print("Building BM25 index")
    bm25, bm25_ids = build_bm25_index(merged_food_index)
    print(f"{len(bm25_ids)} BM25 entries")

    save_indexes(merged_food_index, recipe_index, faiss_index, faiss_ids, bm25, bm25_ids)