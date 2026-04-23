from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
from spacy.matcher import PhraseMatcher
from spacy.language import Language
from spacy.tokens import Span
import pandas as pd
import re

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
    return {
        "data": doc.to_json(),
        "index": food_index
    }