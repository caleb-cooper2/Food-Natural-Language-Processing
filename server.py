from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
import pandas as pd
import re

nlp = spacy.load("en_core_web_sm")

csm_df = pd.read_excel("data/New Zealand FOODfiles 2024/Principal files/Excel files/CSM.FT.XLSX", skiprows=1) # the NZ common serving measure descriptions
csm_df.columns = csm_df.columns.str.strip()

food_df = pd.read_excel("data/New Zealand FOODfiles 2024/Principal files/Excel files/Unabridged/Unabridged DATA.AP.xlsx", skiprows=1)
food_df = food_df[food_df["FoodID"] != "FoodID"]  # drop the units header row
food_df.columns = food_df.columns.str.strip()

app = FastAPI(title="Food NLP API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def extract_keywords(food_name_string):
    cleaned_name = re.sub(r'[™®©]', '', food_name_string)
    paren_terms = re.findall(r'\((.*?)\)', cleaned_name)
    cleaned_name = re.sub(r'\(.*?\)', '', cleaned_name)

    keywords = [p.strip().lower() for p in cleaned_name.split(',') if p.strip()]

    for term in paren_terms:
        keywords.append(term.strip().lower())

    return keywords

def build_index():
    index = {} # simple index in form food_id: {name, servings, nutrients}
    for _, row in food_df.iterrows():
        food_id = row["FoodID"]
        name = row["Food Name"]
        keywords = extract_keywords(name)

        measure = csm_df[csm_df["FoodID"] == food_id][["Measure"]].to_dict("records") if "FoodID" in csm_df.columns else []

        index[food_id] = {
            "name": name,
            "keywords": keywords,
            "key_term": keywords[0],
            "measure": measure, # in grams
            "nutrients": { # for now, just a summary of a few key nutrients
                "energy_kj": str(row.get("Energy, total metabolisable (kJ)")),
                "protein_g": str(row.get("Protein, total; calculated from total nitrogen")),
                "fat_g": str(row.get("Fat, total")),
                "carbs_g": str(row.get("Available carbohydrate, FSANZ")),
                "fibre_g": str(row.get("Fibre, total dietary")),
                "sodium_mg": str(row.get("Sodium")),
            }
        }
    return index

food_index = build_index()

class ExtractRequest(BaseModel):
    text: str

@app.post("/extract")
def extract(req: ExtractRequest):
    doc = nlp(req.text)
    return {
        "data": doc.to_json(),
        "index": food_index # just return one item for now
    }