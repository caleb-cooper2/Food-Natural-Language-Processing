from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy

nlp = spacy.load("en_core_web_sm")

app = FastAPI(title="Food NLP API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ExtractRequest(BaseModel):
    text: str

@app.post("/extract")
def extract(req: ExtractRequest):
    doc = nlp(req.text)
    return {"data": doc.to_json()}