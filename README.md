# Food Natural Language Processing

A web application for turning free-text food descriptions into matched food database entries, estimated quantities, and scaled nutrients. 
The FastAPI backend contains the extraction and retrieval pipeline; `src/` is a small Vite interface for exercising that API during development.

### Prerequisites
- Node.js
- Python 3.10+
- [Ollama](https://ollama.com) with `qwen2.5:7b-instruct-q4_K_M`

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
```

## Getting started

### Clone the Repository
```bash
git clone <repository-url>
cd unstructured-food-input
```

### Create a Virtual Environment and Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Set up .env file (Optional)
If you would like to use OpenRouter, create a `.env` file in the root directory with the following content:
```env
OPENROUTER_API_KEY=<api_key_here>
OPENROUTER_MODEL=claude-opus-4.8
USE_LOCAL_LLM=False
```
> USE_LOCAL_LLM defaults to `True` and will use the local Ollama model if no `.env` file is present.

### Build the Search Indexes
Required once before running the API server

```bash
python -m server.index
```

### Run the Backend Server
```bash
source .venv/bin/activate  # if not already active
uvicorn server.server:app --port 8000
```

### Run the Frontend
```bash
npm install
npm run dev
```

> **Note:** The frontend, backend, and Ollama must all be running concurrently for the application to function properly.

The frontend calls `http://localhost:8000` by default. For a backend on another host, create a `.env.local` file containing `VITE_NLP_API_URL=http://host:port` before running the Vite server.

### Hugging Face Authentication (optional)
Two models are downloaded automatically from Hugging Face on first run: `sentence-transformers/all-MiniLM-L6-v2` (semantic embeddings) and `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranking). Both are publicly available and require no access approval.

Authentication is not required, but setting a Hugging Face token is recommended to avoid anonymous rate limits on Hub downloads, which can cause timeouts if the models aren't already cached locally.

Generate a read token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then either log in via the CLI (persists across sessions):

```bash
hf auth login
```

Or export for the current session:

```bash
export HF_TOKEN=hf_your_token_here
```

Models are cached to `~/.cache/huggingface/` after the first download.

---

## Data Sources and Indexes

`python -m server.index` reads three sources and merges them into one index keyed by a prefixed FoodID, so where an entry came from is always visible:

| Source                       | Prefix | What it brings                                                        |
|------------------------------|--------|-----------------------------------------------------------------------|
| NZ FOODfiles 2024            | `NZ:`  | The primary database: nutrients, serving measures, measured densities |
| AUSNUT 2023                  | `AU:`  | Extra coverage, recipes, and density rows from the measures file      |
| Open Food Facts              | `OFF:` | Branded packaged products, so brand names can be matched directly     |

Both FOODfiles and AUSNUT recipes are stored as ingredient compositions, which is what lets a composite dish's nutrients add up from its parts rather than a single row.

Five index files get written to `data/indexes/`:

| File                | What it's for                                                    |
|---------------------|------------------------------------------------------------------|
| `food_index.json`   | Per-food metadata, keywords, brands, nutrients, serving measures |
| `recipe_index.json` | Recipe -> ingredient composition mapping                         |
| `faiss.index`       | Dense vector index for semantic search                           |
| `faiss_ids.json`    | Ordered FoodIDs matching the FAISS vectors                       |
| `bm25.pkl`          | Sparse BM25 index for lexical search                             |

---

## API

### `POST /extract`
Accepts a JSON body and returns the matched food entities with their nutrients.

```json
{ "text": "two apples and a slice of vogels toast" }
```

Response, one entity per food found in the order they appear:

```json
{
  "text": "two apples and a slice of vogels toast",
  "source": "llm",
  "entities": [
    {
      "text": "apples",
      "resolved_text": "apples",
      "char_start": 4,
      "char_end": 10,
      "quantity": 2.0,
      "quantity_char_start": 0,
      "quantity_char_end": 3,
      "unit": null,
      "unit_char_start": null,
      "grams": 223.5,
      "confidence": 1.0,
      "rerank_source": "cross_encoder",
      "source": "llm",
      "match": { },
      "candidates": [ ]
    }
  ]
}
```

- `source` -> `llm` | `spacy`, which path produced the result. Present per entity as well as at the top level
- `resolved_text` and `rerank_source` -> LLM path only. `resolved_text` is what was actually linked once slang was expanded or a brand stripped
- `rerank_source` -> `rag` | `cross_encoder`
- The `char_*` offsets locate each span in the input for highlighting, and are `null` where a span couldn't be placed

`match` is `candidates[0]`, the chosen one. Up to 10 candidates come back, each with its nutrients already scaled to `grams`:

```json
{
  "food_id": "AU:F000106",
  "name": "Apple, raw, not further defined",
  "score": 122.58,
  "is_recipe": true,
  "recipe_ingredients": [
    { "food_id": "AU:F000101", "name": "Apple, peeled, raw", "weight_fraction": 0.136564 }
  ],
  "nutrients": {
    "energy_kj": 505.93,
    "protein_g": 0.64,
    "fat_g": 0.0,
    "carbs_g": 26.62,
    "fibre_g": 5.4,
    "sodium_mg": 1.93
  },
  "density": {
    "density_g_per_ml": 0.495,
    "density_log_sigma": 0.0998,
    "density_source": "measured"
  }
}
```

- `density` -> only on the chosen match, and it's what the `volume-estimation` server reads. `density_source` is `measured_presentation` | `measured` | `ratio` | `global_median`, strongest first, with `density_log_sigma` widening as the source weakens
- `recipe_ingredients` -> empty unless `is_recipe`, in which case the nutrients above are the blend of these parts
