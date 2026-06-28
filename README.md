# Unstructured Food Input

A web application for processing unstructured food input.

## Getting Started

### Prerequisites
- Node.js
- Python 3.10+
- [Ollama](https://ollama.com) with `qwen2.5:7b-instruct-q4_K_M`

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
```

### Clone the Repository
```bash
git clone <repository-url>
cd unstructured-food-input
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
Required once before running the backend, and after any changes to the source data.
```bash
cd server
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd ..
python -m server.index
```

### Run the Frontend
```bash
npm install
npm run dev
```

### Run the Backend Server
```bash
source server/.venv/bin/activate
uvicorn server.server:app --reload
```

> **Note:** The frontend, backend, and Ollama must all be running concurrently for the application to function properly.

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


## API
### `POST /extract`
Accepts a JSON body and returns matched food entities with nutrients.