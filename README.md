# Unstructured Food Input

A web application for processing unstructured food input.

## Getting Started

### Prerequisites
- Node.js
- Python 3.x
- [Ollama](https://ollama.com) with `qwen2.5:7b-instruct-q4_K_M`

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
```

### Clone the Repository
```bash
git clone <repository-url>
cd unstructured-food-input
```

### Build the Search Indexes
Required once before running the backend, and after any changes to the source data.
```bash
pip install -r requirements.txt
python index.py
```

### Run the Frontend
```bash
npm install
npm run dev
```

### Run the Backend Server
```bash
uvicorn server:app --reload
```

> **Note:** The frontend, backend, and Ollama must all be running concurrently for the application to function properly.