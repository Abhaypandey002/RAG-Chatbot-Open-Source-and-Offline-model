# RAG Chatbot — Streamlit App

A Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**. It lets you upload/reference PDFs, ask questions, and get grounded answers with source citations. The app can run either:

- **Locally with a GGUF model** via `llama.cpp` bindings (no external API), **or**
- **Using a hosted LLM API** (e.g., OpenAI), skipping the local model entirely.

> This README is designed so you can submit it as a project deliverable. It includes setup, configuration, run steps, and troubleshooting (including the `GGUF_MODEL_PATH` error you hit).


## Table of Contents

1. [Features](#features)  
2. [Architecture](#architecture)  
3. [Project Structure](#project-structure)  
4. [Prerequisites](#prerequisites)  
5. [Quick Start](#quick-start)  
6. [Configuration (.env)](#configuration-env)  
7. [Model Options](#model-options)  
   - [Option A — Local GGUF (no API)](#option-a--local-gguf-no-api)  
   - [Option B — Hosted API (OpenAI)](#option-b--hosted-api-openai)  
8. [Run the App](#run-the-app)  
9. [How to Use](#how-to-use)  
10. [Logging & Data](#logging--data)  
11. [Troubleshooting](#troubleshooting)  
12. [FAQ](#faq)  


## Features

- 💬 **Chat UI** using Streamlit with message history.  
- 📄 **RAG pipeline** to answer from PDFs in `data/pdfs`.  
- 🧠 Pluggable **LLM backends**: local GGUF via `llama.cpp` or hosted APIs.  
- 🧾 **Structured logging** to JSONL for auditability.  
- ⚙️ Environment-based configuration with `.env`.  


## Architecture

```
User → Streamlit UI (app.py)
          │
          ├─► RAG core (rag.py): builds prompts, retrieves chunks from PDFs
          │
          ├─► LLM facade (models/llm.py): selects backend (GGUF or API)
          │
          └─► Storage (storage/logger.py): JSONL logs, data directories
```

- `rag.answer_query()` orchestrates retrieval + generation.
- `models/llm.get_llm()` constructs the LLM client based on env vars.
- `storage/logger.JsonlLogger` captures inputs/outputs as JSONL.


## Project Structure

```
<project-root>/
├─ app.py
├─ rag.py
├─ models/
│  └─ llm.py
├─ storage/
│  └─ logger.py
├─ data/
│  └─ pdfs/          # place your PDFs here
├─ models/           # place your .gguf model(s) here (if using local LLM)
├─ .env              # environment configuration (you create this)
├─ requirements.txt  # Python dependencies (ensure this exists/updated)
└─ README.md
```


## Prerequisites

- **Python 3.10+** (works on Windows/macOS/Linux)
- **pip** (or conda/uv/poetry, but examples use pip)
- **Streamlit** and supporting libs (installed via `requirements.txt`)
- If using **local GGUF**:
  - `pip install llama-cpp-python` (CPU) or `pip install llama-cpp-python[cuda]` (GPU)


## Quick Start

> Replace the folder path with your actual project location.

### Windows (PowerShell)

```powershell
cd D:\new_bot
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python scripts\download_model.py
```

### macOS / Linux

```bash
cd /path/to/new_bot
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts\download_model.py
```

Then create your `.env` (see next section) and pick **Model Option A or B**.


## Configuration (.env)

Create a file named **`.env`** at the project root (same level as `app.py`).

### Minimal template (choose A or B later)

```ini
# General
LOGS_DIR=./storage/logs
DATA_DIR=./data/pdfs

# Which backend to use: gguf (local) or openai (hosted)
LLM_PROVIDER=gguf

# If using local GGUF (Option A):
GGUF_MODEL_PATH=./models/YourModel.gguf

# If using OpenAI (Option B):
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-...
# TEXT_MODEL=gpt-4o-mini
# EMBEDDINGS_MODEL=text-embedding-3-small
```

> The app already calls `load_dotenv()`, so values in `.env` are picked up automatically.


## Model Options

### Option A — Local GGUF (no API)

1. **Download a GGUF model** compatible with `llama.cpp` (e.g., an *Instruct* 7B–8B Q4_K_M).  
   Put it under:
   ```
   <project-root>/models/YourModel.gguf
   ```

2. **Set the path in `.env`** (absolute path recommended on Windows):
   ```ini
   LLM_PROVIDER=gguf
   GGUF_MODEL_PATH=D:/new_bot/models/YourModel.gguf
   ```
   > Use forward slashes `D:/...` to avoid escape issues on Windows.

3. **Install llama.cpp bindings**:
   ```bash
   pip install llama-cpp-python         # CPU
   # or
   pip install "llama-cpp-python[cuda]" # NVIDIA GPU
   ```

4. (Optional) Tune threads/ctx in your `llm.py` if those envs are supported.


### Option B — Hosted API (OpenAI)

1. **Set `.env`**:
   ```ini
   LLM_PROVIDER=openai
   OPENAI_API_KEY=sk-...
   TEXT_MODEL=gpt-4o-mini
   EMBEDDINGS_MODEL=text-embedding-3-small
   ```

2. Ensure `models/llm.py` **skips** GGUF checks when `LLM_PROVIDER=openai`.  
   Example logic (illustrative):
   ```python
   provider = os.getenv("LLM_PROVIDER", "gguf").lower()
   if provider == "gguf":
       # require GGUF_MODEL_PATH to exist
   elif provider == "openai":
       # require OPENAI_API_KEY and TEXT_MODEL; do NOT check GGUF path
   else:
       raise ValueError("Unknown LLM_PROVIDER")
   ```


## Run the App

From the project root (with your virtualenv activated):

```bash
streamlit run app.py
```

- Streamlit prints a local URL (e.g., `http://localhost:8501`).  
- Keep the terminal open while using the app.


## How to Use

1. Put your PDFs into `data/pdfs`.  
2. Launch the app (`streamlit run app.py`).  
3. In the UI:
   - Type your question.
   - The app runs RAG: retrieves relevant chunks and generates an answer.
   - You’ll see the answer plus **source citations** (if implemented in your `rag.py`).


## Logging & Data

- **Logs**: structured JSONL files under `storage/logs` (configurable via `LOGS_DIR`).  
- **Data**: PDFs in `data/pdfs` by default (configurable via `DATA_DIR`).  
- These paths are created on first run if they don’t exist.


## Troubleshooting

### ❗ FileNotFoundError: `GGUF_MODEL_PATH` is 'None' or file does not exist

**Cause**: Local GGUF mode is enabled, but the model path is missing or wrong.

**Fix**:
1. Ensure the `.gguf` file exists (e.g., `D:/new_bot/models/Llama-3-8B-Instruct-Q4_K_M.gguf`).
2. Set an absolute path with forward slashes in `.env`:
   ```ini
   LLM_PROVIDER=gguf
   GGUF_MODEL_PATH=D:/new_bot/models/Llama-3-8B-Instruct-Q4_K_M.gguf
   ```
3. Or switch to OpenAI mode by setting:
   ```ini
   LLM_PROVIDER=openai
   OPENAI_API_KEY=sk-...
   TEXT_MODEL=gpt-4o-mini
   ```
   Make sure `llm.py` doesn’t check `GGUF_MODEL_PATH` when provider is `openai`.

### ❗ `ModuleNotFoundError: No module named 'llama_cpp'`

You selected GGUF mode but didn’t install bindings. Install:
```bash
pip install llama-cpp-python
# or GPU:
pip install "llama-cpp-python[cuda]"
```

### ❗ Streamlit fails to find `.env`

- Ensure `.env` is at the **project root** (same level as `app.py`).  
- Confirm `load_dotenv()` is called before reading env vars.

### ❗ PDFs not being used in answers

- Ensure your PDFs are in `data/pdfs`.  
- Check retrieval code in `rag.py` for file globbing/extensions and embeddings setup.

### ❗ Performance is slow with GGUF

- Use a smaller quantized model (e.g., Q4_K_M).  
- Reduce context length if configurable.  
- Enable GPU build (`llama-cpp-python[cuda]`) if you have NVIDIA GPU.


## FAQ

**Q: Can I run without internet?**  
Yes, with **Option A** (local GGUF). Place the model locally and don’t use API backends.

**Q: Which model size should I use?**  
Start with **7B–8B Instruct** quantized to **Q4_K_M** for reasonable performance on CPU; bigger models need more RAM.

**Q: Where are the logs?**  
By default in `storage/logs` as JSONL. Configure with `LOGS_DIR` in `.env`.

**Q: How do I change the theme/UI?**  
`app.py` defines themes/styles at the top. Tweak the CSS blocks or Streamlit config.


---

**That’s it!** If you include this README with your submission, your reviewer can clone the repo, create a `.env`, choose a model option, and run `streamlit run app.py` end to end.
