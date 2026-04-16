# RAG Workshop — AI Customer Support Chatbot

> **Spring 2026 Hands-On Workshop**
> Build a real AI chatbot that answers product questions using Retrieval-Augmented Generation (RAG), Pinecone, and Google Gemini.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [What is RAG?](#what-is-rag)
3. [Folder Structure](#folder-structure)
4. [Prerequisites](#prerequisites)
5. [Workshop Flow](#workshop-flow)
   - [Step 1 — Clone or Fork the Repo](#step-1--clone-or-fork-the-repo)
   - [Step 2 — Create a Virtual Environment](#step-2--create-a-virtual-environment)
   - [Step 3 — Install Dependencies](#step-3--install-dependencies)
   - [Step 4 — Set Up API Keys](#step-4--set-up-api-keys)
   - [Step 5 — Get Your Google AI Studio API Key](#step-5--get-your-google-ai-studio-api-key)
   - [Step 6 — Get Your Pinecone API Key](#step-6--get-your-pinecone-api-key)
   - [Step 7 — Create a Pinecone Index](#step-7--create-a-pinecone-index)
   - [Step 8 — Run Ingestion](#step-8--run-ingestion)
   - [Step 9 — Run the Chatbot](#step-9--run-the-chatbot)
   - [Step 10 — Experiment!](#step-10--experiment)
6. [Sample Prompts to Try](#sample-prompts-to-try)
7. [Common Errors and Fixes](#common-errors-and-fixes)
8. [Extension Ideas](#extension-ideas)

---

## Project Overview

In this workshop you will build an **AI-powered customer support chatbot** for a fictional tech product store.

The chatbot can answer questions like:
- *"What is the cheapest product?"*
- *"Tell me about the smart lamp."*
- *"Do you have anything under $50?"*
- *"What is best for a student desk setup?"*
- *"Compare the noise-canceling headphones and the earbuds."*

Instead of fine-tuning a model on your data (expensive and slow), we use **RAG** — a technique that lets the AI look up relevant information at query time and answer based only on what it finds.

---

## What is RAG?

**RAG = Retrieval-Augmented Generation**

Think of it like an open-book exam:

- Without RAG → the AI answers purely from memory (its training data). It might hallucinate or not know your specific products.
- With RAG → before answering, the AI looks up the most relevant pages in a "book" (your vector database) and then writes its answer based only on those pages.

Here is the pipeline we build in this workshop:

```
User question
     │
     ▼
Embed question           ← sentence-transformers converts text → vector
     │
     ▼
Search Pinecone          ← find the most similar product vectors
     │
     ▼
Retrieved products       ← top 3 most relevant products
     │
     ▼
Build prompt + context   ← tell Gemini "use only these products to answer"
     │
     ▼
Gemini generates answer  ← grounded, accurate response
     │
     ▼
Print to terminal
```

---

## Folder Structure

```
RAG-workshop-Spring26-Pinecone/
│
├── app.py               ← Terminal chatbot entry point
├── streamlit_app.py     ← Visual web chatbot (browser UI)
├── ingest.py            ← One-time script to load products into Pinecone
├── query.py             ← RAG logic: retrieve from Pinecone + generate with Gemini
│
├── utils/
│   ├── embeddings.py    ← Load model, embed text, convert products to text
│   └── pinecone_db.py   ← Init Pinecone, upsert vectors, query index
│
├── data/
│   └── products.json    ← 12 sample tech products
│
├── requirements.txt     ← Python dependencies
├── .env.example         ← Template for your API keys (copy to .env)
└── README.md            ← This file
```

---

## Prerequisites

Before you start, make sure you have:

| Requirement | Version | Check with |
|---|---|---|
| Python | 3.9 or higher | `python --version` |
| pip | bundled with Python | `pip --version` |
| Git | any recent version | `git --version` |
| Internet connection | needed to call APIs | — |

You will also need **free accounts** on:
- [Google AI Studio](https://aistudio.google.com) — for the Gemini API key
- [Pinecone](https://app.pinecone.io) — for the vector database

Both have free tiers that are more than enough for this workshop.

---

## Workshop Flow

Follow these steps in order. Each step builds on the previous one.

---

### Step 1 — Clone or Fork the Repo

**Option A: Fork (recommended for workshops)**

1. Click the **Fork** button at the top right of this GitHub page.
2. Clone your fork:

```bash
git clone https://github.com/YOUR_USERNAME/RAG-workshop-Spring26-Pinecone.git
cd RAG-workshop-Spring26-Pinecone
```

**Option B: Clone directly**

```bash
git clone https://github.com/KaungMyatNaing9/RAG-workshop-Spring26-Pinecone.git
cd RAG-workshop-Spring26-Pinecone
```

---

### Step 2 — Create a Virtual Environment

A virtual environment keeps the project's packages isolated from the rest of your system. Always use one.

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

> You should see `(venv)` at the start of your terminal prompt after activation.

To deactivate the virtual environment later: `deactivate`

---

### Step 3 — Install Dependencies

With your virtual environment active, install all required packages:

```bash
pip install -r requirements.txt
```

> This will download `sentence-transformers`, `pinecone`, `google-generativeai`, `python-dotenv`, and `streamlit`.
> The first install may take 2–3 minutes because sentence-transformers includes PyTorch.

---

### Step 4 — Set Up API Keys

Copy the example environment file:

**macOS / Linux:**
```bash
cp .env.example .env
```

**Windows (Command Prompt):**
```cmd
copy .env.example .env
```

Now open `.env` in any text editor and fill in your three values:

```
GOOGLE_API_KEY=your_google_key_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=rag-workshop
```

> **Important:** Never commit your `.env` file to GitHub. It is already in `.gitignore`.

---

### Step 5 — Get Your Google AI Studio API Key

1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account.
3. Click **Create API key**.
4. Copy the key and paste it as the value of `GOOGLE_API_KEY` in your `.env` file.

> The free tier includes generous usage limits — well beyond what you need for this workshop.

---

### Step 6 — Get Your Pinecone API Key

1. Go to [https://app.pinecone.io](https://app.pinecone.io) and create a free account.
2. From the left sidebar, click **API Keys**.
3. Copy the default API key (or create a new one).
4. Paste it as the value of `PINECONE_API_KEY` in your `.env` file.

---

### Step 7 — Create a Pinecone Index

You need to create an index (a database table for vectors) in Pinecone before ingesting data.

1. In the [Pinecone console](https://app.pinecone.io), click **Indexes** → **Create Index**.
2. Fill in these exact settings:

| Setting | Value |
|---|---|
| Index name | `rag-workshop` |
| Dimensions | `384` |
| Metric | `cosine` |
| Capacity mode | `Serverless` |
| Cloud | `AWS` |
| Region | `us-east-1` (or the closest to you) |

3. Click **Create Index** and wait until the status shows **Ready**.

> **Why 384 dimensions?** The `all-MiniLM-L6-v2` sentence-transformer model we use produces vectors with exactly 384 numbers. The index dimension must match.

> **Why cosine?** Cosine similarity measures the angle between vectors, which works well for comparing meaning in text.

---

### Step 8 — Run Ingestion

This script loads the 12 products from `data/products.json`, converts them to vectors, and stores them in your Pinecone index.

```bash
python ingest.py
```

You should see output like:

```
==================================================
  RAG Workshop — Product Ingestion
==================================================

Loaded 12 products from data/products.json
Loading embedding model: all-MiniLM-L6-v2 ...
Embedding model loaded.

Converting products to text for embedding ...
Embedding product texts (this may take a moment on first run) ...
Built 12 vectors.

Connecting to Pinecone index: 'rag-workshop' ...
Upserting vectors into Pinecone ...
Upsert complete!

==================================================
  Ingestion finished successfully!
  You can now run the chatbot with:  python app.py
==================================================
```

> Run `ingest.py` only once (or again if you change the product data). The data stays in Pinecone until you delete it.

---

### Step 9 — Run the Chatbot

You have two options — both use the same RAG pipeline under the hood.

**Option A: Terminal chatbot**

```bash
python app.py
```

You should see:

```
============================================================
   AI Customer Support Chatbot  (powered by RAG + Gemini)
============================================================
 Ask me anything about our product catalog!
 Examples:
   - What is the cheapest product?
   - Tell me about the smart lamp
   - Do you have anything under $50?
   - What's best for a student desk setup?

 Type 'exit' or 'quit' to stop.
============================================================
```

Then type your question and press **Enter**.

---

**Option B: Streamlit web UI (visual)**

```bash
streamlit run streamlit_app.py
```

Streamlit will open a browser window automatically at `http://localhost:8501`.

Features of the Streamlit app:
- Chat bubble UI with full conversation history
- Collapsible **Sources retrieved from Pinecone** panel after each answer showing which products were retrieved and their similarity scores
- Sidebar with sample prompts and a **Clear conversation** button
- Same RAG pipeline as the terminal app — just a nicer interface

---

### Step 10 — Experiment!

Try the prompts below, explore the code, and work through the [Extension Ideas](#extension-ideas).

---

## Sample Prompts to Try

Copy and paste these into the chatbot one at a time:

```
What is the cheapest product?
```
```
Tell me about the smart lamp
```
```
Do you have anything under $50?
```
```
What is best for a student desk setup?
```
```
Compare the SoundWave Pro Earbuds and the NoiseFree Headphones
```
```
I need better lighting for my desk. What do you recommend?
```
```
What wireless charging options do you have?
```
```
Do you sell any ergonomic accessories?
```
```
What products work well for working from home?
```

---

## Common Errors and Fixes

### `ModuleNotFoundError: No module named 'pinecone'`

You either forgot to activate the virtual environment or forgot to install dependencies.

```bash
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

---

### `PINECONE_API_KEY is not set`

Your `.env` file is missing or the variable is blank.

- Make sure you ran `cp .env.example .env`
- Open `.env` and confirm `PINECONE_API_KEY=your_actual_key`

---

### `NotFoundException: Index 'rag-workshop' not found`

The Pinecone index does not exist yet, or the name in `.env` does not match.

- Go to [app.pinecone.io](https://app.pinecone.io) and check your index name.
- Make sure `PINECONE_INDEX_NAME` in `.env` exactly matches (case-sensitive).

---

### `google.api_core.exceptions.InvalidArgument` or Gemini auth errors

Your `GOOGLE_API_KEY` is wrong or not set.

- Double-check the key in `.env`.
- Make sure there are no extra spaces around the `=` sign.

---

### The embedding model is slow on first run

`sentence-transformers` downloads the model (~90 MB) from HuggingFace on first use and caches it locally. Subsequent runs are fast. If you are offline, the download will fail — make sure you have internet access the first time.

---

### `venv\Scripts\Activate.ps1 cannot be loaded` (Windows PowerShell)

Run this command first to allow script execution:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then retry `venv\Scripts\Activate.ps1`.

---

## Extension Ideas

Finished early? Here are challenges to take the project further:

### 1. Add More Products
Open `data/products.json` and add new products. Re-run `python ingest.py` to update Pinecone. Try asking about your new products.

### 2. Tune the Number of Retrieved Products
In `query.py`, change `top_k=3` in `retrieve_products()` to `top_k=5`. Does the quality of answers improve? What are the trade-offs?

### 3. Compare RAG vs. No RAG
Modify `query.py` to skip the retrieval step and send the question directly to Gemini without any product context. Compare the answers. Does Gemini hallucinate or give generic answers?

### 4. Add Metadata Filtering
Pinecone supports filtering by metadata fields. Modify `query_index()` in `utils/pinecone_db.py` to add a filter — for example, only search products in the `"Audio"` category:

```python
results = index.query(
    vector=query_vector,
    top_k=top_k,
    include_metadata=True,
    filter={"category": {"$eq": "Audio"}}
)
```

### 5. Extend the Streamlit UI
The Streamlit app (`streamlit_app.py`) is already included. Try extending it: add a price range slider in the sidebar that filters products before retrieval, show a product image placeholder, or display a table of all retrieved products with their scores.

### 6. Add Conversation Memory
Right now every question is independent. Try storing the last 2–3 exchanges and including them in the Gemini prompt for multi-turn conversation context.

### 7. Add Similarity Score Threshold
In `query.py`, filter out retrieved products with a low similarity score before passing them to Gemini:

```python
matches = [m for m in matches if m["score"] > 0.4]
```

This prevents irrelevant context from confusing the model.

