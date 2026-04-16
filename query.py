"""
query.py
--------
The RAG (Retrieval-Augmented Generation) core logic.

Two main steps happen here:
  1. RETRIEVE  — embed the user's question and search Pinecone for the most
                 relevant product descriptions
  2. GENERATE  — pass those retrieved products as context to Gemini and ask
                 it to answer the question based only on what was retrieved

This separation keeps the logic clean and easy to understand.
"""

import os
import google.generativeai as genai

from utils.embeddings import embed_text
from utils.pinecone_db import query_index

# ── Configure Gemini ──────────────────────────────────────────────────────────
# We call configure() once here; the model is initialised in ask() so it picks
# up any key set after this module is imported.
GEMINI_MODEL_NAME = "gemini-2.5-flash"


def format_product_context(match):
    """
    Convert a single Pinecone match (with metadata) into a readable
    text block that Gemini can use as context.

    Args:
        match (dict): a Pinecone query match containing 'metadata'

    Returns:
        str: a formatted product description string
    """
    m = match["metadata"]
    return (
        f"Product: {m['name']}\n"
        f"Category: {m['category']}\n"
        f"Price: ${m['price']}\n"
        f"Description: {m['short_description']}\n"
        f"Features: {m['features']}\n"
        f"Use case: {m['use_case']}"
    )


def retrieve_products(question, embed_model, index, top_k=3):
    """
    Embed the user's question and retrieve the top-k most relevant
    products from Pinecone.

    Args:
        question (str): the user's raw question
        embed_model: the loaded sentence-transformer model
        index: the Pinecone index object
        top_k (int): number of products to retrieve (default: 3)

    Returns:
        list[dict]: list of Pinecone matches sorted by relevance
    """
    # Embed the question using the same model used during ingestion
    # (using the same model is critical — vectors must live in the same space)
    question_vector = embed_text(embed_model, question)

    # Search Pinecone for the closest product vectors
    matches = query_index(index, question_vector, top_k=top_k)
    return matches


def build_prompt(question, matches):
    """
    Build the full prompt we send to Gemini.

    We include the retrieved product context and strict instructions
    so Gemini only answers from what was retrieved — not from its
    general training data. This is the core idea behind RAG.

    Args:
        question (str): the user's question
        matches (list[dict]): retrieved Pinecone matches

    Returns:
        str: the complete prompt string
    """
    # Format each retrieved product into readable text
    context_blocks = [format_product_context(m) for m in matches]
    context_text = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are a helpful AI customer support assistant for a tech product store.

Use ONLY the product information provided below to answer the customer's question.
Do not use any outside knowledge. If the answer cannot be found in the provided products,
say: "I'm sorry, I don't have enough information about that in our current catalog."

--- RETRIEVED PRODUCTS ---

{context_text}

--- END OF PRODUCTS ---

Customer question: {question}

Answer:"""

    return prompt


def ask(question, embed_model, index):
    """
    Full RAG pipeline: retrieve relevant products, then generate an answer.

    Args:
        question (str): the user's question
        embed_model: the loaded sentence-transformer model
        index: the Pinecone index object

    Returns:
        tuple[str, list[dict]]:
            - the generated answer text
            - the list of retrieved product matches (for showing sources)
    """
    # ── Step 1: Retrieve relevant products from Pinecone ──────────────────────
    matches = retrieve_products(question, embed_model, index, top_k=3)

    if not matches:
        return "I couldn't find any relevant products for your question.", []

    # ── Step 2: Build the prompt with retrieved context ───────────────────────
    prompt = build_prompt(question, matches)

    # ── Step 3: Call Gemini to generate the answer ────────────────────────────
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    response = model.generate_content(prompt)

    return response.text.strip(), matches
