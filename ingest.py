"""
ingest.py
---------
Run this script ONCE to load the product catalog into Pinecone.

What it does:
  1. Reads product data from data/products.json
  2. Converts each product into a descriptive text string
  3. Embeds each text string into a vector using sentence-transformers
  4. Upserts all vectors (with product metadata) into Pinecone

Usage:
    python ingest.py
"""

import json
import os
from dotenv import load_dotenv

from utils.embeddings import load_embedding_model, embed_batch, product_to_text
from utils.pinecone_db import init_pinecone, get_index, upsert_products

# ── Load environment variables from .env ──────────────────────────────────────
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


def load_products(filepath="data/products.json"):
    """
    Load product records from the JSON file.

    Args:
        filepath (str): path to the products JSON file

    Returns:
        list[dict]: list of product dictionaries
    """
    with open(filepath, "r") as f:
        products = json.load(f)
    print(f"Loaded {len(products)} products from {filepath}")
    return products


def build_vectors(products, embed_model):
    """
    Convert products into Pinecone-ready vector dicts.

    Each vector dict contains:
      - id       : unique product ID string
      - values   : the embedding vector
      - metadata : all product fields stored for retrieval

    Args:
        products (list[dict]): raw product data
        embed_model: the loaded sentence-transformer model

    Returns:
        list[dict]: list of vector dicts ready for Pinecone upsert
    """
    print("Converting products to text for embedding ...")

    # Step 1: Convert each product dict → descriptive text string
    texts = [product_to_text(p) for p in products]

    # Step 2: Embed all texts in one batch call (more efficient than one-by-one)
    print("Embedding product texts (this may take a moment on first run) ...")
    vectors_data = embed_batch(embed_model, texts)

    # Step 3: Pack each vector with its ID and metadata
    vectors = []
    for product, vector in zip(products, vectors_data):
        vectors.append({
            "id": product["id"],
            "values": vector,
            "metadata": {
                "name": product["name"],
                "category": product["category"],
                "price": product["price"],
                "short_description": product["short_description"],
                "features": ", ".join(product["features"]),
                "use_case": product["use_case"],
            }
        })

    print(f"Built {len(vectors)} vectors.\n")
    return vectors


def main():
    # ── Validate environment variables ────────────────────────────────────────
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set. Check your .env file.")
    if not PINECONE_INDEX_NAME:
        raise ValueError("PINECONE_INDEX_NAME is not set. Check your .env file.")

    print("=" * 50)
    print("  RAG Workshop — Product Ingestion")
    print("=" * 50)
    print()

    # ── Step 1: Load products from JSON ───────────────────────────────────────
    products = load_products()

    # ── Step 2: Load the embedding model ──────────────────────────────────────
    embed_model = load_embedding_model()

    # ── Step 3: Build vector payloads ─────────────────────────────────────────
    vectors = build_vectors(products, embed_model)

    # ── Step 4: Connect to Pinecone and upsert ────────────────────────────────
    print(f"Connecting to Pinecone index: '{PINECONE_INDEX_NAME}' ...")
    pc = init_pinecone(PINECONE_API_KEY)
    index = get_index(pc, PINECONE_INDEX_NAME)

    print("Upserting vectors into Pinecone ...")
    response = upsert_products(index, vectors)
    print(f"Upsert complete! Response: {response}\n")

    print("=" * 50)
    print("  Ingestion finished successfully!")
    print("  You can now run the chatbot with:  python app.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
