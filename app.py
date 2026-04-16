"""
app.py
------
Terminal-based AI customer support chatbot entry point.

Run this after you have completed ingestion (python ingest.py).

Usage:
    python app.py

Type your question at the prompt and press Enter.
Type 'exit' or 'quit' to stop the chatbot.
"""

import os
from dotenv import load_dotenv

from utils.embeddings import load_embedding_model
from utils.pinecone_db import init_pinecone, get_index
from query import ask

# ── Load environment variables from .env ──────────────────────────────────────
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def validate_env():
    """Check that all required environment variables are set."""
    missing = []
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if not PINECONE_INDEX_NAME:
        missing.append("PINECONE_INDEX_NAME")
    if not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")

    if missing:
        print("\n[ERROR] The following environment variables are missing:")
        for var in missing:
            print(f"  - {var}")
        print("\nPlease copy .env.example to .env and fill in your API keys.")
        print("See the README for instructions.\n")
        raise SystemExit(1)


def print_welcome():
    """Print the welcome banner."""
    print()
    print("=" * 60)
    print("   AI Customer Support Chatbot  (powered by RAG + Gemini)")
    print("=" * 60)
    print(" Ask me anything about our product catalog!")
    print(" Examples:")
    print("   - What is the cheapest product?")
    print("   - Tell me about the smart lamp")
    print("   - Do you have anything under $50?")
    print("   - What's best for a student desk setup?")
    print()
    print(" Type 'exit' or 'quit' to stop.")
    print("=" * 60)
    print()


def print_sources(matches):
    """
    Print the product names that were retrieved from Pinecone.
    This helps students see which products the RAG pipeline pulled in
    before Gemini generated its answer.

    Args:
        matches (list[dict]): Pinecone query matches
    """
    print("\n  [Sources retrieved from Pinecone]")
    for i, match in enumerate(matches, start=1):
        name = match["metadata"].get("name", "Unknown")
        score = match.get("score", 0)
        print(f"    {i}. {name}  (similarity: {score:.2f})")
    print()


def main():
    # ── Validate environment ───────────────────────────────────────────────────
    validate_env()

    print_welcome()

    # ── Load models and connect to Pinecone (done once at startup) ────────────
    print("Setting up... (loading embedding model and connecting to Pinecone)")
    embed_model = load_embedding_model()

    pc = init_pinecone(PINECONE_API_KEY)
    index = get_index(pc, PINECONE_INDEX_NAME)
    print("Ready! Ask your first question below.\n")

    # ── Main chat loop ────────────────────────────────────────────────────────
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+C or piped input ending gracefully
            print("\n\nGoodbye! Thanks for using the chatbot.")
            break

        # Skip empty input
        if not user_input:
            continue

        # Exit condition
        if user_input.lower() in ("exit", "quit"):
            print("\nGoodbye! Thanks for using the chatbot.")
            break

        # ── RAG pipeline ──────────────────────────────────────────────────────
        try:
            answer, matches = ask(user_input, embed_model, index)

            # Show which products were retrieved (great for learning RAG)
            if matches:
                print_sources(matches)

            print(f"Bot: {answer}")
            print()

        except Exception as e:
            print(f"\n[ERROR] Something went wrong: {e}")
            print("Please check your API keys and internet connection.\n")


if __name__ == "__main__":
    main()
