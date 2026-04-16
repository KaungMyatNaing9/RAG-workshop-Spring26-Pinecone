"""
streamlit_app.py
----------------
Visual chatbot interface for the RAG workshop — runs in a web browser.

This is an alternative to app.py. It uses the exact same RAG pipeline
(query.py) but displays the conversation in a clean Streamlit chat UI
with retrieved sources shown in an expandable panel.

Usage:
    streamlit run streamlit_app.py

Requirements:
    - You must have run python ingest.py first.
    - Your .env file must have all three API keys set.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from utils.embeddings import load_embedding_model
from utils.pinecone_db import init_pinecone, get_index
from query import ask

# ── Load environment variables from .env ──────────────────────────────────────
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# ── Cache heavy resources so they load only once per session ──────────────────
# @st.cache_resource tells Streamlit to keep these objects alive across
# reruns (every time the user sends a message, Streamlit re-runs the script).
# Without caching, the model and Pinecone connection would reload on every message.

@st.cache_resource
def load_models():
    """Load the embedding model and connect to Pinecone once at startup."""
    embed_model = load_embedding_model()
    pc = init_pinecone(PINECONE_API_KEY)
    index = get_index(pc, PINECONE_INDEX_NAME)
    return embed_model, index


def check_env():
    """
    Validate that all required environment variables are present.
    Returns a list of missing variable names.
    """
    missing = []
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if not PINECONE_INDEX_NAME:
        missing.append("PINECONE_INDEX_NAME")
    if not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    return missing


def format_sources(matches):
    """
    Build a markdown string listing retrieved product sources.

    Args:
        matches (list[dict]): Pinecone query matches with metadata

    Returns:
        str: markdown-formatted source list
    """
    lines = []
    for i, match in enumerate(matches, start=1):
        name = match["metadata"].get("name", "Unknown")
        category = match["metadata"].get("category", "")
        price = match["metadata"].get("price", "")
        score = match.get("score", 0)
        lines.append(
            f"**{i}. {name}** — {category} — ${price}  "
            f"*(similarity: {score:.2f})*"
        )
    return "\n\n".join(lines)


# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Product Support Chatbot",
    page_icon="🛍️",
    layout="centered",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("About This App")
    st.markdown(
        """
        This chatbot uses **RAG** (Retrieval-Augmented Generation) to answer
        questions about a tech product catalog.

        **How it works:**
        1. Your question is converted to a vector
        2. Pinecone finds the most similar products
        3. Gemini answers using only those products as context

        **Model:** Gemini 2.5 Flash
        **Embeddings:** all-MiniLM-L6-v2 (384-dim)
        **Vector DB:** Pinecone (cosine similarity)
        """
    )
    st.divider()
    st.markdown("**Try asking:**")
    sample_prompts = [
        "What is the cheapest product?",
        "Tell me about the smart lamp",
        "Do you have anything under $50?",
        "What's best for a student desk?",
        "Compare the earbuds and headphones",
        "What wireless charging options do you have?",
    ]
    for prompt in sample_prompts:
        st.markdown(f"- *{prompt}*")

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ── Main page header ──────────────────────────────────────────────────────────
st.title("AI Customer Support Chatbot")
st.caption("Powered by RAG · Pinecone · Gemini 2.5 Flash")
st.divider()

# ── Environment check ─────────────────────────────────────────────────────────
missing_vars = check_env()
if missing_vars:
    st.error(
        f"Missing environment variables: **{', '.join(missing_vars)}**\n\n"
        "Please copy `.env.example` to `.env` and fill in your API keys, "
        "then restart the app."
    )
    st.stop()

# ── Load models (cached) ──────────────────────────────────────────────────────
with st.spinner("Loading embedding model and connecting to Pinecone..."):
    try:
        embed_model, index = load_models()
    except Exception as e:
        st.error(f"Failed to connect to Pinecone: {e}")
        st.stop()

# ── Session state: store chat history across reruns ───────────────────────────
# Each entry is a dict: {"role": "user" | "assistant", "content": str,
#                        "sources": list | None}
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render existing chat messages ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources under assistant messages if they exist
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources retrieved from Pinecone", expanded=False):
                st.markdown(format_sources(msg["sources"]))

# ── Chat input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask about our products..."):

    # Display and save the user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append(
        {"role": "user", "content": user_input, "sources": None}
    )

    # ── Run the RAG pipeline and stream the response ───────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Searching products and generating answer..."):
            try:
                answer, matches = ask(user_input, embed_model, index)
            except Exception as e:
                answer = f"Sorry, something went wrong: {e}"
                matches = []

        st.markdown(answer)

        # Show retrieved sources in a collapsible panel
        if matches:
            with st.expander("Sources retrieved from Pinecone", expanded=True):
                st.markdown(format_sources(matches))

    # Save the assistant message (including sources) to history
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": matches}
    )
