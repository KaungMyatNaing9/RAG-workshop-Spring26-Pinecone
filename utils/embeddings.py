"""
utils/embeddings.py
-------------------
Handles loading the sentence-transformer model and converting
text into vector embeddings.

We use the 'all-MiniLM-L6-v2' model because it is:
  - Small and fast (good for a workshop / local machine)
  - Produces 384-dimensional vectors
  - High quality for semantic similarity tasks
"""

from sentence_transformers import SentenceTransformer

# The model name we use throughout the project
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    """
    Load the sentence-transformer model from HuggingFace.
    The first time this runs it will download the model (~90 MB).
    Subsequent runs load it from the local cache.

    Returns:
        SentenceTransformer: the loaded model
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded.\n")
    return model


def embed_text(model, text):
    """
    Embed a single string of text into a vector.

    Args:
        model (SentenceTransformer): the loaded embedding model
        text (str): the text to embed

    Returns:
        list[float]: the embedding vector
    """
    # encode() returns a numpy array; .tolist() converts it to a plain Python list
    # which is required by Pinecone's upsert format
    return model.encode(text).tolist()


def embed_batch(model, texts):
    """
    Embed a list of strings in one efficient batch call.

    Args:
        model (SentenceTransformer): the loaded embedding model
        texts (list[str]): list of strings to embed

    Returns:
        list[list[float]]: list of embedding vectors
    """
    return [vec.tolist() for vec in model.encode(texts)]


def product_to_text(product):
    """
    Convert a product dictionary into a single descriptive string
    that captures all the important information for embedding.

    The richer the text, the better the semantic search results.

    Args:
        product (dict): a product entry from products.json

    Returns:
        str: a plain-text representation of the product
    """
    features_text = ", ".join(product.get("features", []))

    text = (
        f"Product: {product['name']}. "
        f"Category: {product['category']}. "
        f"Price: ${product['price']}. "
        f"Description: {product['short_description']} "
        f"Features: {features_text}. "
        f"Use case: {product['use_case']}"
    )
    return text
