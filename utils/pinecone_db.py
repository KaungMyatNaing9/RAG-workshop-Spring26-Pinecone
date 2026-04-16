"""
utils/pinecone_db.py
--------------------
Handles all interactions with Pinecone:
  - Initializing the Pinecone client
  - Getting a reference to the index
  - Upserting (inserting/updating) vectors
  - Querying for similar vectors

Pinecone stores vectors alongside metadata (like product name, price, etc.)
so we can return human-readable results after a search.
"""

from pinecone import Pinecone


def init_pinecone(api_key):
    """
    Initialize and return the Pinecone client.

    Args:
        api_key (str): your Pinecone API key from the .env file

    Returns:
        Pinecone: authenticated Pinecone client instance
    """
    pc = Pinecone(api_key=api_key)
    return pc


def get_index(pc, index_name):
    """
    Get a reference to an existing Pinecone index.

    NOTE: The index must already exist in your Pinecone dashboard.
    Run the setup step in the README before calling this.

    Args:
        pc (Pinecone): the Pinecone client
        index_name (str): the name of your index (from .env)

    Returns:
        pinecone.Index: the index object used for upsert and query
    """
    index = pc.Index(index_name)
    return index


def upsert_products(index, vectors):
    """
    Upsert (insert or update) a list of vectors into the Pinecone index.

    Each vector is a dict with:
      - 'id'       : unique string identifier for the product
      - 'values'   : the embedding vector (list of floats)
      - 'metadata' : a dict of key-value pairs to store alongside the vector
                     (we store the product fields here for retrieval)

    Args:
        index: the Pinecone index object
        vectors (list[dict]): list of vector dicts to upsert

    Returns:
        dict: Pinecone upsert response
    """
    response = index.upsert(vectors=vectors)
    return response


def query_index(index, query_vector, top_k=3):
    """
    Query the Pinecone index for the most similar vectors.

    Args:
        index: the Pinecone index object
        query_vector (list[float]): the embedded question vector
        top_k (int): how many results to return (default: 3)

    Returns:
        list[dict]: list of matches, each with 'id', 'score', and 'metadata'
    """
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True   # we need metadata to reconstruct product info
    )
    return results["matches"]
