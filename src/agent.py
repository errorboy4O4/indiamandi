"""
IndiaMandi — Claude AI Agent with RAG
Takes a user question, retrieves relevant mandi data from ChromaDB,
sends it to Claude for a precise answer.

Supports two modes:
  - Full RAG mode (local): ChromaDB + SentenceTransformer embeddings
  - Fallback mode (cloud): Pandas keyword search (no ChromaDB needed)

Run standalone test: python src/agent.py
"""

import anthropic
import pandas as pd
import os
import re

# ── Paths ──────────────────────────────────────────────────────────
CHROMA_DIR = os.path.join(os.path.dirname(__file__), '..', 'chroma_db')
CLEAN_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean', 'mandi_prices.parquet')
COLLECTION_NAME = "mandi_prices"

# ── System Prompt ──────────────────────────────────────────────────
SYSTEM_PROMPT = """You are IndiaMandi, an expert agricultural market analyst for India.
You have access to daily price data from APMC mandis across Indian states.

Your job is to answer questions about commodity prices using ONLY the data provided below.

Rules:
- Always cite specific prices in ₹/quintal with mandi name and state
- Keep answers to 3-5 sentences, then a bullet list of key numbers
- If recommending where to buy/sell, give top 3 mandis with prices
- If data is insufficient, say so — NEVER guess or make up prices
- Mention the date range of the data you are referencing
- Use ₹ symbol for all prices
- Format prices with commas (e.g., ₹1,840/quintal)
- Be specific and data-driven — farmers depend on accurate information"""


# ── Cached globals ─────────────────────────────────────────────────
_model = None
_collection = None
_df = None
_use_chroma = None  # None = not yet checked


def _check_chroma_available():
    """Check if ChromaDB + sentence-transformers are available."""
    global _use_chroma
    if _use_chroma is not None:
        return _use_chroma
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        # Check if the chroma_db folder exists and has data
        if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            collection = client.get_collection(COLLECTION_NAME)
            if collection.count() > 0:
                _use_chroma = True
                return True
    except Exception:
        pass
    _use_chroma = False
    return False


def _get_embedding_model():
    """Load SentenceTransformer model (cached after first call)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def _get_collection():
    """Get ChromaDB collection (cached after first call)."""
    global _collection
    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _get_dataframe():
    """Load clean dataframe (cached after first call)."""
    global _df
    if _df is None:
        _df = pd.read_parquet(CLEAN_PATH)
    return _df


# ═══════════════════════════════════════════════════════════════════
# RETRIEVAL — ChromaDB (full RAG) or Pandas fallback
# ═══════════════════════════════════════════════════════════════════

def retrieve_context_chroma(question, n_results=15):
    """Search ChromaDB for the most relevant price records."""
    model = _get_embedding_model()
    collection = _get_collection()
    query_embedding = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )
    return results


def retrieve_context_pandas(question, n_results=15):
    """
    Fallback: keyword-based search using Pandas.
    Extracts commodity and state names from the question,
    filters the dataframe, and returns formatted results.
    """
    df = _get_dataframe()
    question_lower = question.lower()

    # Find matching commodity
    commodities = df['Commodity'].unique()
    matched_commodity = None
    for c in commodities:
        if c.lower() in question_lower:
            matched_commodity = c
            break
    # Try partial match
    if not matched_commodity:
        for c in commodities:
            first_word = c.lower().split('(')[0].split()[0]
            if first_word in question_lower and len(first_word) > 3:
                matched_commodity = c
                break

    # Find matching state
    states = df['State'].unique()
    matched_state = None
    for s in states:
        if s.lower() in question_lower:
            matched_state = s
            break

    # Filter dataframe
    filtered = df.copy()
    if matched_commodity:
        filtered = filtered[filtered['Commodity'] == matched_commodity]
    if matched_state:
        filtered = filtered[filtered['State'] == matched_state]

    # If no filters matched, try to find the most relevant records
    if matched_commodity is None and matched_state is None:
        # Default to top commodities
        top_commodities = ['Onion', 'Tomato', 'Potato']
        filtered = df[df['Commodity'].isin(top_commodities)]

    # Sample diverse records (different mandis, dates)
    if len(filtered) > n_results:
        filtered = filtered.sort_values('Date', ascending=False)
        # Get diverse mandis
        sampled = filtered.groupby('Market').head(2).head(n_results)
    else:
        sampled = filtered.head(n_results)

    # Format results to match ChromaDB output structure
    documents = []
    metadatas = []
    for _, row in sampled.iterrows():
        date_str = row['Date'].strftime('%d %B %Y')
        chunk = (
            f"Commodity: {row['Commodity']} | Variety: {row['Variety']}\n"
            f"State: {row['State']} | District: {row['District']} | Mandi: {row['Market']}\n"
            f"Date: {date_str}\n"
            f"Modal Price: Rs {row['Modal_Price']:,.0f}/quintal\n"
            f"Min Price: Rs {row['Min_Price']:,.0f} | Max Price: Rs {row['Max_Price']:,.0f}\n"
            f"7-Day Avg Price: Rs {row['price_7d_avg']:,.0f}/quintal\n"
            f"Vs State Avg: {row['price_vs_state_avg']:+.1f}%"
        )
        documents.append(chunk)
        metadatas.append({
            "commodity": row['Commodity'],
            "state": row['State'],
            "district": row['District'],
            "market": row['Market'],
            "modal_price": float(row['Modal_Price']),
            "date": row['Date'].strftime('%Y-%m-%d'),
            "commodity_slug": row['commodity_slug'],
        })

    # Return in same format as ChromaDB results
    return {
        'documents': [documents],
        'metadatas': [metadatas],
    }


def retrieve_context(question, n_results=15):
    """Auto-select retrieval method based on availability."""
    if _check_chroma_available():
        return retrieve_context_chroma(question, n_results)
    else:
        return retrieve_context_pandas(question, n_results)


# ═══════════════════════════════════════════════════════════════════
# STATS + GENERATION
# ═══════════════════════════════════════════════════════════════════

def get_price_stats(question, results):
    """Calculate additional stats from the dataframe."""
    df = _get_dataframe()

    if not results['metadatas'] or not results['metadatas'][0]:
        return ""

    commodities = [m['commodity'] for m in results['metadatas'][0]]
    states = [m['state'] for m in results['metadatas'][0]]

    top_commodity = max(set(commodities), key=commodities.count)
    top_state = max(set(states), key=states.count)

    commodity_df = df[df['Commodity'] == top_commodity]
    if commodity_df.empty:
        return ""

    stats = []
    stats.append(f"\n--- Additional Stats for {top_commodity} ---")
    stats.append(f"Overall price range: ₹{commodity_df['Modal_Price'].min():,.0f} - ₹{commodity_df['Modal_Price'].max():,.0f}/quintal")
    stats.append(f"National average: ₹{commodity_df['Modal_Price'].mean():,.0f}/quintal")

    cheapest = (
        commodity_df.groupby(['Market', 'State'])['Modal_Price']
        .mean().sort_values().head(5)
    )
    stats.append(f"\nTop 5 cheapest mandis (avg price):")
    for (market, state), price in cheapest.items():
        stats.append(f"  {market}, {state}: ₹{price:,.0f}/quintal")

    expensive = (
        commodity_df.groupby(['Market', 'State'])['Modal_Price']
        .mean().sort_values(ascending=False).head(5)
    )
    stats.append(f"\nTop 5 most expensive mandis (avg price):")
    for (market, state), price in expensive.items():
        stats.append(f"  {market}, {state}: ₹{price:,.0f}/quintal")

    state_avg = commodity_df.groupby('State')['Modal_Price'].mean().sort_values()
    stats.append(f"\nState-wise average prices:")
    for state, price in state_avg.items():
        stats.append(f"  {state}: ₹{price:,.0f}/quintal")

    latest = commodity_df['Date'].max().strftime('%d %B %Y')
    earliest = commodity_df['Date'].min().strftime('%d %B %Y')
    stats.append(f"\nData period: {earliest} to {latest}")

    return "\n".join(stats)


def ask(question, api_key):
    """
    Main function: takes a question, retrieves context, asks Claude.
    Returns: (answer_text, retrieved_results, top_commodity, top_state)
    """
    # Step 1: Retrieve relevant records
    results = retrieve_context(question, n_results=15)

    # Step 2: Format the retrieved chunks as context
    context_chunks = results['documents'][0]
    context_text = "\n\n".join([f"Record {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])

    # Step 3: Get additional stats
    extra_stats = get_price_stats(question, results)

    # Step 4: Build the prompt for Claude
    user_prompt = f"""Here is the relevant mandi price data retrieved from our database:

{context_text}

{extra_stats}

Based on this data, answer the following question:
{question}"""

    # Step 5: Call Claude API
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
    )

    answer = response.content[0].text

    # Find top commodity for chart
    top_commodity = None
    top_state = None
    if results['metadatas'] and results['metadatas'][0]:
        commodities = [m['commodity'] for m in results['metadatas'][0]]
        states = [m['state'] for m in results['metadatas'][0]]
        top_commodity = max(set(commodities), key=commodities.count)
        top_state = max(set(states), key=states.count)

    return answer, results, top_commodity, top_state


def get_chart_data(commodity, state=None):
    """Get price trend data for Plotly chart."""
    df = _get_dataframe()
    filtered = df[df['Commodity'] == commodity]
    if state and state != "All States":
        filtered = filtered[filtered['State'] == state]
    if filtered.empty:
        return None
    chart_df = (
        filtered.groupby('Date')['Modal_Price']
        .mean().reset_index().sort_values('Date')
    )
    chart_df.columns = ['Date', 'Avg_Price']
    return chart_df


# ── Standalone test ───────────────────────────────────────────────
if __name__ == "__main__":
    import toml

    secrets_path = os.path.join(os.path.dirname(__file__), '..', '.streamlit', 'secrets.toml')
    if os.path.exists(secrets_path):
        secrets = toml.load(secrets_path)
        api_key = secrets.get('ANTHROPIC_API_KEY', '')
    else:
        api_key = os.environ.get('ANTHROPIC_API_KEY', '')

    if not api_key:
        print("ERROR: No API key found!")
        exit(1)

    mode = "ChromaDB (Full RAG)" if _check_chroma_available() else "Pandas (Fallback)"
    print(f"{'='*60}")
    print(f"IndiaMandi Agent — Test Mode")
    print(f"Retrieval mode: {mode}")
    print(f"{'='*60}")

    test_questions = [
        "Where is onion cheapest in India right now?",
        "What is the price of tomato in Kerala?",
        "Which mandi has the most expensive potato?",
    ]

    for q in test_questions:
        print(f"\n{'─'*60}")
        print(f"Q: {q}")
        print(f"{'─'*60}")
        answer, results, commodity, state = ask(q, api_key)
        print(f"\nA: {answer}")
        print(f"\n  [Retrieved {len(results['documents'][0])} records, top commodity: {commodity}]")