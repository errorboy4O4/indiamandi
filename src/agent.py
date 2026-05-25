"""
IndiaMandi — Claude AI Agent with RAG
Takes a user question, retrieves relevant mandi data from ChromaDB,
sends it to Claude for a precise answer.
Run standalone test: python src/agent.py
"""

import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import json

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


# ── Load models once (reused across queries) ──────────────────────
_model = None
_collection = None
_df = None


def _get_embedding_model():
    """Load SentenceTransformer model (cached after first call)."""
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def _get_collection():
    """Get ChromaDB collection (cached after first call)."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _get_dataframe():
    """Load clean dataframe (cached after first call)."""
    global _df
    if _df is None:
        _df = pd.read_parquet(CLEAN_PATH)
    return _df


def retrieve_context(question, n_results=15):
    """
    Search ChromaDB for the most relevant price records.
    Returns the text chunks and metadata.
    """
    model = _get_embedding_model()
    collection = _get_collection()

    # Embed the question
    query_embedding = model.encode([question]).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )

    return results


def get_price_stats(question, results):
    """
    Calculate additional stats from the dataframe based on
    the commodity/state found in search results.
    Gives Claude extra context beyond just the retrieved chunks.
    """
    df = _get_dataframe()

    if not results['metadatas'] or not results['metadatas'][0]:
        return ""

    # Find the most common commodity and state in results
    commodities = [m['commodity'] for m in results['metadatas'][0]]
    states = [m['state'] for m in results['metadatas'][0]]

    top_commodity = max(set(commodities), key=commodities.count)
    top_state = max(set(states), key=states.count)

    # Filter dataframe for this commodity
    commodity_df = df[df['Commodity'] == top_commodity]

    if commodity_df.empty:
        return ""

    stats = []
    stats.append(f"\n--- Additional Stats for {top_commodity} ---")

    # Overall price range
    stats.append(f"Overall price range: ₹{commodity_df['Modal_Price'].min():,.0f} - ₹{commodity_df['Modal_Price'].max():,.0f}/quintal")
    stats.append(f"National average: ₹{commodity_df['Modal_Price'].mean():,.0f}/quintal")

    # Top 5 cheapest mandis (average price)
    cheapest = (
        commodity_df.groupby(['Market', 'State'])['Modal_Price']
        .mean()
        .sort_values()
        .head(5)
    )
    stats.append(f"\nTop 5 cheapest mandis (avg price):")
    for (market, state), price in cheapest.items():
        stats.append(f"  {market}, {state}: ₹{price:,.0f}/quintal")

    # Top 5 most expensive mandis
    expensive = (
        commodity_df.groupby(['Market', 'State'])['Modal_Price']
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )
    stats.append(f"\nTop 5 most expensive mandis (avg price):")
    for (market, state), price in expensive.items():
        stats.append(f"  {market}, {state}: ₹{price:,.0f}/quintal")

    # State-wise averages
    state_avg = (
        commodity_df.groupby('State')['Modal_Price']
        .mean()
        .sort_values()
    )
    stats.append(f"\nState-wise average prices:")
    for state, price in state_avg.items():
        stats.append(f"  {state}: ₹{price:,.0f}/quintal")

    # Latest date data
    latest = commodity_df['Date'].max().strftime('%d %B %Y')
    earliest = commodity_df['Date'].min().strftime('%d %B %Y')
    stats.append(f"\nData period: {earliest} to {latest}")

    return "\n".join(stats)


def ask(question, api_key):
    """
    Main function: takes a question, retrieves context, asks Claude.
    Returns: (answer_text, retrieved_results, top_commodity)
    """
    # Step 1: Retrieve relevant records from ChromaDB
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

    # Find top commodity for chart purposes
    top_commodity = None
    top_state = None
    if results['metadatas'] and results['metadatas'][0]:
        commodities = [m['commodity'] for m in results['metadatas'][0]]
        states = [m['state'] for m in results['metadatas'][0]]
        top_commodity = max(set(commodities), key=commodities.count)
        top_state = max(set(states), key=states.count)

    return answer, results, top_commodity, top_state


def get_chart_data(commodity, state=None):
    """
    Get price trend data for Plotly chart.
    Returns a dataframe with Date and average Modal_Price.
    """
    df = _get_dataframe()

    filtered = df[df['Commodity'] == commodity]
    if state and state != "All States":
        filtered = filtered[filtered['State'] == state]

    if filtered.empty:
        return None

    chart_df = (
        filtered.groupby('Date')['Modal_Price']
        .mean()
        .reset_index()
        .sort_values('Date')
    )
    chart_df.columns = ['Date', 'Avg_Price']
    return chart_df


# ── Standalone test ───────────────────────────────────────────────
if __name__ == "__main__":
    import toml

    # Load API key from secrets
    secrets_path = os.path.join(os.path.dirname(__file__), '..', '.streamlit', 'secrets.toml')

    if os.path.exists(secrets_path):
        secrets = toml.load(secrets_path)
        api_key = secrets.get('ANTHROPIC_API_KEY', '')
    else:
        api_key = os.environ.get('ANTHROPIC_API_KEY', '')

    if not api_key:
        print("ERROR: No API key found!")
        print("Add your key to .streamlit/secrets.toml")
        exit(1)

    # Test questions
    test_questions = [
        "Where is onion cheapest in India right now?",
        "What is the price of tomato in Kerala?",
        "Which mandi has the most expensive potato?",
    ]

    print("=" * 60)
    print("IndiaMandi Agent — Test Mode")
    print("=" * 60)

    for q in test_questions:
        print(f"\n{'─' * 60}")
        print(f"Q: {q}")
        print(f"{'─' * 60}")

        answer, results, commodity, state = ask(q, api_key)
        print(f"\nA: {answer}")
        print(f"\n  [Retrieved {len(results['documents'][0])} records, top commodity: {commodity}]")