"""
IndiaMandi — Vector Embedding Pipeline
Converts cleaned mandi price records into text chunks,
embeds them using SentenceTransformer, stores in ChromaDB.
Run: python src/embed.py
"""

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import os
import time

# ── Paths ──────────────────────────────────────────────────────────
CLEAN_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean', 'mandi_prices.parquet')
CHROMA_DIR = os.path.join(os.path.dirname(__file__), '..', 'chroma_db')
COLLECTION_NAME = "mandi_prices"


def load_clean_data():
    """Load the cleaned parquet file."""
    print("=" * 60)
    print("STEP 1: Loading clean data")
    print("=" * 60)

    df = pd.read_parquet(CLEAN_PATH)
    print(f"  Loaded {len(df):,} rows")
    print(f"  Date range: {df['Date'].min().strftime('%d %b %Y')} to {df['Date'].max().strftime('%d %b %Y')}")

    # Only embed the last 90 days (keeps ChromaDB fast and relevant)
    latest_date = df['Date'].max()
    cutoff = latest_date - pd.Timedelta(days=90)
    df = df[df['Date'] >= cutoff]
    print(f"  After 90-day filter: {len(df):,} rows\n")
    return df


def create_text_chunks(df):
    """
    Convert each price record into a human-readable text chunk.
    This is what gets embedded and what Claude will read.
    """
    print("=" * 60)
    print("STEP 2: Creating text chunks from records")
    print("=" * 60)

    chunks = []
    ids = []

    for idx, row in df.iterrows():
        # Format the date nicely
        date_str = row['Date'].strftime('%d %B %Y')

        # Build a rich text description of this price record
        chunk = (
            f"Commodity: {row['Commodity']} | Variety: {row['Variety']}\n"
            f"State: {row['State']} | District: {row['District']} | Mandi: {row['Market']}\n"
            f"Date: {date_str}\n"
            f"Modal Price: Rs {row['Modal_Price']:,.0f}/quintal\n"
            f"Min Price: Rs {row['Min_Price']:,.0f} | Max Price: Rs {row['Max_Price']:,.0f}\n"
            f"7-Day Avg Price: Rs {row['price_7d_avg']:,.0f}/quintal\n"
            f"Vs State Avg: {row['price_vs_state_avg']:+.1f}%"
        )

        # Create a unique ID for each record
        record_id = f"{row['commodity_slug']}_{row['State']}_{row['Market']}_{date_str}_{idx}"
        # ChromaDB IDs can't have spaces or special chars
        record_id = record_id.replace(" ", "_").replace(",", "")[:200]

        chunks.append(chunk)
        ids.append(record_id)

    print(f"  Created {len(chunks):,} text chunks")
    print(f"\n  Sample chunk:")
    print(f"  {'-' * 50}")
    print(f"  {chunks[0]}")
    print(f"  {'-' * 50}\n")
    return chunks, ids, df


def embed_and_store(chunks, ids, df):
    """
    Embed text chunks using SentenceTransformer and store in ChromaDB.
    Uses all-MiniLM-L6-v2: free, fast, 384-dimensional vectors.
    """
    print("=" * 60)
    print("STEP 3: Loading embedding model")
    print("=" * 60)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"  Model: all-MiniLM-L6-v2")
    print(f"  Embedding dimensions: 384")
    print(f"  This model runs locally — no API key needed!\n")

    # ── Set up ChromaDB ──
    print("=" * 60)
    print("STEP 4: Setting up ChromaDB")
    print("=" * 60)

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete old collection if it exists (fresh rebuild each time)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted old '{COLLECTION_NAME}' collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Indian mandi commodity price data"}
    )
    print(f"  Created collection: '{COLLECTION_NAME}'")
    print(f"  Storage: {CHROMA_DIR}\n")

    # ── Embed and store in batches ──
    print("=" * 60)
    print("STEP 5: Embedding and storing (this takes a few minutes)")
    print("=" * 60)

    BATCH_SIZE = 500
    total = len(chunks)
    start_time = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total)
        batch_chunks = chunks[i:batch_end]
        batch_ids = ids[i:batch_end]

        # Get the metadata for this batch
        batch_df = df.iloc[i:batch_end]
        batch_metadata = []
        for _, row in batch_df.iterrows():
            batch_metadata.append({
                "commodity": row['Commodity'],
                "state": row['State'],
                "district": row['District'],
                "market": row['Market'],
                "modal_price": float(row['Modal_Price']),
                "date": row['Date'].strftime('%Y-%m-%d'),
                "commodity_slug": row['commodity_slug'],
            })

        # Embed the batch
        embeddings = model.encode(batch_chunks).tolist()

        # Store in ChromaDB
        collection.add(
            ids=batch_ids,
            documents=batch_chunks,
            embeddings=embeddings,
            metadatas=batch_metadata,
        )

        elapsed = time.time() - start_time
        progress = batch_end / total * 100
        print(f"  Batch {i // BATCH_SIZE + 1}: {batch_end:,}/{total:,} records ({progress:.0f}%) — {elapsed:.0f}s elapsed")

    total_time = time.time() - start_time

    # ── Verify ──
    print(f"\n{'=' * 60}")
    print("STEP 6: Verification")
    print("=" * 60)

    count = collection.count()
    print(f"  Records in ChromaDB: {count:,}")
    print(f"  Total time: {total_time:.0f} seconds")
    print(f"  Speed: {total / total_time:.0f} records/second")

    # Test a sample query
    print(f"\n  Testing query: 'cheapest onion price'")
    results = collection.query(
        query_embeddings=model.encode(["cheapest onion price"]).tolist(),
        n_results=3,
    )

    print(f"  Top 3 results:")
    for i, doc in enumerate(results['documents'][0]):
        price = results['metadatas'][0][i]['modal_price']
        market = results['metadatas'][0][i]['market']
        state = results['metadatas'][0][i]['state']
        print(f"    {i + 1}. {market}, {state} — Rs {price:,.0f}/q")

    print(f"\n{'=' * 60}")
    print("EMBEDDING PIPELINE COMPLETE!")
    print(f"ChromaDB ready at: {CHROMA_DIR}")
    print(f"Total vectors: {count:,}")
    print("=" * 60)


if __name__ == "__main__":
    df = load_clean_data()
    chunks, ids, df = create_text_chunks(df)
    embed_and_store(chunks, ids, df)