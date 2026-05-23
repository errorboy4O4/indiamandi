"""
IndiaMandi — Data Cleaning & Feature Engineering Pipeline
Reads raw mandi CSV, cleans it, engineers features, saves as Parquet.
Run: python src/ingest.py
"""

import pandas as pd
import numpy as np
import os
import re

# ── Paths ──────────────────────────────────────────────────────────
RAW_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'mandi_prices_90days.csv')
CLEAN_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean')
CLEAN_PATH = os.path.join(CLEAN_DIR, 'mandi_prices.parquet')


def load_raw_data():
    """Load the raw CSV file."""
    print("=" * 60)
    print("STEP 1: Loading raw data")
    print("=" * 60)

    df = pd.read_csv(RAW_PATH)
    print(f"  Loaded {len(df):,} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Sample:\n{df.head(2).to_string()}\n")
    return df


def clean_columns(df):
    """Rename ugly columns and standardize text fields."""
    print("=" * 60)
    print("STEP 2: Cleaning column names & text fields")
    print("=" * 60)

    # Rename columns if they have the weird x0020 encoding
    rename_map = {
        'Min_x0020_Price': 'Min_Price',
        'Max_x0020_Price': 'Max_Price',
        'Modal_x0020_Price': 'Modal_Price',
    }
    df = df.rename(columns=rename_map)
    print(f"  Columns after rename: {list(df.columns)}")

    # Standardize State names: "MAHARASHTRA" -> "Maharashtra"
    df['State'] = df['State'].str.strip().str.title()

    # Standardize District names
    df['District'] = df['District'].str.strip().str.title()

    # Standardize Market names
    df['Market'] = df['Market'].str.strip().str.title()

    # Standardize Commodity names: lowercase and strip
    df['Commodity'] = df['Commodity'].str.strip()

    print(f"  Unique States: {df['State'].nunique()}")
    print(f"  Unique Commodities: {df['Commodity'].nunique()}")
    print(f"  Unique Markets: {df['Market'].nunique()}\n")
    return df


def parse_dates(df):
    """Parse Arrival_Date into proper datetime."""
    print("=" * 60)
    print("STEP 3: Parsing dates")
    print("=" * 60)

    df['Date'] = pd.to_datetime(df['Arrival_Date'], dayfirst=True, errors='coerce')

    # Drop rows where date couldn't be parsed
    bad_dates = df['Date'].isna().sum()
    print(f"  Bad dates dropped: {bad_dates}")
    df = df.dropna(subset=['Date'])

    # Drop the old text date column
    df = df.drop(columns=['Arrival_Date'])

    print(f"  Date range: {df['Date'].min().strftime('%d %b %Y')} to {df['Date'].max().strftime('%d %b %Y')}")
    print(f"  Unique dates: {df['Date'].nunique()}\n")
    return df


def clean_prices(df):
    """Remove price outliers (data entry errors)."""
    print("=" * 60)
    print("STEP 4: Cleaning prices (removing outliers)")
    print("=" * 60)

    before = len(df)

    # Convert prices to numeric (in case of any text)
    for col in ['Min_Price', 'Max_Price', 'Modal_Price']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing prices
    df = df.dropna(subset=['Modal_Price'])

    # Remove extreme outliers
    # Below ₹100/quintal is likely a data error
    # Above ₹50,000/quintal is likely a data error (except spices like saffron)
    df = df[(df['Modal_Price'] >= 100) & (df['Modal_Price'] <= 50000)]

    after = len(df)
    print(f"  Rows before: {before:,}")
    print(f"  Rows after:  {after:,}")
    print(f"  Removed:     {before - after:,} outlier rows")
    print(f"  Price range:  ₹{df['Modal_Price'].min():.0f} - ₹{df['Modal_Price'].max():.0f}/quintal\n")
    return df


def create_commodity_slug(df):
    """Create a clean slug for each commodity (for joining/filtering)."""
    print("=" * 60)
    print("STEP 5: Creating commodity slugs")
    print("=" * 60)

    df['commodity_slug'] = (
        df['Commodity']
        .str.lower()
        .str.replace(r'[^a-z0-9]', '_', regex=True)
        .str.replace(r'_+', '_', regex=True)
        .str.strip('_')
    )

    print(f"  Examples:")
    examples = df[['Commodity', 'commodity_slug']].drop_duplicates().head(8)
    for _, row in examples.iterrows():
        print(f"    '{row['Commodity']}' -> '{row['commodity_slug']}'")
    print()
    return df


def engineer_features(df):
    """Add calculated columns for analysis."""
    print("=" * 60)
    print("STEP 6: Feature engineering")
    print("=" * 60)

    # Sort by group and date (required for rolling calculations)
    df = df.sort_values(['State', 'Market', 'commodity_slug', 'Date'])

    # ── 1. 7-day rolling average price ──
    print("  Computing 7-day rolling average...")
    df['price_7d_avg'] = (
        df.groupby(['State', 'Market', 'commodity_slug'])['Modal_Price']
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
        .round(2)
    )

    # ── 2. Price vs State average (%) ──
    print("  Computing price vs state average...")
    state_avg = df.groupby(['State', 'commodity_slug', 'Date'])['Modal_Price'].transform('mean')
    df['price_vs_state_avg'] = ((df['Modal_Price'] - state_avg) / state_avg * 100).round(2)

    # ── 3. Week of year (for seasonal analysis) ──
    print("  Adding week of year...")
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)

    # ── 4. Month (for seasonal analysis) ──
    df['month'] = df['Date'].dt.month

    # ── 5. Day of week ──
    df['day_of_week'] = df['Date'].dt.day_name()

    print(f"\n  New columns added:")
    print(f"    price_7d_avg      - 7 day rolling average")
    print(f"    price_vs_state_avg - % deviation from state average")
    print(f"    week_of_year      - week number (1-52)")
    print(f"    month             - month number (1-12)")
    print(f"    day_of_week       - Monday/Tuesday/etc.")
    print()
    return df


def save_clean_data(df):
    """Save the cleaned dataframe as Parquet."""
    print("=" * 60)
    print("STEP 7: Saving clean data as Parquet")
    print("=" * 60)

    os.makedirs(CLEAN_DIR, exist_ok=True)
    df.to_parquet(CLEAN_PATH, index=False)

    file_size = os.path.getsize(CLEAN_PATH) / (1024 * 1024)
    print(f"  Saved to: {CLEAN_PATH}")
    print(f"  File size: {file_size:.1f} MB")
    print(f"  Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}\n")
    return df


def print_summary(df):
    """Print a summary of the final clean dataset."""
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Total records:  {len(df):,}")
    print(f"  Date range:     {df['Date'].min().strftime('%d %b %Y')} to {df['Date'].max().strftime('%d %b %Y')}")
    print(f"  States:         {df['State'].nunique()}")
    print(f"  Districts:      {df['District'].nunique()}")
    print(f"  Markets:        {df['Market'].nunique()}")
    print(f"  Commodities:    {df['Commodity'].nunique()}")
    print(f"  Avg Modal Price: ₹{df['Modal_Price'].mean():.0f}/quintal")
    print()
    print("  Top 5 commodities by records:")
    top5 = df['Commodity'].value_counts().head(5)
    for commodity, count in top5.items():
        avg_price = df[df['Commodity'] == commodity]['Modal_Price'].mean()
        print(f"    {commodity:30s} {count:>6,} records  avg ₹{avg_price:,.0f}/q")
    print()
    print("  Top 5 states by records:")
    for state, count in df['State'].value_counts().head(5).items():
        print(f"    {state:30s} {count:>6,} records")
    print()
    print("=" * 60)
    print("DATA PIPELINE COMPLETE!")
    print(f"Clean file ready at: data/clean/mandi_prices.parquet")
    print("=" * 60)


if __name__ == "__main__":
    df = load_raw_data()
    df = clean_columns(df)
    df = parse_dates(df)
    df = clean_prices(df)
    df = create_commodity_slug(df)
    df = engineer_features(df)
    df = save_clean_data(df)
    print_summary(df)