"""
IndiaMandi — AI-Powered Agri Price Intelligence Tool
Streamlit web app with RAG-based Q&A, Price Explorer, and live stats.
Run: streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add src to path so imports work
sys.path.insert(0, os.path.dirname(__file__))
from agent import ask, get_chart_data, _get_dataframe

# ── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="IndiaMandi — Agri Price Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (works in both light & dark themes) ─────────────────
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #1a472a 0%, #2d5016 50%, #4a7c28 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        color: white !important;
        font-size: 2.2rem !important;
        margin-bottom: 0.3rem !important;
    }
    .main-header p {
        color: #c8e6c9 !important;
        font-size: 1rem !important;
        margin: 0 !important;
    }

    /* Stat cards — theme-aware */
    .stat-card {
        background: rgba(46, 125, 50, 0.15);
        border-left: 4px solid #4caf50;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }
    .stat-card h3 {
        color: #66bb6a !important;
        font-size: 1.5rem !important;
        margin: 0 !important;
    }
    .stat-card p {
        color: inherit !important;
        opacity: 0.8;
        font-size: 0.85rem !important;
        margin: 0 !important;
    }

    /* Answer box — theme-aware */
    .answer-box {
        background: rgba(46, 125, 50, 0.1);
        border: 1px solid rgba(76, 175, 80, 0.3);
        border-left: 5px solid #4caf50;
        padding: 1.2rem;
        border-radius: 8px;
        font-size: 1rem;
        line-height: 1.7;
        margin: 1rem 0;
        color: inherit;
    }

    /* Price table headers — theme-aware */
    .price-header-cheap {
        background: rgba(46, 125, 50, 0.2);
        padding: 0.6rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-weight: 600;
        color: #66bb6a;
        font-size: 1.1rem;
    }
    .price-header-expensive {
        background: rgba(239, 83, 80, 0.2);
        padding: 0.6rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-weight: 600;
        color: #ef5350;
        font-size: 1.1rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Chart theme helper ─────────────────────────────────────────────
def get_chart_layout():
    """Returns Plotly layout settings that work with Streamlit's theme."""
    return dict(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Segoe UI, sans-serif", size=13, color="#ccc"),
        height=420,
        margin=dict(l=60, r=20, t=60, b=40),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickprefix='₹',
            tickfont=dict(size=12, color="#aaa"),
            title_font=dict(size=13, color="#aaa"),
        ),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(size=12, color="#aaa"),
            title_font=dict(size=13, color="#aaa"),
        ),
        title_font=dict(size=16, color="#e0e0e0"),
    )


# ── Load Data ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load clean dataset (cached for performance)."""
    return _get_dataframe()


df = load_data()
API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")


# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 IndiaMandi")
    st.markdown("*AI-Powered Agri Price Intelligence*")
    st.markdown("---")

    # Live stats
    latest_date = df['Date'].max().strftime('%d %b %Y')
    earliest_date = df['Date'].min().strftime('%d %b %Y')

    st.markdown(f"""
    <div class="stat-card">
        <h3>{df['Market'].nunique():,}</h3>
        <p>Mandis Tracked</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-card">
        <h3>{df['Commodity'].nunique()}</h3>
        <p>Commodities</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-card">
        <h3>{df['State'].nunique()}</h3>
        <p>States Covered</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-card">
        <h3>{latest_date}</h3>
        <p>Latest Data</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Sample questions
    st.markdown("### 💡 Try these questions")

    sample_questions = [
        "Where is onion cheapest in India?",
        "What is the price of tomato in Kerala?",
        "Which state has the most expensive potato?",
        "Compare rice prices across states",
    ]

    for sq in sample_questions:
        if st.button(sq, key=f"sample_{sq}", use_container_width=True):
            st.session_state['user_question'] = sq
            st.session_state['run_question'] = True

    st.markdown("---")
    st.markdown(
        f"📊 **Data range:**  \n{earliest_date} → {latest_date}  \n"
        f"**Total records:** {len(df):,}"
    )
    st.caption("Data source: data.gov.in (Govt of India)")


# ── Main Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌾 IndiaMandi</h1>
    <p>Ask any question about crop prices across India's APMC mandis.
       Powered by RAG + Claude AI with real market data.</p>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🤖 AI Price Assistant", "📊 Price Explorer"])


# ══════════════════════════════════════════════════════════════════
# TAB 1: AI CHAT
# ══════════════════════════════════════════════════════════════════
with tab1:
    # Check for sample question from sidebar
    default_question = st.session_state.pop('user_question', '')

    user_question = st.text_input(
        "Ask anything about crop prices in India:",
        value=default_question,
        placeholder="e.g., Where is onion cheapest in Maharashtra?",
    )

    # Use sidebar question if just clicked, otherwise use text input
    active_question = default_question if default_question else user_question

    if active_question:
        if not API_KEY:
            st.error("⚠️ Anthropic API key not found! Add it to `.streamlit/secrets.toml`")
        else:
            with st.spinner("🔍 Searching mandi data & generating answer..."):
                try:
                    answer, results, top_commodity, top_state = ask(active_question, API_KEY)
                    # Display answer using st.markdown for proper rendering
                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                    # Show price trend chart if we found a commodity
                    if top_commodity:
                        st.markdown(f"### 📈 Price Trend: {top_commodity}")

                        chart_df = get_chart_data(top_commodity, top_state)

                        if chart_df is not None and len(chart_df) > 1:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=chart_df['Date'],
                                y=chart_df['Avg_Price'],
                                mode='lines',
                                name='Avg Price',
                                line=dict(color='#4caf50', width=2.5),
                                fill='tozeroy',
                                fillcolor='rgba(76, 175, 80, 0.15)',
                                hovertemplate='₹%{y:,.0f}/q<br>%{x|%d %b %Y}<extra></extra>',
                            ))
                            layout = get_chart_layout()
                            layout['title'] = (
                                f"{top_commodity} — Average Price Trend"
                                + (f" ({top_state})" if top_state else " (All India)")
                            )
                            fig.update_layout(**layout)
                            st.plotly_chart(fig, use_container_width=True)

                    # Show retrieved records in expander
                    with st.expander("📋 View retrieved data records"):
                        for i, doc in enumerate(results['documents'][0]):
                            meta = results['metadatas'][0][i]
                            st.text(f"Record {i+1}: {meta['commodity']} | "
                                    f"{meta['market']}, {meta['state']} | "
                                    f"₹{meta['modal_price']:,.0f}/q | "
                                    f"{meta['date']}")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Make sure you've run `python src/embed.py` first to build the vector database.")


# ══════════════════════════════════════════════════════════════════
# TAB 2: PRICE EXPLORER
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Explore Commodity Prices Across Mandis")

    # ── Filters ──
    col1, col2, col3 = st.columns(3)

    # Top 20 commodities by record count
    top_commodities = df['Commodity'].value_counts().head(20).index.tolist()

    with col1:
        selected_commodity = st.selectbox("🌽 Commodity", top_commodities)

    with col2:
        states = ["All States"] + sorted(df['State'].unique().tolist())
        selected_state = st.selectbox("📍 State", states)

    with col3:
        period_options = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}
        selected_period_label = st.selectbox("📅 Period", list(period_options.keys()), index=1)
        selected_period = period_options[selected_period_label]

    # ── Filter data ──
    filtered = df[df['Commodity'] == selected_commodity].copy()

    if selected_state != "All States":
        filtered = filtered[filtered['State'] == selected_state]

    cutoff_date = df['Date'].max() - pd.Timedelta(days=selected_period)
    filtered = filtered[filtered['Date'] >= cutoff_date]

    if filtered.empty:
        st.warning(f"No data found for {selected_commodity} in {selected_state} for the last {selected_period} days.")
    else:
        # ── Price trend chart ──
        trend_df = (
            filtered.groupby('Date')['Modal_Price']
            .mean()
            .reset_index()
            .sort_values('Date')
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend_df['Date'],
            y=trend_df['Modal_Price'],
            mode='lines',
            name='Avg Price',
            line=dict(color='#42a5f5', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(66, 165, 245, 0.12)',
            hovertemplate='₹%{y:,.0f}/q<br>%{x|%d %b %Y}<extra></extra>',
        ))
        layout = get_chart_layout()
        layout['title'] = (
            f"{selected_commodity} — Average Modal Price ({selected_period} days)"
            + (f" in {selected_state}" if selected_state != "All States" else " (All India)")
        )
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

        # ── Summary stats ──
        avg_price = filtered['Modal_Price'].mean()
        min_price = filtered['Modal_Price'].min()
        max_price = filtered['Modal_Price'].max()
        num_mandis = filtered['Market'].nunique()

        stat_cols = st.columns(4)
        stat_cols[0].metric("Avg Price", f"₹{avg_price:,.0f}/q")
        stat_cols[1].metric("Min Price", f"₹{min_price:,.0f}/q")
        stat_cols[2].metric("Max Price", f"₹{max_price:,.0f}/q")
        stat_cols[3].metric("Mandis Reporting", f"{num_mandis}")

        # ── Cheapest vs Most Expensive Mandis ──
        st.markdown("---")
        left_col, right_col = st.columns(2)

        mandi_avg = (
            filtered.groupby(['Market', 'State'])['Modal_Price']
            .mean()
            .reset_index()
            .sort_values('Modal_Price')
        )

        with left_col:
            st.markdown('<div class="price-header-cheap">🟢 Top 5 Cheapest Mandis</div>', unsafe_allow_html=True)
            cheapest = mandi_avg.head(5).copy()
            cheapest['Modal_Price'] = cheapest['Modal_Price'].apply(lambda x: f"₹{x:,.0f}/q")
            cheapest.columns = ['Mandi', 'State', 'Avg Price']
            cheapest.index = range(1, len(cheapest) + 1)
            st.table(cheapest)

        with right_col:
            st.markdown('<div class="price-header-expensive">🔴 Top 5 Most Expensive Mandis</div>', unsafe_allow_html=True)
            expensive = mandi_avg.tail(5).sort_values('Modal_Price', ascending=False).copy()
            expensive['Modal_Price'] = expensive['Modal_Price'].apply(lambda x: f"₹{x:,.0f}/q")
            expensive.columns = ['Mandi', 'State', 'Avg Price']
            expensive.index = range(1, len(expensive) + 1)
            st.table(expensive)

        # ── State-wise comparison chart ──
        if selected_state == "All States" and filtered['State'].nunique() > 1:
            st.markdown("### 🗺️ State-wise Price Comparison")

            state_comparison = (
                filtered.groupby('State')['Modal_Price']
                .mean()
                .sort_values(ascending=True)
                .reset_index()
            )

            # Color bars: green for cheap, yellow mid, red expensive
            min_val = state_comparison['Modal_Price'].min()
            max_val = state_comparison['Modal_Price'].max()
            range_val = max_val - min_val if max_val != min_val else 1
            bar_colors = []
            for price in state_comparison['Modal_Price']:
                ratio = (price - min_val) / range_val
                if ratio < 0.33:
                    bar_colors.append('#4caf50')   # green = cheap
                elif ratio < 0.66:
                    bar_colors.append('#ffc107')   # amber = mid
                else:
                    bar_colors.append('#ef5350')   # red = expensive

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=state_comparison['Modal_Price'],
                y=state_comparison['State'],
                orientation='h',
                marker=dict(color=bar_colors, line=dict(width=0)),
                hovertemplate='%{y}: ₹%{x:,.0f}/q<extra></extra>',
                text=[f'₹{p:,.0f}' for p in state_comparison['Modal_Price']],
                textposition='outside',
                textfont=dict(color='#aaa', size=11),
            ))
            layout2 = get_chart_layout()
            layout2['title'] = f"{selected_commodity} — Average Price by State"
            layout2['height'] = max(350, state_comparison.shape[0] * 40)
            layout2['showlegend'] = False
            layout2['yaxis'] = dict(
                tickfont=dict(size=12, color="#ccc"),
                gridcolor='rgba(0,0,0,0)',
            )
            layout2['xaxis'] = dict(
                gridcolor='rgba(255,255,255,0.08)',
                tickprefix='₹',
                tickfont=dict(size=11, color="#aaa"),
            )
            fig2.update_layout(**layout2)
            st.plotly_chart(fig2, use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small style='color: #888;'>Built with Streamlit + ChromaDB + Claude AI | "
    "Data: data.gov.in (Govt of India) | "
    "IndiaMandi © 2025</small></center>",
    unsafe_allow_html=True,
)