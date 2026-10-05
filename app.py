import os
import json
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Page Setup
st.set_page_config(
    page_title="MarketMind | Trading Intelligence",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ MarketMind")
st.caption("AI-Powered Stock Trading Intelligence & Technical Analysis Assistant")

# Initialize Gemini API Client
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found in .env file. Please check your configuration.")
    st.stop()


@st.cache_resource
def get_gemini_client(key: str):
    return genai.Client(api_key=key)


client = get_gemini_client(api_key)


# Pydantic Schema for Strict Trading Intelligence Parsing
class StockOpinionReport(BaseModel):
    ticker: str = Field(description="Ticker symbol analyzed.")
    overall_sentiment: str = Field(
        description="Market stance: 'Strong Bullish', 'Bullish', 'Neutral', 'Bearish', or 'Strong Bearish'"
    )
    confidence_score: int = Field(description="Confidence percentage score (0 to 100).")
    technical_summary: str = Field(description="Executive summary of technical price action, momentum, and trends.")
    key_supports_and_resistances: list[str] = Field(
        description="Key support levels, resistance barriers, or moving average triggers."
    )
    bullish_catalysts: list[str] = Field(description="Factors driving potential upside.")
    bearish_risks: list[str] = Field(description="Factors driving potential downside or risk.")
    trading_insight: str = Field(description="Strategic takeaway for short-term traders and long-term investors.")


SYSTEM_INSTRUCTION = """
You are MarketMind, an elite quantitative stock market analyst and technical trading strategist.
Analyze the provided stock technical performance data (prices, 52-week highs/lows, moving averages, volume).
Provide an objective, data-driven trading opinion structured strictly according to the requested JSON schema.
Maintain an institutional, analytical, and professional tone. Always include standard risk disclosures.
"""

# FIX 1: Map clean UI labels to valid yfinance 'mo' month identifiers
TIMEFRAME_MAP = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y"
}


def fetch_stock_data(ticker_symbol: str, period: str = "6mo"):
    """Fetches real-time price history and key metrics using yfinance."""
    try:
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period=period)
        info = stock.info
        if df.empty:
            return None, None
        return df, info
    except Exception:
        return None, None


def plot_stock_chart(df: pd.DataFrame, symbol: str):
    """Renders an interactive Candlestick chart with Volume using Plotly."""
    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price"
        )
    )

    # 20-Day Moving Average
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['SMA20'],
            mode='lines',
            name='20 SMA',
            line=dict(color='orange', width=1.5)
        )
    )

    fig.update_layout(
        title=f"{symbol.upper()} Price Action & 20-Day Moving Average",
        yaxis_title="Stock Price ($)",
        xaxis_title="Date",
        template="plotly_dark",
        height=450,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def render_opinion_dashboard(report: dict):
    """Renders structured JSON opinion report into an interactive UI."""

    sentiment = report.get("overall_sentiment", "Neutral")
    confidence = report.get("confidence_score", 50)

    # 1. Header Metrics
    m1, m2 = st.columns(2)

    with m1:
        if "Bullish" in sentiment:
            st.success(f"🟢 **Overall Stance:** {sentiment}")
        elif "Bearish" in sentiment:
            st.error(f"🚨 **Overall Stance:** {sentiment}")
        else:
            st.warning(f"⚠️ **Overall Stance:** {sentiment}")

    with m2:
        st.metric(label="AI Model Confidence Score", value=f"{confidence}%")

    st.info(f"**Technical Summary:** {report.get('technical_summary', '')}")
    st.divider()

    # 2. Catalysts vs Risks
    col_bull, col_bear = st.columns(2)

    with col_bull:
        st.subheader("🚀 Bullish Catalysts")
        for catalyst in report.get("bullish_catalysts", []):
            st.markdown(f"- {catalyst}")

    with col_bear:
        st.subheader("⚠️ Bearish Risks")
        for risk in report.get("bearish_risks", []):
            st.markdown(f"- {risk}")

    st.divider()

    # 3. Key Levels & Final Insight
    st.subheader("🎯 Key Technical Levels (Support & Resistance)")
    for level in report.get("key_supports_and_resistances", []):
        st.markdown(f"- 📍 {level}")

    st.subheader("💡 Strategic Trading Insight")
    st.write(report.get("trading_insight", ""))


# Main Search Input
col_input, col_period = st.columns([3, 1])

with col_input:
    ticker_input = st.text_input("Enter Stock Ticker Symbol:", value="AAPL",
                                 placeholder="e.g. AAPL, NVDA, TSLA, MSFT, RELIANCE.NS")

with col_period:
    selected_timeframe_label = st.selectbox("Timeframe", options=list(TIMEFRAME_MAP.keys()), index=2)
    timeframe = TIMEFRAME_MAP[selected_timeframe_label]

if st.button("⚡ Get Market Intelligence", type="primary", use_container_width=True):
    if not ticker_input.strip():
        st.warning("Please enter a stock ticker symbol.")
    else:
        symbol = ticker_input.strip().upper()

        with st.spinner(f"Fetching live market data for {symbol}..."):
            df, info = fetch_stock_data(symbol, period=timeframe)

        if df is None or df.empty:
            st.error(f"Could not fetch data for ticker symbol '{symbol}'. Please verify the symbol.")
        else:
            # Display Stock Header & Interactive Graph
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
            change = current_price - prev_price
            pct_change = (change / prev_price) * 100

            s1, s2, s3 = st.columns(3)
            s1.metric(label=f"{symbol} Current Price", value=f"${current_price:,.2f}",
                      delta=f"{change:+.2f} ({pct_change:+.2f}%)")
            s2.metric(label="52-Week High", value=f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
            s3.metric(label="52-Week Low", value=f"${info.get('fiftyTwoWeekLow', 'N/A')}")

            # Plot Candlestick Graph
            fig = plot_stock_chart(df, symbol)
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # Generate AI Opinion
            with st.spinner("Generating AI technical breakdown and opinion"):
                try:
                    # Construct technical context string for Gemini
                    recent_summary = {
                        "ticker": symbol,
                        "current_price": round(current_price, 2),
                        "52_week_high": info.get("fiftyTwoWeekHigh"),
                        "52_week_low": info.get("fiftyTwoWeekLow"),
                        "market_cap": info.get("marketCap"),
                        "pe_ratio": info.get("trailingPE"),
                        "volume": int(df['Volume'].iloc[-1]),
                        "avg_volume": info.get("averageVolume"),
                        "50_day_ma": info.get("fiftyDayAverage"),
                        "200_day_ma": info.get("twoHundredDayAverage"),
                        "recent_close_prices": df['Close'].tail(10).round(2).tolist()
                    }

                    # FIX 2 & FIX 3: Use valid model ('gemini-2.5-flash') and pass Pydantic schema class directly
                    response = client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=[f"Analyze this live stock technical profile: {json.dumps(recent_summary)}"],
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_INSTRUCTION,
                            temperature=0.1,
                            response_mime_type="application/json",
                            response_schema=StockOpinionReport,
                        ),
                    )

                    report_data = json.loads(response.text)
                    render_opinion_dashboard(report_data)

                except Exception as e:
                    st.error(f"Error generating market intelligence: {e}")