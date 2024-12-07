# Requirements: dlt[duckdb], streamlit, yfinance, pandas, plotly, ta
# pip install dlt[duckdb] streamlit yfinance pandas plotly ta

import dlt
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Iterator, Dict, List
import duckdb
import ta
import numpy as np

class TechnicalAnalysis:
    """Technical analysis calculations"""
    
    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe"""
        if df.empty:
            return df
            
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        # Moving Averages
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Mid'] = bollinger.bollinger_mavg()
        df['BB_Low'] = bollinger.bollinger_lband()
        
        return df

    @staticmethod
    def calculate_returns(df: pd.DataFrame) -> dict:
        """Calculate various return metrics"""
        if df.empty:
            return {}
            
        daily_returns = df['close'].pct_change()
        
        return {
            'total_return': ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100,
            'annualized_return': ((1 + (df['close'].iloc[-1] / df['close'].iloc[0] - 1)) ** (365 / len(df)) - 1) * 100,
            'volatility': daily_returns.std() * np.sqrt(252) * 100,
            'sharpe_ratio': (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0,
            'max_drawdown': ((df['close'] / df['close'].expanding(min_periods=1).max()) - 1).min() * 100
        }

class StockDataPipeline:
    """Data pipeline for stock market data"""
    
    def __init__(self):
        self.pipeline = dlt.pipeline(
            pipeline_name='stock_data',
            destination='duckdb',
            dataset_name='stock_analytics'
        )
        self.conn = duckdb.connect('stock_data.duckdb')
        self.setup_database()

    def setup_database(self):
        """Setup database tables"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                symbol VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                extracted_at TIMESTAMP
            )
        """)

    def get_stock_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Extract stock data for multiple symbols"""
        all_data = []
        
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                data = stock.history(start=start_date, end=end_date)
                
                if data.empty:
                    st.warning(f"No data available for {symbol}")
                    continue
                
                data = data.reset_index()
                df = pd.DataFrame({
                    'symbol': symbol,
                    'date': data['Date'].dt.strftime('%Y-%m-%d'),
                    'open': data['Open'].astype(float),
                    'high': data['High'].astype(float),
                    'low': data['Low'].astype(float),
                    'close': data['Close'].astype(float),
                    'volume': data['Volume'].astype(int),
                    'extracted_at': datetime.now().isoformat()
                })
                all_data.append(df)
                
            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {str(e)}")
        
        return pd.concat(all_data) if all_data else pd.DataFrame()

    def load_stock_data(self, symbols: List[str], start_date: str, end_date: str):
        """Load stock data for multiple symbols"""
        try:
            df = self.get_stock_data(symbols, start_date, end_date)
            if df.empty:
                return None

            # Clear existing data for these symbols
            symbols_str = "', '".join(symbols)
            self.conn.execute(f"DELETE FROM stock_prices WHERE symbol IN ('{symbols_str}')")

            # Insert new data
            self.conn.execute(
                """
                INSERT INTO stock_prices 
                SELECT 
                    symbol,
                    CAST(date AS DATE) as date,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    CAST(extracted_at AS TIMESTAMP) as extracted_at
                FROM df
                """
            )
            return True

        except Exception as e:
            st.error(f"Error in load_stock_data: {str(e)}")
            return None

    def query_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Query data for multiple symbols"""
        try:
            symbols_str = "', '".join(symbols)
            query = f"""
                SELECT symbol, date, open, high, low, close, volume, extracted_at
                FROM stock_prices
                WHERE symbol IN ('{symbols_str}')
                AND date BETWEEN CAST('{start_date}' AS DATE) AND CAST('{end_date}' AS DATE)
                ORDER BY date, symbol
            """
            return self.conn.execute(query).df()

        except Exception as e:
            st.error(f"Error querying data: {str(e)}")
            return pd.DataFrame()

def create_technical_chart(df: pd.DataFrame, symbol: str):
    """Create a technical analysis chart"""
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                       vertical_spacing=0.05,
                       subplot_titles=(f'{symbol} Price and Indicators',
                                     'Volume',
                                     'RSI',
                                     'MACD'),
                       row_heights=[0.4, 0.2, 0.2, 0.2])

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'],
                                high=df['high'], low=df['low'],
                                close=df['close'], name='OHLC'),
                  row=1, col=1)

    # Add Moving Averages
    fig.add_trace(go.Scatter(x=df['date'], y=df['SMA_20'],
                            name='SMA 20', line=dict(color='orange')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['SMA_50'],
                            name='SMA 50', line=dict(color='blue')),
                  row=1, col=1)

    # Add Bollinger Bands
    fig.add_trace(go.Scatter(x=df['date'], y=df['BB_High'],
                            name='BB High', line=dict(color='gray', dash='dash')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['BB_Low'],
                            name='BB Low', line=dict(color='gray', dash='dash')),
                  row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name='Volume'),
                  row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'],
                            name='RSI', line=dict(color='purple')),
                  row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=df['date'], y=df['MACD'],
                            name='MACD', line=dict(color='blue')),
                  row=4, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['MACD_Signal'],
                            name='Signal', line=dict(color='orange')),
                  row=4, col=1)

    fig.update_layout(
        height=1000,
        xaxis_rangeslider_visible=False,
        showlegend=True
    )

    return fig

def create_comparison_chart(df: pd.DataFrame):
    """Create a comparison chart for multiple stocks"""
    # Normalize prices to 100 for comparison
    pivot_df = df.pivot(index='date', columns='symbol', values='close')
    normalized_df = pivot_df / pivot_df.iloc[0] * 100
    
    fig = go.Figure()
    
    for symbol in normalized_df.columns:
        fig.add_trace(go.Scatter(
            x=normalized_df.index,
            y=normalized_df[symbol],
            name=symbol,
            mode='lines'
        ))
    
    fig.update_layout(
        title='Price Performance Comparison (Normalized to 100)',
        xaxis_title='Date',
        yaxis_title='Normalized Price',
        height=500
    )
    
    return fig

def main():
    st.set_page_config(page_title="Stock Market Analysis", layout="wide")
    st.title("Advanced Stock Market Analysis")

    try:
        pipeline = StockDataPipeline()

        # Sidebar inputs
        st.sidebar.header("Settings")
        
        # Multiple stock selection
        default_symbols = ['AAPL', 'MSFT', 'GOOGL']
        symbols_input = st.sidebar.text_input(
            "Enter Stock Symbols (comma-separated)",
            value=", ".join(default_symbols)
        )
        symbols = [s.strip().upper() for s in symbols_input.split(",")]
        
        # Date range selection
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(start_date, end_date),
            max_value=end_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
        
        # Analysis options
        show_technical = st.sidebar.checkbox("Show Technical Indicators", value=True)
        show_comparison = st.sidebar.checkbox("Show Stock Comparison", value=True)
        show_stats = st.sidebar.checkbox("Show Advanced Statistics", value=True)
        
        if st.sidebar.button("Load Data"):
            with st.spinner(f"Loading data for {', '.join(symbols)}..."):
                success = pipeline.load_stock_data(
                    symbols,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                if success:
                    st.sidebar.success("Data loaded successfully!")
                    df = pipeline.query_data(
                        symbols,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d')
                    )
                    
                    if not df.empty:
                        # Individual stock analysis
                        for symbol in symbols:
                            stock_df = df[df['symbol'] == symbol].copy()
                            if not stock_df.empty:
                                st.header(f"{symbol} Analysis")
                                
                                # Technical Analysis
                                if show_technical:
                                    stock_df = TechnicalAnalysis.add_indicators(stock_df)
                                    fig = create_technical_chart(stock_df, symbol)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Advanced Statistics
                                if show_stats:
                                    metrics = TechnicalAnalysis.calculate_returns(stock_df)
                                    st.subheader("Advanced Statistics")
                                    cols = st.columns(5)
                                    
                                    cols[0].metric("Total Return", f"{metrics['total_return']:.2f}%")
                                    cols[1].metric("Annual Return", f"{metrics['annualized_return']:.2f}%")
                                    cols[2].metric("Volatility", f"{metrics['volatility']:.2f}%")
                                    cols[3].metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                                    cols[4].metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
                        
                        # Stock Comparison
                        if show_comparison and len(symbols) > 1:
                            st.header("Stock Comparison")
                            comparison_fig = create_comparison_chart(df)
                            st.plotly_chart(comparison_fig, use_container_width=True)
                        
                        # Raw Data Display
                        if st.checkbox("Show Raw Data"):
                            st.dataframe(df)
                    
                    else:
                        st.warning("No data available for the selected symbols and date range.")
                else:
                    st.error("Failed to load data. Please check the symbols and try again.")
    
    except Exception as e:
        st.error(f"Error in application: {str(e)}")
        st.info("Please make sure all required packages are installed and try again.")

if __name__ == "__main__":
    main()