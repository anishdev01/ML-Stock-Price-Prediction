# data_ingestion.py
import yfinance as yf
from datetime import timedelta, date
import pandas as pd
import numpy as np
import streamlit as st
import time
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class DataIngestion:
    """
    Comprehensive data ingestion class for stock price prediction
    """
    
    def __init__(self):
        self.last_request_time = 0
        self.min_request_interval = 2  # seconds between requests
    
    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def fetch_stock_data(self, ticker: str, prediction_date: str, 
                        lookback_days: int = 365, max_retries: int = 5) -> pd.DataFrame:
        """
        Fetch historical stock data with robust error handling and rate limiting
        
        Parameters:
        - ticker: Stock symbol (e.g., "RELIANCE.NS")
        - prediction_date: String in 'YYYY-MM-DD' format
        - lookback_days: Number of past days to use for training
        - max_retries: Maximum number of retries on rate limit
        
        Returns:
        - DataFrame with historical stock data
        """
        end_date = pd.to_datetime(prediction_date) - timedelta(days=1)
        start_date = end_date - timedelta(days=lookback_days)
        
        progress_bar = None
        status_text = None

        for attempt in range(max_retries):
            try:
                # Rate limiting
                self._rate_limit()
                
                # Add progress indicator
                try:
                    if hasattr(st, '_main') and st._main is not None:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        status_text.text(f"Downloading data for {ticker}...")
                        progress_bar.progress(25)
                except (AttributeError, RuntimeError):
                    print(f"Downloading data for {ticker}... (Attempt {attempt+1}/{max_retries})")

                # Download data
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)

                if progress_bar:
                    progress_bar.progress(50)

                # Check if data is empty (common with rate limiting)
                if df.empty:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"Empty response - likely rate limited. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise ValueError(f"No data found for {ticker}. Please check the symbol or try a different date range.")

                print(f"Available columns for {ticker}: {df.columns.tolist()}")

                # Handle MultiIndex columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)

                # Process columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                available_columns = []
                for col in required_columns:
                    if col in df.columns:
                        available_columns.append(col)

                # Handle adjusted close
                if 'Adj Close' in df.columns:
                    available_columns.append('Adj Close')
                elif 'Close' in df.columns:
                    df['Adj Close'] = df['Close']
                    available_columns.append('Adj Close')
                else:
                    raise ValueError(f"No Close or Adj Close column found in data for {ticker}")

                if progress_bar:
                    progress_bar.progress(75)

                # Select and clean data
                df = df[available_columns].copy()
                df.reset_index(inplace=True)

                # Column mapping
                column_mapping = {
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high', 
                    'Low': 'low',
                    'Close': 'close',
                    'Adj Close': 'adj_close',
                    'Volume': 'volume'
                }
                existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
                df.rename(columns=existing_mapping, inplace=True)
                
                # Clean data
                df = self.clean_dataframe(df)

                # Validation checks
                if len(df) < 50:
                    raise ValueError(f"Insufficient data points ({len(df)}). Need at least 50 days of data.")

                if 'adj_close' not in df.columns and 'close' in df.columns:
                    df['adj_close'] = df['close']

                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date').reset_index(drop=True)

                # Final progress update
                if progress_bar:
                    progress_bar.progress(100)
                    status_text.text("✅ Data downloaded successfully!")
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                else:
                    print("✅ Data downloaded successfully!")

                print(f"Final dataframe shape: {df.shape}")
                print(f"Final columns: {df.columns.tolist()}")

                return df

            except Exception as e:
                # Clean up progress indicators
                if progress_bar:
                    progress_bar.empty()
                if status_text:
                    status_text.empty()
                
                # Handle specific errors
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['rate', 'limit', 'too many', 'yfratelimit']):
                    wait_time = 2 ** attempt
                    print(f"Rate limited. Retrying in {wait_time} seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    if attempt == max_retries - 1:
                        raise Exception(f"Error fetching data for {ticker}: {str(e)}")
                    else:
                        print(f"Error (will retry): {e}")
                        time.sleep(2 ** attempt)
                        continue
        
        raise Exception(f"Failed to fetch data for {ticker} after {max_retries} attempts")

    def prepare_train_test_split(self, df: pd.DataFrame, 
                               test_size: float = 0.2, 
                               time_based_split: bool = True,
                               target_column: str = 'adj_close') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets
        
        Parameters:
        - df: DataFrame with stock data
        - test_size: Proportion of data for testing (0.2 = 20%)
        - time_based_split: If True, use chronological split (recommended for time series)
        - target_column: Name of the target column
        
        Returns:
        - X_train, X_test, y_train, y_test
        """
        
        # Define feature columns (exclude date and target)
        exclude_cols = ['date', target_column]
        feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_columns]
        y = df[target_column]
        
        if time_based_split:
            # Time-based split (recommended for stock data)
            split_idx = int(len(df) * (1 - test_size))
            
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            print(f"Time-based split:")
            print(f"Training data: {len(X_train)} samples")
            print(f"Testing data: {len(X_test)} samples")
            
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=not time_based_split
            )
            print(f"Random split:")
            print(f"Training data: {len(X_train)} samples")
            print(f"Testing data: {len(X_test)} samples")
        
        print(f"Feature columns: {feature_columns}")
        print(f"Target column: {target_column}")
        
        return X_train, X_test, y_train, y_test

    def get_data_with_split(self, ticker: str, 
                           prediction_date: str, 
                           lookback_days: int = 365,
                           test_size: float = 0.2,
                           time_based_split: bool = True,
                           target_column: str = 'adj_close') -> Dict:
        """
        Complete pipeline: fetch data and split into train/test
        
        Returns:
        Dictionary containing all data splits and metadata
        """
        
        print(f"Starting complete data pipeline for {ticker}...") 
        
        # Step 1: Fetch raw data
        print("Step 1: Fetching stock data...")
        raw_data = self.fetch_stock_data(ticker, prediction_date, lookback_days)
        
        # Step 2: Train-test split
        print("Step 2: Splitting data...")
        X_train, X_test, y_train, y_test = self.prepare_train_test_split(
            raw_data, test_size, time_based_split, target_column
        )
        
        # Create result dictionary
        result = {
            'raw_data': raw_data,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': X_train.columns.tolist(),
            'target_column': target_column,
            'ticker': ticker,
            'prediction_date': prediction_date,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'split_type': 'time_based' if time_based_split else 'random'
        }
        
        print("✅ Data pipeline completed successfully!")
        print(f"Raw data shape: {raw_data.shape}")
        print(f"Training set: {len(X_train)} samples")
        print(f"Testing set: {len(X_test)} samples")
        
        return result

    def validate_stock_symbol(self, ticker: str) -> bool:
        """Validate if a stock symbol exists and has data"""
        try:
            self._rate_limit()
            test_data = yf.download(ticker, period="5d", progress=False)
            return not test_data.empty
        except Exception:
            return False

    def get_stock_info(self, ticker: str) -> Dict:
        """Get basic stock information"""
        try:
            self._rate_limit()
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'name': info.get('longName', info.get('shortName', ticker)),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'currency': info.get('currency', 'INR'),
                'current_price': info.get('regularMarketPrice', 'N/A')
            }
        except Exception as e:
            print(f"Error getting stock info: {e}")
            return {
                'name': ticker,
                'sector': 'N/A',
                'industry': 'N/A',
                'market_cap': 'N/A',
                'currency': 'INR',
                'current_price': 'N/A'
            }

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Additional data cleaning function to handle edge cases"""
        df_clean = df.copy()
        
        # Remove any completely empty rows
        df_clean = df_clean.dropna(how='all')
        
        # For numerical columns, replace inf/-inf with NaN then drop
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with NaN in critical columns
        critical_cols = ['close', 'adj_close'] if 'adj_close' in df_clean.columns else ['close']
        existing_critical_cols = [col for col in critical_cols if col in df_clean.columns]
        
        if existing_critical_cols:
            df_clean = df_clean.dropna(subset=existing_critical_cols)
        
        return df_clean

    def get_common_indian_stocks(self) -> list:
        """Returns a list of common Indian stock symbols"""
        return [
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
            "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS",
            "HINDUNILVR.NS", "LT.NS", "AXISBANK.NS", "KOTAKBANK.NS",
            "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS"
        ]

    def batch_validate_symbols(self, symbols: list) -> Dict:
        """Validate multiple stock symbols at once"""
        results = {}
        for symbol in symbols:
            print(f"Validating {symbol}...")
            results[symbol] = self.validate_stock_symbol(symbol)
            time.sleep(1)  # Additional delay between validations
        return results

# Utility functions for easy usage
def test_data_ingestion():
    """Test the complete data ingestion pipeline"""
    ingestion = DataIngestion()
    
    # Test with multiple tickers (fallback if one fails)
    test_tickers = ['TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'RELIANCE.NS']
    
    for ticker in test_tickers:
        try:
            print(f"\n{'='*60}")
            print(f"Testing pipeline for {ticker}")
            print(f"{'='*60}")
            
            result = ingestion.get_data_with_split(
                ticker=ticker,
                prediction_date=str(date.today()),
                lookback_days=100,  # Smaller for testing
                test_size=0.2,
                time_based_split=True
            )
            
            print(f"\n✅ Success with {ticker}!")
            print(f"Raw data shape: {result['raw_data'].shape}")
            print(f"Training samples: {result['train_size']}")
            print(f"Testing samples: {result['test_size']}")
            
            return result  # Return first successful result
            
        except Exception as e:
            print(f"❌ Failed with {ticker}: {e}")
            continue
    
    raise Exception("All test tickers failed")

def quick_fetch(ticker: str, days: int = 30) -> Optional[pd.DataFrame]:
    """Quick data fetch for testing"""
    try:
        ingestion = DataIngestion()
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        return df if not df.empty else None
    except:
        return None

if __name__ == "__main__":
    print("Testing Data Ingestion Module...")
    print("="*60)
    
    # Test the module
    try:
        result = test_data_ingestion()
        
        if result:
            print(f"\n{'='*60}")
            print("DATA INGESTION TEST SUCCESSFUL!")
            print(f"{'='*60}")
            print(f"Ticker: {result['ticker']}")
            print(f"Data shape: {result['raw_data'].shape}")
            print(f"Features: {result['feature_columns']}")
            print(f"Training samples: {result['train_size']}")
            print(f"Testing samples: {result['test_size']}")
            
        # Test validation
        print(f"\n{'='*60}")
        print("VALIDATION TEST")
        print(f"{'='*60}")
        ingestion = DataIngestion()
        common_stocks = ingestion.get_common_indian_stocks()[:3]
        validation_results = ingestion.batch_validate_symbols(common_stocks)
        print("Validation results:", validation_results)
        
    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")
        print("Testing quick fetch as fallback...")
        
        # Fallback test
        test_df = quick_fetch('TCS.NS', 10)
        if test_df is not None and not test_df.empty:
            print("✅ Quick fetch successful!")
            print(f"Data shape: {test_df.shape}")
        else:
            print("❌ All tests failed. Check internet connection and Yahoo Finance availability.")



# Initialize
ingestion = DataIngestion()

# Fetch data with train-test split
result = ingestion.get_data_with_split(
    ticker='TCS.NS',
    prediction_date='2024-01-20',
    lookback_days=365,
    test_size=0.2
)

# Access results
X_train = result['X_train']
y_train = result['y_train']
raw_data = result['raw_data']
df = pd.DataFrame(raw_data)
print(df.head())