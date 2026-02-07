#!/usr/bin/env python3
"""
High-Quality Value Stock Screener

This script uses the API endpoint to filter stocks based on quantitative criteria:
- Profitability: ROE > 15%, positive free cash flow
- Solvency: Debt/Equity < 1.2, Current Ratio > 1.0
- Growth: Positive 1-year earnings growth
- Valuation: PEG < 1.5, Forward PE < 25
- Technical: Price above SMA(200)

Output: CSV file with ranked results
"""

import sys
import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from urllib.parse import urlencode
import json


# API Configuration
API_BASE_URL = "http://192.168.1.17:8006"
DATABASE_ENDPOINT = f"{API_BASE_URL}/database/query"


# The High-Quality Value SQL Query (Relaxed Criteria - Wide Net)
HIGH_QUALITY_VALUE_QUERY = """
SELECT
    c.ticker,
    c.company_name,
    c.sector,
    -- Valuation Metrics
    sf.forward_pe,
    sf.peg_ratio,
    sf.price_to_book,
    -- Quality Metrics
    sf.return_on_equity,
    sf.free_cashflow,
    sf.debt_to_equity,
    sf.current_ratio,
    -- Growth Metrics
    sf.earnings_growth_1y_pct,
    sf.earnings_growth_3y_pct,
    sf.revenue_growth_1y_pct,
    sf.net_income_last_year,
    -- Technical/Price Metrics (informational, not filtered)
    sp.price AS price_latest,
    st.sma_200,
    st.rsi_14,
    -- Market Data
    sf.market_cap AS fundamental_market_cap,
    sf.cap_category,
    -- Price vs Technical Indicator (for reference)
    CASE
        WHEN sp.price > st.sma_200 THEN 'Above 200MA'
        WHEN sp.price <= st.sma_200 THEN 'Below 200MA'
        ELSE 'No Data'
    END AS technical_position,
    -- Growth acceleration flag
    CASE
        WHEN sf.earnings_growth_1y_pct > sf.earnings_growth_3y_pct THEN 'Accelerating'
        WHEN sf.earnings_growth_1y_pct > 0 THEN 'Positive'
        ELSE 'Slowing'
    END AS growth_trend
FROM
    companies c
    INNER JOIN stock_fundamentals sf ON c.ticker = sf.ticker
    LEFT JOIN stock_prices sp ON c.ticker = sp.ticker
    LEFT JOIN stock_technicals st ON c.ticker = st.ticker
WHERE
    -- Relaxed Quality: ROE > 10% (was 15%)
    (sf.return_on_equity > 10.0 OR sf.return_on_equity IS NULL)

    -- Profitability: Positive net income (softer than free cash flow requirement)
    AND (sf.net_income_last_year > 0 OR sf.free_cashflow > 0)

    -- Relaxed Solvency: debt_to_equity < 2.5 (was 1.2)
    AND (sf.debt_to_equity < 2.5 OR sf.debt_to_equity IS NULL)

    -- Basic liquidity: current_ratio > 0.8 (was 1.0)
    AND (sf.current_ratio > 0.8 OR sf.current_ratio IS NULL)

    -- Growth: Positive recent earnings growth OR positive net income
    AND (sf.earnings_growth_1y_pct > 0 OR sf.net_income_last_year > 0)

    -- Relaxed Valuation: PEG < 2.0 (was 1.5), forward_pe < 40 (was 25)
    AND (sf.peg_ratio < 2.0 OR sf.peg_ratio IS NULL)
    AND (sf.forward_pe < 40 OR sf.forward_pe IS NULL)

    -- Must have fundamental data
    AND sf.ticker IS NOT NULL

ORDER BY
    -- Sort by best value scores (nulls last)
    CASE WHEN sf.peg_ratio IS NOT NULL THEN sf.peg_ratio ELSE 999 END ASC,
    CASE WHEN sf.return_on_equity IS NOT NULL THEN sf.return_on_equity ELSE 0 END DESC,
    sf.market_cap DESC
LIMIT 100
"""


def execute_api_query(sql_query: str) -> dict:
    """
    Execute SQL query via API endpoint

    Args:
        sql_query: The SQL query to execute

    Returns:
        dict: API response with results
    """
    try:
        print(f"\n{'='*80}")
        print("EXECUTING QUERY VIA API")
        print(f"{'='*80}")
        print(f"API Endpoint: {DATABASE_ENDPOINT}")

        # Make POST request with SQL in request body
        payload = {'sql': sql_query}
        headers = {'Content-Type': 'application/json'}

        response = requests.post(
            DATABASE_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=30
        )

        # Parse JSON response
        result = response.json()

        # Check for HTTP errors
        if response.status_code != 200:
            error_detail = result.get('detail', result.get('error', 'Unknown error'))
            print(f"✗ API request failed (HTTP {response.status_code}): {error_detail}")
            return None

        if result.get("success"):
            print(f"✓ Query executed successfully")
            print(f"  Rows returned: {result.get('rows', 0)}")
            print(f"  Columns: {len(result.get('columns', []))}")
            return result
        else:
            print(f"✗ Query failed: {result.get('error', 'Unknown error')}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"✗ API request failed: {str(e)}")
        # Try to get error details from response
        try:
            error_data = e.response.json()
            print(f"  Error details: {error_data.get('detail', error_data)}")
        except:
            pass
        return None
    except json.JSONDecodeError as e:
        print(f"✗ Failed to parse JSON response: {str(e)}")
        return None


def convert_to_dataframe(api_result: dict) -> pd.DataFrame:
    """
    Convert API result to pandas DataFrame

    Args:
        api_result: The API response dictionary

    Returns:
        pd.DataFrame: The results as a DataFrame
    """
    if not api_result or not api_result.get("success"):
        return pd.DataFrame()

    data = api_result.get("data", [])
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    print(f"\n{'='*80}")
    print("DATAFRAME CREATED")
    print(f"{'='*80}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add additional derived metrics for analysis

    Args:
        df: Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with additional metrics
    """
    if df.empty:
        return df

    print(f"\n{'='*80}")
    print("ADDING DERIVED METRICS")
    print(f"{'='*80}")

    # Value Score (lower is better for PEG, PE, P/B)
    if 'peg_ratio' in df.columns and 'forward_pe' in df.columns and 'price_to_book' in df.columns:
        df['value_score'] = (
            df['peg_ratio'].rank(ascending=True) +
            df['forward_pe'].rank(ascending=True) +
            df['price_to_book'].rank(ascending=True)
        )
        print("  ✓ Added value_score")

    # Quality Score (higher is better for ROE, lower for debt)
    if 'return_on_equity' in df.columns and 'debt_to_equity' in df.columns and 'current_ratio' in df.columns:
        df['quality_score'] = (
            df['return_on_equity'].rank(ascending=False) +
            df['debt_to_equity'].rank(ascending=True) +
            df['current_ratio'].rank(ascending=False)
        )
        print("  ✓ Added quality_score")

    # Growth Score (higher is better)
    if 'earnings_growth_1y_pct' in df.columns and 'revenue_growth_1y_pct' in df.columns:
        df['growth_score'] = (
            df['earnings_growth_1y_pct'].rank(ascending=False) +
            df['revenue_growth_1y_pct'].rank(ascending=False)
        )
        print("  ✓ Added growth_score")

    # Composite Score (lower is better - combines value, quality, growth)
    if 'value_score' in df.columns and 'quality_score' in df.columns and 'growth_score' in df.columns:
        df['composite_score'] = (
            df['value_score'] * 0.4 +  # 40% weight on value
            df['quality_score'] * 0.4 +  # 40% weight on quality
            df['growth_score'] * 0.2     # 20% weight on growth
        )
        print("  ✓ Added composite_score")

        # Sort by composite score
        df = df.sort_values('composite_score', ascending=True)
        print("  ✓ Sorted by composite_score")

    return df


def save_to_csv(df: pd.DataFrame, output_dir: str = None) -> Path:
    """
    Save DataFrame to CSV file with timestamp

    Args:
        df: The DataFrame to save
        output_dir: Output directory (defaults to script directory)

    Returns:
        Path: Path to the saved CSV file
    """
    if df.empty:
        print("\n✗ No data to save")
        return None

    print(f"\n{'='*80}")
    print("SAVING TO CSV")
    print(f"{'='*80}")

    # Determine output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "output"
    else:
        output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"high_quality_value_stocks_{timestamp}.csv"
    output_path = output_dir / filename

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"✓ Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size:,} bytes")
    print(f"  Rows: {len(df)}")

    return output_path


def print_summary(df: pd.DataFrame):
    """
    Print summary statistics of the screened stocks

    Args:
        df: The DataFrame with stock data
    """
    if df.empty:
        print("\n✗ No stocks found matching criteria")
        return

    print(f"\n{'='*80}")
    print("SCREENING SUMMARY")
    print(f"{'='*80}")

    print(f"\nTotal Qualified Stocks: {len(df)}")

    # Sector breakdown
    if 'sector' in df.columns:
        print("\nSector Distribution:")
        sector_counts = df['sector'].value_counts()
        for sector, count in sector_counts.head(10).items():
            print(f"  {sector}: {count}")

    # Market cap breakdown
    if 'cap_category' in df.columns:
        print("\nMarket Cap Distribution:")
        cap_counts = df['cap_category'].value_counts()
        for cap, count in cap_counts.items():
            print(f"  {cap}: {count}")

    # Top 10 stocks by composite score
    if 'composite_score' in df.columns:
        print("\nTop 10 High-Quality Value Stocks:")
        print("-" * 80)
        top_10 = df.head(10)[['ticker', 'company_name', 'peg_ratio', 'return_on_equity', 'composite_score']]
        print(top_10.to_string(index=False))

    # Key statistics
    print("\nKey Metrics Summary:")
    metrics = ['peg_ratio', 'forward_pe', 'return_on_equity', 'earnings_growth_1y_pct']
    for metric in metrics:
        if metric in df.columns:
            print(f"\n{metric}:")
            print(f"  Mean: {df[metric].mean():.2f}")
            print(f"  Median: {df[metric].median():.2f}")
            print(f"  Min: {df[metric].min():.2f}")
            print(f"  Max: {df[metric].max():.2f}")


def main():
    """Main execution"""
    print("\n{'='*80}")
    print("HIGH-QUALITY VALUE STOCK SCREENER")
    print(f"{'='*80}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Execute query via API
    result = execute_api_query(HIGH_QUALITY_VALUE_QUERY)

    if not result:
        print("\n✗ Failed to execute query")
        return 1

    # Convert to DataFrame
    df = convert_to_dataframe(result)

    if df.empty:
        print("\n✗ No data returned from query")
        return 1

    # Add derived metrics
    df = add_derived_metrics(df)

    # Save to CSV
    output_path = save_to_csv(df)

    if output_path:
        # Print summary
        print_summary(df)

        print(f"\n{'='*80}")
        print("SCREENING COMPLETE")
        print(f"{'='*80}")
        print(f"\n✓ Results saved to: {output_path}")
        print(f"\nNext steps:")
        print(f"  1. Review the CSV file for detailed analysis")
        print(f"  2. Use composite_score to rank stocks")
        print(f"  3. Perform further due diligence on top candidates")
        print(f"  4. Consider diversification across sectors")

        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
