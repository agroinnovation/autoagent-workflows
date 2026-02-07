#!/usr/bin/env python3
"""
Stock Data Importer
Reads stock symbols from CSV, fetches data from API endpoints, and inserts into database
"""

import pandas as pd
import requests
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockDataImporter:
    """Import stock data from API to database"""

    def __init__(self, base_url: str = "http://192.168.1.17:8006"):
        self.base_url = base_url.rstrip('/')
        self.stocks_endpoint = f"{self.base_url}/stocks"
        self.db_endpoint = f"{self.base_url}/database"
        self.success_count = 0
        self.error_count = 0
        self.errors = []

    def fetch_stock_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch detailed stock profile"""
        try:
            response = requests.get(
                f"{self.stocks_endpoint}/profile",
                params={"symbol": symbol},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching profile for {symbol}: {str(e)}")
            return None

    def fetch_stock_technicals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch technical indicators"""
        try:
            response = requests.get(
                f"{self.stocks_endpoint}/technicals",
                params={"symbol": symbol},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching technicals for {symbol}: {str(e)}")
            return None

    def execute_sql(self, sql: str) -> Optional[Dict[str, Any]]:
        """Execute SQL query via database API"""
        try:
            response = requests.post(
                f"{self.db_endpoint}/query",
                json={"sql": sql},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error executing SQL: {str(e)}")
            return None

    def insert_company(self, data: Dict[str, Any]) -> bool:
        """Insert company data"""
        try:
            # Escape single quotes in summary
            summary = data.get('summary', '').replace("'", "''")[:1000]  # Limit length

            sql = f"""
            INSERT INTO companies (ticker, company_name, sector, industry, summary)
            VALUES (
                '{data['ticker']}',
                '{data.get('company_name', '').replace("'", "''")}',
                '{data.get('sector', '')}',
                '{data.get('industry', '')}',
                '{summary}'
            )
            ON DUPLICATE KEY UPDATE
                company_name = VALUES(company_name),
                sector = VALUES(sector),
                industry = VALUES(industry),
                summary = VALUES(summary),
                updated_at = CURRENT_TIMESTAMP
            """

            result = self.execute_sql(sql)
            return result and result.get('success', False)
        except Exception as e:
            logger.error(f"Error inserting company data: {str(e)}")
            return False

    def insert_fundamentals(self, data: Dict[str, Any]) -> bool:
        """Insert stock fundamentals data"""
        try:
            # Helper to format value
            def fmt(val):
                if val is None or val == 'null':
                    return 'NULL'
                return str(val)

            sql = f"""
            INSERT INTO stock_fundamentals (
                ticker, market_cap, cap_category, pe_ratio, forward_pe, peg_ratio,
                eps_ttm, dividend_yield, payout_ratio, beta, price_to_book,
                return_on_equity, debt_to_equity, current_ratio, free_cashflow,
                earnings_growth_1y_pct, earnings_growth_2y_pct, earnings_growth_3y_pct,
                net_income_last_year, revenue_growth_1y_pct, revenue_growth_2y_pct,
                revenue_growth_3y_pct
            )
            VALUES (
                '{data['ticker']}',
                {fmt(data.get('market_cap'))},
                '{data.get('cap_category', '')}',
                {fmt(data.get('pe_ratio'))},
                {fmt(data.get('forward_pe'))},
                {fmt(data.get('peg_ratio'))},
                {fmt(data.get('eps_ttm'))},
                {fmt(data.get('dividend_yield'))},
                {fmt(data.get('payout_ratio'))},
                {fmt(data.get('beta'))},
                {fmt(data.get('price_to_book'))},
                {fmt(data.get('return_on_equity'))},
                {fmt(data.get('debt_to_equity'))},
                {fmt(data.get('current_ratio'))},
                {fmt(data.get('free_cashflow'))},
                {fmt(data.get('earnings_growth_1y_pct_cagr'))},
                {fmt(data.get('earnings_growth_2y_pct_cagr'))},
                {fmt(data.get('earnings_growth_3y_pct_cagr'))},
                {fmt(data.get('net_income_last_year'))},
                {fmt(data.get('revenue_growth_1y_pct_cagr'))},
                {fmt(data.get('revenue_growth_2y_pct_cagr'))},
                {fmt(data.get('revenue_growth_3y_pct_cagr'))}
            )
            """

            result = self.execute_sql(sql)
            return result and result.get('success', False)
        except Exception as e:
            logger.error(f"Error inserting fundamentals data: {str(e)}")
            return False

    def insert_prices(self, data: Dict[str, Any]) -> bool:
        """Insert stock price data"""
        try:
            def fmt(val):
                if val is None or val == 'null':
                    return 'NULL'
                return str(val)

            sql = f"""
            INSERT INTO stock_prices (
                ticker, price, price_change_1y_pct, fifty_two_week_high, fifty_two_week_low
            )
            VALUES (
                '{data['ticker']}',
                {fmt(data.get('price'))},
                {fmt(data.get('1y_price_change_pct'))},
                {fmt(data.get('fifty_two_week_high'))},
                {fmt(data.get('fifty_two_week_low'))}
            )
            """

            result = self.execute_sql(sql)
            return result and result.get('success', False)
        except Exception as e:
            logger.error(f"Error inserting price data: {str(e)}")
            return False

    def insert_technicals(self, symbol: str, data: Dict[str, Any]) -> bool:
        """Insert stock technical indicators"""
        try:
            def fmt(val):
                if val is None or val == 'null':
                    return 'NULL'
                return str(val)

            sql = f"""
            INSERT INTO stock_technicals (
                ticker, price, vwap_approx, price_vs_vwap_pct, rsi_14,
                sma_50, sma_200, sma_trend, macd, macd_signal, macd_hist,
                volume_today, volume_avg_10d, volume_spike_pct
            )
            VALUES (
                '{symbol}',
                {fmt(data.get('price'))},
                {fmt(data.get('vwap_approx'))},
                {fmt(data.get('price_vs_vwap_pct'))},
                {fmt(data.get('rsi_14'))},
                {fmt(data.get('sma_50'))},
                {fmt(data.get('sma_200'))},
                {f"'{data.get('sma_trend')}'" if data.get('sma_trend') else 'NULL'},
                {fmt(data.get('macd'))},
                {fmt(data.get('macd_signal'))},
                {fmt(data.get('macd_hist'))},
                {fmt(data.get('volume_today'))},
                {fmt(data.get('volume_avg_10d'))},
                {fmt(data.get('volume_spike_pct'))}
            )
            """

            result = self.execute_sql(sql)
            return result and result.get('success', False)
        except Exception as e:
            logger.error(f"Error inserting technicals data: {str(e)}")
            return False

    def process_stock(self, symbol: str) -> bool:
        """Process a single stock symbol"""
        logger.info(f"Processing {symbol}...")

        # Fetch profile data
        profile = self.fetch_stock_profile(symbol)
        if not profile:
            self.errors.append(f"{symbol}: Failed to fetch profile")
            return False

        # Fetch technicals
        technicals = self.fetch_stock_technicals(symbol)
        if not technicals:
            logger.warning(f"{symbol}: No technicals available")
            technicals = {}

        # Insert data into tables
        success = True

        # Insert company
        if not self.insert_company(profile):
            logger.error(f"{symbol}: Failed to insert company data")
            success = False

        # Insert fundamentals
        if not self.insert_fundamentals(profile):
            logger.error(f"{symbol}: Failed to insert fundamentals data")
            success = False

        # Insert prices
        if not self.insert_prices(profile):
            logger.error(f"{symbol}: Failed to insert price data")
            success = False

        # Insert technicals
        if technicals and not self.insert_technicals(symbol, technicals):
            logger.error(f"{symbol}: Failed to insert technicals data")
            success = False

        if success:
            logger.info(f"{symbol}: ✓ Successfully imported")
            self.success_count += 1
        else:
            self.errors.append(f"{symbol}: Partial import failure")
            self.error_count += 1

        return success

    def import_from_csv(self, csv_path: str, delay: float = 1.0):
        """Import stocks from CSV file"""
        try:
            # Read CSV
            df = pd.read_csv(csv_path)

            # Validate CSV has symbol column
            if 'symbol' not in df.columns and 'ticker' not in df.columns:
                logger.error("CSV must have 'symbol' or 'ticker' column")
                return

            # Get symbol column name
            symbol_col = 'symbol' if 'symbol' in df.columns else 'ticker'
            symbols = df[symbol_col].dropna().unique()

            logger.info(f"Found {len(symbols)} unique symbols to process")

            # Process each symbol
            for i, symbol in enumerate(symbols, 1):
                logger.info(f"[{i}/{len(symbols)}] Processing {symbol}")
                self.process_stock(symbol.strip().upper())

                # Rate limiting
                if i < len(symbols):
                    time.sleep(delay)

            # Print summary
            logger.info("=" * 60)
            logger.info("IMPORT SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total stocks processed: {len(symbols)}")
            logger.info(f"Successful: {self.success_count}")
            logger.info(f"Failed: {self.error_count}")

            if self.errors:
                logger.info("\nErrors:")
                for error in self.errors:
                    logger.info(f"  - {error}")

        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            sys.exit(1)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python stock_data_importer.py <csv_file> [delay_seconds]")
        print("\nExample: python stock_data_importer.py stocks.csv 1.5")
        print("\nCSV file should have a 'symbol' or 'ticker' column")
        sys.exit(1)

    csv_file = sys.argv[1]
    delay = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    if not Path(csv_file).exists():
        logger.error(f"CSV file not found: {csv_file}")
        sys.exit(1)

    importer = StockDataImporter()
    importer.import_from_csv(csv_file, delay)


if __name__ == "__main__":
    main()
