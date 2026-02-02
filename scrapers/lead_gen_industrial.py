#!/usr/bin/env python3
"""
Lead Generation Industrial Scraper

Enriches Albuquerque industrial parcels with business intelligence
by searching addresses via SearxNG and storing results in CRM schema.

Usage:
    python lead_gen_industrial.py          # Full run (all parcels)
    python lead_gen_industrial.py --test   # Test mode (10 parcels)
    python lead_gen_industrial.py --resume # Resume from checkpoint
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from typing import Any, Optional
from urllib.parse import quote_plus

try:
    import requests
except ImportError:
    print("Error: requests library required. Install with: pip install requests")
    sys.exit(1)

# Configuration
API_BASE = "http://192.168.1.17:8006"
RATE_LIMIT = 1.0  # seconds between requests
CHECKPOINT_FILE = "/tmp/lead_gen_checkpoint.json"
MAX_RESULTS_PER_PARCEL = 5  # Limit results to store per parcel

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def fetch_parcels(limit: Optional[int] = None, offset_pin: Optional[str] = None) -> list[dict]:
    """Fetch parcels from business_parcels_target view."""
    sql = '''
        SELECT "PIN", "STREETNUMB", "STREETNAME", "STREETDESI", "STREETQUAD",
               "LandUseCat", "LandUseDes"
        FROM business_parcels_target
        WHERE "STREETNUMB" NOT IN (0, 99999)
          AND "STREETNAME" IS NOT NULL
          AND "STREETNAME" != ''
    '''

    if offset_pin:
        sql += f" AND \"PIN\" > '{offset_pin}'"

    sql += ' ORDER BY "PIN"'

    if limit:
        sql += f' LIMIT {limit}'

    try:
        response = requests.post(
            f"{API_BASE}/postgresql/query",
            json={"sql": sql},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        if data.get("success"):
            logger.info(f"Fetched {data.get('rows', 0)} parcels")
            return data.get("data", [])
        else:
            logger.error(f"Query failed: {data}")
            return []

    except requests.RequestException as e:
        logger.error(f"Failed to fetch parcels: {e}")
        return []


def format_address(parcel: dict) -> str:
    """Build searchable address string from parcel data."""
    parts = [
        str(parcel.get("STREETNUMB", "")),
        parcel.get("STREETNAME", ""),
        parcel.get("STREETDESI", ""),
        parcel.get("STREETQUAD", ""),
    ]
    street = " ".join(p for p in parts if p).strip()
    return f"{street}, Albuquerque, NM"


def search_address(address: str) -> list[dict]:
    """Search for business information via SearxNG."""
    query = f"{address} business"

    try:
        response = requests.get(
            f"{API_BASE}/searx/search",
            params={"q": query},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        # SearxNG returns nested results structure
        results_obj = data.get("results", {})
        if isinstance(results_obj, dict):
            results = results_obj.get("results", [])
        else:
            results = results_obj

        logger.debug(f"Search returned {len(results)} results for: {query}")
        return results[:MAX_RESULTS_PER_PARCEL]

    except requests.RequestException as e:
        logger.warning(f"Search failed for '{address}': {e}")
        return []


def enrich_results(results: list[dict], address: str, parcel: dict) -> tuple[list[dict], bool]:
    """Enrich search results using LLM filtering.

    Returns:
        Tuple of (enriched_results, llm_used)
    """
    if not results:
        return results, False

    try:
        response = requests.post(
            f"{API_BASE}/crm/enrich",
            json={
                "results": results,
                "address": address,
                "parcel_context": {
                    "pin": parcel.get("PIN"),
                    "land_use_category": parcel.get("LandUseCat"),
                    "land_use_description": parcel.get("LandUseDes")
                }
            },
            timeout=90  # Allow time for LLM processing
        )
        response.raise_for_status()
        data = response.json()

        if data.get("llm_used"):
            logger.info(f"  LLM enriched: {data.get('relevant_count')}/{data.get('total_count')} relevant")
            return data.get("enriched_results", results), True
        else:
            logger.debug(f"  LLM not used: {data.get('error', 'unknown')}")
            return results, False

    except requests.RequestException as e:
        logger.warning(f"Enrichment failed, using regex fallback: {e}")
        return results, False


def extract_phone(text: str) -> Optional[str]:
    """Extract phone number from text."""
    patterns = [
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'\d{3}[-.\s]\d{3}[-.\s]\d{4}',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group()
    return None


def extract_email(text: str) -> Optional[str]:
    """Extract email from text."""
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    match = re.search(pattern, text)
    return match.group() if match else None


def parse_results(results: list[dict], parcel: dict, address: str, query: str, enriched: bool = False) -> list[dict]:
    """Parse search results into business records.

    Args:
        results: Search results (optionally LLM-enriched)
        parcel: Parcel data
        address: Formatted address string
        query: Original search query
        enriched: If True, use LLM-extracted fields; otherwise use regex
    """
    businesses = []

    for result in results:
        title = result.get("title", "")
        content = result.get("content", "")
        url = result.get("url", "")

        # Skip empty results
        if not title and not content:
            continue

        # Use LLM-extracted data if available, otherwise fall back to regex
        if enriched:
            business_name = result.get("business_name") or title
            phone = result.get("phone")
            email = result.get("email")
            website = result.get("website") or url
            business_type = result.get("business_type")
            confidence = result.get("relevance_score", 0.0)
        else:
            # Fall back to regex extraction
            full_text = f"{title} {content}"
            business_name = title
            phone = extract_phone(full_text)
            email = extract_email(full_text)
            website = url
            business_type = None
            score = result.get("score", 0.0)
            confidence = min(score, 1.0) if score else 0.0

        business = {
            "pin": parcel.get("PIN"),
            "address": address,
            "business_name": business_name[:255] if business_name else None,
            "phone": phone,
            "email": email,
            "website": website[:500] if website else None,
            "business_type": business_type,
            "search_query": query[:255],
            "result_url": url[:500] if url else None,
            "result_title": title[:255] if title else None,
            "result_snippet": content[:1000] if content else None,
            "confidence_score": confidence,
            "land_use_category": parcel.get("LandUseCat"),
            "land_use_description": parcel.get("LandUseDes"),
            "enrichment_method": "llm" if enriched else "regex",
        }
        businesses.append(business)

    return businesses


def save_business(business: dict) -> bool:
    """Save a business record to the CRM database."""
    # Build INSERT with ON CONFLICT
    columns = list(business.keys())
    placeholders = []
    values = []

    for col in columns:
        val = business[col]
        if val is None:
            placeholders.append("NULL")
        elif isinstance(val, (int, float)):
            placeholders.append(str(val))
        else:
            # Escape single quotes
            escaped = str(val).replace("'", "''")
            placeholders.append(f"'{escaped}'")

    sql = f'''
        INSERT INTO crm.businesses ({", ".join(columns)})
        VALUES ({", ".join(placeholders)})
        ON CONFLICT (pin, result_url) DO UPDATE SET
            business_name = EXCLUDED.business_name,
            phone = EXCLUDED.phone,
            email = EXCLUDED.email,
            confidence_score = EXCLUDED.confidence_score,
            updated_at = NOW()
    '''

    try:
        response = requests.post(
            f"{API_BASE}/postgresql/query",
            json={"sql": sql},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data.get("success", False)

    except requests.RequestException as e:
        logger.warning(f"Failed to save business: {e}")
        return False


def load_checkpoint() -> Optional[str]:
    """Load the last processed PIN from checkpoint file."""
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
            return data.get("last_pin")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_checkpoint(pin: str) -> None:
    """Save checkpoint with last processed PIN."""
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump({
                "last_pin": pin,
                "timestamp": datetime.now().isoformat()
            }, f)
    except IOError as e:
        logger.warning(f"Failed to save checkpoint: {e}")


def main():
    parser = argparse.ArgumentParser(description="Lead Generation Industrial Scraper")
    parser.add_argument("--test", action="store_true", help="Test mode: process only 10 parcels")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--limit", type=int, help="Limit number of parcels to process")
    args = parser.parse_args()

    # Determine limit
    limit = 10 if args.test else args.limit

    # Check for resume
    offset_pin = None
    if args.resume:
        offset_pin = load_checkpoint()
        if offset_pin:
            logger.info(f"Resuming from PIN: {offset_pin}")
        else:
            logger.info("No checkpoint found, starting from beginning")

    # Fetch parcels
    logger.info("Fetching parcels from business_parcels_target...")
    parcels = fetch_parcels(limit=limit, offset_pin=offset_pin)

    if not parcels:
        logger.error("No parcels to process")
        return 1

    logger.info(f"Processing {len(parcels)} parcels...")

    # Stats
    total_processed = 0
    total_businesses = 0
    total_errors = 0

    for i, parcel in enumerate(parcels):
        pin = parcel.get("PIN")
        address = format_address(parcel)
        query = f"{address} business"

        logger.info(f"[{i+1}/{len(parcels)}] Processing PIN {pin}: {address}")

        # Search for business info
        results = search_address(address)

        if results:
            # Enrich results with LLM
            enriched_results, llm_used = enrich_results(results, address, parcel)

            # Parse and save results
            businesses = parse_results(enriched_results, parcel, address, query, enriched=llm_used)

            for biz in businesses:
                if save_business(biz):
                    total_businesses += 1
                else:
                    total_errors += 1

            logger.info(f"  -> Found {len(businesses)} results (LLM: {llm_used})")
        else:
            logger.info(f"  -> No results found")

        total_processed += 1

        # Save checkpoint periodically
        if total_processed % 10 == 0:
            save_checkpoint(pin)

        # Rate limiting
        if i < len(parcels) - 1:  # Don't sleep after last parcel
            time.sleep(RATE_LIMIT)

    # Final checkpoint
    if parcels:
        save_checkpoint(parcels[-1].get("PIN"))

    # Summary
    logger.info("=" * 50)
    logger.info("Processing complete!")
    logger.info(f"  Parcels processed: {total_processed}")
    logger.info(f"  Business records saved: {total_businesses}")
    logger.info(f"  Errors: {total_errors}")
    logger.info("=" * 50)

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
