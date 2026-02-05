#!/usr/bin/env python3
"""
Classify businesses as sales targets using SQL filters + Qwen LLM.

Pass 1: SQL filter for known good/bad business types
Pass 2: Qwen classification for unclassified businesses

Usage:
    python3 classify_targets.py --sql-only      # Just SQL pass
    python3 classify_targets.py --qwen-only     # Just Qwen pass (assumes SQL done)
    python3 classify_targets.py                 # Full run (SQL + Qwen)
    python3 classify_targets.py --dry-run       # Preview without DB updates
"""

import argparse
import json
import logging
import re
import sys
import time
from typing import List, Dict, Tuple

import requests

# Configuration
API_BASE = "http://192.168.1.17:8006"
BATCH_SIZE = 20  # Businesses per Qwen call
QWEN_DELAY = 1.0  # Seconds between Qwen calls

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Known Business Type Classifications
# ============================================================================

# Types we definitely want to target
TARGET_TYPES = {
    'general_contractor',
    'electrician',
    'plumber',
    'roofing_contractor',
    'hvac_contractor',
    'moving_company',
    'wholesaler',
    'warehouse',
    'machine_shop',
    'painter',
    'locksmith',
    'pest_control',
    'tree_service',
    'landscaper',
    'cleaning_service',
    'security_service',
    'courier_service',
    'printing_service',
    'sign_shop',
    'welder',
    'fabricator',
    'manufacturer',
}

# Types we definitely want to skip
SKIP_TYPES = {
    'restaurant',
    'mexican_restaurant',
    'pizza_restaurant',
    'fast_food_restaurant',
    'cafe',
    'coffee_shop',
    'bar',
    'grocery_store',
    'supermarket',
    'convenience_store',
    'liquor_store',
    'clothing_store',
    'jewelry_store',
    'shoe_store',
    'furniture_store',
    'home_goods_store',
    'discount_store',
    'department_store',
    'shopping_mall',
    'storage',
    'self_storage',
    'museum',
    'amusement_park',
    'movie_theater',
    'gym',
    'spa',
    'salon',
    'barbershop',
    'church',
    'school',
    'university',
    'hospital',
    'doctor',
    'dentist',
    'veterinarian',
    'pharmacy',
    'bank',
    'atm',
    'gas_station',
    'car_wash',
    'parking',
    'hotel',
    'motel',
    'apartment_building',
    'apartment_complex',
    'real_estate_agency',
    'insurance_agency',
    'lawyer',
    'accountant',
    'local_government_office',
    'government_office',
    'post_office',
    'library',
    'park',
    'cemetery',
}

# National chains to skip (by name patterns)
CHAIN_PATTERNS = [
    r'\bwalmart\b',
    r'\btarget\b',
    r'\bcostco\b',
    r'\bhome depot\b',
    r'\blowes\b',
    r'\bmenards\b',
    r'\bsafelite\b',
    r'\bups\b(?!\s+store)',  # UPS but not "UPS Store" franchise
    r'\bfedex\b',
    r'\busps\b',
    r'\bmcdonald',
    r'\bburger king\b',
    r'\bwendy',
    r'\bsubway\b',
    r'\bstarbucks\b',
    r'\bdunkin\b',
    r'\bwalgreens\b',
    r'\bcvs\b',
    r'\brite aid\b',
    r'\bdollar general\b',
    r'\bdollar tree\b',
    r'\bfamily dollar\b',
    r'\bautozone\b',
    r'\boreilly auto\b',
    r'\badvance auto\b',
    r'\bnapa auto\b',
    r'\benterprise rent\b',
    r'\bhertz\b',
    r'\bavis\b',
    r'\bbudget rent\b',
    r'\bu-haul\b',
    r'\bpenske\b',
]

# Nonprofit/Government patterns
NONPROFIT_PATTERNS = [
    r'\bhumane\b',
    r'\bfoundation\b',
    r'\bcharity\b',
    r'\bchurch\b',
    r'\bministry\b',
    r'\bsociety\b',
    r'\bassociation\b',
    r'\bnon-?profit\b',
]

GOVERNMENT_PATTERNS = [
    r'\bdepartment\b',
    r'\bagency\b',
    r'\bbureau\b',
    r'\bcity of\b',
    r'\bcounty of\b',
    r'\bstate of\b',
    r'\bhuman services\b',
    r'\bpublic works\b',
    r'\bfire department\b',
    r'\bpolice\b',
    r'\bsheriff\b',
]


def is_chain(name: str) -> bool:
    """Check if business name matches a known chain pattern."""
    name_lower = name.lower()
    for pattern in CHAIN_PATTERNS:
        if re.search(pattern, name_lower):
            return True
    return False


def is_nonprofit(name: str) -> bool:
    """Check if business name suggests nonprofit."""
    name_lower = name.lower()
    for pattern in NONPROFIT_PATTERNS:
        if re.search(pattern, name_lower):
            return True
    return False


def is_government(name: str) -> bool:
    """Check if business name suggests government entity."""
    name_lower = name.lower()
    for pattern in GOVERNMENT_PATTERNS:
        if re.search(pattern, name_lower):
            return True
    return False


# ============================================================================
# Database Operations
# ============================================================================

def query_db(sql: str) -> Dict:
    """Execute SQL query and return results."""
    response = requests.post(
        f"{API_BASE}/postgresql/query",
        json={"sql": sql},
        timeout=30
    )
    response.raise_for_status()
    return response.json()


def update_targets(ids: List[int], is_target: bool, source: str, dry_run: bool = False) -> int:
    """Update is_sales_target for a list of business IDs."""
    if not ids:
        return 0

    if dry_run:
        logger.info(f"  [DRY RUN] Would update {len(ids)} businesses to is_sales_target={is_target}")
        return len(ids)

    id_list = ','.join(str(i) for i in ids)
    sql = f"""
        UPDATE crm.businesses
        SET is_sales_target = {str(is_target).lower()},
            target_source = '{source}'
        WHERE id IN ({id_list})
    """
    result = query_db(sql)
    return result.get('rowcount', 0)


# ============================================================================
# SQL Classification Pass
# ============================================================================

def sql_pass(dry_run: bool = False) -> Tuple[int, int]:
    """
    Classify businesses based on known business_type values.
    Returns (targets_count, skips_count).
    """
    logger.info("=" * 60)
    logger.info("SQL CLASSIFICATION PASS")
    logger.info("=" * 60)

    targets_marked = 0
    skips_marked = 0

    # Mark known good types as targets
    if TARGET_TYPES:
        types_str = "','".join(TARGET_TYPES)
        sql = f"""
            SELECT id FROM crm.businesses
            WHERE business_type IN ('{types_str}')
            AND is_sales_target IS NULL
        """
        result = query_db(sql)
        ids = [row['id'] for row in result.get('data', [])]
        if ids:
            count = update_targets(ids, True, 'sql_filter', dry_run)
            targets_marked += count
            logger.info(f"  Marked {count} businesses as TARGET (known good types)")

    # Mark known bad types as skip
    if SKIP_TYPES:
        types_str = "','".join(SKIP_TYPES)
        sql = f"""
            SELECT id FROM crm.businesses
            WHERE business_type IN ('{types_str}')
            AND is_sales_target IS NULL
        """
        result = query_db(sql)
        ids = [row['id'] for row in result.get('data', [])]
        if ids:
            count = update_targets(ids, False, 'sql_filter', dry_run)
            skips_marked += count
            logger.info(f"  Marked {count} businesses as SKIP (known bad types)")

    # Mark chains by name pattern
    sql = "SELECT id, business_name FROM crm.businesses WHERE is_sales_target IS NULL"
    result = query_db(sql)

    chain_ids = []
    nonprofit_ids = []
    govt_ids = []

    for row in result.get('data', []):
        name = row.get('business_name', '')
        if is_chain(name):
            chain_ids.append(row['id'])
        elif is_nonprofit(name):
            nonprofit_ids.append(row['id'])
        elif is_government(name):
            govt_ids.append(row['id'])

    if chain_ids:
        count = update_targets(chain_ids, False, 'sql_filter', dry_run)
        skips_marked += count
        logger.info(f"  Marked {count} businesses as SKIP (chain names)")

    if nonprofit_ids:
        count = update_targets(nonprofit_ids, False, 'sql_filter', dry_run)
        skips_marked += count
        logger.info(f"  Marked {count} businesses as SKIP (nonprofit names)")

    if govt_ids:
        count = update_targets(govt_ids, False, 'sql_filter', dry_run)
        skips_marked += count
        logger.info(f"  Marked {count} businesses as SKIP (government names)")

    logger.info(f"\nSQL Pass Complete: {targets_marked} targets, {skips_marked} skips")
    return targets_marked, skips_marked


# ============================================================================
# Qwen Classification Pass
# ============================================================================

QWEN_PROMPT = """Classify for B2B sales. T=target, S=skip.
SKIP: Nonprofits (Humane, Foundation), Govt (Department, Agency), Chains (Safelite, UPS, FedEx), Restaurants, Retail, Consumer services.
TARGET: Local contractors, trades, pest control, HVAC, solar, manufacturing, shredding, recycling, tree service, industrial.

{businesses}

Format: 1:T 2:S ..."""


def classify_batch_with_qwen(businesses: List[Dict]) -> Dict[int, bool]:
    """
    Classify a batch of businesses using Qwen.
    Returns dict of {id: is_target}.
    """
    # Build numbered list
    lines = []
    for i, biz in enumerate(businesses, 1):
        lines.append(f"{i}. {biz['business_name']}")

    prompt = QWEN_PROMPT.format(businesses='\n'.join(lines))

    try:
        response = requests.post(
            f"{API_BASE}/llm/chat",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "provider": "ollama",
                "model": "qwen3:8b"
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()

        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')

        # Parse response like "1:T 2:S 3:T ..."
        results = {}
        matches = re.findall(r'(\d+):([TS])', content, re.IGNORECASE)
        for num_str, classification in matches:
            idx = int(num_str) - 1
            if 0 <= idx < len(businesses):
                biz_id = businesses[idx]['id']
                results[biz_id] = classification.upper() == 'T'

        return results

    except Exception as e:
        logger.warning(f"Qwen classification failed: {e}")
        return {}


def qwen_pass(dry_run: bool = False) -> Tuple[int, int]:
    """
    Classify remaining unclassified businesses using Qwen.
    Returns (targets_count, skips_count).
    """
    logger.info("=" * 60)
    logger.info("QWEN CLASSIFICATION PASS")
    logger.info("=" * 60)

    # Get unclassified businesses
    sql = """
        SELECT id, business_name
        FROM crm.businesses
        WHERE is_sales_target IS NULL
        ORDER BY review_count DESC NULLS LAST
    """
    result = query_db(sql)
    unclassified = result.get('data', [])

    logger.info(f"Found {len(unclassified)} unclassified businesses")

    if not unclassified:
        return 0, 0

    targets_marked = 0
    skips_marked = 0

    # Process in batches
    for i in range(0, len(unclassified), BATCH_SIZE):
        batch = unclassified[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(unclassified) + BATCH_SIZE - 1) // BATCH_SIZE

        logger.info(f"\nBatch {batch_num}/{total_batches} ({len(batch)} businesses)")

        classifications = classify_batch_with_qwen(batch)

        if not classifications:
            logger.warning("  No classifications returned, skipping batch")
            continue

        # Separate targets and skips
        target_ids = [bid for bid, is_target in classifications.items() if is_target]
        skip_ids = [bid for bid, is_target in classifications.items() if not is_target]

        if target_ids:
            count = update_targets(target_ids, True, 'qwen', dry_run)
            targets_marked += count
            logger.info(f"  Marked {count} as TARGET")

        if skip_ids:
            count = update_targets(skip_ids, False, 'qwen', dry_run)
            skips_marked += count
            logger.info(f"  Marked {count} as SKIP")

        # Rate limit
        if i + BATCH_SIZE < len(unclassified):
            time.sleep(QWEN_DELAY)

    logger.info(f"\nQwen Pass Complete: {targets_marked} targets, {skips_marked} skips")
    return targets_marked, skips_marked


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Classify businesses as sales targets')
    parser.add_argument('--sql-only', action='store_true', help='Only run SQL classification pass')
    parser.add_argument('--qwen-only', action='store_true', help='Only run Qwen classification pass')
    parser.add_argument('--dry-run', action='store_true', help='Preview without updating database')
    args = parser.parse_args()

    total_targets = 0
    total_skips = 0

    if args.dry_run:
        logger.info("*** DRY RUN MODE - No database changes will be made ***\n")

    # SQL pass
    if not args.qwen_only:
        targets, skips = sql_pass(args.dry_run)
        total_targets += targets
        total_skips += skips

    # Qwen pass
    if not args.sql_only:
        targets, skips = qwen_pass(args.dry_run)
        total_targets += targets
        total_skips += skips

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("CLASSIFICATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total targets marked: {total_targets}")
    logger.info(f"Total skips marked: {total_skips}")

    # Show remaining
    result = query_db("SELECT COUNT(*) as cnt FROM crm.businesses WHERE is_sales_target IS NULL")
    remaining = result.get('data', [{}])[0].get('cnt', 0)
    logger.info(f"Remaining unclassified: {remaining}")

    # Show target summary
    result = query_db("""
        SELECT is_sales_target, COUNT(*) as cnt
        FROM crm.businesses
        GROUP BY is_sales_target
    """)
    logger.info("\nDatabase totals:")
    for row in result.get('data', []):
        status = row.get('is_sales_target')
        if status is True:
            label = "Targets"
        elif status is False:
            label = "Skips"
        else:
            label = "Unclassified"
        logger.info(f"  {label}: {row.get('cnt', 0)}")


if __name__ == "__main__":
    main()
