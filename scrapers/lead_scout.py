#!/usr/bin/env python3
"""
Lead Scouting Pipeline

Scouts high-value business leads by searching for company intel and scraping websites.
Stores results as structured markdown profiles in data/leads/{slug}/profile.md.

Usage:
    python3 lead_scout.py --help           # Show help
    python3 lead_scout.py --limit 10       # Scout first N leads
    python3 lead_scout.py --id 4823        # Scout specific business
    python3 lead_scout.py --resume         # Resume from checkpoint
    python3 lead_scout.py --test           # Test mode (1 lead)
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("Error: requests library required. Install with: pip install requests")
    sys.exit(1)

# Configuration
API_BASE = os.environ.get("API_BASE", "http://192.168.1.17:8006")
SEARCH_DELAY = 3.0  # seconds between web searches
CHECKPOINT_FILE = "/tmp/lead_scout_checkpoint.json"
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "leads"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Slug Generation
# ============================================================================

def slugify(name: str) -> str:
    """Convert business name to URL-safe slug."""
    if not name:
        return "unknown"
    slug = re.sub(r'[^a-z0-9]+', '-', name.lower())
    return slug.strip('-')[:50]  # Limit length


# ============================================================================
# Web Search
# ============================================================================

def search_company_intel(business_name: str, city: str = "Albuquerque") -> Optional[str]:
    """
    Search for company intel using SearxNG.

    Returns a summary of search results.
    """
    query = f"{business_name} {city}"

    try:
        response = requests.get(
            f"{API_BASE}/searx/search",
            params={"q": query, "num_results": 5},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        # Handle error response
        if data.get("error"):
            logger.warning(f"Search failed: {data.get('error')}")
            return None

        # SearxNG response has nested results.results structure
        results_obj = data.get("results", {})
        if isinstance(results_obj, dict):
            results = results_obj.get("results", [])
        else:
            results = results_obj  # Fallback if it's a list

        if not results:
            return "No search results found."

        # Build summary from results
        summaries = []
        for r in results[:5]:
            title = r.get("title", "")
            content = r.get("content", "")
            if content:
                summaries.append(f"- {content[:200]}")

        return "\n".join(summaries) if summaries else "No detailed information found."

    except requests.Timeout:
        logger.warning(f"Search timed out for: {query}")
        return None
    except requests.RequestException as e:
        logger.warning(f"Search failed for: {query}: {e}")
        return None


# ============================================================================
# Website Scraping (via Playwright MCP)
# ============================================================================

def scrape_website(url: str) -> Optional[str]:
    """
    Scrape website content using the scraper endpoint.

    Returns extracted content or None.
    """
    if not url:
        return None

    # Ensure URL has protocol
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    try:
        response = requests.post(
            f"{API_BASE}/scraper/scrape-url",
            json={"url": url, "timeout": 15000},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        if data.get("success"):
            content = data.get("content", "")
            # Truncate if too long
            if len(content) > 2000:
                content = content[:2000] + "...\n\n*[Content truncated]*"
            return content
        else:
            logger.debug(f"Scrape failed: {data.get('error')}")
            return None

    except requests.Timeout:
        logger.warning(f"Scrape timed out for: {url}")
        return None
    except requests.RequestException as e:
        logger.warning(f"Scrape failed for: {url}: {e}")
        return None


# ============================================================================
# Profile Generation
# ============================================================================

def generate_profile(business: dict, intel: Optional[str], website_content: Optional[str]) -> str:
    """Generate markdown profile for a business."""

    name = business.get("business_name", "Unknown Business")
    address = business.get("address", "N/A")
    phone = business.get("phone", "N/A")
    website = business.get("website", "N/A")
    rating = business.get("rating", "N/A")
    review_count = business.get("review_count", 0)
    business_type = business.get("business_type", "N/A")

    intel_section = intel if intel else "No company intel found."
    website_section = website_content if website_content else "Website not scraped or unavailable."

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    profile = f"""# {name}

## Quick Facts
- **Address:** {address}
- **Phone:** {phone}
- **Website:** {website}
- **Rating:** {rating} ({review_count} reviews)
- **Type:** {business_type}

## Company Intel
{intel_section}

## Website Highlights
{website_section}

---
*Scouted: {timestamp}*
"""
    return profile


def save_profile(slug: str, profile: str) -> str:
    """Save profile to data/leads/{slug}/profile.md. Returns the path."""
    profile_dir = DATA_DIR / slug
    profile_dir.mkdir(parents=True, exist_ok=True)

    profile_path = profile_dir / "profile.md"
    profile_path.write_text(profile)

    # Return relative path from project root
    return f"data/leads/{slug}/profile.md"


# ============================================================================
# Database Functions
# ============================================================================

def fetch_leads(limit: Optional[int] = None, business_id: Optional[int] = None,
                offset_id: Optional[int] = None) -> list[dict]:
    """Fetch high-value leads from database."""

    conditions = ["rating >= 4.5", "review_count >= 10", "scout_status = 'pending'"]

    if business_id:
        conditions = [f"id = {business_id}"]
    elif offset_id:
        conditions.append(f"id > {offset_id}")

    sql = f"""
        SELECT id, business_name, address, phone, website, rating, review_count, business_type
        FROM crm.businesses
        WHERE {' AND '.join(conditions)}
        ORDER BY rating DESC, review_count DESC
    """

    if limit:
        sql += f" LIMIT {limit}"

    try:
        response = requests.post(
            f"{API_BASE}/postgresql/query",
            json={"sql": sql},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        if data.get("success"):
            leads = data.get("data", [])
            logger.info(f"Fetched {len(leads)} leads")
            return leads
        else:
            logger.error(f"Query failed: {data}")
            return []

    except requests.RequestException as e:
        logger.error(f"Failed to fetch leads: {e}")
        return []


def update_scout_status(business_id: int, profile_path: str) -> bool:
    """Update database with scout status and profile path."""

    sql = f"""
        UPDATE crm.businesses SET
            scouted_at = NOW(),
            scout_status = 'scouted',
            profile_path = '{profile_path}'
        WHERE id = {business_id}
    """

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
        logger.warning(f"Failed to update status for business {business_id}: {e}")
        return False


# ============================================================================
# Checkpoint Functions
# ============================================================================

def load_checkpoint() -> Optional[dict]:
    """Load checkpoint data."""
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_checkpoint(business_id: int, stats: dict) -> None:
    """Save checkpoint with last processed business ID and stats."""
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump({
                "last_id": business_id,
                "timestamp": datetime.now().isoformat(),
                "stats": stats
            }, f, indent=2)
    except IOError as e:
        logger.warning(f"Failed to save checkpoint: {e}")


# ============================================================================
# Main Processing
# ============================================================================

def scout_lead(business: dict) -> bool:
    """
    Scout a single business lead.

    Returns True if successful.
    """
    business_id = business.get("id")
    name = business.get("business_name", "Unknown")
    website = business.get("website")

    logger.info(f"Scouting: {name}")

    # 1. Search for company intel
    logger.info(f"  Searching for company intel...")
    intel = search_company_intel(name)

    # Rate limit before website scrape
    time.sleep(SEARCH_DELAY)

    # 2. Scrape website if available
    website_content = None
    if website:
        logger.info(f"  Scraping website: {website}")
        website_content = scrape_website(website)
    else:
        logger.info(f"  No website to scrape")

    # 3. Generate profile
    slug = slugify(name)
    profile = generate_profile(business, intel, website_content)
    profile_path = save_profile(slug, profile)
    logger.info(f"  Profile saved: {profile_path}")

    # 4. Update database
    if update_scout_status(business_id, profile_path):
        logger.info(f"  Database updated")
        return True
    else:
        logger.warning(f"  Failed to update database")
        return False


def main():
    parser = argparse.ArgumentParser(description="Lead Scouting Pipeline")
    parser.add_argument("--test", action="store_true", help="Test mode: scout only 1 lead")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--limit", type=int, help="Limit number of leads to scout")
    parser.add_argument("--id", type=int, help="Scout specific business by ID")
    args = parser.parse_args()

    # Determine limit
    limit = 1 if args.test else args.limit

    # Check for resume
    offset_id = None
    if args.resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            offset_id = checkpoint.get("last_id")
            logger.info(f"Resuming from business ID: {offset_id}")
        else:
            logger.info("No checkpoint found, starting from beginning")

    # Fetch leads
    logger.info("Fetching leads...")
    leads = fetch_leads(limit=limit, business_id=args.id, offset_id=offset_id)

    if not leads:
        logger.info("No leads to scout")
        return 0

    logger.info(f"Scouting {len(leads)} leads...")

    # Stats
    stats = {
        "total_processed": 0,
        "successful": 0,
        "errors": 0
    }

    for i, lead in enumerate(leads):
        business_id = lead.get("id")
        name = lead.get("business_name", "Unknown")

        logger.info(f"[{i+1}/{len(leads)}] {name}")

        try:
            if scout_lead(lead):
                stats["successful"] += 1
            else:
                stats["errors"] += 1

            stats["total_processed"] += 1
            save_checkpoint(business_id, stats)

        except Exception as e:
            logger.error(f"  Error scouting {name}: {e}")
            stats["errors"] += 1

        # Rate limit between leads (except for last)
        if i < len(leads) - 1:
            logger.debug(f"  Rate limiting: {SEARCH_DELAY}s")
            time.sleep(SEARCH_DELAY)

    # Final summary
    logger.info("=" * 50)
    logger.info("Lead Scouting Complete")
    logger.info(f"  Leads processed: {stats['total_processed']}")
    logger.info(f"  Successful: {stats['successful']}")
    logger.info(f"  Errors: {stats['errors']}")
    logger.info("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
