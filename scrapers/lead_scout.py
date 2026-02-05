#!/usr/bin/env python3
"""
Lead Scouting Pipeline v2

Deep research on business leads using:
- SearxNG: Targeted searches (ownership, history, services)
- Playwright: Website scraping (/about, /team, /contact)
- Groq/Ollama: LLM distillation of raw content into structured facts
- Gemini: Person enrichment with Google Search grounding

Usage:
    python3 lead_scout.py --help           # Show help
    python3 lead_scout.py --limit 10       # Scout first N leads
    python3 lead_scout.py --id 4823        # Scout specific business
    python3 lead_scout.py --resume         # Resume from checkpoint
    python3 lead_scout.py --test           # Test mode (1 lead)
    python3 lead_scout.py --provider ollama  # Use Ollama instead of Groq
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
from typing import Optional, Dict, List, Any
from urllib.parse import urljoin, urlparse

try:
    import requests
except ImportError:
    print("Error: requests library required. Install with: pip install requests")
    sys.exit(1)

# Configuration
API_BASE = os.environ.get("API_BASE", "http://192.168.1.17:8006")
SEARCH_DELAY = 2.0  # seconds between searches
SCRAPE_DELAY = 1.0  # seconds between page scrapes
CHECKPOINT_FILE = "/tmp/lead_scout_checkpoint.json"
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "leads"

# LLM Configuration
DEFAULT_LLM_PROVIDER = "groq"  # or "ollama"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1500

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Utilities
# ============================================================================

def slugify(name: str) -> str:
    """Convert business name to URL-safe slug."""
    if not name:
        return "unknown"
    slug = re.sub(r'[^a-z0-9]+', '-', name.lower())
    return slug.strip('-')[:50]


def extract_city_from_address(address: str) -> str:
    """Extract city from address string."""
    if not address:
        return "Albuquerque"
    # Try to find city before state abbreviation
    match = re.search(r',\s*([^,]+),\s*[A-Z]{2}', address)
    if match:
        return match.group(1).strip()
    return "Albuquerque"


# ============================================================================
# SearxNG Search
# ============================================================================

def search(query: str, num_results: int = 8) -> List[Dict[str, str]]:
    """
    Search using SearxNG. Returns list of {title, content, url}.
    """
    try:
        response = requests.get(
            f"{API_BASE}/searx/search",
            params={"q": query, "num_results": num_results},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        if data.get("error"):
            logger.warning(f"Search error: {data.get('error')}")
            return []

        results_obj = data.get("results", {})
        if isinstance(results_obj, dict):
            results = results_obj.get("results", [])
        else:
            results = results_obj

        return [
            {
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "url": r.get("url", "")
            }
            for r in results[:num_results]
            if r.get("content")
        ]

    except Exception as e:
        logger.warning(f"Search failed for '{query}': {e}")
        return []


def gather_search_intel(business_name: str, city: str) -> Dict[str, List[Dict]]:
    """
    Run multiple targeted searches to gather intel.
    Returns dict with search categories.
    """
    searches = {
        "ownership": f'"{business_name}" owner OR founder OR CEO OR president {city}',
        "history": f'"{business_name}" founded OR established OR history OR "since" {city}',
        "services": f'"{business_name}" services OR products OR specializes {city}',
        "reputation": f'"{business_name}" reviews OR awards OR certified {city}',
    }

    intel = {}
    for category, query in searches.items():
        logger.debug(f"  Search [{category}]: {query[:60]}...")
        results = search(query, num_results=5)
        intel[category] = results
        time.sleep(SEARCH_DELAY)

    return intel


# ============================================================================
# Website Scraping
# ============================================================================

def scrape_page(url: str) -> Optional[str]:
    """Scrape a single page using the scraper endpoint."""
    if not url:
        return None

    try:
        # Use the synchronous test-scrape endpoint (GET)
        response = requests.get(
            f"{API_BASE}/scraper/test-scrape",
            params={"url": url},
            timeout=45
        )
        response.raise_for_status()
        data = response.json()

        if data.get("scrape_status") == "success":
            content = data.get("clean_text", "")
            # Limit content size for LLM processing
            return content[:8000] if content else None
        else:
            logger.debug(f"Scrape failed: {data.get('error_message')}")
            return None

    except Exception as e:
        logger.debug(f"Scrape failed for {url}: {e}")
        return None


def scrape_website(base_url: str) -> Dict[str, str]:
    """
    Scrape key pages from a website.
    Returns dict of {page_type: content}.
    """
    if not base_url:
        return {}

    # Normalize URL
    if not base_url.startswith(('http://', 'https://')):
        base_url = 'https://' + base_url

    # Parse base URL
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    # Pages to try (in priority order) - expanded based on real-world patterns
    page_variants = {
        "about": ["/about", "/about-us", "/company", "/who-we-are", "/company-overview",
                  "/our-company", "/our-story", "/history"],
        "team": ["/team", "/our-team", "/leadership", "/staff", "/management",
                 "/executives", "/people", "/meet-the-team"],
        "services": ["/services", "/products", "/what-we-do", "/solutions",
                     "/capabilities", "/offerings", "/industries"],
        "contact": ["/contact", "/contact-us", "/locations", "/find-us",
                    "/branches", "/get-in-touch"],
    }

    scraped = {}

    # First scrape homepage
    logger.debug(f"  Scraping homepage: {base_url}")
    homepage = scrape_page(base_url)
    if homepage:
        scraped["homepage"] = homepage

    # Try to find and scrape key pages
    for page_type, variants in page_variants.items():
        for variant in variants:
            url = urljoin(base, variant)
            content = scrape_page(url)
            if content and len(content) > 200:  # Skip empty/error pages
                scraped[page_type] = content
                logger.debug(f"  Found {page_type}: {variant}")
                break
            time.sleep(SCRAPE_DELAY)

        if page_type not in scraped:
            logger.debug(f"  No {page_type} page found")

    return scraped


# ============================================================================
# Gemini Person Enrichment
# ============================================================================

def extract_owner_name(search_intel: str) -> Optional[Dict[str, str]]:
    """
    Extract owner/founder name from distilled search intel.
    Returns dict with first, last, title if found.
    """
    if not search_intel:
        return None

    # Look for patterns like "Martin Calfee – Founder" or "John Smith, CEO"
    patterns = [
        r'\*\*Owner/Founder\*\*:\s*([A-Z][a-z]+)\s+([A-Z][a-z]+)\s*[–\-—]\s*(\w+)',
        r'\*\*Owner/Founder\*\*:\s*([A-Z][a-z]+)\s+([A-Z][a-z]+)',
        r'([A-Z][a-z]+)\s+([A-Z][a-z]+)\s*[–\-—]\s*(Founder|Owner|CEO|President|Chairman)',
        r'(founder|owner|ceo|president):\s*([A-Z][a-z]+)\s+([A-Z][a-z]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, search_intel, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) >= 2:
                return {
                    "first": groups[0],
                    "last": groups[1],
                    "title": groups[2] if len(groups) > 2 else "Owner"
                }

    return None


def enrich_person_with_gemini(
    first: str,
    last: str,
    company: str,
    location: str = "Albuquerque, NM"
) -> Optional[Dict[str, Any]]:
    """
    Use Gemini 2.0 Flash with Google Search grounding to enrich person info.
    """
    try:
        response = requests.post(
            f"{API_BASE}/gemini/enrich",
            json={
                "first": first,
                "last": last,
                "company": company,
                "location": location
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()

        # Check if we got useful data
        if data.get("bio") or data.get("recent_milestones"):
            return data
        return None

    except Exception as e:
        logger.warning(f"Gemini enrichment failed: {e}")
        return None


# ============================================================================
# LLM Distillation
# ============================================================================

def distill_with_llm(
    content: str,
    prompt: str,
    provider: str = DEFAULT_LLM_PROVIDER
) -> Optional[str]:
    """
    Use Groq/Ollama to distill raw content into structured facts.
    """
    try:
        response = requests.post(
            f"{API_BASE}/llm/chat",
            json={
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": content}
                ],
                "provider": provider,
                "temperature": LLM_TEMPERATURE,
                "max_tokens": LLM_MAX_TOKENS
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return None

    except Exception as e:
        logger.warning(f"LLM distillation failed: {e}")
        return None


def distill_search_results(intel: Dict[str, List[Dict]], provider: str) -> Dict[str, str]:
    """
    Distill search results into structured ownership/history facts.
    """
    # Combine all search results into context
    context_parts = []
    for category, results in intel.items():
        if results:
            context_parts.append(f"=== {category.upper()} ===")
            for r in results:
                context_parts.append(f"[{r['title']}] {r['content']}")

    if not context_parts:
        return {}

    context = "\n".join(context_parts)

    prompt = """You are a business intelligence analyst. Extract ONLY facts that are explicitly stated in the sources.

Extract these fields (use "Unknown" if not found):
- **Owner/Founder**: Name and title of owner, founder, CEO, or president
- **Year Founded**: When the company was established
- **Employees**: Number of employees or size indicator
- **Certifications**: ISO, industry certifications, awards
- **Geographic Coverage**: Cities, states, or regions served
- **Key Services**: Main products or services offered
- **Notable Clients**: Any mentioned customers or industries served

Format as a clean markdown list. Be factual - only include what's explicitly stated."""

    logger.info("  Distilling search results...")
    result = distill_with_llm(context[:6000], prompt, provider)
    return {"search_intel": result} if result else {}


def distill_website_content(pages: Dict[str, str], provider: str) -> Dict[str, str]:
    """
    Distill website content into structured business info.
    """
    if not pages:
        return {}

    # Combine page content
    context_parts = []
    for page_type, content in pages.items():
        # Truncate each page to avoid token limits
        truncated = content[:3000] if len(content) > 3000 else content
        context_parts.append(f"=== {page_type.upper()} PAGE ===\n{truncated}")

    context = "\n\n".join(context_parts)

    prompt = """You are extracting business information from a company website.

Extract these fields (use "Not found" if not present):
- **Company Description**: One paragraph summary of what they do
- **Team/Leadership**: Names and titles of key people mentioned
- **Services Offered**: List of main services or products
- **Unique Value Proposition**: What makes them different
- **Contact Info**: Phone, email, address if visible
- **Company Culture**: Any mission/values statements

Format as clean markdown. Only include facts explicitly stated on the website."""

    logger.info("  Distilling website content...")
    result = distill_with_llm(context[:8000], prompt, provider)
    return {"website_intel": result} if result else {}


# ============================================================================
# Profile Generation
# ============================================================================

def generate_profile(
    business: Dict,
    search_distilled: Dict[str, str],
    website_distilled: Dict[str, str],
    person_intel: Optional[Dict[str, Any]] = None
) -> str:
    """Generate rich markdown profile from distilled intel."""

    name = business.get("business_name", "Unknown Business")
    address = business.get("address", "N/A")
    phone = business.get("phone", "N/A")
    website = business.get("website", "N/A")
    rating = business.get("rating", "N/A")
    review_count = business.get("review_count", 0)
    business_type = business.get("business_type", "N/A")

    search_intel = search_distilled.get("search_intel", "No search intelligence gathered.")
    website_intel = website_distilled.get("website_intel", "Website not analyzed.")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Determine quality score based on what we found
    quality_flags = []
    if search_intel and search_intel != "No search intelligence gathered.":
        quality_flags.append("search_complete")
    if website_intel and website_intel != "Website not analyzed.":
        quality_flags.append("website_complete")
    if person_intel:
        quality_flags.append("owner_enriched")

    quality = "high" if len(quality_flags) >= 2 else "partial" if quality_flags else "minimal"

    # Build owner/decision maker section from Gemini intel
    owner_section = ""
    if person_intel:
        owner_section = "\n## Owner/Decision Maker Intel\n\n"
        if person_intel.get("bio"):
            owner_section += f"**Bio:** {person_intel['bio']}\n\n"
        if person_intel.get("linkedin_url") and person_intel["linkedin_url"] != "unavailable":
            owner_section += f"**LinkedIn:** {person_intel['linkedin_url']}\n\n"
        if person_intel.get("recent_milestones"):
            owner_section += "**Recent Milestones:**\n"
            for milestone in person_intel["recent_milestones"]:
                owner_section += f"- {milestone}\n"
            owner_section += "\n"
        if person_intel.get("ice_breaker"):
            owner_section += f"**Outreach Angle:** {person_intel['ice_breaker']}\n"

    profile = f"""# {name}

> **Scout Quality:** {quality} | **Scouted:** {timestamp}

## Quick Facts

| Field | Value |
|-------|-------|
| **Address** | {address} |
| **Phone** | {phone} |
| **Website** | {website} |
| **Rating** | {rating} ({review_count} reviews) |
| **Category** | {business_type} |

## Company Background

{search_intel}
{owner_section}
## Business Operations

{website_intel}

---

## Scout Metadata
- **Quality Flags:** {', '.join(quality_flags) if quality_flags else 'none'}
- **Sources:** SearxNG search, Gemini grounded search
- **LLM:** Groq distillation + Gemini 2.0 Flash
"""
    return profile


def save_profile(slug: str, profile: str) -> str:
    """Save profile to data/leads/{slug}/profile.md."""
    profile_dir = DATA_DIR / slug
    profile_dir.mkdir(parents=True, exist_ok=True)

    profile_path = profile_dir / "profile.md"
    profile_path.write_text(profile)

    return f"data/leads/{slug}/profile.md"


# ============================================================================
# Database Functions
# ============================================================================

def fetch_leads(
    limit: Optional[int] = None,
    business_id: Optional[int] = None,
    offset_id: Optional[int] = None
) -> List[Dict]:
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

    except Exception as e:
        logger.error(f"Failed to fetch leads: {e}")
        return []


def update_scout_status(business_id: int, profile_path: str, quality: str) -> bool:
    """Update database with scout status."""
    # Escape single quotes in profile_path
    safe_path = profile_path.replace("'", "''")

    sql = f"""
        UPDATE crm.businesses SET
            scouted_at = NOW(),
            scout_status = 'scouted',
            profile_path = '{safe_path}'
        WHERE id = {business_id}
    """

    try:
        response = requests.post(
            f"{API_BASE}/postgresql/query",
            json={"sql": sql},
            timeout=30
        )
        response.raise_for_status()
        return response.json().get("success", False)

    except Exception as e:
        logger.warning(f"Failed to update status: {e}")
        return False


# ============================================================================
# Checkpoint Functions
# ============================================================================

def load_checkpoint() -> Optional[Dict]:
    """Load checkpoint data."""
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_checkpoint(business_id: int, stats: Dict) -> None:
    """Save checkpoint."""
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

def scout_lead(business: Dict, provider: str) -> tuple[bool, str]:
    """
    Scout a single business lead with deep research.
    Returns (success, quality).
    """
    business_id = business.get("id")
    name = business.get("business_name", "Unknown")
    website = business.get("website")
    address = business.get("address", "")

    city = extract_city_from_address(address)
    logger.info(f"Scouting: {name} ({city})")

    # Phase 1: Targeted searches
    logger.info("  Phase 1: Gathering search intelligence...")
    search_intel = gather_search_intel(name, city)

    # Phase 2: Distill search results
    logger.info("  Phase 2: Distilling with Groq...")
    search_distilled = distill_search_results(search_intel, provider)

    # Phase 3: Gemini person enrichment (if owner found)
    person_intel = None
    search_text = search_distilled.get("search_intel", "")
    owner_info = extract_owner_name(search_text)
    if owner_info:
        logger.info(f"  Phase 3: Enriching owner via Gemini: {owner_info['first']} {owner_info['last']}")
        person_intel = enrich_person_with_gemini(
            first=owner_info["first"],
            last=owner_info["last"],
            company=name,
            location=city
        )
        if person_intel:
            logger.info("    Gemini enrichment successful")
        else:
            logger.info("    Gemini enrichment returned no data")
    else:
        logger.info("  Phase 3: No owner name found to enrich")

    # Phase 4: Scrape website
    website_pages = {}
    if website:
        logger.info(f"  Phase 4: Scraping website...")
        website_pages = scrape_website(website)
        logger.info(f"    Found {len(website_pages)} pages")
    else:
        logger.info("  Phase 4: No website to scrape")

    # Phase 5: Distill website content
    website_distilled = {}
    if website_pages:
        logger.info("  Phase 5: Distilling website content...")
        website_distilled = distill_website_content(website_pages, provider)

    # Phase 6: Generate profile
    slug = slugify(name)
    profile = generate_profile(business, search_distilled, website_distilled, person_intel)

    # Determine quality
    quality_score = 0
    if search_distilled.get("search_intel"):
        quality_score += 1
    if person_intel:
        quality_score += 1
    if website_distilled.get("website_intel"):
        quality_score += 1

    quality = "high" if quality_score >= 2 else "partial" if quality_score == 1 else "minimal"

    # Save profile
    profile_path = save_profile(slug, profile)
    logger.info(f"  Profile saved: {profile_path} (quality: {quality})")

    # Update database
    if update_scout_status(business_id, profile_path, quality):
        logger.info("  Database updated")
        return True, quality
    else:
        logger.warning("  Failed to update database")
        return False, quality


def main():
    parser = argparse.ArgumentParser(description="Lead Scouting Pipeline v2")
    parser.add_argument("--test", action="store_true", help="Test mode: scout 1 lead")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--limit", type=int, help="Limit number of leads")
    parser.add_argument("--id", type=int, help="Scout specific business by ID")
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER,
                        choices=["groq", "ollama"], help="LLM provider for distillation")
    args = parser.parse_args()

    limit = 1 if args.test else args.limit
    provider = args.provider

    logger.info(f"Using LLM provider: {provider}")

    # Check for resume
    offset_id = None
    if args.resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            offset_id = checkpoint.get("last_id")
            logger.info(f"Resuming from ID: {offset_id}")

    # Fetch leads
    leads = fetch_leads(limit=limit, business_id=args.id, offset_id=offset_id)

    if not leads:
        logger.info("No leads to scout")
        return 0

    logger.info(f"Scouting {len(leads)} leads...")

    stats = {
        "total": 0,
        "successful": 0,
        "high_quality": 0,
        "partial_quality": 0,
        "minimal_quality": 0,
        "errors": 0
    }

    for i, lead in enumerate(leads):
        business_id = lead.get("id")
        name = lead.get("business_name", "Unknown")

        logger.info(f"\n[{i+1}/{len(leads)}] {name}")

        try:
            success, quality = scout_lead(lead, provider)

            if success:
                stats["successful"] += 1
                stats[f"{quality}_quality"] += 1
            else:
                stats["errors"] += 1

            stats["total"] += 1
            save_checkpoint(business_id, stats)

        except Exception as e:
            logger.error(f"  Error: {e}")
            stats["errors"] += 1

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SCOUT COMPLETE")
    logger.info(f"  Total processed: {stats['total']}")
    logger.info(f"  Successful: {stats['successful']}")
    logger.info(f"  High quality: {stats['high_quality']}")
    logger.info(f"  Partial quality: {stats['partial_quality']}")
    logger.info(f"  Minimal quality: {stats['minimal_quality']}")
    logger.info(f"  Errors: {stats['errors']}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
