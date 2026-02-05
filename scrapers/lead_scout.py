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

try:
    import yaml
except ImportError:
    print("Error: pyyaml library required. Install with: pip install pyyaml")
    sys.exit(1)

# Configuration
API_BASE = os.environ.get("API_BASE", "http://192.168.1.17:8006")
SEARCH_DELAY = 5.0  # seconds between searches (increased to avoid rate limits)
SCRAPE_DELAY = 1.0  # seconds between page scrapes
CHECKPOINT_FILE = "/tmp/lead_scout_checkpoint.json"
DATA_DIR = Path(__file__).parent.parent.parent / "leads" / "data" / "leads"

# LLM Configuration
DEFAULT_LLM_PROVIDER = "groq"  # or "ollama"
LLM_TEMPERATURE = 0.1  # Default (backward compat)
LLM_STORY_TEMP = 0.35  # Higher temp for narrative creativity
LLM_EXTRACT_TEMP = 0.1  # Low temp for deterministic extraction
LLM_MAX_TOKENS = 1500
LLM_STORY_MAX_TOKENS = 2000  # Allow longer story responses

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
# Serper Search (Google Search API)
# ============================================================================

def search(query: str, num_results: int = 8) -> List[Dict[str, str]]:
    """
    Search using Serper (Google Search API). Returns list of {title, content, url}.
    """
    try:
        response = requests.post(
            f"{API_BASE}/serper/search",
            json={"query": query, "num": min(num_results, 100)},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        if data.get("error"):
            logger.warning(f"Search error: {data.get('error')}")
            return []

        # Serper returns organic results in 'organic' key
        results = data.get("organic", [])

        return [
            {
                "title": r.get("title", ""),
                "content": r.get("snippet", ""),  # Serper uses 'snippet' not 'content'
                "url": r.get("link", "")  # Serper uses 'link' not 'url'
            }
            for r in results[:num_results]
            if r.get("snippet")
        ]

    except Exception as e:
        logger.warning(f"Search failed for '{query}': {e}")
        return []


def gather_search_intel(
    business_name: str,
    city: str,
    employee_names: List[str] = None
) -> Dict[str, List[Dict]]:
    """
    Run multiple targeted searches to gather intel.
    Returns dict with search categories.

    If employee_names provided (from Places reviews), uses them for more
    targeted ownership searches instead of generic queries.
    """
    # Use shorter business name for searches (full name often too restrictive)
    short_name = business_name.split(" - ")[0].split(" LLC")[0].split(" Inc")[0].strip()

    searches = {}

    # If we have employee names from Places, search for them specifically
    if employee_names:
        for emp_name in employee_names[:2]:  # Limit to top 2 to avoid rate limits
            searches[f"person_{emp_name.lower().replace(' ', '_')}"] = \
                f'"{emp_name}" "{short_name}" {city}'
    else:
        # Fallback to generic ownership search
        searches["ownership"] = f'{short_name} owner OR founder OR CEO {city}'

    # Always include these
    searches["history"] = f'{short_name} founded OR established OR history {city}'
    searches["services"] = f'{short_name} services OR products OR specializes {city}'

    intel = {}
    for category, query in searches.items():
        logger.debug(f"  Search [{category}]: {query[:60]}...")
        results = search(query, num_results=5)
        intel[category] = results
        time.sleep(SEARCH_DELAY)

    return intel


# ============================================================================
# Google Places API
# ============================================================================

def search_place(business_name: str, address: str = None) -> Optional[str]:
    """
    Search for a business in Google Places and return place_id.
    """
    query = business_name
    if address:
        # Extract city from address for better matching
        city_match = re.search(r',\s*([^,]+),\s*[A-Z]{2}', address)
        if city_match:
            query = f"{business_name} {city_match.group(1)}"

    try:
        response = requests.post(
            f"{API_BASE}/places/search",
            json={"query": query, "max_results": 1},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        if data.get("success") and data.get("places"):
            return data["places"][0].get("place_id")
        return None

    except Exception as e:
        logger.warning(f"Places search failed: {e}")
        return None


def get_place_details(place_id: str, include_reviews: bool = True) -> Optional[Dict]:
    """
    Get detailed information about a place including reviews.
    """
    try:
        response = requests.post(
            f"{API_BASE}/places/details",
            json={"place_id": place_id, "include_reviews": include_reviews},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        if data.get("success"):
            return data.get("place")
        return None

    except Exception as e:
        logger.warning(f"Places details failed: {e}")
        return None


def extract_employees_from_reviews(reviews: List[Dict]) -> List[Dict[str, str]]:
    """
    Parse employee names mentioned in reviews.

    Looks for patterns like:
    - "worked with John" / "John was great"
    - "technician Mike" / "Mike the technician"
    - "Annette/Mike" (paired names)
    - "office staff - Matt and Stephanie"
    - Names followed by role indicators

    Returns list of {name, role, context}
    """
    if not reviews:
        return []

    employees = []
    seen_names = set()

    # Common role indicators
    role_patterns = [
        (r'\b(technician|tech)\s+([A-Z][a-z]+)', 'technician'),
        (r'([A-Z][a-z]+)\s+(?:the\s+)?(technician|tech)\b', 'technician'),
        (r'([A-Z][a-z]+)\s+(?:was|is)\s+(?:our|the|my)\s+(technician|installer|plumber|electrician)', None),
        (r'\b(installer|plumber|electrician)\s+([A-Z][a-z]+)', None),
        (r'([A-Z][a-z]+)\s+(?:from|at)\s+(?:the\s+)?(?:front\s+)?(?:desk|office)', 'office'),
        (r'office\s+(?:staff|personnel)[^.]*?([A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+)?)', 'office'),
        (r'([A-Z][a-z]+)\s+(?:and|&)\s+([A-Z][a-z]+)\s+(?:were|are|was)', None),
        (r'([A-Z][a-z]+)\s+solved|fixed|helped|installed|replaced', 'service'),
        (r'(?:sales|salesperson|rep)[^.]*?([A-Z][a-z]+)', 'sales'),
        (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*[-–—]\s*(?:Owner|Manager|President|CEO)', 'owner'),
        (r'(?:owner|manager)[^.]*?([A-Z][a-z]+)', 'management'),
    ]

    for review in reviews:
        text = review.get("text", "")
        if not text:
            continue

        # Check each pattern
        for pattern, default_role in role_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                # Find the name group (capitalized word)
                for g in groups:
                    if g and re.match(r'^[A-Z][a-z]+', g):
                        name = g.strip()
                        # Handle "Matt and Stephanie" style
                        if ' and ' in name.lower():
                            parts = re.split(r'\s+and\s+', name, flags=re.IGNORECASE)
                            for part in parts:
                                part = part.strip()
                                if part and part not in seen_names:
                                    seen_names.add(part)
                                    employees.append({
                                        "name": part,
                                        "role": default_role or "staff",
                                        "context": text[:100] + "..." if len(text) > 100 else text
                                    })
                        elif name not in seen_names and len(name) > 2:
                            # Filter out common false positives (pronouns, articles, adjectives)
                            if name.lower() not in ['the', 'they', 'this', 'that', 'very', 'great', 'good',
                                                    'our', 'your', 'their', 'his', 'her', 'my', 'its']:
                                seen_names.add(name)
                                employees.append({
                                    "name": name,
                                    "role": default_role or "staff",
                                    "context": text[:100] + "..." if len(text) > 100 else text
                                })
                        break

    return employees


def gather_places_intel(business_name: str, address: str = None) -> Dict[str, Any]:
    """
    Gather intelligence from Google Places API.
    Returns structured data including extracted employee names from reviews.
    """
    logger.info("  Searching Places API...")
    place_id = search_place(business_name, address)

    if not place_id:
        logger.info("    No place found")
        return {}

    logger.info(f"    Found place_id: {place_id[:20]}...")
    details = get_place_details(place_id, include_reviews=True)

    if not details:
        logger.info("    Could not fetch details")
        return {}

    # Extract employees from reviews
    reviews = details.get("reviews", [])
    employees = extract_employees_from_reviews(reviews)
    if employees:
        logger.info(f"    Extracted {len(employees)} employee names from reviews")

    return {
        "place_id": place_id,
        "rating": details.get("rating"),
        "review_count": details.get("rating_count"),
        "business_status": details.get("business_status"),
        "hours": details.get("hours"),
        "types": details.get("types", []),
        "reviews": reviews,
        "employees_from_reviews": employees,
        "address_components": details.get("address_components", {}),
        "google_maps_url": details.get("google_maps_url"),
    }


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


def research_company_with_gemini(
    company: str,
    location: str = "Albuquerque, NM"
) -> Optional[Dict[str, Any]]:
    """
    Use Gemini 2.0 Flash with Google Search grounding for company research.
    Returns leadership, scale, history, news, priorities, and sales opportunities.
    """
    try:
        response = requests.post(
            f"{API_BASE}/gemini/research-company",
            json={
                "company": company,
                "location": location
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()

        # Check if we got useful data
        if data.get("leadership") or data.get("summary") or data.get("scale"):
            return data
        return None

    except Exception as e:
        logger.warning(f"Gemini company research failed: {e}")
        return None


# ============================================================================
# LLM Distillation
# ============================================================================

def distill_with_llm(
    content: str,
    prompt: str,
    provider: str = DEFAULT_LLM_PROVIDER,
    temperature: float = None,
    max_tokens: int = None
) -> Optional[str]:
    """
    Use Groq/Ollama to distill raw content into structured facts.

    Args:
        content: Raw content to distill
        prompt: System prompt for distillation
        provider: LLM provider (groq or ollama)
        temperature: Override default temperature (0.1)
        max_tokens: Override default max tokens (1500)
    """
    temp = temperature if temperature is not None else LLM_TEMPERATURE
    tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS

    try:
        response = requests.post(
            f"{API_BASE}/llm/chat",
            json={
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": content}
                ],
                "provider": provider,
                "temperature": temp,
                "max_tokens": tokens
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


def distill_search_story(intel: Dict[str, List[Dict]], provider: str) -> str:
    """
    Distill search results into narrative story with conversation hooks.
    Returns markdown string for profile.md.
    """
    # Combine all search results into context
    context_parts = []
    for category, results in intel.items():
        if results:
            context_parts.append(f"=== {category.upper()} ===")
            for r in results:
                context_parts.append(f"[{r['title']}] {r['content']}")

    if not context_parts:
        return ""

    context = "\n".join(context_parts)

    prompt = """You are a sales researcher preparing a brief for a relationship manager.

From the search results below, write a 2-3 paragraph STORY about this company:
- Who runs it and what are their backgrounds?
- How did they get here? What's their journey?
- What makes them different from competitors?
- What drives them? What do they care about?

Write like you're briefing a colleague before a first meeting. Focus on the PEOPLE and their journey, not just business specs.

After the story, suggest 3 specific CONVERSATION STARTERS - questions that show you did your homework and care about them as people, not just as a sale.

Finally, list Quick Reference facts: founded, owners, team, services, licenses, reputation signals."""

    logger.info("  Distilling search story...")
    result = distill_with_llm(
        context[:6000], prompt, provider,
        temperature=LLM_STORY_TEMP, max_tokens=LLM_STORY_MAX_TOKENS
    )
    return result if result else ""


def distill_search_structured(intel: Dict[str, List[Dict]], provider: str) -> Dict:
    """
    Distill search results into structured YAML data.
    Returns dict for data.yaml.
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

    prompt = """Extract structured data from these search results into YAML format.

Use this exact schema:
```yaml
company:
  name:
  founded:
  acquired:
leadership:
  - name:
    role:
    background:
team:
  - name:
    role:
services:
  - service
licenses:
  - license
reputation:
  signals:
    - signal
  differentiators:
    - differentiator
conversation_hooks:
  - hook
```

Fill in what you find. Use null for missing fields. Be factual and precise.
Only include information explicitly stated in the sources.
Return ONLY the YAML, no markdown code blocks or explanation."""

    logger.info("  Distilling search structured...")
    result = distill_with_llm(
        context[:6000], prompt, provider,
        temperature=LLM_EXTRACT_TEMP, max_tokens=LLM_MAX_TOKENS
    )

    if not result:
        return {}

    # Parse YAML response
    try:
        # Strip markdown code blocks if present
        yaml_text = result.strip()
        if yaml_text.startswith("```"):
            yaml_text = "\n".join(yaml_text.split("\n")[1:])
        if yaml_text.endswith("```"):
            yaml_text = "\n".join(yaml_text.split("\n")[:-1])
        return yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse YAML: {e}")
        return {}


def distill_search_results(intel: Dict[str, List[Dict]], provider: str) -> Dict[str, str]:
    """
    DEPRECATED: Use distill_search_story() and distill_search_structured() instead.
    Kept for backward compatibility.
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


def distill_website_story(pages: Dict[str, str], provider: str) -> str:
    """
    Distill website content into business operations narrative.
    Returns markdown string to append to profile.md.
    """
    if not pages:
        return ""

    # Combine page content
    context_parts = []
    for page_type, content in pages.items():
        truncated = content[:3000] if len(content) > 3000 else content
        context_parts.append(f"=== {page_type.upper()} PAGE ===\n{truncated}")

    context = "\n\n".join(context_parts)

    prompt = """You are summarizing a company's website for a sales brief.

Write a 1-2 paragraph summary of how this business operates:
- What services/products do they offer?
- Who are their target customers?
- What's their unique approach or value proposition?
- Any culture, mission, or values that stand out?

Write naturally, as if briefing a colleague. Focus on what makes them tick."""

    logger.info("  Distilling website story...")
    result = distill_with_llm(
        context[:8000], prompt, provider,
        temperature=LLM_STORY_TEMP, max_tokens=LLM_STORY_MAX_TOKENS
    )
    return result if result else ""


def distill_website_structured(pages: Dict[str, str], provider: str) -> Dict:
    """
    Distill website content into structured data.
    Returns dict to merge into data.yaml.
    """
    if not pages:
        return {}

    # Combine page content
    context_parts = []
    for page_type, content in pages.items():
        truncated = content[:3000] if len(content) > 3000 else content
        context_parts.append(f"=== {page_type.upper()} PAGE ===\n{truncated}")

    context = "\n\n".join(context_parts)

    prompt = """Extract structured data from this website content into YAML format.

Use this schema:
```yaml
contact:
  phone:
  email:
  address:
  city:
  state:
  zip:
services:
  - service
team:
  - name:
    role:
certifications:
  - certification
values:
  - value statement
```

Fill in what you find. Use null for missing fields. Be factual.
Return ONLY the YAML, no markdown code blocks or explanation."""

    logger.info("  Distilling website structured...")
    result = distill_with_llm(
        context[:8000], prompt, provider,
        temperature=LLM_EXTRACT_TEMP, max_tokens=LLM_MAX_TOKENS
    )

    if not result:
        return {}

    # Parse YAML response
    try:
        yaml_text = result.strip()
        if yaml_text.startswith("```"):
            yaml_text = "\n".join(yaml_text.split("\n")[1:])
        if yaml_text.endswith("```"):
            yaml_text = "\n".join(yaml_text.split("\n")[:-1])
        return yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse website YAML: {e}")
        return {}


def distill_website_content(pages: Dict[str, str], provider: str) -> Dict[str, str]:
    """
    DEPRECATED: Use distill_website_story() and distill_website_structured() instead.
    Kept for backward compatibility.
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
    person_intel: Optional[Dict[str, Any]] = None,
    company_intel: Optional[Dict[str, Any]] = None
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
    if company_intel:
        quality_flags.append("gemini_research")
    if search_intel and search_intel != "No search intelligence gathered.":
        quality_flags.append("search_complete")
    if website_intel and website_intel != "Website not analyzed.":
        quality_flags.append("website_complete")
    if person_intel:
        quality_flags.append("owner_enriched")

    quality = "high" if len(quality_flags) >= 2 else "partial" if quality_flags else "minimal"

    # Build company research section from Gemini
    company_section = ""
    if company_intel:
        company_section = "\n## Company Intelligence (Gemini Research)\n\n"

        # Summary
        if company_intel.get("summary"):
            company_section += f"{company_intel['summary']}\n\n"

        # Leadership
        if company_intel.get("leadership"):
            company_section += "### Leadership\n\n"
            for leader in company_intel["leadership"]:
                leader_name = leader.get("name", "Unknown")
                leader_title = leader.get("title", "")
                leader_bg = leader.get("background", "")
                company_section += f"- **{leader_name}** - {leader_title}"
                if leader_bg:
                    company_section += f" ({leader_bg})"
                company_section += "\n"
            company_section += "\n"

        # Scale
        if company_intel.get("scale"):
            scale = company_intel["scale"]
            company_section += "### Scale & Footprint\n\n"
            if scale.get("employees"):
                company_section += f"- **Employees:** {scale['employees']}\n"
            if scale.get("locations"):
                company_section += f"- **Locations:** {scale['locations']}\n"
            if scale.get("customers"):
                company_section += f"- **Customers:** {scale['customers']}\n"
            if scale.get("revenue_indicator"):
                company_section += f"- **Revenue Indicator:** {scale['revenue_indicator']}\n"
            company_section += "\n"

        # History
        if company_intel.get("history"):
            hist = company_intel["history"]
            company_section += "### History\n\n"
            if hist.get("founded"):
                company_section += f"- **Founded:** {hist['founded']}\n"
            if hist.get("ownership_type"):
                company_section += f"- **Ownership:** {hist['ownership_type']}\n"
            if hist.get("background"):
                company_section += f"- {hist['background']}\n"
            company_section += "\n"

        # Recent News
        if company_intel.get("recent_news"):
            company_section += "### Recent News\n\n"
            for news in company_intel["recent_news"]:
                company_section += f"- {news}\n"
            company_section += "\n"

        # Strategic Priorities
        if company_intel.get("strategic_priorities"):
            company_section += "### Strategic Priorities\n\n"
            for priority in company_intel["strategic_priorities"]:
                company_section += f"- {priority}\n"
            company_section += "\n"

        # Sales Opportunities
        if company_intel.get("sales_opportunities"):
            company_section += "### Sales Opportunities\n\n"
            for opp in company_intel["sales_opportunities"]:
                company_section += f"- {opp}\n"
            company_section += "\n"

    # Build owner/decision maker section from Gemini person intel
    owner_section = ""
    if person_intel:
        owner_section = "\n## Decision Maker Intel\n\n"
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
{company_section}
## Supplementary Research

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
    """Save profile to projects/leads/data/leads/{slug}/profile.md."""
    profile_dir = DATA_DIR / slug
    profile_dir.mkdir(parents=True, exist_ok=True)

    profile_path = profile_dir / "profile.md"
    profile_path.write_text(profile)

    return f"projects/leads/data/leads/{slug}/profile.md"


def save_lead(slug: str, story: str, data: Dict) -> tuple[str, str]:
    """
    Save both profile.md and data.yaml for a lead.

    Args:
        slug: URL-safe business name slug
        story: Markdown content for profile.md
        data: Dict to serialize as data.yaml

    Returns:
        Tuple of (profile_path, data_path)
    """
    profile_dir = DATA_DIR / slug
    profile_dir.mkdir(parents=True, exist_ok=True)

    # Save narrative profile
    profile_path = profile_dir / "profile.md"
    profile_path.write_text(story)

    # Save structured data
    data_path = profile_dir / "data.yaml"
    with open(data_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return f"projects/leads/data/leads/{slug}/profile.md", f"projects/leads/data/leads/{slug}/data.yaml"


def generate_story_profile(
    business: Dict,
    search_story: str,
    website_story: str,
    search_data: Dict,
    website_data: Dict,
    person_intel: Optional[Dict[str, Any]] = None,
    company_intel: Optional[Dict[str, Any]] = None,
    places_intel: Optional[Dict[str, Any]] = None
) -> tuple[str, Dict, List[str]]:
    """
    Generate story-first profile and structured data.

    Returns:
        Tuple of (profile_markdown, structured_data_dict, quality_flags)
    """
    name = business.get("business_name", "Unknown Business")
    address = business.get("address", "N/A")
    phone = business.get("phone", "N/A")
    website = business.get("website", "N/A")
    rating = business.get("rating", "N/A")
    review_count = business.get("review_count", 0)
    business_type = business.get("business_type", "N/A")

    # Override with Places data if available (more reliable)
    if places_intel:
        if places_intel.get("rating"):
            rating = places_intel["rating"]
        if places_intel.get("review_count"):
            review_count = places_intel["review_count"]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Quality flags based on new criteria
    quality_flags = []
    if places_intel and places_intel.get("reviews"):
        quality_flags.append("places_enriched")
    if places_intel and places_intel.get("employees_from_reviews"):
        quality_flags.append("employees_identified")
    if search_story and len(search_story) > 200:
        quality_flags.append("story_complete")
    if "conversation" in search_story.lower() or "?" in search_story:
        quality_flags.append("hooks_found")
    if search_data.get("leadership") or (company_intel and company_intel.get("leadership")):
        quality_flags.append("leadership_identified")
    if website_story and len(website_story) > 100:
        quality_flags.append("website_analyzed")
    if search_data.get("company") or search_data.get("services"):
        quality_flags.append("structured_complete")

    quality = "high" if len(quality_flags) >= 4 else "partial" if len(quality_flags) >= 2 else "minimal"

    # Build The Story section
    story_section = search_story if search_story else "No background story available."

    # Build Business Operations section
    operations_section = website_story if website_story else "Website not analyzed."

    # Build conversation starters - extract from search_story or generate from data
    conversation_starters = ""
    if search_data.get("conversation_hooks"):
        hooks = search_data["conversation_hooks"]
        if hooks:
            conversation_starters = "\n".join(f"{i+1}. {hook}" for i, hook in enumerate(hooks[:3]))

    # Build Quick Reference from structured data
    quick_ref_parts = []

    # From search_data
    if search_data.get("company", {}).get("founded"):
        quick_ref_parts.append(f"- **Founded:** {search_data['company']['founded']}")

    # Leadership
    leaders = []
    if search_data.get("leadership"):
        for l in search_data["leadership"][:3]:
            if l.get("name"):
                leader_str = l["name"]
                if l.get("role"):
                    leader_str += f" ({l['role']})"
                leaders.append(leader_str)
    if leaders:
        quick_ref_parts.append(f"- **Owners:** {', '.join(leaders)}")

    # Team
    team = []
    combined_team = (search_data.get("team") or []) + (website_data.get("team") or [])
    for t in combined_team[:5]:
        if t.get("name"):
            team_str = t["name"]
            if t.get("role"):
                team_str += f" ({t['role']})"
            team.append(team_str)
    if team:
        quick_ref_parts.append(f"- **Team:** {', '.join(team)}")

    # Location
    quick_ref_parts.append(f"- **Location:** {address}")

    # Services
    services = search_data.get("services") or website_data.get("services") or []
    if services:
        quick_ref_parts.append(f"- **Services:** {', '.join(services[:5])}")

    # Reputation
    rep_parts = []
    if rating and rating != "N/A":
        rep_parts.append(f"{rating} stars")
    if review_count:
        rep_parts.append(f"{review_count} reviews")
    if search_data.get("reputation", {}).get("signals"):
        rep_parts.extend(search_data["reputation"]["signals"][:2])
    if rep_parts:
        quick_ref_parts.append(f"- **Reputation:** {', '.join(rep_parts)}")

    quick_reference = "\n".join(quick_ref_parts) if quick_ref_parts else "- No structured data available"

    # Build sources list based on what was actually used
    sources = []
    if places_intel:
        sources.append("google_places")
    sources.append("serper")
    sources.append("playwright")

    # Build profile markdown
    profile = f"""# {name}

> **Scout Quality:** {quality} | **Scouted:** {timestamp}

## The Story

{story_section}

## Conversation Starters

{conversation_starters if conversation_starters else "1. (No specific hooks identified)"}

## Quick Reference

{quick_reference}

## Business Operations

{operations_section}

---

## Scout Metadata
- **Quality Flags:** {', '.join(quality_flags) if quality_flags else 'none'}
- **Sources:** {', '.join(sources)}
- **LLM:** Groq distillation
"""

    # Merge employees from Places reviews into team
    places_employees = []
    if places_intel and places_intel.get("employees_from_reviews"):
        places_employees = [
            {"name": e["name"], "role": e.get("role", "staff"), "source": "places_reviews"}
            for e in places_intel["employees_from_reviews"]
        ]

    all_team = places_employees + combined_team
    # Deduplicate by name
    seen_names = set()
    unique_team = []
    for member in all_team:
        if member.get("name") and member["name"] not in seen_names:
            seen_names.add(member["name"])
            unique_team.append(member)

    # Build structured data dict for data.yaml
    structured_data = {
        "company": {
            "name": name,
            "founded": search_data.get("company", {}).get("founded"),
            "acquired": search_data.get("company", {}).get("acquired"),
            "location": {
                "address": address,
                "city": extract_city_from_address(address),
                "state": "NM",  # Default, could be extracted
                "zip": None
            }
        },
        "leadership": search_data.get("leadership") or [],
        "team": unique_team[:15],
        "contact": {
            "phone": phone,
            "website": website,
            "email": website_data.get("contact", {}).get("email")
        },
        "services": services[:20] if services else [],
        "licenses": search_data.get("licenses") or [],
        "reputation": {
            "rating": rating if rating != "N/A" else None,
            "review_count": review_count,
            "signals": search_data.get("reputation", {}).get("signals") or [],
            "differentiators": search_data.get("reputation", {}).get("differentiators") or []
        },
        "places": {
            "place_id": places_intel.get("place_id") if places_intel else None,
            "business_status": places_intel.get("business_status") if places_intel else None,
            "hours": places_intel.get("hours") if places_intel else None,
            "types": places_intel.get("types") if places_intel else [],
            "google_maps_url": places_intel.get("google_maps_url") if places_intel else None,
        } if places_intel else None,
        "conversation_hooks": search_data.get("conversation_hooks") or [],
        "metadata": {
            "scouted_at": timestamp,
            "quality": quality,
            "quality_flags": quality_flags,
            "sources": sources
        }
    }

    return profile, structured_data, quality_flags


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

    Uses story-first distillation to produce both:
    - profile.md (narrative for humans)
    - data.yaml (structured for automation)

    Pipeline order:
    1. Places API (reliable, structured, reviews with employee names)
    2. SearxNG searches (uses employee names from Places for targeted queries)
    3. Groq distillation (story + structured)
    4. Website scraping
    5. Website distillation
    6. Profile generation
    """
    business_id = business.get("id")
    name = business.get("business_name", "Unknown")
    website = business.get("website")
    address = business.get("address", "")

    city = extract_city_from_address(address)
    logger.info(f"Scouting: {name} ({city})")

    # Phase 1: Google Places API (PRIMARY - reliable, structured data)
    logger.info("  Phase 1: Google Places API...")
    places_intel = gather_places_intel(name, address)
    if places_intel:
        logger.info(f"    Places data: {places_intel.get('rating')}★, {places_intel.get('review_count')} reviews")
        employees = places_intel.get("employees_from_reviews", [])
        if employees:
            logger.info(f"    Employees from reviews: {', '.join(e['name'] for e in employees[:5])}")
    else:
        logger.info("    No Places data found")

    # Phase 2: SearxNG searches (supplementary - use employee names if available)
    logger.info("  Phase 2: Gathering search intelligence...")
    # Build targeted queries using employee names from Places reviews
    employee_names = []
    if places_intel and places_intel.get("employees_from_reviews"):
        employee_names = [e["name"] for e in places_intel["employees_from_reviews"]
                        if e.get("role") in ("owner", "management", "sales", None)][:3]

    search_intel = gather_search_intel(name, city, employee_names=employee_names)

    # Phase 3a: Distill search results into story (narrative)
    logger.info("  Phase 3a: Distilling search story...")
    search_story = distill_search_story(search_intel, provider)

    # Phase 3b: Distill search results into structured data
    logger.info("  Phase 3b: Distilling search structured...")
    search_data = distill_search_structured(search_intel, provider)

    # Gemini is deprioritized - manual enrichment for TOP 50 leads only
    company_intel = None
    person_intel = None

    # Try to enrich leadership from structured data if found
    if search_data.get("leadership"):
        leaders = search_data["leadership"]
        if leaders and len(leaders) > 0:
            leader = leaders[0]
            if leader.get("name"):
                name_parts = leader["name"].split()
                if len(name_parts) >= 2:
                    first, last = name_parts[0], name_parts[-1]
                    logger.info(f"  Phase 4: Enriching owner via Gemini: {first} {last}")
                    person_intel = enrich_person_with_gemini(
                        first=first,
                        last=last,
                        company=name,
                        location=city
                    )
                    if person_intel:
                        logger.info("    Gemini enrichment successful")
                    else:
                        logger.info("    Gemini enrichment returned no data")
    if not person_intel:
        logger.info("  Phase 4: No owner name found to enrich")

    # Phase 5: Scrape website
    website_pages = {}
    if website:
        logger.info(f"  Phase 5: Scraping website...")
        website_pages = scrape_website(website)
        logger.info(f"    Found {len(website_pages)} pages")
    else:
        logger.info("  Phase 5: No website to scrape")

    # Phase 6a: Distill website story
    website_story = ""
    if website_pages:
        logger.info("  Phase 6a: Distilling website story...")
        website_story = distill_website_story(website_pages, provider)

    # Phase 6b: Distill website structured data
    website_data = {}
    if website_pages:
        logger.info("  Phase 6b: Distilling website structured...")
        website_data = distill_website_structured(website_pages, provider)

    # Phase 7: Generate story-first profile and structured data
    slug = slugify(name)
    profile, structured_data, quality_flags = generate_story_profile(
        business, search_story, website_story, search_data, website_data,
        person_intel, company_intel, places_intel
    )

    # Determine quality from flags
    quality = "high" if len(quality_flags) >= 4 else "partial" if len(quality_flags) >= 2 else "minimal"

    # Save both profile.md and data.yaml
    profile_path, data_path = save_lead(slug, profile, structured_data)
    logger.info(f"  Profile saved: {profile_path} (quality: {quality})")
    logger.info(f"  Data saved: {data_path}")

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
