#!/usr/bin/env python3
"""
Google Places Business Discovery Pipeline

Discovers businesses near parcels using Google Places API.
Stores results in crm.businesses with source='google_places'.

Usage:
    python3 places_discovery.py          # Full run (all parcels)
    python3 places_discovery.py --test   # Test mode (10 parcels)
    python3 places_discovery.py --resume # Resume from checkpoint
    python3 places_discovery.py --limit 50  # Process 50 parcels
    python3 places_discovery.py --enrich    # Call details API (costs more)
    python3 places_discovery.py --cost-limit 5.00  # Stop at $5 estimated cost
    python3 places_discovery.py --query "SELECT ..." --limit 10  # Custom query
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Optional, Tuple

try:
    import requests
except ImportError:
    print("Error: requests library required. Install with: pip install requests")
    sys.exit(1)

# Configuration (8000=dev, 8006=prod)
API_BASE = os.environ.get("API_BASE", "http://192.168.1.17:8006")
RATE_LIMIT = 0.5  # seconds between Places API requests
CHECKPOINT_FILE = "/tmp/places_discovery_checkpoint.json"
DEFAULT_RADIUS = 100  # meters (parcels are precise)
MAX_RESULTS = 5

# Cost estimation (per 1000 requests)
COST_NEARBY = 0.032  # $32 per 1K for Nearby Search (New)
COST_DETAILS = 0.017  # $17 per 1K for Place Details (Basic)

# Default SQL query for facilities with addresses
# Note: query must output columns: facility_id, pin, address
# Uses facility layer to avoid duplicate API calls for shared addresses
DEFAULT_QUERY = """
    SELECT
        facility_id,
        pin,
        CONCAT(address, ', ', city, ', ', state) as address
    FROM crm.business_targets_v
"""

# Legacy query using lat/lng for backward compatibility
LEGACY_QUERY = """
    SELECT
        "PIN" as pin,
        ST_Y(ST_Centroid(geometry)) as lat,
        ST_X(ST_Centroid(geometry)) as lng
    FROM business_parcels_target
"""

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Address Matching
# ============================================================================

def parse_address(address: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract street number and normalized street name from an address.

    Returns:
        Tuple of (street_number, street_name) or (None, None) if parsing fails

    Examples:
        "6612 ACOMA RD SE" -> ("6612", "ACOMA RD SE")
        "6608 Acoma Rd SE, Albuquerque, NM 87108, USA" -> ("6608", "ACOMA RD SE")
    """
    if not address:
        return None, None

    # Normalize: uppercase, remove extra spaces
    addr = " ".join(address.upper().split())

    # Remove city, state, zip suffix (everything after first comma)
    if "," in addr:
        addr = addr.split(",")[0].strip()

    # Extract street number (first numeric token)
    match = re.match(r'^(\d+)\s+(.+)$', addr)
    if not match:
        return None, None

    street_num = match.group(1)
    street_name = match.group(2).strip()

    # Normalize street suffixes
    street_name = re.sub(r'\bSTREET\b', 'ST', street_name)
    street_name = re.sub(r'\bAVENUE\b', 'AV', street_name)
    street_name = re.sub(r'\bAVE\b', 'AV', street_name)
    street_name = re.sub(r'\bROAD\b', 'RD', street_name)
    street_name = re.sub(r'\bBOULEVARD\b', 'BLVD', street_name)
    street_name = re.sub(r'\bDRIVE\b', 'DR', street_name)
    street_name = re.sub(r'\bLANE\b', 'LN', street_name)
    street_name = re.sub(r'\bCOURT\b', 'CT', street_name)
    street_name = re.sub(r'\bPLACE\b', 'PL', street_name)

    # Remove unit/suite info
    street_name = re.sub(r'\s+(UNIT|STE|SUITE|APT|#)\s*\S*$', '', street_name)

    return street_num, street_name


def addresses_match(facility_addr: str, business_addr: str, tolerance: int = 10) -> bool:
    """
    Check if a business address matches the facility address.

    Args:
        facility_addr: The facility/parcel address
        business_addr: The business address from Places API
        tolerance: Max difference in street numbers (for large buildings)

    Returns:
        True if addresses match within tolerance
    """
    fac_num, fac_street = parse_address(facility_addr)
    biz_num, biz_street = parse_address(business_addr)

    if not all([fac_num, fac_street, biz_num, biz_street]):
        return False

    # Street names must match
    if fac_street != biz_street:
        return False

    # Street numbers must be within tolerance
    try:
        num_diff = abs(int(fac_num) - int(biz_num))
        return num_diff <= tolerance
    except ValueError:
        return fac_num == biz_num


# ============================================================================
# Cost Tracking
# ============================================================================

class CostTracker:
    """Track API costs and enforce limits."""

    def __init__(self, cost_limit: Optional[float] = None):
        self.nearby_calls = 0
        self.details_calls = 0
        self.cost_limit = cost_limit

    def add_nearby(self) -> None:
        self.nearby_calls += 1

    def add_details(self) -> None:
        self.details_calls += 1

    @property
    def estimated_cost(self) -> float:
        return (self.nearby_calls * COST_NEARBY / 1000 +
                self.details_calls * COST_DETAILS / 1000)

    def check_limit(self) -> bool:
        """Return True if we should stop due to cost limit."""
        if self.cost_limit is None:
            return False
        return self.estimated_cost >= self.cost_limit

    def summary(self) -> str:
        return (f"Nearby: {self.nearby_calls}, Details: {self.details_calls}, "
                f"Est. cost: ${self.estimated_cost:.4f}")


# ============================================================================
# Google Places API Functions
# ============================================================================

def search_places_by_address(
    address: str,
    max_results: int = 5,
    cost_tracker: Optional[CostTracker] = None
) -> list[dict]:
    """
    Search for places by address using Places API Text Search.

    Args:
        address: Street address to search (e.g., "123 Main St, Albuquerque, NM")
        max_results: Max places to return (default 5)
        cost_tracker: Optional cost tracker

    Returns:
        List of parsed place dictionaries
    """
    payload = {
        "query": f"businesses at {address}",
        "max_results": max_results
    }

    try:
        response = requests.post(
            f"{API_BASE}/places/search",
            json=payload,
            timeout=30
        )

        # Track cost (text search uses same tier as nearby)
        if cost_tracker:
            cost_tracker.add_nearby()

        # Handle rate limiting with retry
        if response.status_code == 429:
            for delay in [1, 5, 30]:
                logger.warning(f"Rate limited, waiting {delay}s...")
                time.sleep(delay)
                response = requests.post(
                    f"{API_BASE}/places/search",
                    json=payload,
                    timeout=30
                )
                if cost_tracker:
                    cost_tracker.add_nearby()
                if response.status_code != 429:
                    break
            else:
                logger.error("Rate limit exceeded after retries")
                return []

        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            logger.warning(f"Places API error: {data.get('message')}")
            return []

        places = data.get("places", [])
        logger.debug(f"Places API returned {len(places)} places for address: {address}")

        return places

    except requests.Timeout:
        logger.warning(f"Places API timed out for address: {address}")
        return []
    except requests.RequestException as e:
        logger.warning(f"Places API request failed for address: {address}: {e}")
        return []
    except (ValueError, KeyError) as e:
        logger.warning(f"Failed to parse Places API response: {e}")
        return []


def search_places_nearby(
    lat: float,
    lng: float,
    radius: int = 100,
    max_results: int = 5,
    cost_tracker: Optional[CostTracker] = None
) -> list[dict]:
    """
    Search for places near coordinates using Places API (legacy method).

    Args:
        lat: Latitude
        lng: Longitude
        radius: Search radius in meters (default 100)
        max_results: Max places to return (default 5)
        cost_tracker: Optional cost tracker

    Returns:
        List of parsed place dictionaries
    """
    payload = {
        "latitude": lat,
        "longitude": lng,
        "radius": radius,
        "max_results": max_results
    }

    try:
        response = requests.post(
            f"{API_BASE}/places/nearby",
            json=payload,
            timeout=30
        )

        # Track cost
        if cost_tracker:
            cost_tracker.add_nearby()

        # Handle rate limiting with retry
        if response.status_code == 429:
            for delay in [1, 5, 30]:
                logger.warning(f"Rate limited, waiting {delay}s...")
                time.sleep(delay)
                response = requests.post(
                    f"{API_BASE}/places/nearby",
                    json=payload,
                    timeout=30
                )
                if cost_tracker:
                    cost_tracker.add_nearby()
                if response.status_code != 429:
                    break
            else:
                logger.error("Rate limit exceeded after retries")
                return []

        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            logger.warning(f"Places API error: {data.get('message')}")
            return []

        places = data.get("places", [])
        logger.debug(f"Places API returned {len(places)} places for ({lat}, {lng})")

        return places

    except requests.Timeout:
        logger.warning(f"Places API timed out for ({lat}, {lng})")
        return []
    except requests.RequestException as e:
        logger.warning(f"Places API request failed for ({lat}, {lng}): {e}")
        return []
    except (ValueError, KeyError) as e:
        logger.warning(f"Failed to parse Places API response: {e}")
        return []


def get_place_details(
    place_id: str,
    cost_tracker: Optional[CostTracker] = None
) -> Optional[dict]:
    """
    Get detailed information about a place.

    Args:
        place_id: Google Place ID
        cost_tracker: Optional cost tracker

    Returns:
        Place details dict or None
    """
    payload = {
        "place_id": place_id,
        "include_reviews": False
    }

    try:
        response = requests.post(
            f"{API_BASE}/places/details",
            json=payload,
            timeout=30
        )

        if cost_tracker:
            cost_tracker.add_details()

        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            logger.warning(f"Details API error for {place_id}: {data.get('message')}")
            return None

        return data.get("place")

    except requests.RequestException as e:
        logger.warning(f"Details API request failed for {place_id}: {e}")
        return None


def parse_place_to_business(place: dict, pin: str, facility_id: str = None) -> dict:
    """
    Map a Places API response to a crm.businesses record.

    Schema mapping:
    | Places API       | crm.businesses    |
    |------------------|-------------------|
    | place_id         | result_url        |
    | name             | business_name     |
    | phone            | phone             |
    | website          | website           |
    | primary_type     | business_type     |
    | rating           | rating            |
    | rating_count     | review_count      |
    | location.lat/lng | latitude/longitude|
    | types            | types (TEXT[])    |
    | business_status  | business_status   |
    | google_maps_url  | google_maps_url   |
    | hours            | hours             |
    | (full response)  | attributes (JSONB)|
    | -                | source='google_places' |
    """
    place_id = place.get("place_id", place.get("id", ""))

    # Extract name - can be displayName.text (new API) or name (our normalized response)
    display_name = place.get("displayName")
    if isinstance(display_name, dict):
        name = display_name.get("text", "")
    else:
        name = place.get("name", "")

    # Extract phone - can be in various formats
    phone = (place.get("phone") or
             place.get("formatted_phone_number") or
             place.get("national_phone_number"))

    # Build address from components
    address = place.get("address") or place.get("formatted_address")

    # Website can be in different fields
    website = place.get("website") or place.get("websiteUri") or ""

    # Extract location coordinates
    location = place.get("location", {})
    latitude = location.get("lat") if isinstance(location, dict) else None
    longitude = location.get("lng") if isinstance(location, dict) else None

    # Extract types array
    types = place.get("types", [])

    # Extract hours - format weekday_text if available
    hours_data = place.get("hours", {})
    if isinstance(hours_data, dict):
        weekday_text = hours_data.get("weekday_text", [])
        hours = "\n".join(weekday_text) if weekday_text else None
    else:
        hours = None

    # Build attributes with full raw data for future extraction
    # Sanitize to remove problematic Unicode characters for SQL_ASCII databases
    attributes = sanitize_dict({
        "raw_response": place,
        "address_components": place.get("address_components"),
        "open_now": hours_data.get("open_now") if isinstance(hours_data, dict) else None,
    })

    return {
        "pin": pin,
        "facility_id": facility_id,
        "business_name": name[:255] if name else None,
        "phone": phone,
        "email": None,  # Places API doesn't provide email
        "website": website[:500] if website else None,
        "business_type": place.get("primary_type") or place.get("primaryType"),
        "result_url": f"places://{place_id}",
        "result_title": name,
        "address": address,
        "source": "google_places",
        "confidence_score": 0.9,  # Slightly lower than OSM (paid API, might have stale data)
        "rating": place.get("rating"),
        "review_count": place.get("rating_count") or place.get("userRatingCount"),
        # New fields
        "latitude": latitude,
        "longitude": longitude,
        "business_status": place.get("business_status"),
        "types": types,
        "google_maps_url": place.get("google_maps_url"),
        "hours": hours,
        "attributes": attributes,
    }


# ============================================================================
# Database Functions
# ============================================================================

def fetch_parcels(
    query: str,
    limit: Optional[int] = None,
    offset_facility: Optional[str] = None
) -> list[dict]:
    """Fetch parcels/facilities using provided SQL query."""
    sql = query.strip()

    # Remove trailing semicolon if present
    sql = sql.rstrip(';')

    # Build clauses
    clauses = []
    if offset_facility:
        clauses.append(f"facility_id > '{offset_facility}'")

    # Add WHERE clause if we have conditions
    if clauses:
        if " WHERE " in sql.upper():
            sql += f" AND ({' AND '.join(clauses)})"
        else:
            sql += f" WHERE {' AND '.join(clauses)}"

    sql += " ORDER BY facility_id"

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
            logger.info(f"Fetched {data.get('rows', 0)} parcels")
            return data.get("data", [])
        else:
            logger.error(f"Query failed: {data}")
            return []

    except requests.RequestException as e:
        logger.error(f"Failed to fetch parcels: {e}")
        return []


def sanitize_text(text: str) -> str:
    """Sanitize text for SQL insertion, handling Unicode and escaping."""
    if not text:
        return text
    # Replace problematic Unicode spaces with regular spaces
    text = text.replace('\u202f', ' ')  # narrow no-break space
    text = text.replace('\u2009', ' ')  # thin space
    text = text.replace('\u00a0', ' ')  # non-breaking space
    text = text.replace('\u2013', '-')  # en dash
    text = text.replace('\u2014', '-')  # em dash
    # Escape backslashes and single quotes for SQL
    text = text.replace("\\", "\\\\").replace("'", "''")
    return text


def sanitize_dict(obj: Any) -> Any:
    """Recursively sanitize all strings in a dict/list structure."""
    if isinstance(obj, dict):
        return {k: sanitize_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_dict(v) for v in obj]
    elif isinstance(obj, str):
        # Just replace Unicode, don't SQL-escape here (json.dumps handles that)
        obj = obj.replace('\u202f', ' ')
        obj = obj.replace('\u2009', ' ')
        obj = obj.replace('\u00a0', ' ')
        obj = obj.replace('\u2013', '-')
        obj = obj.replace('\u2014', '-')
        return obj
    else:
        return obj


def save_business(business: dict) -> bool:
    """Save a business record to the CRM database."""
    # Filter out None values, but keep empty lists
    filtered = {k: v for k, v in business.items() if v is not None}
    columns = list(filtered.keys())
    placeholders = []

    for col in columns:
        val = filtered[col]
        if isinstance(val, (int, float)):
            placeholders.append(str(val))
        elif isinstance(val, dict):
            # JSONB fields (attributes) - use ensure_ascii to handle Unicode
            json_str = json.dumps(val, ensure_ascii=True)
            escaped = json_str.replace("'", "''")
            placeholders.append(f"'{escaped}'::jsonb")
        elif isinstance(val, list):
            # TEXT[] array fields (types)
            escaped_items = [sanitize_text(str(v)) for v in val]
            array_literal = "ARRAY[" + ", ".join(f"'{item}'" for item in escaped_items) + "]::text[]"
            placeholders.append(array_literal)
        else:
            escaped = sanitize_text(str(val))
            placeholders.append(f"'{escaped}'")

    sql = f"""
        INSERT INTO crm.businesses ({", ".join(columns)})
        VALUES ({", ".join(placeholders)})
        ON CONFLICT (pin, result_url) DO UPDATE SET
            facility_id = EXCLUDED.facility_id,
            business_name = EXCLUDED.business_name,
            phone = EXCLUDED.phone,
            website = EXCLUDED.website,
            business_type = EXCLUDED.business_type,
            rating = EXCLUDED.rating,
            review_count = EXCLUDED.review_count,
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude,
            business_status = EXCLUDED.business_status,
            types = EXCLUDED.types,
            google_maps_url = EXCLUDED.google_maps_url,
            hours = EXCLUDED.hours,
            attributes = EXCLUDED.attributes,
            updated_at = NOW()
    """

    try:
        response = requests.post(
            f"{API_BASE}/postgresql/query",
            json={"sql": sql},
            timeout=30
        )
        if response.status_code != 200:
            logger.warning(f"SQL error: {response.text[:500]}")
        response.raise_for_status()
        data = response.json()
        return data.get("success", False)

    except requests.RequestException as e:
        logger.warning(f"Failed to save business: {e}")
        return False


# ============================================================================
# Checkpoint Functions
# ============================================================================

def load_checkpoint() -> Optional[dict]:
    """Load checkpoint data including last PIN and cost info."""
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_checkpoint(facility_id: str, stats: dict, cost_tracker: CostTracker) -> None:
    """Save checkpoint with last processed facility_id, stats, and cost info."""
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump({
                "last_facility": facility_id,
                "timestamp": datetime.now().isoformat(),
                "stats": stats,
                "cost": {
                    "nearby_calls": cost_tracker.nearby_calls,
                    "details_calls": cost_tracker.details_calls,
                    "estimated_cost": cost_tracker.estimated_cost
                }
            }, f, indent=2)
    except IOError as e:
        logger.warning(f"Failed to save checkpoint: {e}")


# ============================================================================
# Main Processing
# ============================================================================

def process_parcel(
    parcel: dict,
    radius: int,
    max_results: int,
    enrich: bool,
    cost_tracker: CostTracker,
    use_address: bool = True
) -> list[dict]:
    """
    Query Places API for businesses at/near a facility/parcel.

    Args:
        parcel: Dict with 'facility_id', 'pin' and either 'address' or 'lat'/'lng'
        radius: Search radius in meters (only used for lat/lng mode)
        max_results: Max places to return
        enrich: Whether to call details API
        cost_tracker: Cost tracker instance
        use_address: If True, use address-based text search (default)
                     If False, use legacy lat/lng nearby search

    Returns list of business records ready for save_business().
    """
    pin = parcel.get("pin")
    facility_id = parcel.get("facility_id")

    if use_address:
        # Address-based text search (preferred)
        address = parcel.get("address")
        if not address or address.strip() == "":
            logger.warning(f"PIN {pin}: Missing address, skipping")
            return []

        # Clean up address (remove extra spaces)
        address = " ".join(address.split())

        places = search_places_by_address(
            address,
            max_results=max_results,
            cost_tracker=cost_tracker
        )
    else:
        # Legacy lat/lng nearby search
        lat = parcel.get("lat")
        lng = parcel.get("lng")

        if lat is None or lng is None:
            logger.warning(f"PIN {pin}: Missing coordinates, skipping")
            return []

        places = search_places_nearby(
            lat, lng,
            radius=radius,
            max_results=max_results,
            cost_tracker=cost_tracker
        )

    if not places:
        return []

    # Filter to only businesses that match the facility address
    facility_addr = parcel.get("address", "")
    matched_places = []
    for place in places:
        biz_addr = place.get("address") or place.get("formatted_address") or ""
        if addresses_match(facility_addr, biz_addr):
            matched_places.append(place)
        else:
            biz_name = place.get("name", "Unknown")
            logger.debug(f"  Filtered out '{biz_name}' - address mismatch: {biz_addr}")

    if not matched_places:
        logger.debug(f"  No businesses matched facility address")
        return []

    logger.info(f"  {len(matched_places)}/{len(places)} businesses matched address")

    # Convert to business records
    businesses = []
    for place in matched_places:
        # Optionally enrich with details
        if enrich:
            place_id = place.get("place_id") or place.get("id")
            if place_id:
                details = get_place_details(place_id, cost_tracker)
                if details:
                    place.update(details)

        business = parse_place_to_business(place, pin, facility_id)
        if business.get("business_name"):  # Only include named places
            businesses.append(business)

    return businesses


def main():
    parser = argparse.ArgumentParser(description="Google Places Business Discovery Pipeline")
    parser.add_argument("--test", action="store_true", help="Test mode: process only 10 parcels")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--limit", type=int, help="Limit number of parcels to process")
    parser.add_argument("--query", type=str, help="Custom SQL query for parcels (must return pin, address)")
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS, help=f"Search radius in meters (default: {DEFAULT_RADIUS}, only for legacy mode)")
    parser.add_argument("--max-results", type=int, default=MAX_RESULTS, help=f"Max places per parcel (default: {MAX_RESULTS})")
    parser.add_argument("--enrich", action="store_true", help="Call details API for each place (costs more)")
    parser.add_argument("--cost-limit", type=float, help="Stop at estimated cost threshold (e.g., 5.00)")
    parser.add_argument("--legacy", action="store_true", help="Use legacy lat/lng nearby search instead of address-based text search")
    args = parser.parse_args()

    # Determine limit
    limit = 10 if args.test else args.limit

    # Determine search mode
    use_address = not args.legacy

    # Use custom query or default (different defaults for address vs legacy mode)
    if args.query:
        query = args.query
    else:
        query = DEFAULT_QUERY if use_address else LEGACY_QUERY

    # Initialize cost tracker
    cost_tracker = CostTracker(cost_limit=args.cost_limit)

    # Check for resume
    offset_facility = None
    if args.resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            offset_facility = checkpoint.get("last_facility")
            # Restore cost tracking from checkpoint
            if "cost" in checkpoint:
                cost_tracker.nearby_calls = checkpoint["cost"].get("nearby_calls", 0)
                cost_tracker.details_calls = checkpoint["cost"].get("details_calls", 0)
            logger.info(f"Resuming from facility: {offset_facility}")
            logger.info(f"Previous cost: {cost_tracker.summary()}")
        else:
            logger.info("No checkpoint found, starting from beginning")

    # Fetch facilities
    logger.info("Fetching facilities...")
    parcels = fetch_parcels(query=query, limit=limit, offset_facility=offset_facility)

    if not parcels:
        logger.error("No facilities to process")
        return 1

    search_mode = "address-based text search" if use_address else f"lat/lng nearby search ({args.radius}m radius)"
    logger.info(f"Processing {len(parcels)} facilities with {search_mode}...")
    if args.enrich:
        logger.info("Enrichment enabled (details API will be called)")
    if args.cost_limit:
        logger.info(f"Cost limit: ${args.cost_limit:.2f}")

    # Stats
    stats = {
        "total_processed": 0,
        "total_businesses": 0,
        "parcels_with_results": 0,
        "errors": 0
    }

    for i, parcel in enumerate(parcels):
        # Check cost limit
        if cost_tracker.check_limit():
            logger.warning(f"Cost limit reached: {cost_tracker.summary()}")
            break

        pin = parcel.get("pin")
        facility_id = parcel.get("facility_id")

        # Log based on search mode
        if use_address:
            address = parcel.get("address", "").strip()
            address_short = address[:50] + "..." if len(address) > 50 else address
            logger.info(f"[{i+1}/{len(parcels)}] {facility_id}: {address_short}")
        else:
            lat = parcel.get("lat")
            lng = parcel.get("lng")
            logger.info(f"[{i+1}/{len(parcels)}] {facility_id}: ({lat:.6f}, {lng:.6f})")

        try:
            businesses = process_parcel(
                parcel,
                args.radius,
                args.max_results,
                args.enrich,
                cost_tracker,
                use_address=use_address
            )

            if businesses:
                stats["parcels_with_results"] += 1
                for biz in businesses:
                    if save_business(biz):
                        stats["total_businesses"] += 1
                        logger.info(f"  + {biz.get('business_name')} ({biz.get('business_type')})")
                    else:
                        stats["errors"] += 1

            stats["total_processed"] += 1

            # Save checkpoint after each facility
            save_checkpoint(facility_id, stats, cost_tracker)

        except Exception as e:
            logger.error(f"  Error processing {facility_id}: {e}")
            stats["errors"] += 1

        # Rate limit (except for last parcel)
        if i < len(parcels) - 1:
            logger.debug(f"  Rate limiting: {RATE_LIMIT}s")
            time.sleep(RATE_LIMIT)

    # Final summary
    logger.info("=" * 50)
    logger.info("Places Discovery Complete")
    logger.info(f"  Facilities processed: {stats['total_processed']}")
    logger.info(f"  Facilities with results: {stats['parcels_with_results']}")
    logger.info(f"  Businesses discovered: {stats['total_businesses']}")
    logger.info(f"  Errors: {stats['errors']}")
    logger.info(f"  Cost: {cost_tracker.summary()}")
    logger.info("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
