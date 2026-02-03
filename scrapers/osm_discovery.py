#!/usr/bin/env python3
"""
OSM Business Discovery Pipeline

Discovers businesses near parcels using OpenStreetMap Overpass API.
Stores results in crm.businesses with source='osm'.

Usage:
    python3 osm_discovery.py          # Full run (all parcels)
    python3 osm_discovery.py --test   # Test mode (10 parcels)
    python3 osm_discovery.py --resume # Resume from checkpoint
    python3 osm_discovery.py --limit 50  # Process 50 parcels
    python3 osm_discovery.py --query "SELECT ..." --limit 10  # Custom query
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from typing import Any, Optional

try:
    import requests
except ImportError:
    print("Error: requests library required. Install with: pip install requests")
    sys.exit(1)

# Configuration
API_BASE = "http://192.168.1.17:8006"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
RATE_LIMIT = 10.0  # seconds between Overpass requests
CHECKPOINT_FILE = "/tmp/osm_discovery_checkpoint.json"
DEFAULT_RADIUS = 200  # meters

# Default SQL query for parcels with coordinates
# Note: query must output columns: pin, lat, lng
DEFAULT_QUERY = """
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
# OSM Overpass Query Functions
# ============================================================================

def query_overpass(lat: float, lng: float, radius_m: int = 200) -> list[dict]:
    """
    Query Overpass API for named POIs near coordinates.

    Args:
        lat: Latitude
        lng: Longitude
        radius_m: Search radius in meters (default 200)

    Returns:
        List of parsed business dictionaries
    """
    query = f"""
    [out:json][timeout:25];
    (
      nwr(around:{radius_m},{lat},{lng})["name"]["shop"];
      nwr(around:{radius_m},{lat},{lng})["name"]["amenity"]["amenity"!~"parking|bench|waste_basket|toilets|drinking_water"];
      nwr(around:{radius_m},{lat},{lng})["name"]["office"];
      nwr(around:{radius_m},{lat},{lng})["name"]["craft"];
    );
    out body qt;
    """

    try:
        response = requests.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        elements = data.get("elements", [])
        logger.debug(f"Overpass returned {len(elements)} elements for ({lat}, {lng})")

        # Parse and deduplicate by OSM ID
        seen_ids = set()
        results = []
        for element in elements:
            osm_id = element.get("id")
            if osm_id and osm_id not in seen_ids:
                seen_ids.add(osm_id)
                parsed = parse_osm_element(element)
                if parsed.get("name"):  # Only include named POIs
                    results.append(parsed)

        return results

    except requests.Timeout:
        logger.warning(f"Overpass query timed out for ({lat}, {lng})")
        return []
    except requests.RequestException as e:
        logger.warning(f"Overpass query failed for ({lat}, {lng}): {e}")
        return []
    except (ValueError, KeyError) as e:
        logger.warning(f"Failed to parse Overpass response: {e}")
        return []


def parse_osm_element(element: dict) -> dict:
    """Extract business fields from an OSM element."""
    tags = element.get("tags", {})

    # Get coordinates - nodes have lat/lon directly, ways/relations need center
    lat = element.get("lat")
    lng = element.get("lon")
    if lat is None and "center" in element:
        lat = element["center"].get("lat")
        lng = element["center"].get("lon")

    return {
        "osm_id": element.get("id"),
        "osm_type": element.get("type"),
        "name": tags.get("name"),
        "phone": tags.get("phone") or tags.get("contact:phone"),
        "website": tags.get("website") or tags.get("contact:website"),
        "email": tags.get("email") or tags.get("contact:email"),
        "business_type": map_osm_type(tags),
        "lat": lat,
        "lng": lng,
        "address": format_osm_address(tags),
        "raw_tags": tags
    }


def map_osm_type(tags: dict) -> str:
    """Map OSM tags to a business type string."""
    for key in ["shop", "amenity", "office", "craft"]:
        if key in tags:
            return f"{key}:{tags[key]}"
    return "unknown"


def format_osm_address(tags: dict) -> Optional[str]:
    """Format address from OSM addr:* tags."""
    parts = []

    housenumber = tags.get("addr:housenumber")
    street = tags.get("addr:street")
    if housenumber and street:
        parts.append(f"{housenumber} {street}")
    elif street:
        parts.append(street)

    city = tags.get("addr:city")
    if city:
        parts.append(city)

    state = tags.get("addr:state")
    postcode = tags.get("addr:postcode")
    if state and postcode:
        parts.append(f"{state} {postcode}")
    elif state:
        parts.append(state)
    elif postcode:
        parts.append(postcode)

    return ", ".join(parts) if parts else None


# ============================================================================
# Database Functions
# ============================================================================

def fetch_parcels(
    query: str,
    limit: Optional[int] = None,
    offset_pin: Optional[str] = None
) -> list[dict]:
    """Fetch parcels using provided SQL query."""
    sql = query.strip()

    # Remove trailing semicolon if present
    sql = sql.rstrip(';')

    # Build clauses
    clauses = []
    if offset_pin:
        # Use quoted column name for Postgres
        clauses.append(f"\"PIN\" > '{offset_pin}'")

    # Add WHERE clause if we have conditions
    if clauses:
        # Check if query already has WHERE
        if " WHERE " in sql.upper():
            sql += f" AND ({' AND '.join(clauses)})"
        else:
            sql += f" WHERE {' AND '.join(clauses)}"

    sql += " ORDER BY \"PIN\""

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


def save_business(business: dict) -> bool:
    """Save a business record to the CRM database."""
    # Skip attributes field - can have problematic characters
    filtered = {k: v for k, v in business.items() if k != "attributes"}
    columns = list(filtered.keys())
    placeholders = []

    for col in columns:
        val = filtered[col]
        if val is None:
            placeholders.append("NULL")
        elif isinstance(val, (int, float)):
            placeholders.append(str(val))
        elif isinstance(val, dict):
            # JSON fields
            escaped = json.dumps(val).replace("'", "''")
            placeholders.append(f"'{escaped}'::jsonb")
        else:
            # Escape single quotes and backslashes
            escaped = str(val).replace("\\", "\\\\").replace("'", "''")
            placeholders.append(f"'{escaped}'")

    sql = f"""
        INSERT INTO crm.businesses ({", ".join(columns)})
        VALUES ({", ".join(placeholders)})
        ON CONFLICT (pin, result_url) DO UPDATE SET
            business_name = EXCLUDED.business_name,
            phone = EXCLUDED.phone,
            website = EXCLUDED.website,
            business_type = EXCLUDED.business_type,
            updated_at = NOW()
    """

    try:
        response = requests.post(
            f"{API_BASE}/postgresql/query",
            json={"sql": sql},
            timeout=30
        )
        if response.status_code != 200:
            logger.debug(f"SQL: {sql[:200]}...")
            logger.debug(f"Response: {response.text[:200]}")
        response.raise_for_status()
        data = response.json()
        return data.get("success", False)

    except requests.RequestException as e:
        logger.warning(f"Failed to save business: {e}")
        return False


# ============================================================================
# Checkpoint Functions
# ============================================================================

def load_checkpoint() -> Optional[str]:
    """Load the last processed PIN from checkpoint file."""
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
            return data.get("last_pin")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_checkpoint(pin: str, stats: dict) -> None:
    """Save checkpoint with last processed PIN and stats."""
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump({
                "last_pin": pin,
                "timestamp": datetime.now().isoformat(),
                "stats": stats
            }, f, indent=2)
    except IOError as e:
        logger.warning(f"Failed to save checkpoint: {e}")


# ============================================================================
# Main Processing
# ============================================================================

def process_parcel(parcel: dict, radius: int) -> list[dict]:
    """
    Query OSM for businesses near a parcel and format for storage.

    Returns list of business records ready for save_business().
    """
    pin = parcel.get("pin")
    lat = parcel.get("lat")
    lng = parcel.get("lng")

    if lat is None or lng is None:
        logger.warning(f"PIN {pin}: Missing coordinates, skipping")
        return []

    # Query OSM Overpass API
    pois = query_overpass(lat, lng, radius_m=radius)

    if not pois:
        return []

    # Convert to business records
    businesses = []
    for poi in pois:
        osm_type = poi.get("osm_type", "node")
        osm_id = poi.get("osm_id")

        # Address: use OSM address if available, otherwise construct from coordinates
        address = poi.get("address")
        if not address:
            # Use coordinates as fallback address
            poi_lat = poi.get("lat", lat)
            poi_lng = poi.get("lng", lng)
            address = f"Near ({poi_lat:.5f}, {poi_lng:.5f})"

        business = {
            "pin": pin,
            "business_name": poi.get("name", "")[:255] if poi.get("name") else None,
            "phone": poi.get("phone"),
            "email": poi.get("email"),
            "website": poi.get("website", "")[:500] if poi.get("website") else None,
            "business_type": poi.get("business_type"),
            "result_url": f"osm://{osm_type}/{osm_id}",
            "result_title": poi.get("name"),
            "address": address,
            "source": "osm",
            "confidence_score": 1.0,  # OSM data is authoritative
        }
        businesses.append(business)

    return businesses


def main():
    parser = argparse.ArgumentParser(description="OSM Business Discovery Pipeline")
    parser.add_argument("--test", action="store_true", help="Test mode: process only 10 parcels")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--limit", type=int, help="Limit number of parcels to process")
    parser.add_argument("--query", type=str, help="Custom SQL query for parcels (must return pin, lat, lng)")
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS, help=f"Search radius in meters (default: {DEFAULT_RADIUS})")
    args = parser.parse_args()

    # Determine limit
    limit = 10 if args.test else args.limit

    # Use custom query or default
    query = args.query if args.query else DEFAULT_QUERY

    # Check for resume
    offset_pin = None
    if args.resume:
        offset_pin = load_checkpoint()
        if offset_pin:
            logger.info(f"Resuming from PIN: {offset_pin}")
        else:
            logger.info("No checkpoint found, starting from beginning")

    # Fetch parcels
    logger.info("Fetching parcels...")
    parcels = fetch_parcels(query=query, limit=limit, offset_pin=offset_pin)

    if not parcels:
        logger.error("No parcels to process")
        return 1

    logger.info(f"Processing {len(parcels)} parcels with {args.radius}m radius...")

    # Stats
    stats = {
        "total_processed": 0,
        "total_businesses": 0,
        "parcels_with_results": 0,
        "errors": 0
    }

    for i, parcel in enumerate(parcels):
        pin = parcel.get("pin")
        lat = parcel.get("lat")
        lng = parcel.get("lng")

        logger.info(f"[{i+1}/{len(parcels)}] PIN {pin}: ({lat:.6f}, {lng:.6f})")

        try:
            businesses = process_parcel(parcel, args.radius)

            if businesses:
                stats["parcels_with_results"] += 1
                for biz in businesses:
                    if save_business(biz):
                        stats["total_businesses"] += 1
                        logger.info(f"  + {biz.get('business_name')} ({biz.get('business_type')})")
                    else:
                        stats["errors"] += 1

            stats["total_processed"] += 1

            # Save checkpoint after each parcel
            save_checkpoint(pin, stats)

        except Exception as e:
            logger.error(f"  Error processing PIN {pin}: {e}")
            stats["errors"] += 1

        # Rate limit (except for last parcel)
        if i < len(parcels) - 1:
            logger.debug(f"  Rate limiting: {RATE_LIMIT}s")
            time.sleep(RATE_LIMIT)

    # Final summary
    logger.info("=" * 50)
    logger.info("OSM Discovery Complete")
    logger.info(f"  Parcels processed: {stats['total_processed']}")
    logger.info(f"  Parcels with results: {stats['parcels_with_results']}")
    logger.info(f"  Businesses discovered: {stats['total_businesses']}")
    logger.info(f"  Errors: {stats['errors']}")
    logger.info("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
