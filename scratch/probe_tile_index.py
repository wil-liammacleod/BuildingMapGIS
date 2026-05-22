"""
scratch/probe_tile_index.py

Probes the Ontario GeoHub / ArcGIS REST services to:
  1. Find which feature service holds the 1km LiDAR tile index
  2. Return all tiles intersecting a given bbox
  3. Print field names and sample download URLs

Run with:
    uv run scratch/probe_tile_index.py
"""

import requests
import json

# Ward 1 bbox [min_lon, min_lat, max_lon, max_lat]
WARD_1_BBOX = [-79.9462, 43.2417, -79.8743, 43.2940]

# Candidate tile index feature service endpoints to probe
# (Ontario GeoHub uses ArcGIS Online hosted feature layers)
CANDIDATE_SERVICES = [
    # ArcGIS Online — Ontario LIO hosted services
    "https://services5.arcgis.com/mnr4HNsHJwZhFrVR/arcgis/rest/services/Ontario_Classified_Point_Cloud_Tile_Index/FeatureServer/0",
    "https://services5.arcgis.com/mnr4HNsHJwZhFrVR/arcgis/rest/services/Ontario_PointCloud_Tile_Index/FeatureServer/0",
    "https://services1.arcgis.com/qAo8M4a4363qUoSg/arcgis/rest/services/Ontario_Classified_Point_Cloud/FeatureServer/0",
    # GeoHub direct services
    "https://ws.geoservices.lrc.gov.on.ca/arcgis5/rest/services/Elevation/Ontario_PointCloud_Tiles/FeatureServer/0",
    "https://ws.geoservices.lrc.gov.on.ca/arcgis5/rest/services/Elevation/Ontario_PointCloud_Tiles/MapServer/0",
]


def probe_service(service_url: str, bbox: list) -> dict | None:
    """Query a candidate feature service and return result if it works."""
    # First check the service exists
    try:
        r = requests.get(service_url, params={"f": "json"}, timeout=15)
        r.raise_for_status()
        meta = r.json()
        if "error" in meta:
            print(f"  ✗ {service_url.split('/')[-3]}: {meta['error'].get('message','error')}")
            return None
        print(f"  ✓ Service exists: {meta.get('name', '?')} | Fields: {[f['name'] for f in meta.get('fields', [])]}")
    except Exception as e:
        print(f"  ✗ {service_url}: {e}")
        return None

    # Query for tiles intersecting the bbox
    query_url = service_url + "/query"
    params = {
        "f": "json",
        "geometry": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "resultRecordCount": 5,  # just peek at a few
        "returnGeometry": False,
    }
    try:
        r = requests.get(query_url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            print(f"    Query error: {data['error']}")
            return None
        features = data.get("features", [])
        print(f"    Tiles found (first 5 of ?): {len(features)}")
        if features:
            print(f"    Sample attributes: {json.dumps(features[0]['attributes'], indent=6)}")
        return data
    except Exception as e:
        print(f"    Query failed: {e}")
        return None


def count_tiles(service_url: str, bbox: list) -> int:
    """Count total tiles in bbox."""
    query_url = service_url + "/query"
    params = {
        "f": "json",
        "geometry": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "OBJECTID",
        "returnCountOnly": True,
    }
    try:
        r = requests.get(query_url, params=params, timeout=15)
        data = r.json()
        return data.get("count", -1)
    except Exception:
        return -1


def list_all_tiles(service_url: str, bbox: list) -> list[dict]:
    """Return all tile attribute records in bbox."""
    query_url = service_url + "/query"
    params = {
        "f": "json",
        "geometry": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": False,
        "resultRecordCount": 200,
    }
    r = requests.get(query_url, params=params, timeout=30)
    data = r.json()
    return [f["attributes"] for f in data.get("features", [])]


# ── Also probe the CanElevation FTP/API for point clouds ─────────────────────
CANELE_COLLECTION_ID = "7069387e-9986-4297-9f55-0288e9676947"

def probe_canelevation():
    """Check if CanElevation Open Government Portal has an API for tile discovery."""
    url = f"https://open.canada.ca/data/en/api/3/action/package_show?id={CANELE_COLLECTION_ID}"
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        resources = data.get("result", {}).get("resources", [])
        print(f"\n📦 CanElevation API: {len(resources)} resources found")
        for res in resources[:5]:
            print(f"  - {res.get('name','?')} | {res.get('format','?')} | {res.get('url','?')[:80]}")
    except Exception as e:
        print(f"  CanElevation probe failed: {e}")


if __name__ == "__main__":
    print("=" * 70)
    print("🔍 Ontario LiDAR Tile Index Probe")
    print(f"   Bbox (Ward 1): {WARD_1_BBOX}")
    print("=" * 70)

    working_service = None

    print("\n── Probing candidate ArcGIS feature services ──")
    for svc in CANDIDATE_SERVICES:
        print(f"\n→ {svc}")
        result = probe_service(svc, WARD_1_BBOX)
        if result and result.get("features"):
            working_service = svc
            break  # found one that works

    if working_service:
        count = count_tiles(working_service, WARD_1_BBOX)
        print(f"\n✅ Working service: {working_service}")
        print(f"   Total tiles in Ward 1 bbox: {count}")

        print("\n── All tile attributes ──")
        tiles = list_all_tiles(working_service, WARD_1_BBOX)
        for t in tiles:
            print(f"  {t}")

        # Show which field looks like a download URL
        if tiles:
            keys = list(tiles[0].keys())
            url_keys = [k for k in keys if any(w in k.lower() for w in ["url", "link", "download", "href", "path"])]
            name_keys = [k for k in keys if any(w in k.lower() for w in ["name", "tile", "file", "id"])]
            print(f"\n  📋 Likely URL fields:  {url_keys}")
            print(f"  📋 Likely name fields: {name_keys}")
    else:
        print("\n⚠️  No working ArcGIS feature service found.")
        print("   Falling back to CanElevation check...")

    probe_canelevation()

    print("\n" + "=" * 70)
    print("Probe complete.")
