import requests

def test_ontario_elevation_api():
    """
    Tests the Ontario GeoHub ArcGIS REST API specifically for the LiDAR services.
    """
    print("🔍 Testing Ontario GeoHub Elevation API...")
    
    # The root folder for all elevation services
    base_url = "https://ws.geoservices.lrc.gov.on.ca/arcgis5/rest/services/Elevation"
    params = {'f': 'json'}
    
    try:
        # 1. Check if the server is online and list services
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        services = [svc['name'] for svc in data.get('services', [])]
        
        # 2. Check for the specific Lidar datasets we need
        dsm_service = "Elevation/Ontario_DSM_LidarDerived"
        dtm_service = "Elevation/Ontario_DTM_LidarDerived"
        
        print("✅ Connected to Ontario GeoHub Server.")
        
        if dsm_service in services and dtm_service in services:
            print("✅ Found LiDAR-Derived DSM and DTM ImageServers!")
            
            # 3. Test the specific DSM service to get its metadata
            dsm_url = f"{base_url}/Ontario_DSM_LidarDerived/ImageServer"
            dsm_meta = requests.get(dsm_url, params=params).json()
            
            print(f"   - Max Export Size: {dsm_meta.get('maxImageWidth')} x {dsm_meta.get('maxImageHeight')} pixels")
            print(f"   - Pixel Type: {dsm_meta.get('pixelType')}")
            print(f"   - Spatial Reference (WKID): {dsm_meta.get('spatialReference', {}).get('wkid')}")
            return True
        else:
            print("❌ Server is online, but Lidar services are missing.")
            return False
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API Test Failed. Error: {e}")
        return False

if __name__ == "__main__":
    test_ontario_elevation_api()
