https://geohub.lio.gov.on.ca/pages/ontario-elevation-mapping-program

https://tubvsig-so2sat-vm1.srv.mwn.de/

https://www.whiteboxgeo.com/learning-resources/

https://www.whiteboxgeo.com/manual/wbw-user-manual/book/installing.html

https://wiki.openstreetmap.org/wiki/3D

https://maplibre.org/maplibre-gl-js/docs/examples/display-buildings-in-3d/

### Features
- **LiDAR Rooftop Cleaning:** Phase 3b implementation in `src/main.py`. Processes raw segmentation into simplified, non-overlapping 3D blocks.
- **Output:** `[city]_lidar_rooftops_clean_3d.geojson`
- **Utility Scripts:** See `scratch/` for building-specific analysis tools.

### Whitebox Workflows Testing & Footprint Extraction
- Detailed notes, workaround for unlicensed WbW-Pro functions, and tree/vegetation filtering strategies can be found in [whiteboxtest/README.md](file:///Users/liammacleod/Nextcloud/MASC/BuildingMapGIS/whiteboxtest/README.md).

