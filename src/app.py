import streamlit as st
import pydeck as pdk
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import socket
import json

# Global variables for background HTTP server
GLOBAL_SERVER_PORT = None
GLOBAL_SERVER_THREAD = None

def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port

class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
        super().end_headers()
        
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path.startswith('/log_error'):
            from urllib.parse import urlparse, parse_qs
            query = urlparse(self.path).query
            params = parse_qs(query)
            msg = params.get('msg', [''])[0]
            print(f"[JS ERROR] {msg}", flush=True)
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"OK")
            return
        super().do_GET()

def start_server_in_thread(directory, port):
    class Handler(CORSHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
            
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.serve_forever()

def ensure_http_server():
    if "GLOBAL_SERVER_PORT" not in st.session_state:
        port = get_free_port()
        output_dir = Path(__file__).parent.parent / "whiteboxtest" / "output"
        thread = threading.Thread(
            target=start_server_in_thread, 
            args=(str(output_dir), port), 
            daemon=True
        )
        thread.start()
        st.session_state["GLOBAL_SERVER_PORT"] = port
    return st.session_state["GLOBAL_SERVER_PORT"]

@st.cache_data
def load_building_offsets():
    footprints_path = Path("./whiteboxtest/output/building_footprints.shp")
    dtm_path = Path("./whiteboxtest/output/DTM.tif")
    if not footprints_path.exists() or not dtm_path.exists():
        return []
        
    gdf = gpd.read_file(footprints_path)
    building_data = []
    
    with rasterio.open(dtm_path) as dtm_src:
        for idx, row in gdf.iterrows():
            poly = row.geometry
            if poly is None or poly.is_empty:
                continue
            X_c = poly.centroid.x
            Y_c = poly.centroid.y
            try:
                Z_base = float(next(dtm_src.sample([(X_c, Y_c)]))[0])
                if np.isnan(Z_base) or Z_base < -100:
                    Z_base = 0.0
            except Exception:
                Z_base = 0.0
            building_data.append({
                "id": int(idx),
                "dx": float(X_c - 587000.0),
                "dy": float(Y_c - 4790000.0),
                "dz": float(Z_base)
            })
    return building_data

st.set_page_config(layout="wide", page_title="Ontario 3D Building GIS")

# View Mode Selection
view_mode = st.sidebar.radio(
    "Select View Mode:",
    ["Scientific Analytics (Plotly)", "Interactive 3D Map (PyDeck)", "Dataset Comparison (StatCan vs Overture)", "3D Building GLB Viewer (Whitebox Mode)"],
    index=1,
    help="Switch between analyzing the raw LiDAR data pixel-by-pixel, and viewing the fully extruded 3D building models."
)

city_name = "Hamilton"
province_name = "Ontario"
ward_name = "Ward 1"

ward_slug = ward_name.lower().replace(" ", "_") if ward_name else ""
if ward_slug:
    processed_dir = Path(f"./data/{province_name}/{city_name}/{ward_slug}/processed")
    file_prefix = ward_slug
else:
    processed_dir = Path(f"./data/{province_name}/{city_name}/processed")
    file_prefix = city_name.lower()

statcan_path = processed_dir / f"{file_prefix}_statcan_buildings_3d.geojson"
overture_path = processed_dir / f"{file_prefix}_overture_buildings_3d.geojson"
lidar_path = processed_dir / f"{file_prefix}_lidar_buildings_3d.geojson"
rooftops_path = processed_dir / f"{file_prefix}_lidar_rooftops_3d.geojson"
rooftops_clean_path = processed_dir / f"{file_prefix}_lidar_rooftops_clean_3d.geojson"

def _get_file_size_mb(path) -> float:
    return path.stat().st_size / (1024 * 1024)

def load_data_with_status(path):
    """Wrapper around load_data() that shows a size warning before loading.
    st.toast() cannot be called inside @st.cache_data, so we do it here."""
    if path.exists():
        size_mb = _get_file_size_mb(path)
        if size_mb > 10:
            st.toast(f"⏳ Loading large dataset ({size_mb:.0f} MB) — simplifying geometries...", icon="🗺️")
    return load_data(path)

@st.cache_data
def load_data(path):
    if not path.exists():
        return None
    
    file_size_mb = _get_file_size_mb(path)
    needs_simplification = file_size_mb > 10 and "clean" not in path.name
    
    gdf = gpd.read_file(path)
    
    if needs_simplification:
        original_crs = gdf.crs
        gdf = gdf.to_crs("EPSG:2958")
        
        # Drop tiny fragments (< 10 m²)
        gdf = gdf[gdf.geometry.area >= 10.0].copy()
        
        # Simplify geometry vertices (0.5m tolerance)
        gdf.geometry = gdf.geometry.simplify(tolerance=0.5, preserve_topology=True)
        gdf = gdf.to_crs(original_crs)
        
        # Drop columns not needed for visualization to reduce payload
        keep_cols = {'geometry', 'height_p90', 'height_max', 'address', 'type',
                     'AVE_HGT', 'SLOPE', 'ASPECT', 'AREA', 'VALUE',
                     # Method D metric columns
                     'num_floors', 'internal_area_sqft', 'internal_area_m2',
                     'clean_area', 'clean_volume', 'clean_volume_total',
                     'clean_surface_area', 'total_internal_sqft',
                     'max_floors', 'bldg_use', 'floor_height', 'BUILDING',
                     # Quality & S3DB columns
                     'lqs', 'q_coverage', 'q_canopy', 'tree_percentage',
                     'is_fallback', 'osm_id', 'parent_id', 'height'}
        drop_cols = [c for c in gdf.columns if c not in keep_cols]
        if drop_cols:
            gdf = gdf.drop(columns=drop_cols)
    
    # Generate safe colors based on height
    def calculate_color(height):
        r = 255
        g = max(0, 255 - int(height * 4))
        b = max(0, 255 - int(height * 6))
        return [r, g, b, 200]
        
    def calculate_lqs_color(lqs):
        if lqs is None or np.isnan(lqs):
            return [150, 150, 150, 200] # Gray for missing
        if lqs >= 0.85:
            return [39, 174, 96, 200] # Green
        elif lqs >= 0.70:
            return [241, 196, 15, 200] # Yellow
        else:
            return [192, 57, 43, 200] # Red
        
    # Ensure height columns exist
    if 'height_p90' not in gdf.columns:
        if 'height' in gdf.columns:
            gdf['height_p90'] = gdf['height']
        else:
            gdf['height_p90'] = 10.0
            
    if 'height_max' not in gdf.columns:
        if 'height' in gdf.columns:
            gdf['height_max'] = gdf['height']
        else:
            gdf['height_max'] = 10.0
            
    if 'lqs' not in gdf.columns:
        gdf['lqs'] = np.nan
        
    gdf['color_height'] = gdf['height_p90'].apply(calculate_color)
    gdf['color_lqs'] = gdf['lqs'].apply(calculate_lqs_color)
    
    # Keep legacy color reference pointing to height color for compatibility
    gdf['color'] = gdf['color_height']
    
    gdf['height_p90'] = gdf['height_p90'].round(1)
    gdf['height_max'] = gdf['height_max'].round(1)
    
    # CRITICAL: Explode multipolygons for PyDeck PolygonLayer compatibility
    gdf = gdf.explode(index_parts=False)
    return gdf

@st.cache_data
def get_raster_layer(tif_name, png_name, cmap, vmin=0, vmax=60):
    tif_path = processed_dir / tif_name
    png_path = processed_dir / png_name
    
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        bounds = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
        
    left, bottom, right, top = bounds
    pydeck_bounds = [[left, bottom], [left, top], [right, top], [right, bottom]]
    
    if not png_path.exists():
        import matplotlib.cm as cm
        import matplotlib.colors as colors
        from PIL import Image
        
        # Mask out no-data regions
        data[data < 0] = np.nan
        
        # Robustly map to RGBA
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        rgba = mapper.to_rgba(data, alpha=1.0)
        rgba[..., 3] = np.where(np.isnan(data), 0.0, 1.0)
        
        img_uint8 = (rgba * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(png_path)
        
    return pydeck_bounds, str(png_path)

if view_mode == "Dataset Comparison (StatCan vs Overture)":
    st.sidebar.markdown("---")
    st.sidebar.header("Comparison Settings")
    st.sidebar.info("🔴 **Red** = Dataset 1\\n🔵 **Blue** = Dataset 2\\n🟪 **Purple** = Both agree")
    
    source_map = {
        "Overture (Modern)": overture_path,
        "LiDAR (Auto-Extracted)": lidar_path,
        "StatCan (Legacy)": statcan_path,
        "LiDAR (Rooftop Raw)": rooftops_path,
        "LiDAR (Method D Healed)": rooftops_clean_path
    }
    
    all_sources = ["Overture (Modern)", "LiDAR (Auto-Extracted)", "LiDAR (Rooftop Raw)", "LiDAR (Method D Healed)", "StatCan (Legacy)"]
    ds1 = st.sidebar.selectbox("Dataset 1 (Red):", all_sources, index=0)
    ds2 = st.sidebar.selectbox("Dataset 2 (Blue):", all_sources, index=2)
    
    gdf_statcan = load_data_with_status(source_map[ds1])
    gdf_overture = load_data_with_status(source_map[ds2])
    
    layers = []
    
    if gdf_statcan is not None:
        layers.append(pdk.Layer(
            "PolygonLayer",
            gdf_statcan,
            get_polygon="geometry.coordinates",
            opacity=0.5,
            stroked=False,
            filled=True,
            extruded=False,
            get_fill_color=[255, 0, 0, 150]
        ))
        
    if gdf_overture is not None:
        layers.append(pdk.Layer(
            "PolygonLayer",
            gdf_overture,
            get_polygon="geometry.coordinates",
            opacity=0.5,
            stroked=False,
            filled=True,
            extruded=False,
            get_fill_color=[0, 150, 255, 150]
        ))
        
    view_state = pdk.ViewState(
        longitude=gdf_statcan.to_crs("EPSG:2958").geometry.centroid.to_crs("EPSG:4326").x.mean() if gdf_statcan is not None else -79.91025,
        latitude=gdf_statcan.to_crs("EPSG:2958").geometry.centroid.to_crs("EPSG:4326").y.mean() if gdf_statcan is not None else 43.26785,
        zoom=14.5, 
        pitch=0, 
        bearing=0
    )
    
    map_height = st.sidebar.slider(
        "Map Window Height (px)",
        min_value=400,
        max_value=1200,
        value=750,
        step=50,
        help="Adjust the vertical height of the map window."
    )
    
    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=pdk.map_styles.DARK
    ), use_container_width=True, height=map_height)

elif view_mode == "Interactive 3D Map (PyDeck)":
    st.sidebar.markdown("---")
    st.sidebar.header("Map Layers")
    
    s3db_path = processed_dir / f"{file_prefix}_lidar_rooftops_s3db_3d.geojson"
    blended_path = processed_dir / f"{file_prefix}_lidar_rooftops_blended_3d.geojson"
    
    dataset_choice = st.sidebar.selectbox(
        "Footprint Source:",
        ["Overture (Modern)", "LiDAR (Auto-Extracted)", "LiDAR (Rooftop Raw)", "LiDAR (Method D Healed)", "LiDAR (S3DB Parents & Parts)", "LiDAR (Blended OSM)", "StatCan (Legacy)"],
        index=4
    )
    
    source_map = {
        "Overture (Modern)": overture_path,
        "LiDAR (Auto-Extracted)": lidar_path,
        "StatCan (Legacy)": statcan_path,
        "LiDAR (Rooftop Raw)": rooftops_path,
        "LiDAR (Method D Healed)": rooftops_clean_path,
        "LiDAR (S3DB Parents & Parts)": s3db_path,
        "LiDAR (Blended OSM)": blended_path
    }
    active_path = source_map[dataset_choice]
    
    show_3d = st.sidebar.checkbox("Show 3D Buildings", value=True)
    show_2d = st.sidebar.checkbox("Show 2D Footprints", value=False)
    show_ndsm = st.sidebar.checkbox("Show nDSM (Height Raster)", value=False)
    show_roughness = st.sidebar.checkbox("Show Roughness (Trees Raster)", value=False)
    
    color_theme = st.sidebar.selectbox("Color Theme:", ["Height Gradient", "LiDAR Quality Score (LQS)"])
    color_col = "color_lqs" if color_theme == "LiDAR Quality Score (LQS)" else "color_height"

    st.sidebar.markdown("---")
    st.sidebar.header("Display Settings")
    map_height = st.sidebar.slider(
        "Map Window Height (px)",
        min_value=400,
        max_value=1200,
        value=750,
        step=50,
        help="Adjust the vertical height of the map window."
    )
    
    elevation_scale = 1.0
    if show_3d:
        elevation_scale = st.sidebar.slider(
            "3D Elevation Scale",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Multiply building heights by this factor for vertical exaggeration."
        )

    gdf = load_data_with_status(active_path)
    layers = []

    # Calculate sidebar stats if LQS is in gdf
    if gdf is not None and 'lqs' in gdf.columns:
        valid_lqs = gdf['lqs'].dropna()
        if not valid_lqs.empty:
            avg_lqs = valid_lqs.mean()
            st.sidebar.metric("Average LiDAR Quality", f"{avg_lqs * 100:.1f}%")
            
            if 'is_fallback' in gdf.columns:
                # Group by building id to count buildings, not individual parts
                bldg_fallback = gdf.groupby('BUILDING')['is_fallback'].first()
                fallbacks = bldg_fallback.sum()
                total_bldgs = len(bldg_fallback)
                if total_bldgs > 0:
                    st.sidebar.metric("Low Quality Fallbacks", f"{fallbacks} / {total_bldgs}", f"{fallbacks/total_bldgs*100:.1f}% of bldgs")

    if show_ndsm:
        bounds, img_path = get_raster_layer(f"{file_prefix}_ndsm_output.tif", "ndsm_v2.png", "plasma", vmin=0, vmax=30)
        import base64
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        image_url = f"data:image/png;base64,{encoded_string}"
        
        layers.append(pdk.Layer(
            "BitmapLayer",
            image=pdk.types.String(image_url),
            bounds=bounds,
            opacity=0.7
        ))

    if show_roughness:
        bounds, img_path = get_raster_layer(f"{file_prefix}_roughness_output.tif", "roughness_v2.png", "viridis", vmin=0, vmax=5)
        import base64
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        image_url = f"data:image/png;base64,{encoded_string}"
        
        layers.append(pdk.Layer(
            "BitmapLayer",
            image=pdk.types.String(image_url),
            bounds=bounds,
            opacity=0.7
        ))

    if gdf is not None:
        # Safely copy rename colons to underscores to prevent format specifier errors in tooltip rendering
        if 'building:levels' in gdf.columns:
            gdf['building_levels'] = gdf['building:levels']
        else:
            gdf['building_levels'] = gdf.get('levels', gdf.get('num_floors', 3))
            
        if 'building:part' in gdf.columns:
            gdf['building_part'] = gdf['building:part']

        # Separate parent footprints and building parts to prevent rendering double overlapping volumes
        if 'type' in gdf.columns:
            parents_gdf = gdf[gdf['type'] == 'building']
            parts_gdf = gdf[gdf['type'] == 'building_part']
            
            layer_2d_gdf = parents_gdf
            
            # Extrude parts + any parent buildings that don't have parts (e.g. unmatched OSM buildings)
            has_parts_parent_ids = set(parts_gdf['parent_id'].unique() if 'parent_id' in parts_gdf.columns else parts_gdf['BUILDING'].unique())
            unmatched_parents = parents_gdf[~parents_gdf['BUILDING'].isin(has_parts_parent_ids) | (parents_gdf['BUILDING'] == -1)]
            layer_3d_gdf = gpd.GeoDataFrame(pd.concat([parts_gdf, unmatched_parents], ignore_index=True), crs=gdf.crs)
        else:
            layer_2d_gdf = gdf
            layer_3d_gdf = gdf

        if show_2d:
            layers.append(pdk.Layer(
                "PolygonLayer",
                layer_2d_gdf,
                get_polygon="geometry.coordinates",
                pickable=True,
                auto_highlight=True,
                opacity=0.5,
                stroked=True,
                filled=True,
                extruded=False,
                get_fill_color=color_col,
                get_line_color=[255, 255, 255]
            ))
    
        if show_3d:
            layers.append(pdk.Layer(
                "PolygonLayer",
                layer_3d_gdf,
                get_polygon="geometry.coordinates",
                pickable=True,
                auto_highlight=True,
                opacity=0.8,
                stroked=False,
                filled=True,
                extruded=True,
                wireframe=True,
                get_elevation="height_p90",
                elevation_scale=elevation_scale,
                get_fill_color=color_col,
                get_line_color=[255, 255, 255]
            ))
    
        _gdf_proj = gdf.to_crs("EPSG:2958")
        view_state = pdk.ViewState(
            longitude=_gdf_proj.geometry.centroid.to_crs("EPSG:4326").x.mean(),
            latitude=_gdf_proj.geometry.centroid.to_crs("EPSG:4326").y.mean(),
            zoom=14.5, 
            pitch=45 if show_3d else 0, 
            bearing=0
        )
    else:
        view_state = pdk.ViewState(longitude=-79.91025, latitude=43.26785, zoom=14.5)

    # Build tooltip — rich metadata for Method D healed, basic for all other sources
    if gdf is not None and 'lqs' in gdf.columns and not gdf['lqs'].isna().all():
        tooltip_html = """
            <b>📍 {address}</b><br/>
            <hr/>
            <b>Feature Type:</b> {type}<br/>
            <b>Height:</b> {height_p90} m<br/>
            <b>Floors:</b> {building_levels}<br/>
            <b>LiDAR Quality Score (LQS):</b> {lqs}<br/>
            <b>- Footprint Coverage:</b> {q_coverage}<br/>
            <b>- Tree Canopy Overlap:</b> {q_canopy} ({tree_percentage}% cover)<br/>
            <b>Fallback Applied:</b> {is_fallback}<br/>
        """
        if 'clean_area' in gdf.columns:
            tooltip_html += """
                <hr/>
                <b>Segment Area:</b> {clean_area} m²<br/>
                <b>Segment Floors:</b> {num_floors}<br/>
                <b>Segment Int. Area:</b> {internal_area_sqft} sq ft<br/>
            """
        if 'total_internal_sqft' in gdf.columns:
            tooltip_html += """
                <hr/>
                <b>BUILDING TOTALS:</b><br/>
                <b>Estimated Use:</b> {bldg_use}<br/>
                <b>Max Floors:</b> {max_floors}<br/>
                <b>Total Floor Area:</b> {total_internal_sqft} sq ft<br/>
                <b>Sealed Surface Area:</b> {clean_surface_area} m²<br/>
                <b>Total Volume:</b> {clean_volume_total} m³
            """
        if 'osm_id' in gdf.columns:
            tooltip_html += """
                <hr/>
                <b>OSM Building ID:</b> {osm_id}<br/>
                <b>OSM Name:</b> {name}
            """
    elif gdf is not None and 'num_floors' in gdf.columns:
        # Rich Method D tooltip matching visualize_comparison_all.py
        tooltip_html = """
            <b>📍 {address}</b><br/>
            <hr/>
            <b>Segment Height:</b> {height_p90} m<br/>
            <b>Segment Area:</b> {clean_area} m²<br/>
            <b>Segment Floors:</b> {num_floors}<br/>
            <b>Segment Int. Area:</b> {internal_area_sqft} sq ft<br/>
            <hr/>
            <b>BUILDING TOTALS:</b><br/>
            <b>Estimated Use:</b> {bldg_use}<br/>
            <b>Max Floors:</b> {max_floors}<br/>
            <b>Total Floor Area:</b> {total_internal_sqft} sq ft<br/>
            <b>Sealed Surface Area:</b> {clean_surface_area} m²<br/>
            <b>Total Volume:</b> {clean_volume_total} m³
        """
    else:
        tooltip_html = "<b>📍 {address}</b><br/>Type: {type}<br/><hr/>📏 <b>Height:</b> {height_p90} m<br/>📐 <b>Max Peak:</b> {height_max} m"
        if gdf is not None and 'SLOPE' in gdf.columns:
            tooltip_html += "<br/>⛰️ <b>Slope:</b> {SLOPE}°<br/>🧭 <b>Aspect:</b> {ASPECT}°"

    tooltip = {
        "html": tooltip_html,
        "style": {
            "backgroundColor": "#222222",
            "color": "white",
            "font-family": "Helvetica, Arial, sans-serif",
            "padding": "10px",
            "border-radius": "8px"
        }
    }

    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=pdk.map_styles.DARK,
        tooltip=tooltip
    ), use_container_width=True, height=map_height)

elif view_mode == "3D Building GLB Viewer (Whitebox Mode)":
    st.header("🗄️ 3D Building GLB Viewer (Whitebox Mode)")
    st.write("This tool loads the sloped 3D roof segments and flat gap fills extruded from the LiDAR rooftop analysis.")
    
    import base64
    import glob
    import os
    
    # Path to GLB files generated by wbwbuildingmapping.py
    whitebox_glb_dir = Path("./whiteboxtest/output/glb")
    
    if not whitebox_glb_dir.exists():
        st.warning("⚠️ No GLB folder found at `whiteboxtest/output/glb`. Please run the footprint mapping script first:")
        st.code("uv run whiteboxtest/wbwbuildingmapping.py")
    else:
        glb_files = sorted(glob.glob(str(whitebox_glb_dir / "building_*.glb")), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        
        if not glb_files:
            st.warning("⚠️ No `.glb` files found in `whiteboxtest/output/glb`. Run the script to generate them.")
        else:
            # View type selection
            glb_view_type = st.sidebar.radio(
                "GLB Visualization Mode:",
                ["Single Building Detail", "All Buildings 3D Map"],
                index=1
            )
            
            if glb_view_type == "Single Building Detail":
                building_ids = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in glb_files]
                selected_id = st.sidebar.selectbox("Select Building ID to View:", building_ids)
                
                # Find selected glb path
                selected_glb_path = whitebox_glb_dir / f"building_{selected_id}.glb"
                
                if selected_glb_path.exists():
                    st.subheader(f"🏢 Building {selected_id} Model")
                    
                    # Load and base64-encode the GLB file
                    with open(selected_glb_path, "rb") as f:
                        glb_bytes = f.read()
                        
                    b64_glb = base64.b64encode(glb_bytes).decode('utf-8')
                    data_url = f"data:model/gltf-binary;base64,{b64_glb}"
                    
                    # HTML with Google's <model-viewer>
                    html_code = f"""
                    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
                    <style>
                        body {{ margin: 0; padding: 0; background-color: #0e1117; overflow: hidden; display: flex; justify-content: center; align-items: center; }}
                        model-viewer {{
                            width: 100%;
                            height: 500px;
                            background-color: #0e1117;
                            --poster-color: transparent;
                            border-radius: 12px;
                            border: 1px solid #30363d;
                        }}
                    </style>
                    <model-viewer 
                        src="{data_url}"
                        camera-controls
                        auto-rotate
                        shadow-intensity="1.5"
                        shadow-softness="0.5"
                        environment-image="neutral"
                        exposure="1.2"
                        interaction-prompt="auto"
                        style="width: 100%; height: 500px;"
                        ar>
                    </model-viewer>
                    """
                    
                    st.components.v1.html(html_code, height=520)
                    
                    st.markdown(f"""
                    ### Model Specifications:
                    * **File Name:** `building_{selected_id}.glb`
                    * **Location:** `{selected_glb_path}`
                    * **File Size:** `{len(glb_bytes)/1024:.2f} KB`
                    * **Interactive Controls:** Click and drag to rotate, scroll/pinch to zoom, right click and drag to pan.
                    """)
            else:
                # All Buildings 3D Map
                st.subheader("🏢 All Buildings Interactive 3D Map")
                
                # Check for combined GLB
                combined_glb_path = Path("./whiteboxtest/output/buildings_combined.glb")
                if not combined_glb_path.exists():
                    st.info("⚠️ Combined GLB model not found. Generating it now using the footprint layout...")
                    try:
                        with st.spinner("Compiling combined master GLB..."):
                            import sys
                            # Add whiteboxtest to path to import combine_glbs
                            whitebox_path = os.path.abspath("./whiteboxtest")
                            if whitebox_path not in sys.path:
                                sys.path.append(whitebox_path)
                            import combine_glbs
                            combine_glbs.main()
                        st.success("Successfully generated combined GLB!")
                    except Exception as e:
                        st.error(f"Error compiling combined GLB: {e}")
                
                # Select map loading mode
                map_loader_mode = st.sidebar.selectbox(
                    "Map Loading Mode:",
                    ["Combined Master GLB (Fast)", "Separate Individual GLBs (Detailed - Performance Test)"],
                    index=0,
                    help="Combined mode loads a single pre-merged 3D file (~61MB) for maximum speed and smoothness. Separate mode loads 273 individual files sequentially to demonstrate performance differences."
                )
                
                # Heading offset adjustment slider (default to 90 degrees since GLB exporter rotated them)
                heading_correction_deg = st.sidebar.slider(
                    "Manual Heading Correction (degrees):",
                    min_value=-180.0,
                    max_value=180.0,
                    value=90.0,
                    step=1.0,
                    help="Rotate all building models to align perfectly with the map streets. Default is 90 degrees."
                )
                
                # Start HTTP server
                port = ensure_http_server()
                
                # Load building data for individual positioning
                building_data = load_building_offsets()
                
                # Render MapLibre component
                loader_mode_val = "combined" if "Combined" in map_loader_mode else "individual"
                
                # Pass data to Javascript
                buildings_json = json.dumps(building_data)
                
                # HTML with MapLibre GL JS + Three.js custom layer
                html_code = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8" />
                    <title>3D Buildings Map</title>
                    <meta name="viewport" content="initial-scale=1,maximum-scale=1,user-scalable=no" />
                    <link href="https://unpkg.com/maplibre-gl@4.1.2/dist/maplibre-gl.css" rel="stylesheet" />
                    <script src="https://unpkg.com/maplibre-gl@4.1.2/dist/maplibre-gl.js"></script>
                    <script src="https://unpkg.com/three@0.147.0/build/three.min.js"></script>
                    <script src="https://unpkg.com/three@0.147.0/examples/js/loaders/GLTFLoader.js"></script>
                    <style>
                        body {{ margin: 0; padding: 0; background-color: #0e1117; }}
                        #map {{ position: absolute; top: 0; bottom: 0; width: 100%; height: 100%; }}
                        #status-box {{
                            position: absolute;
                            top: 20px;
                            left: 20px;
                            background: rgba(14, 17, 23, 0.95);
                            color: #e2e8f0;
                            padding: 12px 18px;
                            border-radius: 8px;
                            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                            font-size: 13px;
                            z-index: 10;
                            border: 1px solid #30363d;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
                            pointer-events: none;
                            transition: opacity 0.3s ease;
                        }}
                        .spinner {{
                            display: inline-block;
                            width: 12px;
                            height: 12px;
                            border: 2px solid rgba(255,255,255,0.3);
                            border-radius: 50%;
                            border-top-color: #fff;
                            animation: spin 1s ease-in-out infinite;
                            margin-right: 8px;
                            vertical-align: middle;
                        }}
                        @keyframes spin {{
                            to {{ transform: rotate(360deg); }}
                        }}
                    </style>
                </head>
                <body>
                    <div id="map"></div>
                    <div id="status-box">
                        <span id="spinner" class="spinner"></span>
                        <span id="status-text">Initializing 3D Map...</span>
                    </div>
                    <div id="error-console" style="position: absolute; bottom: 20px; left: 20px; right: 20px; max-height: 150px; overflow-y: auto; background: rgba(255,0,0,0.85); color: white; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 12px; z-index: 1000; display: none; border: 1px solid red; box-shadow: 0 0 10px rgba(0,0,0,0.5);"></div>

                    <script>
                        function logDiagnostic(msg) {{
                            const errDiv = document.getElementById('error-console');
                            if (errDiv) {{
                                errDiv.style.display = 'block';
                                errDiv.innerHTML += '<div>[DIAG] ' + msg + '</div>';
                            }}
                        }}

                        window.onerror = function(message, source, lineno, colno, error) {{
                            console.error(message, error);
                            const errDiv = document.getElementById('error-console');
                            if (errDiv) {{
                                errDiv.style.display = 'block';
                                errDiv.innerHTML += '<div style="color: yellow;">[ERROR] ' + message + ' (' + source + ':' + lineno + ')</div>';
                            }}
                            fetch("http://127.0.0.1:" + port + "/log_error?msg=" + encodeURIComponent(message + ' at ' + source + ':' + lineno + ':' + colno + (error ? ' | ' + error.stack : '')));
                        }};
                        window.addEventListener('unhandledrejection', function(event) {{
                            console.error(event.reason);
                            const errDiv = document.getElementById('error-console');
                            if (errDiv) {{
                                errDiv.style.display = 'block';
                                errDiv.innerHTML += '<div style="color: orange;">[REJECTION] ' + (event.reason ? event.reason.message || event.reason : 'unknown') + '</div>';
                            }}
                            fetch("http://127.0.0.1:" + port + "/log_error?msg=" + encodeURIComponent('Unhandled Rejection: ' + (event.reason ? event.reason.message || event.reason : 'unknown')));
                        }});


                        const statusBox = document.getElementById('status-box');
                        const statusText = document.getElementById('status-text');
                        const spinner = document.getElementById('spinner');

                        function updateStatus(text, showSpinner = true) {{
                            statusText.innerText = text;
                            spinner.style.display = showSpinner ? 'inline-block' : 'none';
                            statusBox.style.opacity = '1';
                        }}

                        function hideStatus() {{
                            statusBox.style.opacity = '0';
                        }}

                        // Configuration
                        const port = {port};
                        const loaderMode = "{loader_mode_val}";
                        const buildings = {buildings_json};
                        const headingCorrection = {heading_correction_deg} * Math.PI / 180;

                        // Initialize MapLibre Map using CORS-free raster tiles for the base map
                        const map = new maplibregl.Map({{
                            container: 'map',
                            style: {{
                                "version": 8,
                                "sources": {{
                                    "cartodb-dark": {{
                                        "type": "raster",
                                        "tiles": [
                                            "https://a.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}.png",
                                            "https://b.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}.png",
                                            "https://c.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}.png",
                                            "https://d.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}.png"
                                        ],
                                        "tileSize": 256,
                                        "attribution": "© OpenStreetMap contributors, © CartoDB"
                                    }}
                                }},
                                "layers": [
                                    {{
                                        "id": "cartodb-tiles",
                                        "type": "raster",
                                        "source": "cartodb-dark",
                                        "minzoom": 0,
                                        "maxzoom": 20
                                    }}
                                ]
                            }},
                            center: [-79.925004, 43.260006],
                            zoom: 16.0,
                            pitch: 60,
                            bearing: -20,
                            antialias: true
                        }});

                        map.addControl(new maplibregl.NavigationControl());

                        // Geographic reference point of the UTM grid origin (587000.0, 4790000.0, 0.0)
                        const modelOrigin = [-79.92814012, 43.25779592];
                        const modelAltitude = 0;
                        
                        // Z-axis rotation to correct for UTM grid convergence (-0.732016 degrees in radians)
                        const gridConvergenceAngle = -0.732016 * Math.PI / 180;
                        
                        // Coordinate transforms matching the Web Mercator projection
                        const modelAsMercator = maplibregl.MercatorCoordinate.fromLngLat(
                            modelOrigin,
                            modelAltitude
                        );

                        const modelTransform = {{
                            translateX: modelAsMercator.x,
                            translateY: modelAsMercator.y,
                            translateZ: modelAsMercator.z,
                            rotateX: Math.PI / 2, // Rotate to make Y-up map to Z-up
                            rotateY: 0,
                            rotateZ: 0,
                            scale: modelAsMercator.meterInMercatorCoordinateUnits()
                        }};

                        const THREE = window.THREE;

                        // Custom WebGL Layer to integrate Three.js
                        const custom3DLayer = {{
                            id: '3d-buildings-layer',
                            type: 'custom',
                            renderingMode: '3d',
                            onAdd: function (map, gl) {{
                                this.camera = new THREE.Camera();
                                this.scene = new THREE.Group();
                                this.rootScene = new THREE.Scene();
                                this.rootScene.add(this.scene);

                                // Add premium dual-directional lighting
                                const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.85);
                                dirLight1.position.set(200, 400, 200);
                                this.rootScene.add(dirLight1);

                                const dirLight2 = new THREE.DirectionalLight(0x99ccff, 0.45);
                                dirLight2.position.set(-200, 200, -200);
                                this.rootScene.add(dirLight2);

                                const ambientLight = new THREE.AmbientLight(0xffffff, 0.45);
                                this.rootScene.add(ambientLight);

                                this.renderer = new THREE.WebGLRenderer({{
                                    canvas: map.getCanvas(),
                                    context: gl,
                                    antialias: true
                                }});
                                this.renderer.autoClear = false;

                                logDiagnostic("Three.js loaded. resetState: " + (typeof this.renderer.resetState) + " | state.reset: " + (this.renderer.state ? typeof this.renderer.state.reset : "no state"));


                                // Load building meshes
                                const loader = new THREE.GLTFLoader();
                                const startTime = performance.now();

                                if (loaderMode === "combined") {{
                                    updateStatus("Loading combined 3D model (61.6 MB)...");
                                    const url = `http://127.0.0.1:${{port}}/buildings_combined.glb`;

                                    loader.load(url, (gltf) => {{
                                        const model = gltf.scene;
                                        
                                        // Rotate around Y-axis to align Grid North with Mercator North + manual heading adjustment
                                        model.rotation.y = gridConvergenceAngle + headingCorrection;
                                        
                                        this.scene.add(model);
                                        const loadTime = ((performance.now() - startTime) / 1000).toFixed(2);
                                        updateStatus(`Loaded 273 buildings in ${{loadTime}}s (Combined)`, false);
                                        setTimeout(hideStatus, 4000);
                                    }}, (xhr) => {{
                                        if (xhr.total) {{
                                            const pct = Math.round((xhr.loaded / xhr.total) * 100);
                                            updateStatus(`Downloading combined model: ${{pct}}%`);
                                        }}
                                    }}, (err) => {{
                                        updateStatus("Error loading combined GLB mesh", false);
                                        console.error(err);
                                    }});
                                }} else {{
                                    let loadedCount = 0;
                                    const totalCount = buildings.length;
                                    updateStatus(`Loading 273 separate GLB files (0/${{totalCount}})...`);

                                    buildings.forEach((bldg) => {{
                                        const url = `http://127.0.0.1:${{port}}/glb/building_${{bldg.id}}.glb`;
                                        loader.load(url, (gltf) => {{
                                            const model = gltf.scene;

                                            // Position relative to UTM origin
                                            model.position.x = bldg.dx;
                                            model.position.y = bldg.dz;   // Elevation is Y in Three.js (after rotationX)
                                            model.position.z = -bldg.dy;  // Northing is -Z in Three.js (after rotationX)

                                            // Align Grid North + manual heading adjustment
                                            model.rotation.y = gridConvergenceAngle + headingCorrection;

                                            this.scene.add(model);
                                            loadedCount++;

                                            updateStatus(`Loading individual GLBs: ${{loadedCount}}/${{totalCount}}`);

                                            if (loadedCount === totalCount) {{
                                                const loadTime = ((performance.now() - startTime) / 1000).toFixed(2);
                                                updateStatus(`Loaded all ${{totalCount}} buildings in ${{loadTime}}s (Separate)`, false);
                                                setTimeout(hideStatus, 4000);
                                            }}
                                        }}, undefined, (err) => {{
                                            console.error(`Error loading building_${{bldg.id}}.glb:`, err);
                                            loadedCount++;
                                            if (loadedCount === totalCount) {{
                                                const loadTime = ((performance.now() - startTime) / 1000).toFixed(2);
                                                updateStatus(`Loaded ${{loadedCount}} buildings with errors in ${{loadTime}}s`, false);
                                                setTimeout(hideStatus, 4000);
                                            }}
                                        }});
                                    }});
                                }}
                            }},
                            render: function (gl, matrix) {{
                                const rotationX = new THREE.Matrix4().makeRotationX(modelTransform.rotateX);
                                const rotationY = new THREE.Matrix4().makeRotationY(modelTransform.rotateY);
                                const rotationZ = new THREE.Matrix4().makeRotationZ(modelTransform.rotateZ);

                                const m = new THREE.Matrix4().fromArray(matrix);
                                const l = new THREE.Matrix4()
                                    .makeTranslation(
                                        modelTransform.translateX,
                                        modelTransform.translateY,
                                        modelTransform.translateZ
                                    )
                                    .scale(
                                        new THREE.Vector3(
                                            modelTransform.scale,
                                            -modelTransform.scale,
                                            modelTransform.scale
                                        )
                                    )
                                    .multiply(rotationX)
                                    .multiply(rotationY)
                                    .multiply(rotationZ);

                                this.camera.projectionMatrix = m.multiply(l);
                                if (typeof this.renderer.resetState === 'function') {{
                                    this.renderer.resetState();
                                }} else if (this.renderer.state && typeof this.renderer.state.reset === 'function') {{
                                    this.renderer.state.reset();
                                }}
                                this.renderer.render(this.rootScene, this.camera);
                                map.triggerRepaint();
                            }}
                        }};

                        map.on('style.load', () => {{
                            map.addLayer(custom3DLayer);
                        }});
                    </script>
                </body>
                </html>
                """
                
                st.components.v1.html(html_code, height=750)
                
                st.markdown(f"""
                ### Map Specifications:
                * **Map Engine:** MapLibre GL JS + Three.js Integration
                * **Visual Style:** CartoDB Dark Matter base map
                * **UTM Origin Reference:** `Lon: -79.92814012, Lat: 43.25779592` (E587000.0, N4790000.0)
                * **Grid Convergence Correction:** `-0.732016°` (Z-axis rotation)
                * **Local Server Port:** `{port}`
                * **Interactive Controls:** Click and drag to pan, scroll/pinch to zoom, right click and drag (or Ctrl+drag) to change pitch/bearing.
                """)

else:
    # Plotly Scientific Analytics Mode
    st.sidebar.markdown("---")
    st.sidebar.header("Scientific Tools")
    
    data_choice = st.sidebar.selectbox("Select Data Source:", [
        "Building Footprints (2D)", 
        "nDSM (Heights)", 
        "Roughness Index (Trees)"
    ])
    
    with st.spinner(f"Loading {data_choice} for Analytics..."):
        if data_choice == "Building Footprints (2D)":
            dataset_choice = st.sidebar.selectbox("Footprint Source:", ["Overture (Modern)", "LiDAR (Auto-Extracted)", "LiDAR (Rooftop Raw)", "LiDAR (Method D Healed)", "StatCan (Legacy)"])
            source_map = {
                "Overture (Modern)": overture_path,
                "LiDAR (Auto-Extracted)": lidar_path,
                "StatCan (Legacy)": statcan_path,
                "LiDAR (Rooftop Raw)": rooftops_path,
                "LiDAR (Method D Healed)": rooftops_clean_path
            }
            active_path = source_map[dataset_choice]
            
            gdf = load_data_with_status(active_path)
            if gdf is not None:
                fig = px.choropleth_map(
                    gdf,
                    geojson=gdf.geometry,
                    locations=gdf.index,
                    color='height_p90',
                    hover_name='address' if 'address' in gdf.columns else gdf.index,
                    hover_data={'height_p90': True, 'height_max': True},
                    map_style='carto-darkmatter',
                    zoom=14.5,
                    center={'lat': gdf.geometry.centroid.y.mean(), 'lon': gdf.geometry.centroid.x.mean()},
                    color_continuous_scale='plasma',
                    labels={'height_p90': 'Height'}
                )
                
                # Format the colorbar for buildings
                fig.update_layout(
                    margin=dict(l=0, r=0, t=50, b=0),
                    height=700,
                    coloraxis_colorbar=dict(
                        title="", # Remove "Scale"
                        ticksuffix=" m", # Add the unit!
                        thicknessmode="pixels", thickness=20,
                        lenmode="pixels", len=400,
                        yanchor="middle", y=0.5,
                        xanchor="left", x=1.02
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.info("💡 **Tip:** Hover your mouse over any building to see its exact metadata. Plotly natively supports 2D building analytics but relies on PyDeck (the other tab) for 3D extrusion!")
            else:
                st.error("Dataset not found. Please run main.py to generate the data.")

        else:
            # Raster Analytics
            if data_choice == "nDSM (Heights)":
                tif_name = f"{file_prefix}_ndsm_output.tif"
                cmap = 'plasma'
                title = "nDSM Raw Height Map"
                cmax = 30
                unit_suffix = " m"
            else:
                tif_name = f"{file_prefix}_roughness_output.tif"
                cmap = 'viridis'
                title = "Surface Roughness Index (Tree Filter)"
                cmax = 5
                unit_suffix = "" # Roughness is a unitless index
                
            with rasterio.open(processed_dir / tif_name) as src:
                data = src.read(1)
            
            # Downsample slightly to ensure the browser doesn't lag with millions of hover points
            data = data[::2, ::2]
            data[data < 0] = np.nan
            
            fig = px.imshow(
                data, 
                color_continuous_scale=cmap, 
                title=title,
                labels={'color': 'Value'},
                zmax=cmax
            )
            
            # Optimize hover tooltips
            fig.update_traces(
                hovertemplate=f"X: %{{x}}<br>Y: %{{y}}<br><b>Value: %{{z:.2f}}{unit_suffix}</b><extra></extra>"
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=50, b=0),
                height=700,
                coloraxis_colorbar=dict(
                    title="", # Remove "Scale" title
                    ticksuffix=unit_suffix, # Add the unit!
                    thicknessmode="pixels", thickness=20,
                    lenmode="pixels", len=400,
                    yanchor="middle", y=0.5,
                    xanchor="left", x=1.02
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 **Tip:** Hover your mouse over any pixel on the map to see the exact recorded value. You can also zoom in by clicking and dragging a box!")


