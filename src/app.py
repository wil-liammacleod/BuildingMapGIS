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

st.set_page_config(layout="wide", page_title="Ontario 3D Building GIS")

st.title("🏙️ Ontario 3D Building Explorer")
st.sidebar.header("Navigation")

# View Mode Selection
view_mode = st.sidebar.radio(
    "Select View Mode:",
    ["Scientific Analytics (Plotly)", "Interactive 3D Map (PyDeck)", "Dataset Comparison (StatCan vs Overture)"],
    index=1,
    help="Switch between analyzing the raw LiDAR data pixel-by-pixel, and viewing the fully extruded 3D building models."
)

city_name = "McMaster"
province_name = "Ontario"
processed_dir = Path(f"./data/{province_name}/{city_name}/processed")
statcan_path = processed_dir / f"{city_name.lower()}_statcan_buildings_3d.geojson"
overture_path = processed_dir / f"{city_name.lower()}_overture_buildings_3d.geojson"
lidar_path = processed_dir / f"{city_name.lower()}_lidar_buildings_3d.geojson"
rooftops_path = processed_dir / f"{city_name.lower()}_lidar_rooftops_3d.geojson"
rooftops_clean_path = processed_dir / f"{city_name.lower()}_lidar_rooftops_clean_3d.geojson"

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
        longitude=gdf_statcan.to_crs("EPSG:2958").geometry.centroid.to_crs("EPSG:4326").x.mean() if gdf_statcan is not None else -79.918,
        latitude=gdf_statcan.to_crs("EPSG:2958").geometry.centroid.to_crs("EPSG:4326").y.mean() if gdf_statcan is not None else 43.261,
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
    
    s3db_path = processed_dir / f"{city_name.lower()}_lidar_rooftops_s3db_3d.geojson"
    blended_path = processed_dir / f"{city_name.lower()}_lidar_rooftops_blended_3d.geojson"
    
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
        bounds, img_path = get_raster_layer(f"{city_name.lower()}_ndsm_output.tif", "ndsm_v2.png", "plasma", vmin=0, vmax=30)
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
        bounds, img_path = get_raster_layer(f"{city_name.lower()}_roughness_output.tif", "roughness_v2.png", "viridis", vmin=0, vmax=5)
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
        view_state = pdk.ViewState(longitude=-79.918, latitude=43.261, zoom=14.5)

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
                tif_name = f"{city_name.lower()}_ndsm_output.tif"
                cmap = 'plasma'
                title = "nDSM Raw Height Map"
                cmax = 30
                unit_suffix = " m"
            else:
                tif_name = f"{city_name.lower()}_roughness_output.tif"
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


