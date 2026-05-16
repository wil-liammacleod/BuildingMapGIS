import streamlit as st
import pydeck as pdk
import geopandas as gpd
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
    
    # For very large files (e.g. rooftop segmentation), simplify to stay
    # under Streamlit's 200MB message size limit.
    file_size_mb = _get_file_size_mb(path)
    needs_simplification = file_size_mb > 10
    # NOTE: No st.* calls allowed inside @st.cache_data — notify the caller instead.
    
    gdf = gpd.read_file(path)
    
    if needs_simplification:
        # Simplify in a projected CRS (meters) for a meaningful tolerance,
        # then convert back. 0.5m tolerance is invisible at city scale.
        original_crs = gdf.crs
        gdf = gdf.to_crs("EPSG:2958")
        
        # Drop tiny fragments (< 10 m²) that clutter the view
        gdf = gdf[gdf.geometry.area >= 10.0].copy()
        
        # Simplify geometry vertices (0.5m tolerance)
        gdf.geometry = gdf.geometry.simplify(tolerance=0.5, preserve_topology=True)
        gdf = gdf.to_crs(original_crs)
        
        # Drop columns not needed for visualization to reduce payload
        keep_cols = {'geometry', 'height_p90', 'height_max', 'address', 'type',
                     'AVE_HGT', 'SLOPE', 'ASPECT', 'AREA', 'VALUE'}
        drop_cols = [c for c in gdf.columns if c not in keep_cols]
        if drop_cols:
            gdf = gdf.drop(columns=drop_cols)
    
    # Generate safe colors based on height
    def calculate_color(height):
        r = 255
        g = max(0, 255 - int(height * 4))
        b = max(0, 255 - int(height * 6))
        return [r, g, b, 200]
        
    # Ensure height columns exist
    if 'height_p90' not in gdf.columns:
        gdf['height_p90'] = 10.0 # Fallback
    if 'height_max' not in gdf.columns:
        gdf['height_max'] = 10.0
        
    gdf['color'] = gdf['height_p90'].apply(calculate_color)
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
        "LiDAR (Cleaned Blocks)": rooftops_clean_path
    }
    
    all_sources = ["Overture (Modern)", "LiDAR (Auto-Extracted)", "LiDAR (Rooftop Raw)", "LiDAR (Cleaned Blocks)", "StatCan (Legacy)"]
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
        longitude=gdf_statcan.geometry.centroid.x.mean() if gdf_statcan is not None else -79.918,
        latitude=gdf_statcan.geometry.centroid.y.mean() if gdf_statcan is not None else 43.261,
        zoom=14.5, 
        pitch=0, 
        bearing=0
    )
    
    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=pdk.map_styles.DARK
    ), use_container_width=True)

elif view_mode == "Interactive 3D Map (PyDeck)":
    st.sidebar.markdown("---")
    st.sidebar.header("Map Layers")
    
    dataset_choice = st.sidebar.selectbox("Footprint Source:", ["Overture (Modern)", "LiDAR (Auto-Extracted)", "LiDAR (Rooftop Raw)", "LiDAR (Cleaned Blocks)", "StatCan (Legacy)"])
    
    source_map = {
        "Overture (Modern)": overture_path,
        "LiDAR (Auto-Extracted)": lidar_path,
        "StatCan (Legacy)": statcan_path,
        "LiDAR (Rooftop Raw)": rooftops_path,
        "LiDAR (Cleaned Blocks)": rooftops_clean_path
    }
    active_path = source_map[dataset_choice]
    
    show_3d = st.sidebar.checkbox("Show 3D Buildings", value=True)
    show_2d = st.sidebar.checkbox("Show 2D Footprints", value=False)
    show_ndsm = st.sidebar.checkbox("Show nDSM (Height Raster)", value=False)
    show_roughness = st.sidebar.checkbox("Show Roughness (Trees Raster)", value=False)

    gdf = load_data_with_status(active_path)
    layers = []

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
        if show_2d:
            layers.append(pdk.Layer(
                "PolygonLayer",
                gdf,
                get_polygon="geometry.coordinates",
                pickable=True,
                auto_highlight=True,
                opacity=0.5,
                stroked=True,
                filled=True,
                extruded=False,
                get_fill_color="color",
                get_line_color=[255, 255, 255]
            ))
    
        if show_3d:
            layers.append(pdk.Layer(
                "PolygonLayer",
                gdf,
                get_polygon="geometry.coordinates",
                pickable=True,
                auto_highlight=True,
                opacity=0.8,
                stroked=False,
                filled=True,
                extruded=True,
                wireframe=True,
                get_elevation="height_p90",
                get_fill_color="color",
                get_line_color=[255, 255, 255]
            ))
    
        view_state = pdk.ViewState(
            longitude=gdf.geometry.centroid.x.mean(),
            latitude=gdf.geometry.centroid.y.mean(),
            zoom=14.5, 
            pitch=45 if show_3d else 0, 
            bearing=0
        )
    else:
        view_state = pdk.ViewState(longitude=-79.918, latitude=43.261, zoom=14.5)

    # Build tooltip — include slope/aspect if available (rooftop segments)
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
    ), use_container_width=True)

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
            dataset_choice = st.sidebar.selectbox("Footprint Source:", ["Overture (Modern)", "LiDAR (Auto-Extracted)", "LiDAR (Rooftop Raw)", "LiDAR (Cleaned Blocks)", "StatCan (Legacy)"])
            source_map = {
                "Overture (Modern)": overture_path,
                "LiDAR (Auto-Extracted)": lidar_path,
                "StatCan (Legacy)": statcan_path,
                "LiDAR (Rooftop Raw)": rooftops_path,
                "LiDAR (Cleaned Blocks)": rooftops_clean_path
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
