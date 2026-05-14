# 🏙️ BuildingMapGIS: Multi-Source 3D Geospatial Dashboard

An interactive GIS pipeline and dashboard for extracting, analyzing, and visualizing 3D building footprints from LiDAR and satellite data. This project specifically focuses on the McMaster University campus and surrounding Ontario regions.

## 🚀 Quick Start

1.  **Install `uv`** (if not already installed):
    Refer to the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

2.  **Install Dependencies**:
    ```bash
    uv sync
    ```

3.  **Run the Processing Pipeline**:
    This script downloads raw LiDAR data, calculates building heights, and extracts native footprints.
    ```bash
    uv run src/main.py
    ```

4.  **Launch the Dashboard**:
    Open the interactive web interface to explore the data.
    ```bash
    uv run streamlit run src/app.py
    ```

---

## 🛠️ Project Structure

### Core Scripts
*   **`src/main.py`**: The central orchestration script. It handles the end-to-end data pipeline, including coordinate reprojection, LiDAR raster math (nDSM/Roughness), and building height extraction.
*   **`src/app.py`**: The Streamlit web application. Features three modes:
    *   **Scientific Analytics**: Pixel-level raster analysis using Plotly.
    *   **Interactive 3D Map**: 3D building extrusions and raster overlays using PyDeck.
    *   **Dataset Comparison**: Side-by-side comparison of different footprint sources.

### Modules
*   **`src/importer.py`**: Handles API requests to the Ontario GeoHub (Elevation) and streaming data from Overture Maps Foundation (Amazon S3).
*   **`src/api_test.py`**: A diagnostic utility to verify connectivity to Ontario's ArcGIS ImageServer APIs.

---

## 📊 Dataset Sources

The application allows you to compare three distinct building footprint sources:

1.  **Overture (Modern)**: Latest building footprints from the Overture Maps Foundation (via Amazon S3). High coverage but occasionally lacks complex roof detail.
2.  **LiDAR (Auto-Extracted)**: Generated natively by this project using Whitebox Workflows. These footprints are traced directly from LiDAR height maps, providing the highest accuracy for multi-tiered roofs and perfect alignment.
3.  **StatCan (Legacy)**: Footprints from the Statistics Canada Open Database of Buildings. Reliable but often several years out of date.

---

## 🔍 Features

*   **3D Extrusions**: Visualize buildings in 3D with colors mapped to their actual physical height.
*   **nDSM Calculation**: Automatically generates a Normalized Digital Surface Model (Building Height = Ground - Surface).
*   **Vegetation Filtering**: Uses a Surface Ruggedness Index to distinguish between buildings and trees.
*   **Multi-Tiered Detection**: Native LiDAR extraction shatters complex buildings into sub-polygons based on roof height differences.
*   **Comparison Mode**: Highlights discrepancies between satellite-derived and LiDAR-derived datasets using color-coded overlays.
