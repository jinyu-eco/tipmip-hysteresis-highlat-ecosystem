"""Reusable analysis functions live here for explicit imports from notebooks and scripts."""

from pathlib import Path
import re
import sys
import numpy as np
import xarray as xr

# Constants
SECONDS_PER_DAY = 86_400
KG_PER_PGC = 1e12


def find_project_root(start=None):
    """Find the project root directory by searching upward for the 'utils' folder."""
    folder = Path(start or Path.cwd()).resolve()
    for path in (folder, *folder.parents):
        if (path / "utils").is_dir():
            return path
    raise FileNotFoundError("Cannot find project root with utils.")

# Grid Utility: Infer Cell Edges from Centers
def center_edges(centers, lower_limit=None, upper_limit=None):
    """
    Infer 1-D cell edges from grid cell center coordinates.
    Parameters:
    ------------------
    centers : array-like
        1-D array of grid cell center coordinates.
    lower_limit : float, optional
        Physical lower bound to enforce on the first edge (e.g., min latitude 60).
    upper_limit : float, optional
        Physical upper bound to enforce on the last edge (e.g., max latitude 90).
    Returns
    -----------------
    np.ndarray
        1-D array of cell edges with length equal to len(centers) + 1.
    """
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("centers must be one-dimensional and contain at least two values")
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    edges[0] = centers[0] - (centers[1] - centers[0]) / 2
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2
    if lower_limit is not None:
        edges[0] = lower_limit
    if upper_limit is not None:
        edges[-1] = upper_limit
    return edges
    
def _edges(centres, low=None, high=None):
    """Compatibility name used by the earlier helpers.py API."""
    return center_edges(centres, lower_limit=low, upper_limit=high)


def load_ec_land_area(project_root):
    """Load EC land area (areacella × sftlf / 100) in m²."""
    fx = Path(project_root) / "fx"
    area_file = fx / "areacella_APfx_EC-Earth3-ESM-1_esm-up2p0_r1i1p1f1_gr.nc"
    land_file = fx / "sftlf_APfx_EC-Earth3-ESM-1_esm-up2p0_r1i1p1f1_gr.nc"
    with xr.open_dataset(area_file) as ds:
        area = ds["areacella"].load()
    with xr.open_dataset(land_file) as ds:
        land_percent = ds["sftlf"].load()
    np.testing.assert_allclose(area.lat, land_percent.lat, rtol=0, atol=1e-8)
    np.testing.assert_allclose(area.lon, land_percent.lon, rtol=0, atol=1e-8)
    land_percent = land_percent.assign_coords(lat=area.lat, lon=area.lon)
    land_area = area * land_percent / 100
    land_area.name = "land_area"
    land_area.attrs["units"] = "m2"
    return land_area


# egrid Utility: Conservative Area Remapping across ESM Grids
def remap_land_area_to_esm(ec_area, target_lat, target_lon):
    """
    Conservatively remap 2D regular-grid land area data to a target grid.
    Parameters:
    ------------------
    ec_area : xr.DataArray
        Source grid land area array (with 'lat' and 'lon' coordinates).
    target_lat : array-like
        1D target latitude coordinates.
    target_lon : array-like
        1D target longitude coordinates.
    Returns:
    -----------------
    xr.DataArray
        Conservatively remapped land area on the target grid (unit: m²).
    """
    src_lat = np.asarray(ec_area.lat, dtype=float)
    src_lon_original = np.asarray(ec_area.lon, dtype=float)
    dst_lat = np.asarray(target_lat, dtype=float)
    dst_lon_original = np.asarray(target_lon, dtype=float)

    src_lon = np.mod(src_lon_original, 360)
    src_order = np.argsort(src_lon)
    src_lon = src_lon[src_order]
    dst_lon = np.mod(dst_lon_original, 360)
    dst_order = np.argsort(dst_lon)
    dst_lon = dst_lon[dst_order]
    src_lat_edges = center_edges(src_lat, -90, 90)
    dst_lat_edges = center_edges(dst_lat, -90, 90)
    src_lon_edges = center_edges(src_lon, 0, 360)
    dst_lon_edges = center_edges(dst_lon, 0, 360)

    sy0, sy1 = np.minimum(src_lat_edges[:-1], src_lat_edges[1:]), np.maximum(src_lat_edges[:-1], src_lat_edges[1:])
    dy0, dy1 = np.minimum(dst_lat_edges[:-1], dst_lat_edges[1:]), np.maximum(dst_lat_edges[:-1], dst_lat_edges[1:])
    overlap_y = np.maximum(
        0,
        np.sin(np.deg2rad(np.minimum(dy1[:, None], sy1)))
        - np.sin(np.deg2rad(np.maximum(dy0[:, None], sy0))),
    )
    source_y = np.sin(np.deg2rad(sy1)) - np.sin(np.deg2rad(sy0))
    y_fraction = np.divide(overlap_y, source_y, out=np.zeros_like(overlap_y), where=source_y > 0)
    overlap_x = np.maximum(
        0,
        np.minimum(dst_lon_edges[1:, None], src_lon_edges[1:])
        - np.maximum(dst_lon_edges[:-1, None], src_lon_edges[:-1]),
    )
    source_x = np.diff(src_lon_edges)
    x_fraction = np.divide(overlap_x, source_x, out=np.zeros_like(overlap_x), where=source_x > 0)
    remapped = y_fraction @ ec_area.isel(lon=src_order).values @ x_fraction.T
    remapped = remapped[:, np.argsort(dst_order)]
    return xr.DataArray(
        remapped, dims=("lat", "lon"), coords={"lat": target_lat, "lon": target_lon},
        name="land_area", attrs={"units": "m2", "comment": "Conservatively remapped land area"},
    )


def remap_cell_area(source_area, source_lat_bounds, source_lon_bounds, target_lat, target_lon):
    """Remap cell area conservatively to a target grid using explicit source bounds.
    Parameters:
    ------------------
    source_area : xr.DataArray
        Source grid cell area data.
    source_lat_bounds : array-like
        2D array of source latitude bounds shape (N_lat, 2).
    source_lon_bounds : array-like
        2D array of source longitude bounds shape (N_lon, 2).
    target_lat : array-like
        1D target latitude coordinates.
    target_lon : array-like
        1D target longitude coordinates.
    Returns:
    -----------------
    xr.DataArray
        Conservatively remapped cell area (areacella) on the target grid.
    """
    source_lat_bounds = np.asarray(source_lat_bounds, dtype=float)
    source_lon_bounds = np.asarray(source_lon_bounds, dtype=float)
    target_lat = np.asarray(target_lat, dtype=float)
    target_lon = np.asarray(target_lon, dtype=float)
    lat_edges = center_edges(target_lat, -90, 90)
    seam = (target_lon[-1] + target_lon[0] + 360) / 2
    lon_edges = np.r_[seam - 360, (target_lon[:-1] + target_lon[1:]) / 2, seam]

    south = np.minimum(source_lat_bounds[:, 0], source_lat_bounds[:, 1])
    north = np.maximum(source_lat_bounds[:, 0], source_lat_bounds[:, 1])
    overlap_south = np.maximum(lat_edges[:-1, None], south[None, :])
    overlap_north = np.minimum(lat_edges[1:, None], north[None, :])
    source_measure = np.sin(np.deg2rad(north)) - np.sin(np.deg2rad(south))
    overlap_measure = (
        np.sin(np.deg2rad(np.maximum(overlap_north, overlap_south)))
        - np.sin(np.deg2rad(np.minimum(overlap_north, overlap_south)))
    )
    lat_weights = np.divide(
        overlap_measure, source_measure[None, :], out=np.zeros_like(overlap_measure),
        where=source_measure[None, :] > 0,
    )
    lat_weights[overlap_north <= overlap_south] = 0

    lon_weights = np.zeros((target_lon.size, source_lon_bounds.shape[0]))
    for source_i, bounds in enumerate(source_lon_bounds):
        west, east = float(min(bounds)) % 360, float(max(bounds)) % 360
        width = float(max(bounds) - min(bounds))
        segments = [(west, east)] if west < east else [(west, 360), (0, east)]
        for target_i in range(target_lon.size):
            target_west, target_east = lon_edges[target_i:target_i + 2]
            overlap = sum(
                max(0, min(target_east, seg_east) - max(target_west, seg_west))
                for seg_west, seg_east in segments
            )
            lon_weights[target_i, source_i] = overlap / width

    values = lat_weights @ source_area.values @ lon_weights.T
    return xr.DataArray(
        values, dims=("lat", "lon"), coords={"lat": target_lat, "lon": target_lon},
        name="areacella", attrs={"units": "m2", "source": "Conservatively remapped cell area"},
    )


def map_annual_to_gwl(annual, model, member, leg, project_root):
    """
    Apply the TIPMIP annual-to-GWL mapping using the external 'tipmip-gwl' tool.
    Parameters:
    ------------------
    annual : xr.DataArray or pd.DataFrame
        Annual aggregated dataset
    model : str
        ESM Model name (e.g., 'EC-Earth3-ESM-1', 'UKESM1-2-LL')
    member : str
        Ensemble member ID (e.g., 'r1i1p1f1')
    leg : str
        TIPMIP experiment leg (e.g., 'Exp1', 'Exp2', 'Exp4')
    project_root : str or Path
        Root directory of the project

    Returns:
    ------------------
    xr.DataArray or pd.DataFrame
        Data resampled onto GWL axes
    """
    source = str(Path(project_root) / "tipmip-gwl" / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from tipmip_gwl import load_mapping, resample_to_gwl
    mapping = load_mapping(model, member=member, leg=leg)
    return resample_to_gwl(mapping, annual)

    gwl_src_path = Path(project_root) / "tipmip-gwl" / "src"
    if not gwl_src_path.exists():
        raise FileNotFoundError(
            f"The external dependency 'tipmip-gwl' was not found at '{gwl_src_path}'.\n"
            f"Please clone it first via: git clone https://bgithub.com/tipmip-hub/tipmip-gwl.git\n"
        )
    # Add source path to sys.path if not present
    gwl_src_str = str(gwl_src_path)
    if gwl_src_str not in sys.path:
        sys.path.insert(0, gwl_src_str)
    # Import functions from external module
    try:
        from tipmip_gwl import load_mapping, resample_to_gwl
    except ImportError as e:
        raise ImportError(
            f"Failed to import 'tipmip_gwl'. Ensure requirements for tipmip-gwl are installed.\n"
        ) from e        
    # Execute GWL mapping and resampling
    mapping = load_mapping(model, member=member, leg=leg)
    return resample_to_gwl(mapping, annual)
