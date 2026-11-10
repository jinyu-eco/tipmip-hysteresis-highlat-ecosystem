"""Reusable analysis functions for Vegetation Carbon (Cveg) data processing live here for explicit imports from notebooks and scripts."""

from pathlib import Path
import numpy as np
import xarray as xr
from common import KG_PER_PGC, center_edges


def annual_total_carbon(files, land_area, variable="cVeg", lat_min=60, lat_max=90):
    """Return annual regional vegetation carbon stock in Pg C."""
    files = sorted(map(str, files))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found for {variable}.")
    coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_mfdataset(
        files, combine="by_coords", data_vars="minimal", coords="minimal",
        compat="override", chunks={"time": 12}, decode_times=coder, parallel=False,
    ) as ds:
        carbon = ds[variable]
        np.testing.assert_allclose(carbon.lat, land_area.lat, rtol=0, atol=1e-7)
        np.testing.assert_allclose(carbon.lon, land_area.lon, rtol=0, atol=1e-7)
        carbon = carbon.assign_coords(lat=land_area.lat, lon=land_area.lon)
        carbon = carbon.where((carbon.lat >= lat_min) & (carbon.lat <= lat_max), drop=True)
        area = land_area.sel(lat=carbon.lat, lon=carbon.lon)
        days = carbon.time.dt.days_in_month.astype("float64")
        weighted_sum = (carbon.fillna(0) * days).groupby("time.year").sum("time", skipna=True)
        valid_days = days.where(carbon.notnull()).groupby("time.year").sum("time", skipna=True)
        annual_density = weighted_sum / valid_days
        annual = (annual_density * area).sum(("lat", "lon"), skipna=True, min_count=1) / KG_PER_PGC
        annual = annual.rename("total_cVeg")
        annual.attrs.update({"units": "Pg C", "long_name": f"Annual total {variable} over {lat_min}–{lat_max}°N"})
        return annual.compute()


def load_cmip_variable(base_dir, experiment, var_name):
    """Open one NASA CMIP variable."""
    nasa_dir = Path(base_dir) / f"{experiment}_data" / "NASA"
    if var_name.startswith("cVeg"):
        folder = nasa_dir / f"{experiment}_NASA_cVeg" / f"{experiment}_NASA_{var_name}"
    else:
        folder = nasa_dir / f"{experiment}_NASA_{var_name}"
    files = sorted(folder.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files found in folder: {folder}")
    coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_mfdataset(files, combine="by_coords", decode_times=coder, chunks={"time": 12})
    if var_name not in ds.data_vars:
        available = list(ds.data_vars)
        ds.close()
        raise KeyError(f"{var_name} not found in {folder}; available variables: {available}")
    return ds[var_name]


def calculate_total_cveg(base_dir, experiment, pfts=("Tree", "Shrub", "Grass")):
    """Combine cVeg tiles weighted by tile fraction (%). """
    components = []
    for pft in pfts:
        carbon = load_cmip_variable(base_dir, experiment, f"cVeg{pft}")
        fraction = load_cmip_variable(base_dir, experiment, f"{pft.lower()}Frac")
        carbon, fraction = xr.align(carbon, fraction, join="exact")
        components.append((carbon * fraction / 100).fillna(0))
    total = xr.concat(components, dim="pft").sum("pft")
    total.name = "cVeg"
    total.attrs = dict(components[0].attrs)
    total.attrs.update({
        "long_name": "Grid-cell vegetation carbon density weighted by vegetation-tile fractions",
        "units": "kg m-2",
    })
    return total
