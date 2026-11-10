"""Near-surface Air Temperature (TAS) helpers"""

import numpy as np
import xarray as xr

def annual_regional_tas(files, land_area, lat_min=60, lat_max=90, batch_size=25):
    """Return annual land-area-weighted tas in °C."""
    files = sorted(map(str, files))
    if not files:
        raise FileNotFoundError("No NetCDF files found for this experiment.")
    coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    annual_parts = []
    for start in range(0, len(files), batch_size):
        batch = files[start:start + batch_size]
        with xr.open_mfdataset(
            batch, combine="by_coords", data_vars="minimal", coords="minimal",
            compat="override", join="override", chunks={"time": 12},
            decode_times=coder, parallel=False,
        ) as ds:
            tas = ds["tas"]
            units = tas.attrs.get("units", "").strip().lower()
            if units not in {"k", "kelvin", "degree_celsius", "degrees_celsius", "degc"}:
                raise ValueError(f"Unsupported tas units: {tas.attrs.get('units')!r}")
            np.testing.assert_allclose(tas.lat, land_area.lat, rtol=0, atol=1e-7)
            np.testing.assert_allclose(tas.lon, land_area.lon, rtol=0, atol=1e-7)
            tas = tas.assign_coords(lat=land_area.lat, lon=land_area.lon)
            lat_idx = np.flatnonzero((tas.lat >= lat_min) & (tas.lat <= lat_max))
            tas, area = tas.isel(lat=lat_idx), land_area.isel(lat=lat_idx)
            weights = area.where(tas.notnull())
            monthly = (tas * weights).sum(("lat", "lon"), skipna=True) / weights.sum(
                ("lat", "lon"), skipna=True
            )
            monthly = monthly.load()
        days = monthly.time.dt.days_in_month.astype("float64")
        month_count = monthly.time.groupby("time.year").count()
        valid_days = days.where(monthly.notnull()).groupby("time.year").sum("time", skipna=True)
        annual = (monthly * days).groupby("time.year").sum("time", skipna=True) / valid_days
        annual = annual.where(month_count == 12, drop=True)
        if units in {"k", "kelvin"}:
            annual = annual - 273.15
        annual_parts.append(annual.load())
    result = xr.concat(annual_parts, dim="year", join="override").sortby("year").rename("tas")
    result.attrs.update({"units": "°C", "long_name": f"Annual land-area-weighted tas over {lat_min}–{lat_max}°N"})
    return result
