"""Net Primary Productivity (NPP) helpers"""

import numpy as np
import xarray as xr
from common import KG_PER_PGC, SECONDS_PER_DAY

def annual_regional_npp(files, land_area, lat_min=60, lat_max=90):
    """Return annual area-integrated NPP in Pg C yr⁻¹."""
    files = sorted(map(str, files))
    if not files:
        raise FileNotFoundError("No NetCDF files found for this experiment.")
    coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_mfdataset(
        files, combine="by_coords", data_vars="minimal", coords="minimal",
        compat="override", chunks={"time": 12}, decode_times=coder, parallel=False,
    ) as ds:
        npp = ds["npp"]
        units = npp.attrs.get("units", "").replace(" ", "").lower()
        if units != "kgm-2s-1":
            raise ValueError(f"Expected npp units kg m-2 s-1, got {npp.attrs.get('units')!r}")
        np.testing.assert_allclose(npp.lat, land_area.lat, rtol=0, atol=1e-7)
        np.testing.assert_allclose(npp.lon, land_area.lon, rtol=0, atol=1e-7)
        npp = npp.assign_coords(lat=land_area.lat, lon=land_area.lon)
        area = land_area.sel(lat=npp.lat, lon=npp.lon)
        npp = npp.where((npp.lat >= lat_min) & (npp.lat <= lat_max), drop=True)
        area = area.sel(lat=npp.lat)
        seconds = npp.time.dt.days_in_month.astype("float64") * SECONDS_PER_DAY
        monthly_kg = npp * area * seconds
        annual = monthly_kg.groupby("time.year").sum("time", skipna=True).sum(
            ("lat", "lon"), skipna=True
        ) / KG_PER_PGC
        months = npp.time.groupby("time.year").count()
        annual = annual.where(months == 12, drop=True).rename("npp")
        annual.attrs.update({"units": "Pg C yr-1", "long_name": f"Annual total NPP over {lat_min}–{lat_max}°N"})
        return annual.load()
