"""Vegetation fraction helpers"""

from pathlib import Path
import re
import numpy as np
import xarray as xr


def annual_area_weighted_fraction(files, variable, area, lat_min=60, lat_max=90, batch_size=12):
    """Return complete-year area/day-weighted means in source units."""
    files = sorted(map(str, files))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found for {variable}.")
    coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    annual_parts = []
    batches = (files[i:i + batch_size] for i in range(0, len(files), batch_size))
    for batch in batches:
        with xr.open_mfdataset(
            batch, combine="by_coords", data_vars="minimal", coords="minimal",
            compat="override", chunks={"time": 12}, decode_times=coder, parallel=False,
        ) as ds:
            fraction = ds[variable]
            fraction = fraction.where((fraction.lat >= lat_min) & (fraction.lat <= lat_max), drop=True)
            np.testing.assert_allclose(fraction.lat, area.lat.sel(lat=fraction.lat), rtol=0, atol=1e-7)
            np.testing.assert_allclose(fraction.lon, area.lon.sel(lon=fraction.lon), rtol=0, atol=1e-7)
            fraction = fraction.assign_coords(
                lat=area.lat.sel(lat=fraction.lat), lon=area.lon.sel(lon=fraction.lon)
            )
            weights = area.sel(lat=fraction.lat, lon=fraction.lon)
            days = fraction.time.dt.days_in_month.astype("float64")
            numerator = (fraction.fillna(0) * weights * days).groupby("time.year").sum(
                ("time", "lat", "lon"), skipna=True, min_count=1
            )
            denominator = (weights.where(fraction.notnull(), 0) * days).groupby("time.year").sum(
                ("time", "lat", "lon"), skipna=True, min_count=1
            )
            annual = (numerator / denominator).where(denominator > 0)
            months = fraction.time.groupby("time.year").count()
            annual = annual.where(months == 12, drop=True).rename(variable)
            annual.attrs.update({
                "units": fraction.attrs.get("units", "%"),
                "long_name": f"Annual area-weighted {variable} over {lat_min}–{lat_max}°N",
            })
            annual_parts.append(annual.load())
    if not annual_parts:
        raise ValueError(f"No complete years found for {variable}.")
    annual = xr.concat(annual_parts, dim="year").sortby("year")
    if annual.get_index("year").has_duplicates:
        raise ValueError(f"Duplicate annual values for {variable}; check file overlaps.")
    return annual


def files_for(experiment, variable, experiment_folders, model, member=None):
    """List files for an experiment/model/variable."""
    folder = Path(experiment_folders[experiment]) / f"{experiment}_{model}_{variable}"
    pattern = f"*_{member}_*.nc" if member else "*.nc"
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found for {experiment} {variable}: {folder}")
    return files


def file_year_bounds(file):
    """Parse YYYYMM-YYYYMM from a CMIP filename."""
    match = re.search(r"_(\d{6})-(\d{6})\.nc$", Path(file).name)
    if match is None:
        return None
    start, end = match.groups()
    return int(start[:4]), int(start[4:6]), int(end[:4]), int(end[4:6])


def complete_years_in_files(files, use_filename_bounds=True):
    """Return years with 12 monthly records."""
    month_counts = {}
    coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    for file in files:
        bounds = file_year_bounds(file) if use_filename_bounds else None
        if bounds is None:
            with xr.open_dataset(file, decode_times=coder) as ds:
                years, counts = np.unique(np.asarray(ds.time.dt.year.values, dtype=int), return_counts=True)
                for year, count in zip(years, counts):
                    month_counts[int(year)] = month_counts.get(int(year), 0) + int(count)
            continue
        y0, m0, y1, m1 = bounds
        for year in range(y0, y1 + 1):
            first = m0 if year == y0 else 1
            last = m1 if year == y1 else 12
            month_counts[year] = month_counts.get(year, 0) + last - first + 1
    return np.array(sorted(y for y, count in month_counts.items() if count == 12), dtype=int)


def gwl_window_years(experiment, mappings, tolerance=0.05):
    """Return years/GWL values."""
    axis = mappings[experiment]["gwl_axis"]
    years, values = np.asarray(axis.year.values, dtype=int), np.asarray(axis.values, dtype=float)
    selected = np.isfinite(values) & (np.abs(values) <= tolerance)
    if not selected.any():
        raise ValueError(f"{experiment} has no mapped years in ±{tolerance:.1f}°C")
    return years[selected], values[selected]


def mean_spatial_field(files, variable, selected_years, use_filename_prefilter=True, require_all_years=True):
    """Return mean annual field across selected complete years."""
    selected_years = np.asarray(sorted(set(map(int, selected_years))), dtype=int)
    if not selected_years.size:
        raise ValueError(f"No years selected for {variable}.")
    annual_fields = []
    coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    for file in files:
        bounds = file_year_bounds(file) if use_filename_prefilter else None
        if bounds and not np.any((selected_years >= bounds[0]) & (selected_years <= bounds[2])):
            continue
        with xr.open_dataset(file, decode_times=coder) as ds:
            field = ds[variable]
            file_years = np.asarray(field.time.dt.year.values, dtype=int)
            keep = np.intersect1d(np.unique(file_years), selected_years)
            if not keep.size:
                continue
            field = field.where(field.time.dt.year.isin(keep), drop=True)
            days = field.time.dt.days_in_month.astype("float64")
            numerator = (field * days).groupby("time.year").sum("time", skipna=True, min_count=1)
            denominator = days.where(field.notnull()).groupby("time.year").sum("time", skipna=True)
            annual = (numerator / denominator).where(denominator > 0)
            months = field.time.groupby("time.year").count()
            annual = annual.where(months == 12, drop=True).load()
            if annual.sizes.get("year", 0):
                annual_fields.append(annual)
    if not annual_fields:
        raise ValueError(f"No complete annual fields found for {variable}.")
    combined = xr.concat(annual_fields, dim="year").sortby("year")
    found_years = np.asarray(combined.year.values, dtype=int)
    if require_all_years and not np.array_equal(found_years, selected_years):
        missing = sorted(set(selected_years) - set(found_years))
        raise ValueError(f"Missing complete annual fields for {variable}: {missing}")
    return combined.mean("year", skipna=True).load(), found_years


def describe_window(label, years, gwl=None, print_result=True):
    """Format a selected-period summary."""
    years = np.asarray(years)
    if not years.size:
        raise ValueError("years must contain at least one year")
    message = f"{label}: {years[0]}–{years[-1]} ({len(years)} years)"
    if gwl is not None:
        message += f", mapped GWL {np.min(gwl):.3f} to {np.max(gwl):.3f}°C"
    if print_result:
        print(message)
    return message


def calculate_fraction_spatial_changes(
    model_configs,
    project_root,
    variables,
    mappings,
    exp2_years=30,
    gwl_tolerance=0.2,
    lat_min=60,
    lat_max=90,
):
    """Calculate Exp2/Exp4 grid-cell changes from an Exp1 baseline for configured models.
    """
    project_root = Path(project_root)
    difference_fields = {}
    periods_by_model = {}
    experiments = ("Exp1", "Exp2", "Exp4")

    for config in model_configs:
        model, family = config["name"], config["family"]
        experiment_folders = {
            experiment: project_root / f"{experiment}_data" / family
            for experiment in experiments
        }
        member = config.get("data_member")
        exp1_years, exp1_gwl = gwl_window_years(
            "Exp1", mappings[model], tolerance=gwl_tolerance
        )
        exp4_years, exp4_gwl = gwl_window_years(
            "Exp4", mappings[model], tolerance=gwl_tolerance
        )
        exp2_files = files_for(
            "Exp2", variables[0], experiment_folders, family, member=member
        )
        exp2_complete_years = complete_years_in_files(exp2_files)
        if exp2_complete_years.size < exp2_years:
            raise ValueError(
                f"{model} Exp2 has only {exp2_complete_years.size} complete years; "
                f"need {exp2_years}."
            )
        selected_exp2_years = exp2_complete_years[-exp2_years:]
        selected_years = {
            "Exp1": exp1_years,
            "Exp2": selected_exp2_years,
            "Exp4": exp4_years,
        }
        periods_by_model[model] = {
            **selected_years,
            "Exp1_GWL": exp1_gwl,
            "Exp4_GWL": exp4_gwl,
        }
        difference_fields[model] = {}

        for variable in variables:
            annual_means = {}
            used_years = {}
            for experiment in experiments:
                files = files_for(
                    experiment, variable, experiment_folders, family, member=member
                )
                annual_means[experiment], used_years[experiment] = mean_spatial_field(
                    files, variable, selected_years[experiment]
                )

            exp1 = annual_means["Exp1"]
            for experiment in ("Exp2", "Exp4"):
                field = annual_means[experiment]
                np.testing.assert_allclose(exp1.lat, field.lat, rtol=0, atol=1e-7)
                np.testing.assert_allclose(exp1.lon, field.lon, rtol=0, atol=1e-7)
                annual_means[experiment] = field.assign_coords(lat=exp1.lat, lon=exp1.lon)

            north = (exp1.lat >= lat_min) & (exp1.lat <= lat_max)
            difference_fields[model][variable] = {
                "Exp2_minus_Exp1": (annual_means["Exp2"] - exp1).where(north),
                "Exp4_return_minus_Exp1": (annual_means["Exp4"] - exp1).where(north),
            }
            print(
                f"{model} {variable}: "
                + ", ".join(
                    f"{experiment} {years[0]}–{years[-1]} ({len(years)} years)"
                    for experiment, years in used_years.items()
                )
            )

    return difference_fields, periods_by_model


def save_fraction_spatial_plot_data(
    difference_fields, periods_by_model, model_configs, output_dir, lat_min=60, lat_max=90
):
    """
    Save spatial fraction differences and plot metadata as one NetCDF per model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = {}

    for config in model_configs:
        model = config["name"]
        periods = periods_by_model[model]
        data_vars = {}
        for variable, comparisons in difference_fields[model].items():
            for comparison, field in comparisons.items():
                field_name = f"{variable}_{comparison.lower()}"
                type_value = field.coords["type"].item()
                if isinstance(type_value, bytes):
                    type_value = type_value.decode()

                # Rebuild with only spatial coordinates; source files carry a
                # scalar `type` coordinate that differs between vegetation variables.
                field_2d = field.squeeze("type", drop=True) if "type" in field.dims else field
                spatial_coords = {
                    dim: field_2d.coords[dim]
                    for dim in field_2d.dims
                    if dim in field_2d.coords
                }
                field_to_save = xr.DataArray(
                    field_2d.data,
                    dims=field_2d.dims,
                    coords=spatial_coords,
                    name=field_name,
                    attrs={**field_2d.attrs, "vegetation_type": type_value},
                )
                data_vars[field_name] = field_to_save

        plot_data = xr.Dataset(data_vars)
        plot_data.attrs.update({
            "model": model,
            "dataset_family": config["family"],
            "latitude_band": f"{lat_min}–{lat_max}°N",
            "units": "percentage points",
            "exp1_years": f"{periods['Exp1'][0]}–{periods['Exp1'][-1]}",
            "exp1_mapped_gwl_range": (
                f"{periods['Exp1_GWL'].min():.3f} to "
                f"{periods['Exp1_GWL'].max():.3f} °C"
            ),
            "exp2_years": f"{periods['Exp2'][0]}–{periods['Exp2'][-1]}",
            "exp4_years": f"{periods['Exp4'][0]}–{periods['Exp4'][-1]}",
            "exp4_mapped_gwl_range": (
                f"{periods['Exp4_GWL'].min():.3f} to "
                f"{periods['Exp4_GWL'].max():.3f} °C"
            ),
            "description": (
                "Grid-cell vegetation-fraction changes. Monthly fields were "
                "day-weighted to annual means and averaged over the selected years."
            ),
        })
        for variable in difference_fields[model]:
            plot_data[f"{variable}_exp2_minus_exp1"].attrs.update({
                "units": "percentage points",
                "long_name": (
                    f"{variable}: Exp2 selected years minus Exp1 near-0°C baseline"
                ),
            })
            plot_data[f"{variable}_exp4_return_minus_exp1"].attrs.update({
                "units": "percentage points",
                "long_name": (
                    f"{variable}: Exp4 near-0°C return period minus Exp1 baseline"
                ),
            })

        filename = f"{model.replace('/', '_').replace(' ', '_')}_frac_spatial_plot_data.nc"
        output_file = output_dir / filename
        plot_data.to_netcdf(output_file)
        saved_files[model] = output_file
        print(f"Saved spatial plot data: {output_file}")

    return saved_files
