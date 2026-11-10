"""NASA-GISS vegetation-carbon preprocessing.
Prepare total grid-cell vegetation carbon from the NASA-GISS tile-level
carbon-density and vegetation-fraction NetCDF files. 
"""

from contextlib import ExitStack
from pathlib import Path

import xarray as xr


NASA_VEGETATION_TILES = {
    "Tree": "treeFrac",
    "Shrub": "shrubFrac",
    "Grass": "grassFrac",
}


def _open_nasa_variable(stack, project_root, experiment, variable):
    """Open one variable from the NASA experiment's NetCDF files."""
    nasa_dir = Path(project_root) / f"{experiment}_data" / "NASA"
    if variable.startswith("cVeg"):
        variable_dir = nasa_dir / f"{experiment}_NASA_cVeg" / f"{experiment}_NASA_{variable}"
    else:
        variable_dir = nasa_dir / f"{experiment}_NASA_{variable}"

    files = sorted(variable_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found for {variable}: {variable_dir}")

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    dataset = stack.enter_context(
        xr.open_mfdataset(
            files,
            combine="by_coords",
            data_vars="minimal",
            coords="minimal",
            decode_times=time_coder,
            chunks={"time": 12},
            parallel=False,
        )
    )
    if variable not in dataset.data_vars:
        raise KeyError(
            f"{variable} not found in {variable_dir}; "
            f"available variables: {list(dataset.data_vars)}"
        )
    return dataset[variable]


def preprocess_nasa_total_cveg(project_root, experiment, output_file=None):
    """Calculate and save NASA-GISS total grid-cell vegetation carbon.

    Each tile's carbon density is weighted by its fractional cover before the
    three tiles are summed:

    ``cVeg = (cVegTree * treeFrac + cVegShrub * shrubFrac +
    cVegGrass * grassFrac) / 100``

    The fraction variables are stored in percent. Source carbon units are
    retained in the output. Input files are read from
    ``<project_root>/<experiment>_data/NASA``. By default, the output is written
    to ``<project_root>/<experiment>_NASA_cVeg.nc``; pass ``output_file`` to
    save it elsewhere (for example, outside a GitHub repository).

    Args:
        project_root: Project directory containing the experiment data folders.
        experiment: Experiment name, such as ``"Exp2"``.
        output_file: Optional destination NetCDF path.

    Returns:
        Path to the written NetCDF file.
    """
    project_root = Path(project_root).expanduser().resolve()
    output_file = (
        Path(output_file).expanduser()
        if output_file is not None
        else project_root / f"{experiment}_NASA_cVeg.nc"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    weighted_tiles = []
    source_attrs = None
    with ExitStack() as stack:
        for tile, fraction_name in NASA_VEGETATION_TILES.items():
            carbon_name = f"cVeg{tile}"
            carbon = _open_nasa_variable(stack, project_root, experiment, carbon_name)
            fraction = _open_nasa_variable(stack, project_root, experiment, fraction_name)
            carbon, fraction = xr.align(carbon, fraction, join="exact")
            if source_attrs is None:
                source_attrs = dict(carbon.attrs)

            # The tile carbon density contributes only over its fractional cover.
            weighted_tiles.append((carbon * fraction / 100.0).fillna(0))

        weighted_tiles = xr.align(*weighted_tiles, join="exact")
        total_cveg = xr.concat(
            weighted_tiles,
            dim=xr.IndexVariable("vegetation_tile", tuple(NASA_VEGETATION_TILES)),
        ).sum("vegetation_tile", skipna=True)
        total_cveg = total_cveg.rename("cVeg")
        total_cveg.attrs = source_attrs
        total_cveg.attrs.update({
            "long_name": "Total grid-cell vegetation carbon density",
            "comment": (
                "Calculated as the sum of cVegTree, cVegShrub, and cVegGrass, "
                "each weighted by its corresponding vegetation fraction. "
                "Vegetation fractions are percentages."
            ),
        })

        output = total_cveg.to_dataset()
        output.attrs.update({
            "model": "GISS-E2-1-G-CC2",
            "dataset_family": "NASA",
            "experiment": experiment,
            "processing_method": (
                "Sum of tree, shrub, and grass carbon densities weighted by "
                "their fractional cover."
            ),
        })
        output.to_netcdf(output_file)

    return output_file
