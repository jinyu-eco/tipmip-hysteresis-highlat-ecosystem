# Evaluating High-Latitude Ecosystem Hysteresis under Climate Overshoot

This repository contains the analysis notebooks and supporting Python utilities for comparing high-latitude ecosystem responses in Earth system models (ESMs) during a 2 °C climate overshoot experiment. The analysis covers land areas from 60°N to 90°N and examines near-surface air temperature (`tas`), net primary production (`npp`), tree/shrub/grass fractions, and total vegetation carbon (`cVeg`). It compares the ramp-up and ramp-down branches at equivalent Global Warming Levels (GWLs), and evaluates the zero-emission phase.

## Experimental design and GWL

The analysis uses three phases of the TIPMIP Phase 1 protocol:

| Local experiment | TIPMIP experiment ID | Role in this analysis |
| --- | --- | --- |
| `Exp1` | `esm-up2p0` | Ramp-up under positive CO₂ emissions to 2 °C warming. |
| `Exp2` | `esm-up2p0-gwl2p0` | Zero-emission run branched at 2 °C. |
| `Exp4` | `esm-up2p0-gwl2p0-50y-dn2p0` | Ramp-down under negative emissions, branched 50 years into the zero-emission run. |

In climate research, a Global Warming Level is a change in global mean temperature relative to a pre-industrial reference; IPCC assessments commonly express warming relative to 1850–1900. This repository uses the TIPMIP model simulations and the `tipmip-gwl` mapping products: each model's GWL axis is derived from its simulated global mean surface air temperature (GMSAT) relative to a piControl-based reference. Therefore, the plotted GWL coordinates are model-specific simulated warming levels, not an observational GMST series. The bundled mappings re-index annual model output onto GWL coordinates (0.02 °C spacing); they do not contain the ecosystem variables themselves. See the mapping metadata for the baseline used by each model.

The selected models and ensemble members are:

| Model | Data-family folder | Member | Vegetation representation |
| --- | --- | --- | --- |
| EC-Earth3-ESM-1 | `EC` | `r1i1p1f1` | Dynamic vegetation (LPJ-GUESS) |
| UKESM1-2-LL | `UKESM` | `r1i1p1f1` | Dynamic vegetation (TRIFFID) |
| GISS-E2-1-G-CC2 | `NASA` | `r1i1p1f3` | Prescribed vegetation cover |

## Repository contents

```text
tipmip-hysteresis-highlat-ecosystem/
├── Exp1_data/                 # Ramp-up inputs
├── Exp2_data/                 # Zero-emission inputs
├── Exp4_data/                 # Ramp-down inputs
├── figures/                   # Figures created by the plotting notebook
├── fx/                        # Static area and land-fraction fields
├── notebooks/                 # Analysis, plotting, and results notebooks
├── tipmip-gwl/                # External GWL mapping package and mapping products
└── utils/                     # Reusable analysis and preprocessing functions
```

The notebooks are the analysis entry points; reusable calculations are in `utils/`. The calculation notebooks write compact intermediate NetCDF summaries to a sibling directory named `tipmip-hysteresis-highlat-ecosystem_plot_data/`; the plotting and results notebooks read those summaries.

## Data access and provenance

The model output is from the CMIP6Plus TIPMIP collection on the Earth System Grid Federation (ESGF). Discover available files through the [ESGF node list](https://esgf.github.io/nodes.html) or the [CMIP6Plus data access page](https://www.wcrp-cmip.org/cmip-phases/cmip6plus/). In ESGF, filter by project `CMIP6Plus`, model/source ID, experiment ID, member, variable, frequency, and grid. The analysis expects monthly NetCDF input for the selected members and experiments, arranged in the `Exp*_data/{EC,NASA,UKESM}/` folders using the variable-folder names shown in the notebooks (for example, `Exp1_EC_tas/` and `Exp4_UKESM_npp/`). The static grid-area inputs used by the analysis are in `fx/`.

Model output is large and remains subject to the data provider's access and reuse terms. Obtain the files from ESGF and retain the exact dataset version and its metadata. For a publication, cite each ESGF dataset actually used using the citation/DOI supplied on its dataset record; include the model, experiment, member, variable, version, and access date in the data-provenance record. The ESGF node directory is a discovery route, not a substitute for the dataset-specific citation. See the [WCRP guidance on CMIP data access and citation](https://www.wcrp-cmip.org/cmip-data-access/).

### Preprocess GISS-E2-1-G-CC2 `cVeg`

For GISS-E2-1-G-CC2, `cVeg` is assembled from the tile-level carbon-density fields and vegetation fractions before the cVeg notebook is run. For each of `Exp1`, `Exp2`, and `Exp4`, provide the monthly `cVegTree`, `cVegShrub`, and `cVegGrass` fields and the corresponding `treeFrac`, `shrubFrac`, and `grassFrac` fields. Fractions are percentages. The utility computes:

```text
cVeg = (cVegTree × treeFrac + cVegShrub × shrubFrac + cVegGrass × grassFrac) / 100
```

From the repository root, run the existing helper to create the combined files at the paths expected by `notebooks/cveg.ipynb`:

```python
from pathlib import Path
from utils.preprocessing import preprocess_nasa_total_cveg

project_root = Path.cwd().resolve()
for experiment in ("Exp1", "Exp2", "Exp4"):
    output_file = (
        project_root / f"{experiment}_data" / "NASA" /
        f"{experiment}_NASA_cVeg.nc"
    )
    preprocess_nasa_total_cveg(project_root, experiment, output_file=output_file)
```

This helper requires the tile and fraction files in their expected experiment folders. It preserves the input carbon-density units; the analysis then applies the grid-cell area and converts the regional total to Pg C.

## Installation and dependencies

Clone the analysis repository, create the Conda environment, and clone the GWL tool into the repository root:

```bash
git clone https://github.com/jinyu-eco/tipmip-hysteresis-highlat-ecosystem.git
cd tipmip-hysteresis-highlat-ecosystem
conda env create -f environment.yml
conda activate tipmip-hysteresis
git clone https://github.com/tipmip-hub/tipmip-gwl.git
git -C tipmip-gwl checkout 484169ca183ca71efaa9947e1fa28ec85d138881
```

The GWL code is imported from `tipmip-gwl/src` by the analysis utilities. The checkout above pins the GWL implementation used when preparing this README; the bundled mapping products are version `v1`. Record the GWL repository revision and mapping version alongside any published results.

The Conda environment is specified in [`environment.yml`](environment.yml). It includes the packages used by the notebooks and the dependencies declared by `tipmip-gwl`. To open the notebooks, run `jupyter lab` from the repository root and select the `tipmip-hysteresis` kernel. This file specifies the top-level packages; archive a platform-specific lockfile with a release if exact package builds are required.

## Reproducing the analysis

1. Download the required monthly NetCDF files from ESGF and place them in the expected `Exp1_data/`, `Exp2_data/`, and `Exp4_data/` folders. Confirm the model, member, experiment, grid, and time coverage against the configuration in each notebook.
2. For GISS-E2-1-G-CC2, create the three combined `cVeg` files using the preprocessing helper above.
3. Run `notebooks/tas.ipynb`, `notebooks/npp.ipynb`, `notebooks/frac.ipynb`, and `notebooks/cveg.ipynb` to calculate annual regional summaries and map the ramp-up/ramp-down series to GWL.
4. Run `notebooks/frac_change.ipynb` for the spatial vegetation-fraction comparisons, then `notebooks/plots.ipynb` to create the figures. Run `notebooks/results.ipynb` to calculate and display the summary tables.

The regional analyses use 60–90°N. Monthly fields are aggregated to annual values using the methods in `utils/`; regional carbon and productivity are area-integrated, while temperature and vegetation fractions are area-weighted. Exp1 and Exp4 are compared along their mapped GWL axes. Exp2 is analysed as the zero-emission phase; the spatial fraction comparison uses its final 30 complete years, as configured in `frac_change.ipynb`.

The intermediate summary files are written outside the repository to `../tipmip-hysteresis-highlat-ecosystem_plot_data/` when the project directory has its default name. Keep these generated products available when running `plots.ipynb` and `results.ipynb`.

## Citation

If you use this code or its analysis, cite the TIPMIP protocol and the exact ESGF model datasets used. The protocol reference is:

> Jones, C., Bossert, I., Dennis, D. P., et al. (2026). The TIPMIP Earth system model experiment protocol: phase 1. *Geoscientific Model Development*, 19, 6941–6966. https://doi.org/10.5194/gmd-19-6941-2026.

For the GWL axis, cite the [`tipmip-gwl` software repository](https://github.com/tipmip-hub/tipmip-gwl) and report its code revision and mapping version (`v1`). For background on the general GWL convention and the 1850–1900 reference period, cite IPCC (2021), Chapter 1, *Framing, Context and Methods*, in *Climate Change 2021: The Physical Science Basis*. https://doi.org/10.1017/9781009157896.003. Add the final citation and DOI for the paper associated with this analysis when available. For a citable software release, archive the exact repository version and include its release DOI in the manuscript.
