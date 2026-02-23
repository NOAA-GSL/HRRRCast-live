# HRRRCast (Live Pipeline)

HRRRCast is a neural network-based, high‑resolution regional weather forecasting system leveraging HRRR analyses/forecasts and GFS boundary conditions. The live pipeline now features unified logging utilities, per‑variable/level normalization, enhanced APCP (precipitation) sourcing, HRRR→model downsampling, GFS→HRRR interpolation, diffusion (probabilistic) and deterministic model support, and NetCDF→GRIB2 export.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Ensemble and PMM Support](#ensemble-and-pmm-support)
- [End‑to‑End Pipeline](#end-to-end-pipeline)
- [Model Usage](#model-usage)
- [Data & Channels](#data--channels)
- [Diagnostic Variables](#diagnostic-variables)
- [APCP Handling Logic](#apcp-handling-logic)
- [GRIB2 Export](#grib2-export)
- [Outputs & Naming](#outputs--naming)
- [Available Models](#available-models)
- [Logging & Utilities](#logging--utilities)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Support](#support)

## Installation

### Prerequisites

- Miniconda3 or Anaconda
- CUDA-compatible GPU (recommended) or CPU
- Internet connection (for initial setup)

### Standard Installation (GPU/CPU with Internet)

1. Install Miniconda3 if not already installed
2. Clone this repository and navigate to the project directory
3. Install the environment using the provided configuration:

```bash
conda env create -f environment.yaml
conda activate hrrrcast
```

### HPC Installation (No Internet on Compute Nodes)

For HPC environments like Ursa where compute nodes lack internet access:

```bash
./install_env_ursa.sh
```

This script handles CUDA availability simulation on login nodes.

### Post-Installation Configuration

1. **Configure Environment Paths**: Edit the environment files in the `etc/` directory to match your conda installation directory

2. **Download Cartopy Shapefiles** (for plotting functionality):
   ```bash
   python -c "import cartopy.io.shapereader as shpreader; shpreader.natural_earth()"
   ```

## Quick Start

### Running Forecasts

Use the provided submission script to run forecasts:

```bash
./submit_all.sh <INIT_TIME> <LEAD_HOUR> <N_ENSEMBLES> <N_GPUS> <ACCNR>
```

- `INIT_TIME`: Initialization time in format `YYYY-MM-DDTHH` (e.g., `2024-05-06T23`)
- `LEAD_HOUR`: Number of forecast hours (e.g., `6`)
- `N_ENSEMBLES`: Number of ensemble members to run (default: `1`)
- `N_GPUS`: Number of GPUs to use for parallel forecast jobs (default: `1`)
- `ACCNR`: (Optional) Account number for SLURM jobs (default: `gsd-hpcs`)

**Example**: Run a 6-hour ensemble forecast with 10 members on 2 GPUs starting from May 6, 2024 at 23:00 UTC:
```bash
./submit_all.sh 2024-05-06T23 6 10 2
```

### Manual Forecast, Plotting & GRIB Export

#### Forecast

You can run the forecast script directly:

```bash
python src/fcst.py <model_path> <inittime> <lead_hours> --members 0-2 --output_dir <output_dir> [--no_diffusion] [--base_dir <dir>]
```

- `model_path`: Path to the trained model (e.g., `net-diffusion/model.keras`)
- `inittime`: Initialization time (e.g., `2024-05-06T23`)
- `lead_hours`: Number of forecast hours (e.g., `6`)
- `--members`: List or range of ensemble member IDs (e.g., `0-2 4 6-7`)
- `--no_diffusion`: Use deterministic model (default is diffusion/ensemble)
- `--base_dir`: Base directory for input files (default: `./`)
- `--output_dir`: Output directory for forecast files (default: `./`)

#### Plotting

To plot the forecast output for all hours 1 to N for each member:

```bash
python src/plot.py <inittime> <lead_hour> --members 0-2 --forecast_dir <forecast_dir> --output_dir <output_dir>
```

- `inittime`: Initialization time (e.g., `2024-05-06T23`)
- `lead_hour`: Maximum forecast hour to plot (e.g., `6`)
- `--members`: List or range of member IDs (e.g., `0-2 4 pmm`)
- `--forecast_dir`: Directory containing forecast files (default: `./`)
- `--output_dir`: Output directory for plots (default: `./`)

**Note:** This will generate plots for all hours from 1 to `lead_hour` (inclusive) for each member, saving each hour's plots in a separate subdirectory.

#### GRIB2 Output (default)

Forecasts run via `src/fcst.py` write **both NetCDF and GRIB2** outputs by default (per-hour files during rollout). GRIB2 export uses `grib2io`, `eccodes`, and system `wgrib2`.

If you need a standalone conversion utility, use `src/nc2grib.py` (see `Netcdf2Grib`).

## Ensemble and PMM Support

- For diffusion/ensemble forecasts, use `--members` to specify which ensemble members to run and plot.
- The system supports ranges (e.g., `0-2`), comma-separated, and non-integer IDs (e.g., `pmm` for ensemble mean).
- The PMM (Probability-Matched Mean) is computed and plotted automatically when running in ensemble mode.

## End-to-End Pipeline

| Stage | Script | Key Actions |
|-------|--------|-------------|
| 1. Download HRRR analyses + prior hour f01 surface | `src/get_ics.py` | Fetches pressure & surface GRIB plus previous hour 1h surface forecast (for APCP fallback) |
| 2. Build IC dataset | `src/make_ics.py` | Reads HRRR GRIB, applies per‑variable / per‑level normalization, log transforms, APCP replacement strategy, saves `.npz` |
| 3. Download GFS boundary GRIBs | `src/get_bcs.py` | Selects appropriate synoptic cycle(s); can ensure required f006 and window coverage |
| 4. Build BC dataset | `src/make_bcs.py` | Interpolates GFS fields to downsampled HRRR grid (xESMF), normalizes, APCP future synoptic sourcing, saves `.npz` |
| 5. Run forecast | `src/fcst.py` | Loads IC + BC arrays, assembles inputs, runs deterministic or diffusion model, writes per-hour NetCDF **and GRIB2** outputs |
| 6. Plot results | `src/plot.py` | Parallel (per lead hour) map plots for pressure & surface variables + summary panels |
| 7. (Optional) Standalone GRIB2 export | `src/nc2grib.py` | Converts NetCDF member/mean outputs to GRIB2 with parameter metadata |

All scripts use centralized utilities in `src/utils.py` for logging (`setup_logging`), directory creation, datetime validation, and resilient downloading.

## Model Usage

### Loading Models

Load trained models using TensorFlow/Keras:

```python
import tensorflow as tf

model = tf.keras.models.load_model("net-deterministic/model.keras", safe_mode=False, compile=False)
```

### Input/Output Dimensions

The spatial grid (530×900) represents every other grid point from the original HRRR grid (1059×1799).

## Data & Channels

Channel counts are dynamic and driven by configuration in `make_ics.py` / `make_bcs.py`. Use those scripts (or `fcst.py`) to confirm the exact channel counts for a given model. The default configuration in `make_ics.py` is:

| Category | Components | Count (default) |
|----------|------------|-----------------|
| Pressure-level variables | 6 vars × 20 levels (UGRD,VGRD,VVEL,TMP,HGT,SPFH) | 120 |
| Surface dynamic variables | 18 (PRES, MSLMA, REFC, T2M, UGRD10M, VGRD10M, UGRD80M, VGRD80M, D2M, TCDC, LCDC, MCDC, HCDC, VIS, APCP, HGTCC, CAPE, CIN) | 18 |
| Static constants | LAND, OROG | 2 |
| Lead time (per step, autoregressive) | 1 | 1 |
| Total model input (IC) | 120 + 18 + 2 + 1 | 141 |

The forecast model typically predicts only the dynamic meteorological fields (pressure-level + surface set, excluding static + lead-time). The exact predicted channel count is inferred automatically in `fcst.py` and depends on the model configuration.

## Diagnostic Variables

Diagnostics are computed in [src/diagnostics.py](src/diagnostics.py) via `compute_diagnostics()`. You can run all diagnostics or select a subset with `include`/`exclude` flags.

Available diagnostic groups (see function docstrings for full variable lists):

- **Surface thermodynamics**: R2M, SPFH2M, POT2M
- **Column-integrated**: PWAT
- **Precipitation diagnostics**: CRAIN, CFRZR, and related masks/fractions
- **Wind diagnostics**: GUST, GUST_FACTOR, GUST_CONV, WIND_10M, WIND_MAX
- **Convective diagnostics**: shear, helicity, vorticity, storm motion, updraft helicity, and vertical velocity extrema
- **Vertical profile**: 0°C isotherm height/pressure and RH_0C

## APCP Handling Logic

Accumulated precipitation (APCP / total precipitation) is not reliable directly from the HRRR analysis or isolated GFS lead files for sub‑hour windows, so the pipeline applies tiered sourcing:

1. **Initial Conditions (`make_ics.py`)**: Replace analysis APCP with prior hour 1‑hour forecast accumulation file (`*_surface_f01.grib2`) downloaded by `get_ics.py`.
2. **Boundary Conditions (`make_bcs.py`)**: For each valid time, attempt to replace APCP with the field from the nearest future synoptic GFS cycle (> valid time). If that GRIB file exists it is interpolated and substituted; otherwise keep current lead’s APCP.
3. **(Optional future)**: If cumulative fields from consecutive future hours are available, compute 1‑hour increments (difference of cumulative precipitation); current implementation substitutes directly (documented for transparency).

Logging clearly notes when APCP is substituted (INFO) or when fallback occurs (DEBUG/WARNING).

## GRIB2 Export

GRIB2 export is handled in `src/fcst.py` during forecasts. For standalone conversion, `nc2grib.py` converts NetCDF forecast outputs to GRIB2 with:
* Parameter overrides (`GRIB_PARAM_OVERRIDE`) and center metadata
* Cube attribute mapping (`ATTR_MAPS`)
* Optional index generation via `wgrib2` (`.idx` files)

Dependencies: `grib2io`, `eccodes`, `wgrib2`. These are optional and not required for core inference/plotting.

## Outputs & Naming

Forecast outputs are written per hour into:

```
<output_dir>/<YYYYMMDD>/<HH>/
```

Where `<YYYYMMDD>` and `<HH>` come from the initialization time.

**NetCDF** (per hour):

- Members: `hrrrcast_mem<NN>_f<HH>.nc` (e.g., `hrrrcast_mem0_f03.nc`)
- Ensemble mean: `hrrrcast_avg_f<HH>.nc`

**GRIB2** (per hour):

- Members: `hrrrcast.m<NN>.t<HH>z.pgrb2.f<HH>`
- Ensemble mean: `hrrrcast.avg.t<HH>z.pgrb2.f<HH>`

Hour `f00` is written for the initial state when per-hour outputs are enabled.

## Available Models

| Model | Use |
|-------|------------|
| net-diffusion | For probabilistic forecast |

## Logging & Utilities

All major scripts (`get_ics.py`, `make_ics.py`, `get_bcs.py`, `make_bcs.py`, `fcst.py`, `plot.py`, `nc2grib.py`) use centralized helpers in `src/utils.py`:

| Function | Purpose |
|----------|---------|
| `setup_logging(level)` | Idempotent root logger config |
| `validate_datetime(str)` | Flexible datetime parsing → padded components |
| `make_directory(path)` | Recursive directory creation |
| `download_file_with_retry(url, path, ...)` | Simple resilient downloader with progress |

Customize log verbosity with `--log_level` on each CLI.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Use the smaller model or reduce batch size
2. **Missing Cartopy Shapefiles**: Run the cartopy download command in post-installation
3. **Environment Path Issues**: Verify conda paths in `etc/` configuration files
4. **Missing Optional Libraries**: Plotting works without Cartopy (falls back); GRIB2 export requires extra libs
5. **Model Loading Errors**: Ensure `safe_mode=False` when loading models

### Performance Tips

- Use GPU acceleration when available
- For large-scale runs, consider batch processing
- Monitor memory usage during rollout forecasts

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License
MIT License. See [LICENSE](LICENSE).

## Citation

If you use HRRRCast in your research, please cite:

    @misc{abdi2025hrrrcastdatadrivenemulatorregional,
          title={HRRRCast: a data-driven emulator for regional weather forecasting at convection allowing scales}, 
          author={Daniel Abdi and Isidora Jankov and Paul Madden and Vanderlei Vargas and Timothy A. Smith and Sergey Frolov and Montgomery Flora and Corey Potvin},
          year={2025},
          eprint={2507.05658},
          archivePrefix={arXiv},
          primaryClass={physics.ao-ph},
          url={https://arxiv.org/abs/2507.05658}, 
    }

## Support

For questions or issues not covered in this README, please open an issue in the repository or contact the development team.

---

*This README reflects the live pipeline as of 2026-02-23. Refer to source code and the cited paper for deeper architectural details.*
