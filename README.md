# HRRRCast (Live Pipeline)

HRRRCast is a neural network-based, high‑resolution regional weather forecasting system leveraging HRRR analyses/forecasts and GFS boundary conditions. The live pipeline now features unified logging utilities, per‑variable/level normalization, enhanced APCP (precipitation) sourcing, HRRR→model downsampling, GFS→HRRR interpolation, diffusion (probabilistic) and deterministic model support, and NetCDF→GRIB2 export.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [End‑to‑End Pipeline](#end-to-end-pipeline)
- [Model Usage](#model-usage)
- [Data & Channels](#data--channels)
- [APCP Handling Logic](#apcp-handling-logic)
- [GRIB2 Export](#grib2-export)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

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
conda activate hrrrcast-live
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

#### GRIB2 Conversion (optional)

After generating NetCDF forecasts you can convert to GRIB2 (requires `cf-units`, `iris`, `iris-grib`, `eccodes`, `wgrib2`):

```bash
python src/nc2grib.py  # (create a small driver if needed, see class Netcdf2Grib)
```

The `Netcdf2Grib` class applies GRIB template tweaks (centre, parameter overrides) and writes index (`.idx`) files via `wgrib2`.

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
| 5. Run forecast | `src/fcst.py` | Loads IC + BC arrays, assembles inputs, runs deterministic or diffusion model, writes NetCDF outputs (`hrrrcast_memX.nc`) |
| 6. Plot results | `src/plot.py` | Parallel (per lead hour) map plots for pressure & surface variables + summary panels |
| 7. (Optional) GRIB2 export | `src/nc2grib.py` | Converts NetCDF member/mean outputs to GRIB2 with parameter metadata |

All scripts use centralized utilities in `src/utils.py` for logging (`setup_logging`), directory creation, datetime validation, and resilient downloading.

## Model Usage

### Loading Models

Load trained models using TensorFlow/Keras:

```python
import tensorflow as tf

model = tf.keras.models.load_model("net-deterministic/model.keras", safe_mode=False, compile=False)
```

### Input/Output Dimensions

- **Input**: `(batch_size, 530, 900, 77)`
- **Output**: `(batch_size, 530, 900, 74)`

The spatial grid (530×900) represents every other grid point from the original HRRR grid (1059×1799).

## Data & Channels

Channel counts are now dynamic and driven by configuration in `make_ics.py` / `make_bcs.py`:

| Category | Components | Count (default) |
|----------|------------|-----------------|
| Pressure-level variables | 6 vars × 20 levels (UGRD,VGRD,VVEL,TMP,HGT,SPFH) | 120 |
| Surface dynamic variables | 15 (PRES, MSLMA, REFC, T2M, UGRD10M, VGRD10M, UGRD80M, VGRD80M, D2M, R2M, TCDC, VIS, APCP, HGTCC, CAPE, CIN) | 15 |
| Static constants | LAND, OROG | 2 |
| Lead time (per step, autoregressive) | 1 | 1 |
| Total model input (IC) | 120 + 15 + 2 + 1 | 138 |

The forecast model typically predicts only the dynamic meteorological fields (pressure-level + surface set, excluding static + lead-time). This predicted channel count is inferred automatically in `fcst.py`.

### Lead Time Encoding
Lead time channel value per step: `lead_hours / MAX_STEP` (default max step = 6 for base model cadence). Autoregressive rollouts update this channel each inference slice.

### Normalization

All inputs are normalized using:
```
x_normalized = (x - mean) / std
```

Per-variable / per-level statistics are stored in `normalize.nc` (pressure variables stored with shape `(2, nLevels)`; surface as `(2, ...)`).

#### Static Variables Normalization
```python
c_mean = constants.mean(("lat", "lon"))
c_std = constants.std(("lat", "lon"))
constants_normalized = (constants - c_mean) / c_std
constants_normalized = constants_normalized.fillna(0)
```

### Output Denormalization

Model outputs are normalized and must be denormalized:
```
x = x_normalized * std + mean
```

### Extended Forecasts

For forecasts beyond 6 hours, use rollout prediction:

**Example**: 16-hour forecast decomposition:
- 2 × 6-hour steps
- 1 × 4-hour step  

## APCP Handling Logic

Accumulated precipitation (APCP / total precipitation) is not reliable directly from the HRRR analysis or isolated GFS lead files for sub‑hour windows, so the pipeline applies tiered sourcing:

1. **Initial Conditions (`make_ics.py`)**: Replace analysis APCP with prior hour 1‑hour forecast accumulation file (`*_surface_f01.grib2`) downloaded by `get_ics.py`.
2. **Boundary Conditions (`make_bcs.py`)**: For each valid time, attempt to replace APCP with the field from the nearest future synoptic GFS cycle (> valid time). If that GRIB file exists it is interpolated and substituted; otherwise keep current lead’s APCP.
3. **(Optional future)**: If cumulative fields from consecutive future hours are available, compute 1‑hour increments (difference of cumulative precipitation); current implementation substitutes directly (documented for transparency).

Logging clearly notes when APCP is substituted (INFO) or when fallback occurs (DEBUG/WARNING).

## GRIB2 Export

`nc2grib.py` converts NetCDF forecast outputs to GRIB2 with:
* Parameter overrides (`GRIB_PARAM_OVERRIDE`) and center metadata
* Cube attribute mapping (`ATTR_MAPS`)
* Optional index generation via `wgrib2` (`.idx` files)

Dependencies: `iris`, `iris-grib`, `cf-units`, `eccodes`, `wgrib2`. These are optional and not required for core inference/plotting.

## Available Models

| Model | Use |
|-------|------------|
| net-diffusion | For probabilistic forecast |
| net-deterministic | For deterministic forecast |

## Examples

### Basic Prediction

```python
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model("net-deterministic/model.keras", safe_mode=False, compile=False)

# Prepare input (example dimensions)
batch_size = 1
input_data = np.random.randn(batch_size, 530, 900, 77)

# Run prediction
prediction = model.predict(input_data)
print(f"Prediction shape: {prediction.shape}")  # (1, 530, 900, 74)
```

### Multi-Step Forecast

```python
def rollout_forecast(model, initial_state, target_hours):
    """
    Perform multi-step forecast using rollout approach
    """
    current_state = initial_state.copy()
    forecasts = []
    
    remaining_hours = target_hours
    
    while remaining_hours > 0:
        if remaining_hours >= 6:
            lead_time = 6
        else:
            lead_time = remaining_hours
            
        # Set lead time channel
        current_state[:, :, :, -1] = lead_time / 6.0
        
        # Predict
        forecast = model.predict(current_state)
        forecasts.append(forecast)
        
        # Update state for next iteration
        current_state[:, :, :, :74] = forecast
        remaining_hours -= lead_time
    
    return forecasts
```

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

[Add your license information here]

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

*This README reflects the live (feature/hrrrcast_v2) pipeline. Refer to source code and the cited paper for deeper architectural details.*
