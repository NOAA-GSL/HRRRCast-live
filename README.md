# HRRRCast

HRRRCast is a neural network-based weather forecasting system that provides high-resolution predictions using trained models on HRRR (High-Resolution Rapid Refresh) weather data.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Usage](#model-usage)
- [Input/Output Specifications](#inputoutput-specifications)
- [Available Models](#available-models)
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
./submit_all.sh <YEAR> <MONTH> <DAY> <HOUR> <LEAD_HOURS>
```

**Example**: Run a 6-hour forecast starting from May 6, 2024 at 23:00 UTC:
```bash
./submit_all.sh 2024 05 06 23 6
```

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

## Input/Output Specifications

### Primary Input Variables (74 channels)

#### 3D Atmospheric Fields (72 channels)
Variables at 12 pressure levels each:

**Variables**:
- `UGRD`: U-component of wind (m/s)
- `VGRD`: V-component of wind (m/s) 
- `VVEL`: Vertical velocity (Pa/s)
- `TMP`: Temperature (K)
- `HGT`: Geopotential height (m)
- `SPFH`: Specific humidity (kg/kg)

**Pressure Levels (hPa)**:
```
200, 300, 475, 800, 825, 850, 875, 900, 925, 950, 975, 1000
```

#### 2D Surface Fields (2 channels)
- `T2M`: 2-meter temperature (K)
- `REFC`: Composite reflectivity (dBZ)

### Static Input Variables (2 channels)
- `LAND`: Land mask (binary)
- `OROG`: Orography/terrain height (m)

### Lead Time Channel (1 channel)
- Normalized lead time: `lead_time / 6.0`
- For lead times 1-6 hours: feed directly
- For lead times >6 hours: use rollout approach (see [Extended Forecasts](#extended-forecasts))

### Data Normalization

All inputs are normalized using:
```
x_normalized = (x - mean) / std
```

Normalization parameters are provided in `normalize.nc`.

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
- 1 × 3-hour step  
- 1 × 1-hour step

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
        elif remaining_hours >= 3:
            lead_time = 3
        else:
            lead_time = 1
            
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

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Use the smaller model or reduce batch size
2. **Missing Cartopy Shapefiles**: Run the cartopy download command in post-installation
3. **Environment Path Issues**: Verify conda paths in `etc/` configuration files
4. **Model Loading Errors**: Ensure `safe_mode=False` when loading models

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

[Add citation information if this is research code]

## Support

For questions or issues not covered in this README, please open an issue in the repository or contact the development team.

---

*This README provides comprehensive documentation for HRRRCast. For additional technical details, refer to the source code and associated research papers.*
