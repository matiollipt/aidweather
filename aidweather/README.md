# AidWeather

**AidWeather** is a Python client for retrieving and visualizing NASA POWER daily weather data. It enables streamlined access to key agro-meteorological parameters with flexible filtering, aggregation, and plotting tools.

## Features

- 📡 Fetch daily weather data directly from the NASA POWER API
- 📊 Plot and overlay time series of weather and agrimetric data
- 🔍 Filter by date ranges and custom value thresholds
- 🧠 Built-in parameter descriptions for interpretability
- 🗺️ Interactive location mapping
- 🎨 Configurable visualization themes

## Getting Started

### Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/your-username/aidweather.git
cd aidweather
pip install -r requirements.txt
```

### Configuration

Modify `config.json` to adjust:

- Default weather parameters
- Colormaps and plot style
- NASA POWER base API URL

### Usage

```python
from aidweather.core import AidWeather
from aidweather.dataviz import Visualizer
from aidweather.config import cfg

client = AidWeather("MySite", lat=-10.5, lon=-55.3, start="2023-01-01", end="2023-12-31")
df = client.get(long_names=True)

viz = Visualizer("MySite", cfg["POWER_PARAM_DESCRIPTIONS"], client.get)
viz.plot(cols=["T2M", "PRECTOTCORR"], ma=7, dual=True)
```

## Example Output

- 📈 Time-series plots with moving averages
- 🌦 Overlay plots for crop and weather data
- 🔁 Monthly aggregation with correlation analysis

## File Structure

```text
aidweather/
│
├── config.py         # Loads and manages configuration
├── config.json       # Contains weather param definitions and plot settings
├── core.py           # Main client for NASA POWER data
├── dataviz.py        # Visualization utilities
```

## Requirements

- Python ≥ 3.8

- `pandas`, `requests`, `matplotlib`, `folium`, `numpy`

See `requirements.txt` for a full list.

## Author

Cleverson Matiolli — [github.com/matiollipt](https://github.com/matiollipt)

---

## License

This project is licensed under the MIT License.

```text
MIT License

Copyright (c) 2025 Cleverson Matiolli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      
copies of the Software, and to permit persons to whom the Software is          
furnished to do so, subject to the following conditions:                       

The above copyright notice and this permission notice shall be included in     
all copies or substantial portions of the Software.                            

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN      
THE SOFTWARE.
```
