# Experiment 002: Paper Benchmark Datasets (CSV)

Extracted from `data_NM_PB_LT_DSAA2016.Rdata` (Moniz et al., 2017 - *Resampling Strategies for Imbalanced Time Series Forecasting*).

## Directory Structure

Each file has two columns:
- `time_index`: Date / timestamp string or observation index
- `target`: The continuous target time series value

## Manifest of Datasets

| File | Domain | Name | Frequency | Observations | Missing Values (NA) |
|---|---|---|---|---|---|
| `DS01_bike_daily_temp.csv` | Bike Sharing | Temperature | Daily | 730 | No |
| `DS02_bike_daily_humidity.csv` | Bike Sharing | Humidity | Daily | 730 | No |
| `DS03_bike_daily_windspeed.csv` | Bike Sharing | Windspeed | Daily | 730 | No |
| `DS04_bike_daily_count.csv` | Bike Sharing | Count of Bike Rentals | Daily | 730 | No |
| `DS05_bike_hourly_temp.csv` | Bike Sharing | Temperature | Hourly | 17,378 | No |
| `DS06_bike_hourly_humidity.csv` | Bike Sharing | Humidity | Hourly | 17,378 | No |
| `DS07_bike_hourly_windspeed.csv` | Bike Sharing | Windspeed | Hourly | 17,378 | No |
| `DS08_bike_hourly_count.csv` | Bike Sharing | Count of Bike Rentals | Hourly | 17,378 | No |
| `DS09_icelandic_river_flow.csv` | Icelandic River | Flow of Vatnsdalsa River | Daily | 1,095 | No |
| `DS10_porto_min_temp.csv` | Porto Weather | Minimum Temperature | Daily | 1,456 | No |
| `DS11_porto_max_temp.csv` | Porto Weather | Maximum Temperature | Daily | 1,456 | No |
| `DS12_porto_max_steady_wind.csv` | Porto Weather | Maximum Steady Wind | Daily | 1,456 | Yes (197 NAs) |
| `DS13_porto_max_wind_gust.csv` | Porto Weather | Maximum Wind Gust | Daily | 1,456 | Yes (201 NAs) |
| `DS14_istanbul_sp.csv` | Istanbul Stock Exch. | S&P | Daily | 536 | No |
| `DS15_istanbul_dax.csv` | Istanbul Stock Exch. | DAX | Daily | 536 | No |
| `DS16_istanbul_ftse.csv` | Istanbul Stock Exch. | FTSE | Daily | 536 | No |
| `DS17_istanbul_nikkei.csv` | Istanbul Stock Exch. | NIKKEI | Daily | 536 | No |
| `DS18_istanbul_bovespa.csv` | Istanbul Stock Exch. | BOVESPA | Daily | 536 | No |
| `DS19_istanbul_eu.csv` | Istanbul Stock Exch. | EU | Daily | 536 | No |
| `DS20_istanbul_em.csv` | Istanbul Stock Exch. | Emerging Markets | Daily | 536 | No |
| `DS21_australian_demand.csv` | Australian Electricity | Total Demand | Half-Hourly | 239,602 | No |
| `DS22_australian_rrp.csv` | Australian Electricity | Retail Price (RRP) | Half-Hourly | 239,602 | No |
| `DS23_porto_water_pedroucos.csv` | Oporto Water | Pedrouços Consumption | Half-Hourly | 51,208 | Yes |
| `DS24_porto_water_rotunda.csv` | Oporto Water | Rotunda AEP Consumption | Half-Hourly | 51,208 | Yes |

## Quick Usage in Python / Pandas

```python
import pandas as pd

df = pd.read_csv("DS01_bike_daily_temp.csv")
print(df.head())
```
