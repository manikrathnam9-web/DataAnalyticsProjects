# EV Demand Forecasting

Machine learning project that forecasts **EV charging station demand** (occupied ports) using an ensemble of three regressors combined with stacking. Includes a Flask web app for interactive predictions.

## Project Overview

This project predicts how many charging ports will be occupied at a station based on:

- Time features (month, day, hour, weekend, peak hour)
- Station capacity and availability
- Weather (temperature, precipitation)
- Traffic congestion and gas prices
- Operational status and local events

**Target variable:** `demand_count` (number of occupied ports)

**Ensemble models:**

1. Random Forest Regressor
2. Gradient Boosting Regressor
3. Ridge Regression (with scaling)

Predictions are combined with **weighted averaging** across the three models for improved, more stable results.

## Project Structure

```
EV_Demand_Forecasting/
├── main.py                          # Main entry point
├── requirements.txt
├── README.md
├── data/
│   └── ev_charging_station_data.csv # Raw dataset
├── outputs/
│   ├── prepared_ev_demand.csv       # Feature-engineered data
│   └── model_metrics.json           # Training metrics
├── models/
│   └── ev_demand_ensemble.joblib    # Saved ensemble model
├── src/
│   ├── config.py                    # Paths and feature definitions
│   ├── data_prep.py                 # Data cleaning & feature engineering
│   ├── ensemble_model.py            # Ensemble model builder
│   └── train.py                     # Training & prediction utilities
└── webapp/
    ├── app.py                       # Flask application
    └── templates/
        └── index.html               # Prediction UI
```

## Setup

1. **Create a virtual environment (recommended):**

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run everything (prepare → train → web app)

```bash
python main.py --all
```

### Option 2: Step by step

**1. Prepare data**

```bash
python main.py --prepare-data
```

**2. Train ensemble model**

```bash
python main.py --train
```

Train on a smaller sample for faster runs:

```bash
python main.py --train --sample-fraction 0.05
```

**3. Start Flask web app**

```bash
python main.py --serve
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### API Endpoints

| Endpoint   | Method | Description              |
|-----------|--------|--------------------------|
| `/`       | GET    | Prediction web UI        |
| `/predict`| POST   | JSON prediction API      |
| `/health` | GET    | Health & model status    |

**Example API request:**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"month\":7,\"day\":15,\"day_of_week\":2,\"hour\":9,\"minute\":0,\"is_weekend\":0,\"is_peak_hour\":1,\"has_event\":0,\"station_operational\":1,\"power_output_kw\":19.2,\"ports_total\":6,\"ports_in_service\":6,\"available_share\":0.5,\"temperature_f\":85,\"precipitation_mm\":0,\"traffic_congestion_index\":6,\"gas_price_per_gallon\":4.75}"
```

## Dataset

The raw dataset (`data/ev_charging_station_data.csv`) contains time-series records from **150 EV charging stations** with 30-minute intervals, including:

- Station metadata (network, location, charger type, power output)
- Port availability and utilization
- Weather and traffic conditions
- Pricing and operational status

## Model Persistence

The trained ensemble is saved with **joblib** to:

```
models/ev_demand_ensemble.joblib
```

The bundle includes the model, feature column names, target column, and evaluation metrics.

## Evaluation Metrics

After training, metrics are printed to the console and saved to `outputs/model_metrics.json`:

- **MAE** – Mean Absolute Error (average ports off by)
- **RMSE** – Root Mean Squared Error
- **R²** – Coefficient of determination

## Tech Stack

- Python 3.10+
- pandas, NumPy
- scikit-learn (ensemble voting)
- joblib (model serialization)
- Flask (web app)

## License

This project is for educational and portfolio use.
