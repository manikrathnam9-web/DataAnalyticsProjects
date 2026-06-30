from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "ev_charging_station_data.csv"
PREPARED_DATA_PATH = PROJECT_ROOT / "outputs" / "prepared_ev_demand.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "ev_demand_ensemble.joblib"
METRICS_PATH = PROJECT_ROOT / "outputs" / "model_metrics.json"

TARGET_COLUMN = "demand_count"

FEATURE_COLUMNS = [
    "month",
    "day",
    "day_of_week",
    "hour",
    "minute",
    "is_weekend",
    "is_peak_hour",
    "has_event",
    "station_operational",
    "power_output_kw",
    "ports_total",
    "ports_in_service",
    "available_share",
    "temperature_f",
    "precipitation_mm",
    "traffic_congestion_index",
    "gas_price_per_gallon",
]

DEFAULT_SAMPLE_FRACTION = 0.05
TEST_SIZE = 0.2
RANDOM_STATE = 42
