import pandas as pd
from pathlib import Path


def load_raw_ev_data(csv_path: Path) -> pd.DataFrame:
    """Load the EV charging station dataset and make the timestamp usable."""
    df = pd.read_csv(csv_path)

    # Timestamp is the backbone for time-series forecasting, validate early.
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    if df['timestamp'].isna().any():
        raise ValueError('Found invalid timestamp values in raw data')

    return df


def validate_ev_data(df: pd.DataFrame) -> None:
    """Sanity checks for the raw EV dataset before we add features."""
    required_cols = {
        'timestamp',
        'station_id',
        'ports_total',
        'ports_available',
        'ports_occupied',
        'utilization_rate',
        'temperature_f',
        'precipitation_mm',
        'traffic_congestion_index',
    }

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f'Missing required columns: {sorted(missing_cols)}')

    if df.duplicated(subset=['timestamp', 'station_id']).any():
        raise ValueError('Duplicate timestamp/station_id rows detected')

    # For forecast modeling we want a station-level time series with reasonable granularity.
    if df['station_id'].nunique() <= 1:
        raise ValueError('Dataset appears to have only one station; check coverage')


def engineer_features(df: pd.DataFrame, target_col: str = 'ports_occupied') -> pd.DataFrame:
    """Create model-ready features from the EV dataset."""
    df = df.copy()

    # Sort by station and timestamp to preserve time order for later splitting.
    df = df.sort_values(['station_id', 'timestamp']).reset_index(drop=True)

    # If data contains multiple cities/timezones, keeping local time semantics is safer than forcing UTC.
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Use a simpler demand target for forecasting and keep utilization as a derived target.
    df['demand_count'] = df[target_col].astype(int)
    df['demand_rate'] = df['utilization_rate'].astype(float)

    # Availability features are highly relevant for in-station demand.
    df['ports_in_service'] = df['ports_total'] - df['ports_out_of_service']
    df['available_share'] = df['ports_available'] / df['ports_total']

    # Convert logical fields into numeric flags.
    df['is_peak_hour'] = df['is_peak_hour'].astype(int)
    df['has_event'] = (df['local_event'].astype(str).str.lower() != 'none').astype(int)
    df['station_operational'] = (df['station_status'].astype(str).str.lower() == 'operational').astype(int)

    # Keep a small set of time-invariant station metadata for later grouping.
    station_meta_cols = [
        'station_id',
        'station_name',
        'network',
        'city',
        'state',
        'location_type',
        'charger_type',
        'power_output_kw',
        'pricing_type',
    ]

    # Create a stable identifier if it does not already exist.
    df['station_location'] = df['city'].fillna('unk') + '_' + df['state'].fillna('unk')

    # Drop columns that are likely not needed for traditional tree-based models.
    # Keep timestamp so we can do a proper time-based split later.
    drop_cols = [
        'station_status',
        'local_event',
        'station_name',
        'amenities_nearby',
    ]
    df_processed = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df_processed


def save_prepared_data(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_path = project_root / 'data' / 'ev_charging_station_data.csv'
    out_path = project_root / 'outputs' / 'prepared_ev_demand.csv'

    df = load_raw_ev_data(raw_path)
    validate_ev_data(df)
    prepared = engineer_features(df)
    save_prepared_data(prepared, out_path)

    print(f'Prepared data saved to: {out_path}')


if __name__ == '__main__':
    main()
