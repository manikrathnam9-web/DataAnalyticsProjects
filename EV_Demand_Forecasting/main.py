"""Main entry point for the EV Demand Forecasting project."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_SAMPLE_FRACTION
from src.data_prep import main as prepare_data
from src.train import train_ensemble


def run_webapp(host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
    from webapp.app import app

    print(f"Starting Flask app at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EV Charging Demand Forecasting - prepare data, train ensemble, or serve web app."
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Clean raw CSV and build model-ready features.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the ensemble model and save it with joblib.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the Flask prediction web app.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=DEFAULT_SAMPLE_FRACTION,
        help="Fraction of prepared data used for training (default: 0.15).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Flask host (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=5000, help="Flask port (default: 5000).")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run data prep, training, then start the web app.",
    )

    args = parser.parse_args()

    if not any([args.prepare_data, args.train, args.serve, args.all]):
        parser.print_help()
        return

    if args.all or args.prepare_data:
        print("Preparing data...")
        prepare_data()

    if args.all or args.train:
        print("Training ensemble model...")
        metrics = train_ensemble(sample_fraction=args.sample_fraction)
        print("Training complete.")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  R2:   {metrics['r2']:.4f}")

    if args.all or args.serve:
        run_webapp(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
