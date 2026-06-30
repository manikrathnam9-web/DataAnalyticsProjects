from pathlib import Path
import sys

from flask import Flask, jsonify, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FEATURE_COLUMNS, MODEL_PATH
from src.train import load_model, predict_demand

app = Flask(__name__)
model_bundle = None


def get_model_bundle():
    global model_bundle
    if model_bundle is None:
        model_bundle = load_model(MODEL_PATH)
    return model_bundle


@app.route("/")
def index():
    metrics = {}
    try:
        metrics = get_model_bundle().get("metrics", {})
    except FileNotFoundError:
        metrics = {"error": "Model not trained yet. Run `python main.py --train`."}
    return render_template("index.html", features=FEATURE_COLUMNS, metrics=metrics)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        prediction = predict_demand(payload)
        return jsonify(
            {
                "success": True,
                "predicted_demand": round(prediction, 2),
                "unit": "ports occupied",
            }
        )
    except FileNotFoundError as exc:
        return jsonify({"success": False, "error": str(exc)}), 503
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"success": False, "error": f"Invalid input: {exc}"}), 400


@app.route("/health")
def health():
    trained = MODEL_PATH.exists()
    return jsonify({"status": "ok", "model_ready": trained})


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
