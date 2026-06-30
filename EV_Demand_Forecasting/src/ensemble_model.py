from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_ensemble_model() -> VotingRegressor:
    """Combine three regressors with weighted averaging for robust predictions."""
    return VotingRegressor(
        estimators=[
            (
                "random_forest",
                RandomForestRegressor(
                    n_estimators=60,
                    max_depth=12,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "gradient_boosting",
                GradientBoostingRegressor(
                    n_estimators=60,
                    max_depth=6,
                    learning_rate=0.08,
                    random_state=42,
                ),
            ),
            (
                "ridge",
                Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        ("regressor", Ridge(alpha=1.0)),
                    ]
                ),
            ),
        ],
        weights=[2, 2, 1],
        n_jobs=-1,
    )
