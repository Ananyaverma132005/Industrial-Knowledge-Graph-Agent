"""Predictive-maintenance model.

RandomForestClassifier inside a scikit-learn Pipeline whose ColumnTransformer scales the
continuous inputs (Air Temperature [K], Process Temperature [K], Rotational Speed [rpm],
Torque [Nm], Tool Wear [min]) and one-hot encodes the categorical machine tier (L/M/H).

Training data: if data/ai4i2020.csv exists (the public UCI "AI4I 2020 Predictive
Maintenance" dataset) it is used; otherwise a synthetic dataset following the same
failure rules is generated so the project runs offline.

    python train_model.py
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import config

from ml_features import CATEGORICAL, DERIVED, NUMERIC, add_features

TARGET = "Machine failure"


def synthesize(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Synthetic data using the AI4I 2020 failure-mode rules (TWF, HDF, PWF, OSF, RNF)."""
    rng = np.random.default_rng(seed)
    typ = rng.choice(["L", "M", "H"], size=n, p=[0.6, 0.3, 0.1])
    air = rng.normal(300, 2, n)
    proc = air + 10 + rng.normal(0, 1, n)
    rpm = np.clip(rng.normal(1540, 180, n), 1150, 2900)
    torque = np.clip(rng.normal(40, 10, n), 3, 77)
    wear = rng.uniform(0, 253, n)

    power = torque * rpm * 2 * np.pi / 60
    osf_limit = np.select([typ == "L", typ == "M"], [11000, 12000], 13000)
    twf = (wear >= 200) & (wear <= 240) & (rng.random(n) < 0.08)
    hdf = ((proc - air) < 8.6) & (rpm < 1380)
    pwf = (power < 3500) | (power > 9000)
    osf = wear * torque > osf_limit
    rnf = rng.random(n) < 0.001
    failure = (twf | hdf | pwf | osf | rnf).astype(int)

    return pd.DataFrame({
        "Type": typ, NUMERIC[0]: air, NUMERIC[1]: proc, NUMERIC[2]: rpm,
        NUMERIC[3]: torque, NUMERIC[4]: wear, TARGET: failure,
    })


def load_data() -> tuple[pd.DataFrame, str]:
    real = config.DATA_DIR / "ai4i2020.csv"
    if real.exists():
        return pd.read_csv(real), "AI4I 2020 (UCI)"
    return synthesize(), "synthetic AI4I-style"


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC + DERIVED),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    clf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    return Pipeline([("features", FunctionTransformer(add_features)), ("pre", pre), ("clf", clf)])


def train() -> Pipeline:
    df, source = load_data()
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = build_pipeline().fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    print(f"[ml] data: {source}, {len(df)} rows, failure rate {y.mean():.1%}")
    print(classification_report(y_te, model.predict(X_te), target_names=["ok", "failure"], digits=3))
    print(f"[ml] ROC-AUC: {roc_auc_score(y_te, proba):.3f}")

    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": model, "features": NUMERIC + CATEGORICAL, "source": source}, config.ML_MODEL_FILE)
    print(f"[ml] saved {config.ML_MODEL_FILE.relative_to(config.BASE_DIR)}")
    return model


if __name__ == "__main__":
    train()
