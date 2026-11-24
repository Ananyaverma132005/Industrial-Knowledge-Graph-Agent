"""Feature engineering used inside the saved scikit-learn pipeline.

Kept in its own module so the joblib file can be unpickled by any process (a function
defined in a script run as __main__ can't be found again at load time).
"""
import numpy as np
import pandas as pd

NUMERIC = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
CATEGORICAL = ["Type"]
DERIVED = ["Temp difference [K]", "Power [W]", "Wear x Torque"]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Physics-based features commonly used on AI4I: heat-dissipation gap, mechanical power, strain."""
    X = X.copy()
    X["Temp difference [K]"] = X["Process temperature [K]"] - X["Air temperature [K]"]
    X["Power [W]"] = X["Torque [Nm]"] * X["Rotational speed [rpm]"] * 2 * np.pi / 60
    X["Wear x Torque"] = X["Tool wear [min]"] * X["Torque [Nm]"]
    return X
