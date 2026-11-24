"""Simulated IoT telemetry + RandomForest failure scoring.

Every tick each machine gets thermal/vibration readings and AI4I-style process features.
Occasionally one machine spikes past 90 deg C so the UI's alert-pulse can be demonstrated.
"""
from __future__ import annotations

import random
import time

import joblib
import pandas as pd

import config


class FailureModel:
    def __init__(self):
        if not config.ML_MODEL_FILE.exists():
            print("[ml] model not found - training it now (python train_model.py)")
            import train_model

            train_model.train()
        bundle = joblib.load(config.ML_MODEL_FILE)
        self.pipeline = bundle["pipeline"]
        self.features = bundle["features"]
        self.source = bundle.get("source", "unknown")

    def predict_proba(self, rows: list[dict]) -> list[float]:
        df = pd.DataFrame(rows)[self.features]
        return [round(float(p), 3) for p in self.pipeline.predict_proba(df)[:, 1]]


class TelemetrySimulator:
    def __init__(self, machines: list[dict], model: FailureModel, seed: int | None = None):
        self.rng = random.Random(seed)
        self.model = model
        self.state = {}
        for m in machines:
            degraded = m.get("status") in ("Degraded", "Under Maintenance")
            self.state[m["id"]] = {
                "name": m["name"], "tier": m.get("tier", "M"),
                "base_temp": 72 if degraded else 64,
                "base_vib": 4.5 if degraded else 2.6,
                "tool_wear": self.rng.uniform(40, 190 if degraded else 120),
            }
        self.spike_machine, self.spike_left = None, 0

    def tick(self) -> dict:
        if self.spike_left == 0 and self.rng.random() < 0.18:
            self.spike_machine, self.spike_left = self.rng.choice(list(self.state)), self.rng.randint(2, 4)
        machines, feature_rows = [], []
        for mid, st in self.state.items():
            spiking = mid == self.spike_machine and self.spike_left > 0
            temp = st["base_temp"] + self.rng.gauss(0, 4) + (self.rng.uniform(22, 30) if spiking else 0)
            vib = max(0.5, st["base_vib"] + self.rng.gauss(0, 0.6) + (3.0 if spiking else 0))
            st["tool_wear"] = (st["tool_wear"] + self.rng.uniform(0.5, 2.0)) % 250
            air_k = 300 + self.rng.gauss(0, 1.5)
            row = {
                "Type": st["tier"],
                "Air temperature [K]": air_k,
                # a hot machine dissipates poorly -> small process/air gap (heat-dissipation failure mode)
                "Process temperature [K]": air_k + (7.0 if spiking else 10.5) + self.rng.gauss(0, 0.5),
                "Rotational speed [rpm]": min(1330, 1250 + self.rng.gauss(0, 35)) if spiking else 1550 + self.rng.gauss(0, 50),
                "Torque [Nm]": (55 if spiking else 40) + self.rng.gauss(0, 5),
                "Tool wear [min]": st["tool_wear"],
            }
            feature_rows.append(row)
            machines.append({
                "machine_id": mid, "name": st["name"],
                "temperature_c": round(temp, 1), "vibration_mm_s": round(vib, 2),
                "rpm": round(row["Rotational speed [rpm]"]), "torque_nm": round(row["Torque [Nm]"], 1),
                "tool_wear_min": round(st["tool_wear"]), "alert": temp > config.ALERT_TEMP_C,
            })
        for m, p in zip(machines, self.model.predict_proba(feature_rows)):
            m["failure_probability"] = p
        if self.spike_left:
            self.spike_left -= 1
        return {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "machines": machines}
