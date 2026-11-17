"""Validation-only temperature scaling and calibrated probability output."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def softmax_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    logits = np.asarray(logits, dtype=float) / temperature
    logits -= np.max(logits, axis=1, keepdims=True)
    exp = np.exp(np.clip(logits, -745.0, 50.0))
    return exp / np.sum(exp, axis=1, keepdims=True)


class TemperatureScaler:
    def fit(self, validation_logits: np.ndarray, validation_y: np.ndarray, classes: np.ndarray) -> "TemperatureScaler":
        logits = np.asarray(validation_logits, dtype=float)
        y = np.asarray(validation_y)
        self.classes_ = np.asarray(classes)
        if logits.shape != (len(y), len(self.classes_)):
            raise ValueError("Validation logits must have one column per supplied class.")
        class_to_index = {value: idx for idx, value in enumerate(self.classes_.tolist())}
        try:
            y_index = np.asarray([class_to_index[value] for value in y.tolist()], dtype=int)
        except KeyError as exc:
            raise ValueError(f"Validation label {exc.args[0]!r} is absent from classes.") from exc

        def objective(log_temperature: float) -> float:
            probability = softmax_temperature(logits, float(np.exp(log_temperature)))
            return float(-np.mean(np.log(np.clip(probability[np.arange(len(y)), y_index], 1e-15, 1.0))))

        result = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded", options={"xatol": 1e-8})
        if not result.success:
            raise RuntimeError(f"Temperature scaling failed: {result.message}")
        self.temperature_ = float(np.exp(result.x))
        self.validation_nll_ = float(result.fun)
        self.validation_row_count_ = int(len(y))
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        if not hasattr(self, "temperature_"):
            raise RuntimeError("TemperatureScaler must be fitted on validation logits.")
        return softmax_temperature(logits, self.temperature_)

    def audit_record(self) -> dict:
        return {
            "temperature": float(self.temperature_),
            "validation_nll": float(self.validation_nll_),
            "validation_row_count": int(self.validation_row_count_),
            "fit_scope": "validation_only",
        }
