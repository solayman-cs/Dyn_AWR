#!/usr/bin/env python3
"""ProcessMiner three-representation early-warning study (trimmed results version).

Representations : raw, dynamics_only, dynamics_augmented
Models          : HGB, XGBoost, LSTM, TCN
Threshold       : operational-utility maximisation subject to time-under-alarm <= budget,
                  selected on a dedicated chronological calibration split.

This version runs exactly the same data loading, splits, feature construction,
model training, threshold selection and test evaluation as v36, but saves ONLY:

  <output>/seed_<s>/model_ablation_metrics.csv      observation + event metrics per
                                                   model / representation / horizon
  <output>/seed_<s>/test_predictions.csv           per-observation test probabilities,
                                                   targets, thresholds and alarms
  <output>/seed_<s>/threshold_policy_ablation.csv  four threshold policies on the test set
  <output>/seed_<s>/complexity_efficiency_metrics.csv
                                                   computational cost per component

No figures and no other tables are written. All figures and paper tables are built
afterwards from these files.

The warning-model risk surrogate (_score_candidate_with_warning_surrogate) is kept
in the code for the candidate-adjustment case study but is not executed here.
"""

from __future__ import annotations

import argparse
import gc
import io
import math
import os
import pickle
import random
import threading
import time
import tracemalloc
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover
    torch = None
    nn = None
    Dataset = object
    DataLoader = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


# Exact column-order mapping verified between the legacy ProcessMiner MTS CSV
# (time, y, x1..x61) and the named ProcessMiner export over the shared
# 18,398-observation benchmark interval. Renaming changes labels only; values,
# splits, feature construction, models, thresholds, and evaluation remain unchanged.
PROCESSMINER_X_TO_PHYSICAL = {
    "x1": "RSashScanAvg",
    "x2": "CT#1 BLADE PSI",
    "x3": "P4 CT#2 BLADE PSI",
    "x4": "Bleached GWD Flow",
    "x5": "ShwerTemp",
    "x6": "BlndStckFloTPD",
    "x7": "C1 BW SPREAD CD",
    "x8": "RS BW SPREAD CD",
    "x9": "C1 BW SPREAD MD",
    "x10": "RS BW SPREAD MD",
    "x11": "C1 BW SCAN AVG",
    "x12": "RS BW SCAN AVG",
    "x13": "CoatBrkFlo",
    "x14": "Clay Flow",
    "x15": "CouchLoVac",
    "x16": "COUCH VAC",
    "x17": "4PrsTopLd",
    "x18": "4PrsBotLod",
    "x19": "CalndrDrw",
    "x20": "2DryrDrw",
    "x21": "3DryrDrw",
    "x22": "4DryrDraw",
    "x23": "1PrsTopDrw",
    "x24": "4PrsBotDrw",
    "x25": "FanPmpSpd",
    "x26": "FlBxHdrVac",
    "x27": "FlatBxVac",
    "x28": "Grade&Bwt",
    "x29": "UnblGWDFlo",
    "x30": "Hdbox pH",
    "x31": "HdBxLiqLvl",
    "x32": "TotHead\"",
    "x33": "HorzSlcPos",
    "x34": "KraftFlow",
    "x35": "CouchLoad",
    "x36": "C1MoSprdCD",
    "x37": "RSMoSprdCD",
    "x38": "C1MoSprdMD",
    "x39": "RSMoSprdMD",
    "x40": "RL MoisAct",
    "x41": "PrScrRjFlo",
    "x42": "RwBrkFlo",
    "x43": "RcycFbrFlo",
    "x44": "RetnAidFlo",
    "x45": "RUSH DRAG",
    "x46": "Rush/Drag",
    "x47": "SilicaFlo",
    "x48": "HBxSlcTemp",
    "x49": "SodAlumFlo",
    "x50": "CouchSpd",
    "x51": "MachSpd",
    "x52": "1PrsTopSpd",
    "x53": "4PrsBotSpd",
    "x54": "WtNStarFlo",
    "x55": "BasWgtFlo",
    "x56": "TMP Flow",
    "x57": "HBxTotHead",
    "x58": "TrayCons",
    "x59": "UpprHdTmpRL",
    "x60": "VertSlcPos",
    "x61": "EventPress",
}


@dataclass(frozen=True)
class Config:
    data_path: Path
    output_dir: Path
    horizons: tuple[int, ...] = (1, 3, 5, 10)

    train_fraction: float = 0.60
    validation_fraction: float = 0.10
    threshold_calibration_fraction: float = 0.10
    # Remaining 20% is untouched test data.

    rolling_windows: tuple[int, ...] = (5, 15, 30)
    vae_components: int = 6
    vae_hidden_size: int = 32
    vae_epochs: int = 50
    vae_learning_rate: float = 1e-3
    vae_beta: float = 1e-3

    ftle_window: int = 120
    ftle_horizon: int = 5
    ftle_stride: int = 5
    ftle_ridge: float = 1e-3

    sequence_length: int = 30
    neural_batch_size: int = 256
    neural_epochs: int = 30
    neural_patience: int = 6
    neural_learning_rate: float = 1e-3
    neural_hidden_size: int = 64
    neural_dropout: float = 0.20

    utility_false_alarm_weight: float = 0.02
    utility_alarm_time_weight: float = 1.00
    utility_lead_weight: float = 0.05
    max_time_under_alarm: float = 0.10
    threshold_candidates: int = 300
    event_merge_gap: int = 1

    random_seed: int = 42
    device: str = "cpu"
    feature_settings: tuple[str, ...] = ("raw", "dynamics_only", "dynamics_augmented")
    models: tuple[str, ...] = ("HGB", "XGB", "LSTM", "TCN")
    fast: bool = False

    enable_efficiency_analysis: bool = True
    enable_threshold_ablation: bool = True

    # Used only by the (not executed) warning-model risk surrogate.
    advisory_model: str = "HGB"
    advisory_feature_setting: str = "dynamics_augmented"

    include_eventpress: bool = True
    named_data_window: str = "legacy"


def parse_csv_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(v.strip()) for v in text.split(",") if v.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return values


def parse_csv_strings(text: str) -> tuple[str, ...]:
    values = tuple(v.strip() for v in text.split(",") if v.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


def load_processminer_like_csv(
    path: Path,
    include_eventpress: bool = True,
    named_data_window: str = "legacy",
) -> tuple[pd.DataFrame, list[str]]:
    """Accept either the legacy manuscript CSV or the named ProcessMiner CSV.

    Legacy format:
        time, y, x1, x2, ..., x61

    Named format:
        DateTime, SheetBreak, <named process variables>, EventPress

    The named format preserves the original numeric column order. EventPress is
    included by default so the named-data experiment retains the same number of
    process channels as the earlier x1..x61 experiment. It can be excluded
    explicitly with --exclude-eventpress for a sensitivity analysis.
    """
    df = pd.read_csv(path)

    if {"time", "y"}.issubset(df.columns):
        out = df.copy()
        out["time"] = parse_time(out["time"])
        out["y"] = pd.to_numeric(out["y"], errors="coerce").fillna(0).astype(int)
        sensor_cols = [c for c in out.columns if c.startswith("x")]
        if not sensor_cols:
            raise ValueError("No sensor columns beginning with 'x' were found.")
        out = out[["time", "y"] + sensor_cols].copy()
        out = out.sort_values("time", kind="mergesort").reset_index(drop=True)
        print(f"[INFO] Loaded legacy ProcessMiner format with {len(sensor_cols)} sensor columns.")
        return out, sensor_cols

    if {"DateTime", "SheetBreak"}.issubset(df.columns):
        out = df.copy()
        out["time"] = parse_time(out["DateTime"])
        out["y"] = pd.to_numeric(out["SheetBreak"], errors="coerce").fillna(0).astype(int)

        excluded = {"DateTime", "SheetBreak", "time", "y"}
        sensor_cols = [
            c for c in out.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(out[c])
        ]

        if not include_eventpress and "EventPress" in sensor_cols:
            sensor_cols.remove("EventPress")

        if not sensor_cols:
            raise ValueError("No numeric process variables were found in the named ProcessMiner CSV.")

        out = out[["time", "y"] + sensor_cols].copy()
        out = out.sort_values("time", kind="mergesort").reset_index(drop=True)

        # IMPORTANT: the named CSV contains 20,458 observations through May 31,
        # while the original public time/y/x1..x61 benchmark contains 18,398
        # observations and ends at 1999-05-29 00:06. Recomputing 60/10/10/20
        # chronological splits on the extended file changes the train/cal/test
        # periods and therefore changes the Pareto operating points.
        mode = str(named_data_window).strip().lower()
        if mode not in {"legacy", "full"}:
            raise ValueError("named_data_window must be either 'legacy' or 'full'.")
        if mode == "legacy":
            legacy_end = pd.Timestamp("1999-05-29 00:06:00")
            out = out[out["time"] <= legacy_end].copy().reset_index(drop=True)
            if len(out) != 18398:
                warnings.warn(
                    f"Expected 18,398 rows in the legacy benchmark window, found {len(out)}. "
                    "Verify that this is the mapped public ProcessMiner dataset."
                )
            print(f"[INFO] Named dataset restricted to legacy benchmark window: {len(out)} rows, ending {out['time'].iloc[-1]}.")
        else:
            print(f"[INFO] Using full extended named dataset: {len(out)} rows, ending {out['time'].iloc[-1]}.")

        print(f"[INFO] Loaded named ProcessMiner format with {len(sensor_cols)} sensor columns.")
        if "EventPress" in sensor_cols:
            print("[INFO] EventPress INCLUDED to preserve comparability with the legacy x1..x61 experiment.")
        else:
            print("[INFO] EventPress EXCLUDED by --exclude-eventpress (sensitivity mode).")
        return out, sensor_cols

    raise ValueError(
        "Dataset must contain either ['time','y',x1..] or ['DateTime','SheetBreak',named variables] columns."
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def parse_time(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%m/%d/%y %H:%M", errors="coerce")
    if parsed.isna().any():
        parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        raise ValueError("Some timestamps could not be parsed.")
    return parsed


def chronological_splits(n: int, cfg: Config) -> dict[str, np.ndarray]:
    a = int(n * cfg.train_fraction)
    b = int(n * (cfg.train_fraction + cfg.validation_fraction))
    c = int(n * (cfg.train_fraction + cfg.validation_fraction + cfg.threshold_calibration_fraction))
    if not (0 < a < b < c < n):
        raise ValueError("Invalid chronological split fractions.")
    return {
        "train": np.arange(0, a),
        "validation": np.arange(a, b),
        "threshold_calibration": np.arange(b, c),
        "test": np.arange(c, n),
    }


def future_event_target_within_split(y_split: np.ndarray, horizon: int) -> np.ndarray:
    """Target at t is 1 when an event occurs in t+1,...,t+h within the same split."""
    out = np.full(len(y_split), np.nan, dtype=float)
    for t in range(len(y_split) - horizon):
        out[t] = float(np.any(y_split[t + 1:t + horizon + 1] == 1))
    return out


def event_onsets(y: np.ndarray) -> np.ndarray:
    return np.flatnonzero((y == 1) & (np.r_[0, y[:-1]] == 0))


def count_episodes(mask: np.ndarray, merge_gap: int = 1) -> int:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return 0
    return 1 + int(np.sum(np.diff(idx) > merge_gap + 1))


def duration_days(times: pd.Series) -> float:
    if len(times) < 2:
        return 1.0 / 720.0
    seconds = (pd.Timestamp(times.iloc[-1]) - pd.Timestamp(times.iloc[0])).total_seconds()
    return max(seconds / 86400.0, 1.0 / 720.0)


def event_metrics(
    original_y: np.ndarray,
    alarm: np.ndarray,
    times: pd.Series,
    horizon: int,
    merge_gap: int,
) -> dict[str, float]:
    events = event_onsets(original_y)
    warning_union = np.zeros(len(alarm), dtype=bool)
    leads: list[int] = []
    detected = 0

    for event in events:
        start = max(0, event - horizon)
        warning_union[start:event] = True
        hits = np.flatnonzero(alarm[start:event]) + start
        if len(hits):
            detected += 1
            leads.append(event - int(hits[0]))

    false_alarm_mask = alarm & ~warning_union & (original_y == 0)
    return {
        "Events": int(len(events)),
        "EventsDetected": int(detected),
        "EventHitRate": float(detected / len(events)) if len(events) else np.nan,
        "MeanLeadObservations": float(np.mean(leads)) if leads else np.nan,
        "FalseAlarmEpisodesPerDay": float(
            count_episodes(false_alarm_mask, merge_gap) / duration_days(times)
        ),
        "TimeUnderAlarm": float(np.mean(alarm)),
    }


def operational_utility(metrics: dict[str, float], horizon: int, cfg: Config) -> float:
    hit = float(metrics["EventHitRate"]) if np.isfinite(metrics["EventHitRate"]) else 0.0
    lead = metrics["MeanLeadObservations"]
    normalized_lead = float(np.clip(lead / max(horizon, 1), 0, 1)) if np.isfinite(lead) else 0.0
    return (
        hit
        + cfg.utility_lead_weight * hit * normalized_lead
        - cfg.utility_false_alarm_weight * float(metrics["FalseAlarmEpisodesPerDay"])
        - cfg.utility_alarm_time_weight * float(metrics["TimeUnderAlarm"])
    )


def choose_utility_threshold(
    scores: np.ndarray,
    local_positions: np.ndarray,
    original_y: np.ndarray,
    times: pd.Series,
    horizon: int,
    cfg: Config,
) -> tuple[float, dict[str, float]]:
    """Choose one threshold by utility maximization subject only to time-under-alarm <= budget.

    False-alarm episodes/day remains part of the utility penalty, but it is NOT
    used as a hard feasibility constraint.
    """
    finite = np.isfinite(scores)
    if finite.sum() == 0:
        raise ValueError("No finite calibration scores available.")
    valid_scores = scores[finite]
    quantiles = np.linspace(0.001, 0.999, cfg.threshold_candidates)
    candidates = np.unique(np.r_[0.0, np.quantile(valid_scores, quantiles), 1.0])

    feasible_records = []
    fallback_records = []

    for threshold in candidates:
        alarm = np.zeros(len(original_y), dtype=bool)
        alarm[local_positions] = scores >= threshold
        metrics = event_metrics(original_y, alarm, times, horizon, cfg.event_merge_gap)
        utility = operational_utility(metrics, horizon, cfg)
        fa = float(metrics["FalseAlarmEpisodesPerDay"])
        alarm_time = float(metrics["TimeUnderAlarm"])
        hit = metrics["EventHitRate"] if np.isfinite(metrics["EventHitRate"]) else -np.inf
        lead = metrics["MeanLeadObservations"] if np.isfinite(metrics["MeanLeadObservations"]) else -np.inf

        # Single hard constraint: time under alarm <= 10% by default.
        if alarm_time <= cfg.max_time_under_alarm:
            key = (utility, hit, lead, -fa, -alarm_time)
            feasible_records.append((key, float(threshold), metrics))

        # If no threshold satisfies the budget, choose the closest alarm-time
        # operating point, then prefer higher utility/hit rate.
        violation = max(0.0, alarm_time - cfg.max_time_under_alarm)
        fallback_records.append(
            (violation, -utility, -hit, -lead, fa, alarm_time, float(threshold), metrics)
        )

    if feasible_records:
        _, best_threshold, best_metrics = max(feasible_records, key=lambda row: row[0])
        return best_threshold, best_metrics

    fallback_records.sort(key=lambda row: row[:-2])
    *_, best_threshold, best_metrics = fallback_records[0]
    return float(best_threshold), best_metrics


# -----------------------------------------------------------------------------
# Threshold-policy ablation
# -----------------------------------------------------------------------------


def _candidate_thresholds(scores: np.ndarray, n_candidates: int) -> np.ndarray:
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return np.asarray([0.5], dtype=float)
    quantiles = np.linspace(0.001, 0.999, max(int(n_candidates), 10))
    return np.unique(np.r_[0.0, 0.5, np.quantile(finite, quantiles), 1.0])


def choose_max_f1_threshold(
    scores: np.ndarray,
    targets: np.ndarray,
    cfg: Config,
) -> float:
    """Select a threshold maximizing observation-level F1 on calibration data."""
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(targets, dtype=int)
    finite = np.isfinite(scores)
    if finite.sum() == 0:
        return 0.5
    s = scores[finite]
    y = targets[finite]
    best = (-np.inf, -np.inf, -np.inf, 0.5)
    for threshold in _candidate_thresholds(s, cfg.threshold_candidates):
        pred = s >= threshold
        f1 = f1_score(y, pred, zero_division=0)
        precision = precision_score(y, pred, zero_division=0)
        recall = recall_score(y, pred, zero_division=0)
        key = (float(f1), float(recall), float(precision), float(threshold))
        if key > best:
            best = key
    return float(best[-1])


def choose_hit_rate_budget_threshold(
    scores: np.ndarray,
    local_positions: np.ndarray,
    original_y: np.ndarray,
    times: pd.Series,
    horizon: int,
    cfg: Config,
) -> tuple[float, dict[str, float]]:
    """Maximize event hit rate subject only to Time Under Alarm <= configured budget."""
    finite = np.isfinite(scores)
    if finite.sum() == 0:
        raise ValueError("No finite calibration scores available.")

    candidates = _candidate_thresholds(scores[finite], cfg.threshold_candidates)
    feasible: list[tuple[tuple[float, ...], float, dict[str, float]]] = []
    fallback: list[tuple] = []

    for threshold in candidates:
        alarm = np.zeros(len(original_y), dtype=bool)
        alarm[local_positions] = scores >= threshold
        metrics = event_metrics(original_y, alarm, times, horizon, cfg.event_merge_gap)
        hit = float(metrics["EventHitRate"]) if np.isfinite(metrics["EventHitRate"]) else -np.inf
        lead = float(metrics["MeanLeadObservations"]) if np.isfinite(metrics["MeanLeadObservations"]) else -np.inf
        fa = float(metrics["FalseAlarmEpisodesPerDay"])
        alarm_time = float(metrics["TimeUnderAlarm"])
        utility = operational_utility(metrics, horizon, cfg)

        if alarm_time <= cfg.max_time_under_alarm:
            # Hit rate is the primary objective; lead, utility and nuisance burden break ties.
            feasible.append(((hit, lead, utility, -fa, -alarm_time), float(threshold), metrics))

        violation = max(0.0, alarm_time - cfg.max_time_under_alarm)
        fallback.append((violation, -hit, -lead, -utility, fa, alarm_time, float(threshold), metrics))

    if feasible:
        _, threshold, metrics = max(feasible, key=lambda row: row[0])
        return float(threshold), metrics

    fallback.sort(key=lambda row: row[:-2])
    *_, threshold, metrics = fallback[0]
    return float(threshold), metrics


def evaluate_threshold_on_split(
    threshold: float,
    scores: np.ndarray,
    local_positions: np.ndarray,
    targets: np.ndarray,
    original_y: np.ndarray,
    times: pd.Series,
    horizon: int,
    cfg: Config,
) -> dict[str, float]:
    """Evaluate one fixed threshold without any re-selection on the evaluated split."""
    alarm = np.zeros(len(original_y), dtype=bool)
    alarm[local_positions] = scores >= threshold
    event = event_metrics(original_y, alarm, times, horizon, cfg.event_merge_gap)
    obs = observation_metrics(targets, scores, threshold)
    return {
        "Threshold": float(threshold),
        "F1": float(obs["F1"]),
        "Precision": float(obs["Precision"]),
        "Recall": float(obs["Recall"]),
        "EventHitRate": float(event["EventHitRate"]),
        "MeanLeadObservations": float(event["MeanLeadObservations"]),
        "FalseAlarmEpisodesPerDay": float(event["FalseAlarmEpisodesPerDay"]),
        "TimeUnderAlarm": float(event["TimeUnderAlarm"]),
        "OperationalUtility": float(operational_utility(event, horizon, cfg)),
    }


def threshold_policy_ablation_rows(
    model_name: str,
    feature_setting: str,
    horizon: int,
    score_by_split: dict[str, np.ndarray],
    local_by_split: dict[str, np.ndarray],
    target_by_split: dict[str, np.ndarray],
    y_original: np.ndarray,
    times: pd.Series,
    splits: dict[str, np.ndarray],
    cfg: Config,
) -> list[dict]:
    """Compare calibration-only threshold policies on the untouched test split."""
    cal_idx = splits["threshold_calibration"]
    test_idx = splits["test"]
    cal_y = y_original[cal_idx]
    cal_times = times.iloc[cal_idx].reset_index(drop=True)
    test_y = y_original[test_idx]
    test_times = times.iloc[test_idx].reset_index(drop=True)

    utility_thr, _ = choose_utility_threshold(
        score_by_split["threshold_calibration"],
        local_by_split["threshold_calibration"],
        cal_y,
        cal_times,
        horizon,
        cfg,
    )
    maxf1_thr = choose_max_f1_threshold(
        score_by_split["threshold_calibration"],
        target_by_split["threshold_calibration"],
        cfg,
    )
    hit_thr, _ = choose_hit_rate_budget_threshold(
        score_by_split["threshold_calibration"],
        local_by_split["threshold_calibration"],
        cal_y,
        cal_times,
        horizon,
        cfg,
    )

    policies = {
        "Utility_A<=10pct": utility_thr,
        "HitRate_A<=10pct": hit_thr,
        "MaxF1_calibration": maxf1_thr,
        "Fixed_0.5": 0.5,
    }

    rows: list[dict] = []
    for method, threshold in policies.items():
        cal_metrics = evaluate_threshold_on_split(
            threshold,
            score_by_split["threshold_calibration"],
            local_by_split["threshold_calibration"],
            target_by_split["threshold_calibration"],
            cal_y,
            cal_times,
            horizon,
            cfg,
        )
        test_metrics = evaluate_threshold_on_split(
            threshold,
            score_by_split["test"],
            local_by_split["test"],
            target_by_split["test"],
            test_y,
            test_times,
            horizon,
            cfg,
        )
        row = {
            "Model": model_name,
            "FeatureSetting": feature_setting,
            "Horizon": int(horizon),
            "ThresholdPolicy": method,
            "Threshold": float(threshold),
        }
        row.update({f"Calibration{k}": v for k, v in cal_metrics.items() if k != "Threshold"})
        row.update({f"Test{k}": v for k, v in test_metrics.items() if k != "Threshold"})
        rows.append(row)
    return rows


# -----------------------------------------------------------------------------
# Computational-efficiency measurement
# -----------------------------------------------------------------------------


def _process_rss_mb() -> float:
    try:
        import psutil
        return float(psutil.Process(os.getpid()).memory_info().rss / (1024.0 ** 2))
    except Exception:
        return np.nan


class _PeakRSSMonitor:
    """Sample process RSS during a bounded operation and report peak increase.

    Unlike tracemalloc, RSS includes native-library and PyTorch CPU allocations, so it
    is suitable for comparing HGB/XGBoost with LSTM/TCN on host memory.
    """
    def __init__(self, interval_sec: float = 0.02):
        self.interval_sec = float(interval_sec)
        self.baseline_mb = _process_rss_mb()
        self.peak_mb = self.baseline_mb
        self._stop_event = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop_event.is_set():
            value = _process_rss_mb()
            if np.isfinite(value):
                if not np.isfinite(self.peak_mb):
                    self.peak_mb = value
                else:
                    self.peak_mb = max(self.peak_mb, value)
            self._stop_event.wait(self.interval_sec)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> float:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        final = _process_rss_mb()
        if np.isfinite(final):
            self.peak_mb = max(self.peak_mb, final) if np.isfinite(self.peak_mb) else final
        if np.isfinite(self.baseline_mb) and np.isfinite(self.peak_mb):
            return float(max(0.0, self.peak_mb - self.baseline_mb))
        return np.nan


def serialized_model_size_mb(model, kind: str) -> float:
    """Approximate stored model size; useful for deployment comparison."""
    try:
        if kind == "neural" and torch is not None:
            buf = io.BytesIO()
            torch.save(model.state_dict(), buf)
            return float(buf.getbuffer().nbytes / (1024.0 ** 2))
        return float(len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) / (1024.0 ** 2))
    except Exception:
        return np.nan


def model_parameter_count(model, kind: str) -> float:
    if kind == "neural" and torch is not None:
        return float(sum(p.numel() for p in model.parameters()))
    return np.nan


def model_tree_count(model_name: str, model) -> float:
    try:
        estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
        if model_name == "XGB" and hasattr(estimator, "get_booster"):
            return float(len(estimator.get_booster().get_dump()))
        if model_name == "HGB" and hasattr(estimator, "n_iter_"):
            return float(estimator.n_iter_)
    except Exception:
        pass
    return np.nan


def theoretical_complexity_label(model_name: str) -> str:
    labels = {
        "HGB": "Approx. O(T*N*D*log N)",
        "XGB": "Approx. O(T*N*D*log N)",
        "LSTM": "O(N*L*H*(D+H))",
        "TCN": "O(N*L*K*H^2*B)",
    }
    return labels.get(model_name, "Model dependent")


def efficiency_record(
    model_name: str,
    feature_setting: str,
    horizon: int,
    kind: str,
    model,
    feature_count: int,
    train_samples: int,
    test_samples: int,
    train_input_memory_mb: float,
    train_time_sec: float,
    inference_time_sec: float,
    peak_python_memory_mb: float,
    rss_delta_mb: float,
    sequence_length: int,
    peak_host_memory_mb: float = np.nan,
    peak_gpu_memory_mb: float = np.nan,
) -> dict:
    latency_ms = 1000.0 * inference_time_sec / max(test_samples, 1)
    throughput = test_samples / inference_time_sec if inference_time_sec > 0 else np.nan
    return {
        "Model": model_name,
        "FeatureSetting": feature_setting,
        "Horizon": int(horizon),
        "ModelKind": kind,
        "FeatureCount": int(feature_count),
        "SequenceLength": int(sequence_length if kind == "neural" else 1),
        "TrainSamples": int(train_samples),
        "TestSamples": int(test_samples),
        "TrainInputMemoryMB": float(train_input_memory_mb),
        "PeakPythonMemoryMB": float(peak_python_memory_mb),
        "RSSDeltaMB": float(rss_delta_mb) if np.isfinite(rss_delta_mb) else np.nan,
        "PeakHostMemoryMB": float(peak_host_memory_mb) if np.isfinite(peak_host_memory_mb) else np.nan,
        "PeakGPUMemoryMB": float(peak_gpu_memory_mb) if np.isfinite(peak_gpu_memory_mb) else np.nan,
        "TrainTimeSec": float(train_time_sec),
        "TestInferenceTimeSec": float(inference_time_sec),
        "InferenceLatencyMsPerSample": float(latency_ms),
        "InferenceThroughputSamplesPerSec": float(throughput),
        "ModelSizeMB": serialized_model_size_mb(model, kind),
        "ParameterCount": model_parameter_count(model, kind),
        "TreeCount": model_tree_count(model_name, model) if kind == "tree" else np.nan,
        "TheoreticalComplexity": theoretical_complexity_label(model_name),
    }


# -----------------------------------------------------------------------------
# Feature construction
# -----------------------------------------------------------------------------


class TabularVAE(nn.Module if nn is not None else object):
    """Compact VAE used only to learn nonlinear train-only latent process states."""
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        if nn is None:
            raise RuntimeError(f"PyTorch is unavailable: {TORCH_IMPORT_ERROR}")
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def fit_or_apply_vae_features(
    all_scaled: np.ndarray,
    train_idx: np.ndarray,
    cfg: Config,
    state: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit the VAE on train rows only, or reuse a previously fitted VAE state.

    The posterior mean is the deterministic latent process state.  The returned
    dynamic descriptors include coordinate-wise latent velocity/acceleration,
    their norms, reconstruction error, and causal rolling summaries.
    """
    if torch is None or nn is None:
        raise RuntimeError(
            "VAE dynamics require PyTorch. Install/use PyTorch or remove "
            "dynamics_only/dynamics_augmented/vae_latent from --feature-settings."
        )
    set_seed(cfg.random_seed + 4242)
    device = torch.device(cfg.device)
    input_dim = int(all_scaled.shape[1])
    latent_dim = min(int(cfg.vae_components), input_dim)

    if state is None:
        model = TabularVAE(input_dim, int(cfg.vae_hidden_size), latent_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.vae_learning_rate, weight_decay=1e-5)
        x_train = torch.as_tensor(all_scaled[train_idx], dtype=torch.float32)
        loader = DataLoader(
            torch.utils.data.TensorDataset(x_train),
            batch_size=min(cfg.neural_batch_size, max(32, len(x_train))),
            shuffle=True, num_workers=0,
        )
        model.train()
        n_epochs = min(cfg.vae_epochs, 5) if cfg.fast else cfg.vae_epochs
        for _ in range(max(1, int(n_epochs))):
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                recon, mu, logvar = model(batch)
                recon_loss = torch.mean((recon - batch) ** 2)
                kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + float(cfg.vae_beta) * kl
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        state = {
            "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "input_dim": input_dim, "hidden_dim": int(cfg.vae_hidden_size), "latent_dim": latent_dim,
        }
    else:
        model = TabularVAE(
            int(state["input_dim"]), int(state["hidden_dim"]), int(state["latent_dim"])
        ).to(device)
        model.load_state_dict(state["model_state"])

    model.eval()
    all_tensor = torch.as_tensor(all_scaled, dtype=torch.float32, device=device)
    mus, recons = [], []
    batch_size = max(64, int(cfg.neural_batch_size))
    with torch.no_grad():
        for start in range(0, len(all_scaled), batch_size):
            xb = all_tensor[start:start + batch_size]
            mu, _ = model.encode(xb)
            recon = model.decoder(mu)
            mus.append(mu.cpu().numpy())
            recons.append(recon.cpu().numpy())
    latent = np.concatenate(mus, axis=0)
    reconstructed = np.concatenate(recons, axis=0)
    recon_error = np.sqrt(np.mean((all_scaled - reconstructed) ** 2, axis=1))

    z = pd.DataFrame(latent, columns=[f"vae_latent_{i+1}" for i in range(latent.shape[1])])
    vel = z.diff()
    acc = vel.diff()
    feat = pd.DataFrame(index=z.index)
    feat["vae_reconstruction_error"] = recon_error
    feat["vae_latent_speed_norm"] = np.sqrt((vel ** 2).sum(axis=1))
    feat["vae_latent_acceleration_norm"] = np.sqrt((acc ** 2).sum(axis=1))
    for col in z.columns:
        feat[f"{col}__velocity"] = vel[col]
        feat[f"{col}__acceleration"] = acc[col]
    for window in cfg.rolling_windows:
        feat[f"vae_speed_mean_{window}"] = feat["vae_latent_speed_norm"].rolling(window, min_periods=2).mean()
        feat[f"vae_speed_std_{window}"] = feat["vae_latent_speed_norm"].rolling(window, min_periods=3).std(ddof=0)
        feat[f"vae_reconstruction_error_mean_{window}"] = feat["vae_reconstruction_error"].rolling(window, min_periods=2).mean()
        feat[f"vae_reconstruction_error_slope_{window}"] = (
            feat["vae_reconstruction_error"] - feat["vae_reconstruction_error"].shift(window - 1)
        ) / max(window - 1, 1)
    z.index = feat.index
    return z, feat, state


def causal_local_linear_ftle(
    latent: np.ndarray,
    window: int,
    finite_horizon: int,
    stride: int,
    ridge: float,
) -> pd.DataFrame:
    """Causal local-linear reduced-order FTLE approximation.

    At evaluation time t, fit z_{k+1} ~= A_t z_k using only the preceding
    window. The leading feature is log(sigma_max(A_t^H))/H.
    """
    n, r = latent.shape
    ftle = np.full(n, np.nan)
    spectral_radius = np.full(n, np.nan)
    log_condition = np.full(n, np.nan)
    eye = np.eye(r)

    first_t = max(window, 3)
    for t in range(first_t, n, max(stride, 1)):
        hist = latent[t - window:t]
        x = hist[:-1]
        y = hist[1:]
        x0 = x - np.mean(x, axis=0, keepdims=True)
        y0 = y - np.mean(y, axis=0, keepdims=True)
        gram = x0.T @ x0 + ridge * eye
        try:
            transition = np.linalg.solve(gram, x0.T @ y0).T
            finite_map = np.linalg.matrix_power(transition, finite_horizon)
            singular_values = np.linalg.svd(finite_map, compute_uv=False)
            sigma = max(float(singular_values[0]), 1e-12)
            ftle[t] = math.log(sigma) / finite_horizon
            eigenvalues = np.linalg.eigvals(transition)
            spectral_radius[t] = float(np.max(np.abs(eigenvalues)))
            log_condition[t] = float(np.log1p(np.linalg.cond(transition)))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue

    output = pd.DataFrame({
        "reduced_ftle_leading": pd.Series(ftle).ffill(),
        "local_transition_spectral_radius": pd.Series(spectral_radius).ffill(),
        "local_transition_log_condition": pd.Series(log_condition).ffill(),
    })
    output["reduced_ftle_derivative"] = output["reduced_ftle_leading"].diff()
    return output


def build_feature_sets(
    raw_sensor_frame: pd.DataFrame,
    train_idx: np.ndarray,
    cfg: Config,
    transform_state: dict | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """Construct raw, dynamics-only, dynamics-aware, and VAE ablation representations.

    PCA is not used.  VAE fitting, imputation, and scaling use training rows only.
    The local-linear FTLE/instability descriptors are computed in VAE latent space.
    """
    base = raw_sensor_frame.astype(float).replace([np.inf, -np.inf], np.nan)

    current = base.copy(); current.columns = [f"{c}__current" for c in base.columns]
    lag1 = base.shift(1); lag1.columns = [f"{c}__lag1" for c in base.columns]
    raw_tabular = pd.concat([current, lag1], axis=1)

    diff1 = base.diff(1); diff1.columns = [f"{c}__diff1" for c in base.columns]
    diff2 = base.diff(1).diff(1); diff2.columns = [f"{c}__diff2" for c in base.columns]

    statistical_blocks = [raw_tabular, diff1, diff2]
    dynamic_blocks = [diff1, diff2]
    for window in cfg.rolling_windows:
        roll_mean = base.rolling(window, min_periods=2).mean(); roll_mean.columns = [f"{c}__mean{window}" for c in base.columns]
        roll_std = base.rolling(window, min_periods=3).std(ddof=0); roll_std.columns = [f"{c}__std{window}" for c in base.columns]
        diff_energy = diff1.pow(2).rolling(window, min_periods=2).mean(); diff_energy.columns = [f"{c}__diff_energy{window}" for c in base.columns]
        statistical_blocks.extend([roll_mean, roll_std, diff_energy])
        dynamic_blocks.extend([roll_std, diff_energy])
    statistical = pd.concat(statistical_blocks, axis=1)
    statistical_dynamics_only = pd.concat(dynamic_blocks, axis=1)

    if transform_state is None:
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_imputed = imputer.fit_transform(base.iloc[train_idx])
        scaler.fit(train_imputed)
        transform_state = {"imputer": imputer, "scaler": scaler, "vae": None}
    else:
        imputer = transform_state["imputer"]
        scaler = transform_state["scaler"]

    all_scaled = scaler.transform(imputer.transform(base))
    vae_latent_df, vae_features, vae_state = fit_or_apply_vae_features(
        all_scaled, train_idx, cfg, state=transform_state.get("vae")
    )
    vae_latent_df.index = base.index
    vae_features.index = base.index
    transform_state["vae"] = vae_state

    # FTLE/local transition descriptors now use the nonlinear VAE latent trajectory.
    ftle = causal_local_linear_ftle(
        vae_latent_df.to_numpy(dtype=float),
        window=cfg.ftle_window, finite_horizon=cfg.ftle_horizon,
        stride=cfg.ftle_stride, ridge=cfg.ftle_ridge,
    )
    ftle.index = base.index
    ftle = ftle.rename(columns={
        "reduced_ftle_leading": "vae_reduced_ftle_leading",
        "local_transition_spectral_radius": "vae_local_transition_spectral_radius",
        "local_transition_log_condition": "vae_local_transition_log_condition",
        "reduced_ftle_derivative": "vae_reduced_ftle_derivative",
    })

    # VAE-latent ablation retains absolute nonlinear latent state in addition to statistical features.
    vae_latent_features = pd.concat([statistical, vae_latent_df, vae_features], axis=1)

    # Dynamics-only excludes raw levels and absolute VAE coordinates.
    dynamics_only = pd.concat([statistical_dynamics_only, vae_features, ftle], axis=1)
    dynamics_augmented = pd.concat([raw_tabular, dynamics_only], axis=1)

    feature_sets = {
        "raw": raw_tabular,
        "dynamics_only": dynamics_only,
        "dynamics_augmented": dynamics_augmented,
        "statistical": statistical,
        "vae_latent": vae_latent_features,
    }
    return feature_sets, ftle, transform_state


# -----------------------------------------------------------------------------
# Classical models
# -----------------------------------------------------------------------------


def make_hgb(cfg: Config, seed_offset: int = 0) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=120 if cfg.fast else 350,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=cfg.random_seed + seed_offset,
        )),
    ])


def make_xgboost(cfg: Config, y_train: np.ndarray, seed_offset: int = 0):
    if XGBClassifier is None:
        raise RuntimeError(f"xgboost is unavailable: {XGBOOST_IMPORT_ERROR}")
    positives = max(int(np.sum(y_train == 1)), 1)
    negatives = max(int(np.sum(y_train == 0)), 1)
    scale_pos_weight = negatives / positives
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", XGBClassifier(
            n_estimators=100 if cfg.fast else 450,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.80,
            min_child_weight=3,
            reg_lambda=2.0,
            reg_alpha=0.05,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            n_jobs=max(1, min(8, os.cpu_count() or 1)),
            random_state=cfg.random_seed + seed_offset,
        )),
    ])


# -----------------------------------------------------------------------------
# Neural sequence models
# -----------------------------------------------------------------------------


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.x = torch.as_tensor(sequences, dtype=torch.float32)
        self.y = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(self.dropout(output[:, -1, :])).squeeze(-1)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        return torch.relu(self.net(x) + self.residual(x))


class TCNClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            TCNBlock(input_size, hidden_size, kernel_size=3, dilation=1, dropout=dropout),
            TCNBlock(hidden_size, hidden_size, kernel_size=3, dilation=2, dropout=dropout),
            TCNBlock(hidden_size, hidden_size, kernel_size=3, dilation=4, dropout=dropout),
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Input: batch, time, channels -> batch, channels, time
        z = self.network(x.transpose(1, 2))
        return self.head(z[:, :, -1]).squeeze(-1)


def make_sequences_for_split(
    values: np.ndarray,
    split_indices: np.ndarray,
    targets_within_split: np.ndarray,
    original_y_split: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create strictly within-split sequences ending at eligible non-event t."""
    sequences: list[np.ndarray] = []
    targets: list[int] = []
    local_positions: list[int] = []
    for local_t in range(sequence_length - 1, len(split_indices)):
        if not np.isfinite(targets_within_split[local_t]):
            continue
        if original_y_split[local_t] != 0:
            continue
        global_t = split_indices[local_t]
        global_start = global_t - sequence_length + 1
        # Strictly keep sequence inside the split.
        if global_start < split_indices[0]:
            continue
        seq = values[global_start:global_t + 1]
        if len(seq) != sequence_length:
            continue
        sequences.append(seq)
        targets.append(int(targets_within_split[local_t]))
        local_positions.append(local_t)
    if not sequences:
        return (
            np.empty((0, sequence_length, values.shape[1]), dtype=np.float32),
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
        )
    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=int),
        np.asarray(local_positions, dtype=int),
    )


def fit_neural_model(
    model_name: str,
    input_size: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    cfg: Config,
    seed_offset: int,
):
    if torch is None:
        raise RuntimeError(f"PyTorch is unavailable: {TORCH_IMPORT_ERROR}")
    set_seed(cfg.random_seed + seed_offset)
    device = torch.device(cfg.device)

    if model_name == "LSTM":
        model = LSTMClassifier(input_size, cfg.neural_hidden_size, cfg.neural_dropout)
    elif model_name == "TCN":
        model = TCNClassifier(input_size, cfg.neural_hidden_size, cfg.neural_dropout)
    else:
        raise ValueError(model_name)
    model.to(device)

    positives = max(float(np.sum(train_y == 1)), 1.0)
    negatives = max(float(np.sum(train_y == 0)), 1.0)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.neural_learning_rate, weight_decay=1e-4)

    train_loader = DataLoader(
        SequenceDataset(train_x, train_y),
        batch_size=cfg.neural_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    best_state = None
    best_ap = -np.inf
    best_loss = np.inf
    no_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.neural_epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_prob = predict_neural(model, val_x, cfg)
        val_ap = average_precision_score(val_y, val_prob) if len(np.unique(val_y)) > 1 else 0.0
        val_brier = brier_score_loss(val_y, val_prob)
        train_loss = float(np.mean(losses)) if losses else np.nan
        history.append({
            "Epoch": epoch,
            "TrainLoss": train_loss,
            "ValidationAUPRC": val_ap,
            "ValidationBrier": val_brier,
        })

        improved = (val_ap > best_ap + 1e-6) or (
            abs(val_ap - best_ap) <= 1e-6 and val_brier < best_loss
        )
        if improved:
            best_ap = val_ap
            best_loss = val_brier
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= cfg.neural_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def predict_neural(model, sequences: np.ndarray, cfg: Config) -> np.ndarray:
    if len(sequences) == 0:
        return np.empty(0, dtype=float)
    device = torch.device(cfg.device)
    model.eval()
    loader = DataLoader(
        SequenceDataset(sequences, np.zeros(len(sequences), dtype=int)),
        batch_size=cfg.neural_batch_size,
        shuffle=False,
        num_workers=0,
    )
    outputs = []
    with torch.no_grad():
        for batch_x, _ in loader:
            logits = model(batch_x.to(device))
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs)


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------


def observation_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = scores >= threshold
    return {
        "AUPRC": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else np.nan,
        "AUROC": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else np.nan,
        "Brier": float(brier_score_loss(y_true, scores)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
    }


def eligible_tabular_split(
    split_indices: np.ndarray,
    y_original: np.ndarray,
    horizon: int,
) -> dict[str, np.ndarray]:
    y_split = y_original[split_indices]
    target = future_event_target_within_split(y_split, horizon)
    mask = np.isfinite(target) & (y_split == 0)
    return {
        "global_indices": split_indices[mask],
        "local_positions": np.flatnonzero(mask),
        "target": target[mask].astype(int),
        "target_full": target,
    }


def feature_frame_for_sequence(
    feature_setting: str,
    feature_sets: dict[str, pd.DataFrame],
    raw_sensors: pd.DataFrame,
) -> pd.DataFrame:
    """Return channels for sequence models while avoiding obvious duplication."""
    if feature_setting == "raw":
        return raw_sensors.copy()
    frame = feature_sets[feature_setting].copy()
    # Sequence history already supplies lagged context. Remove explicit lag-1 and
    # rolling-mean channels, but retain raw current state in dynamics_augmented.
    keep = [c for c in frame.columns if not c.endswith("__lag1") and "__mean" not in c]
    return frame[keep]


def fit_sequence_preprocessor(frame: pd.DataFrame, train_idx: np.ndarray):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(frame.iloc[train_idx])
    scaler.fit(train_imputed)
    all_values = scaler.transform(imputer.transform(frame)).astype(np.float32)
    return all_values, imputer, scaler


# -----------------------------------------------------------------------------
# Warning-model risk surrogate (kept for the candidate-adjustment case study;
# not executed in this results-only version)
# -----------------------------------------------------------------------------


# Candidate process variables used only for the offline recommendation case study.
# The public ProcessMiner data do not provide actuator/setpoint metadata, engineering
# units, or admissible control limits. Therefore these tags are treated strictly as
# candidate manipulable variables for predictive stress testing, not as validated
# plant actuators or prescriptive control commands.
DEFAULT_CANDIDATE_PROCESS_PARAMETERS = (
    "Clay Flow", "KraftFlow", "RetnAidFlo", "SilicaFlo", "SodAlumFlo",
    "CouchSpd", "MachSpd", "HorzSlcPos", "VertSlcPos", "HBxSlcTemp",
)



def _candidate_parameter_metadata(raw_sensors: pd.DataFrame, train_idx: np.ndarray) -> pd.DataFrame:
    """Build train-only scaling/bound metadata for candidate process variables.

    Candidate magnitudes are normalized by each variable's training-set IQR and
    proposed values are restricted to the training 1st--99th percentile range.
    This function does not infer actuator status or engineering safety limits.
    """
    train = raw_sensors.iloc[np.asarray(train_idx, dtype=int)]
    rows = []
    for name in DEFAULT_CANDIDATE_PROCESS_PARAMETERS:
        if name not in raw_sensors.columns:
            continue
        s = (
            pd.to_numeric(train[name], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(s) < 20:
            continue
        q01, q25, q75, q99 = s.quantile([0.01, 0.25, 0.75, 0.99]).to_numpy(dtype=float)
        iqr = float(q75 - q25)
        if not np.isfinite(iqr) or iqr <= 1e-12:
            continue
        rows.append({
            "Parameter": name,
            "TrainQ01": float(q01),
            "TrainQ25": float(q25),
            "TrainQ75": float(q75),
            "TrainQ99": float(q99),
            "TrainIQR": iqr,
            "ControlStatus": "candidate_only_public_data_no_actuator_metadata",
        })
    return pd.DataFrame(rows)


def _score_candidate_with_warning_surrogate(
    raw_sensors: pd.DataFrame,
    global_index: int,
    parameter_index: int | None,
    adjusted_value: float | None,
    horizons: tuple[int, ...],
    fitted_tree_models: dict,
    feature_transform_state: dict,
    cfg: Config,
) -> tuple[float, dict[int, float]]:
    """Evaluate a candidate adjustment with the already-fitted warning model.

    No separate process-transition model is trained.  The selected advisory warning
    model (HGB by default) is reused as a *risk surrogate*: the candidate parameter is
    perturbed at the current observation, the exact feature-construction pipeline is
    recomputed using the frozen train-only transform state, and the fitted horizon-
    specific warning classifiers return counterfactual risk scores.

    This is a local predictive sensitivity / consistency analysis.  It is not a
    dynamical plant simulator and must not be interpreted as causal intervention
    evidence or as a six-step physical process forecast.
    """
    context = max(int(cfg.ftle_window) + 10, max(cfg.rolling_windows) + 10, cfg.sequence_length + 10)
    start = max(0, int(global_index) - context + 1)
    local_raw = raw_sensors.iloc[start:int(global_index) + 1].copy().reset_index(drop=True)

    # Median-fill only the evaluated terminal state, matching the recommendation
    # module's treatment of missing current measurements.
    if len(local_raw) == 0:
        return np.nan, {int(h): np.nan for h in horizons}
    terminal = local_raw.iloc[-1].to_numpy(dtype=float)
    med = raw_sensors.iloc[: max(int(global_index), 1)].median(axis=0, skipna=True).to_numpy(dtype=float)
    terminal = np.where(np.isfinite(terminal), terminal, med)
    if parameter_index is not None and adjusted_value is not None:
        terminal[int(parameter_index)] = float(adjusted_value)
    local_raw.iloc[-1, :] = terminal

    # transform_state was fitted only on the original training partition.  Supplying
    # it here prevents any refit on the intervention/test observation.
    local_train_idx = np.arange(min(len(local_raw), max(3, len(local_raw) - 1)), dtype=int)
    local_feature_sets, _, _ = build_feature_sets(
        local_raw,
        local_train_idx,
        cfg,
        transform_state=feature_transform_state,
    )
    setting = cfg.advisory_feature_setting
    score_row = local_feature_sets[setting].iloc[[-1]]

    scores: dict[int, float] = {}
    for h in horizons:
        key = (cfg.advisory_model, setting, int(h))
        model = fitted_tree_models.get(key)
        if model is None:
            scores[int(h)] = np.nan
            continue
        try:
            scores[int(h)] = float(model.predict_proba(score_row)[:, 1][0])
        except Exception:
            scores[int(h)] = np.nan

    weights = {10: 0.40, 20: 0.30, 30: 0.20, 50: 0.10}
    vals, wts = [], []
    for h, value in scores.items():
        if np.isfinite(value):
            vals.append(value)
            wts.append(weights.get(int(h), 1.0 / max(len(horizons), 1)))
    if not vals:
        return np.nan, scores
    weighted = float(np.average(np.asarray(vals), weights=np.asarray(wts)))
    return weighted, scores


# -----------------------------------------------------------------------------
# Experiment
# -----------------------------------------------------------------------------


def run_experiment(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/evaluate every model x representation x horizon for one seed.

    Saves only: model_ablation_metrics.csv, test_predictions.csv,
    threshold_policy_ablation.csv and complexity_efficiency_metrics.csv.
    """
    set_seed(cfg.random_seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.fast:
        cfg = replace(
            cfg,
            neural_epochs=min(cfg.neural_epochs, 3),
            neural_patience=min(cfg.neural_patience, 2),
            neural_hidden_size=min(cfg.neural_hidden_size, 24),
            threshold_candidates=min(cfg.threshold_candidates, 80),
            vae_epochs=min(cfg.vae_epochs, 5),
        )

    allowed_features = {"raw", "dynamics_only", "dynamics_augmented"}
    unknown_features = set(cfg.feature_settings) - allowed_features
    if unknown_features:
        raise ValueError(f"Unknown feature settings: {sorted(unknown_features)}")
    allowed_models = {"HGB", "XGB", "LSTM", "TCN"}
    unknown_models = set(cfg.models) - allowed_models
    if unknown_models:
        raise ValueError(f"Unknown models: {sorted(unknown_models)}")

    df, sensor_cols = load_processminer_like_csv(
        cfg.data_path,
        include_eventpress=cfg.include_eventpress,
        named_data_window=cfg.named_data_window,
    )

    # Legacy x1..x61 columns are relabelled with their physical tag names
    # (labels only; values and column order are unchanged).
    if all(c in PROCESSMINER_X_TO_PHYSICAL for c in sensor_cols):
        used_names = [PROCESSMINER_X_TO_PHYSICAL[c] for c in sensor_cols]
    else:
        used_names = list(sensor_cols)

    raw_sensors = df[sensor_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    raw_sensors.columns = used_names
    y_original = df["y"].astype(int).to_numpy()
    times = df["time"]
    splits = chronological_splits(len(df), cfg)
    feature_sets, _, _ = build_feature_sets(raw_sensors, splits["train"], cfg)

    result_rows: list[dict] = []
    prediction_rows: list[dict] = []
    threshold_ablation_records: list[dict] = []
    complexity_rows: list[dict] = []

    for horizon in cfg.horizons:
        split_info = {
            name: eligible_tabular_split(indices, y_original, horizon)
            for name, indices in splits.items()
        }

        for feature_setting in cfg.feature_settings:
            tabular_frame = feature_sets[feature_setting]
            sequence_frame = feature_frame_for_sequence(feature_setting, feature_sets, raw_sensors)
            sequence_values, _, _ = fit_sequence_preprocessor(sequence_frame, splits["train"])

            # Prepare sequence data once per horizon and feature setting.
            sequence_data = {}
            for split_name, split_indices in splits.items():
                y_split = y_original[split_indices]
                target_full = future_event_target_within_split(y_split, horizon)
                sequence_data[split_name] = make_sequences_for_split(
                    sequence_values,
                    split_indices,
                    target_full,
                    y_split,
                    cfg.sequence_length,
                )

            for model_index, model_name in enumerate(cfg.models):
                label = f"{model_name}_{feature_setting}"
                seed_offset = 1000 * horizon + 100 * model_index + 10 * list(cfg.feature_settings).index(feature_setting)

                if model_name in {"HGB", "XGB"}:
                    train = split_info["train"]
                    if model_name == "HGB":
                        model = make_hgb(cfg, seed_offset)
                    else:
                        model = make_xgboost(cfg, train["target"], seed_offset)
                    train_frame = tabular_frame.iloc[train["global_indices"]]
                    train_input_memory_mb = float(train_frame.memory_usage(deep=True).sum() / (1024.0 ** 2))
                    gc.collect()
                    rss_before = _process_rss_mb()
                    rss_monitor = _PeakRSSMonitor().start()
                    tracemalloc.start()
                    train_t0 = time.perf_counter()
                    model.fit(train_frame, train["target"])
                    train_time_sec = time.perf_counter() - train_t0
                    peak_host_memory_mb = rss_monitor.stop()
                    peak_gpu_memory_mb = np.nan
                    _, peak_bytes = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    rss_after = _process_rss_mb()
                    peak_python_memory_mb = float(peak_bytes / (1024.0 ** 2))
                    rss_delta_mb = max(0.0, rss_after - rss_before) if np.isfinite(rss_before) and np.isfinite(rss_after) else np.nan

                    score_by_split = {}
                    test_inference_time_sec = np.nan
                    for split_name in ["validation", "threshold_calibration", "test"]:
                        info = split_info[split_name]
                        pred_frame = tabular_frame.iloc[info["global_indices"]]
                        pred_t0 = time.perf_counter()
                        score_by_split[split_name] = model.predict_proba(pred_frame)[:, 1]
                        pred_elapsed = time.perf_counter() - pred_t0
                        if split_name == "test":
                            test_inference_time_sec = pred_elapsed
                    local_by_split = {
                        split_name: split_info[split_name]["local_positions"]
                        for split_name in ["validation", "threshold_calibration", "test"]
                    }
                    target_by_split = {
                        split_name: split_info[split_name]["target"]
                        for split_name in ["validation", "threshold_calibration", "test"]
                    }
                    if cfg.enable_efficiency_analysis:
                        complexity_rows.append(efficiency_record(
                            model_name=model_name,
                            feature_setting=feature_setting,
                            horizon=horizon,
                            kind="tree",
                            model=model,
                            feature_count=tabular_frame.shape[1],
                            train_samples=len(train["target"]),
                            test_samples=len(target_by_split["test"]),
                            train_input_memory_mb=train_input_memory_mb,
                            train_time_sec=train_time_sec,
                            inference_time_sec=float(test_inference_time_sec),
                            peak_python_memory_mb=peak_python_memory_mb,
                            rss_delta_mb=rss_delta_mb,
                            sequence_length=1,
                            peak_host_memory_mb=peak_host_memory_mb,
                            peak_gpu_memory_mb=peak_gpu_memory_mb,
                        ))
                else:
                    train_x, train_y, _ = sequence_data["train"]
                    val_x, val_y, _ = sequence_data["validation"]
                    if len(train_x) == 0 or len(val_x) == 0:
                        warnings.warn(f"Skipping {label} at h={horizon}: no sequences.")
                        continue
                    train_input_memory_mb = float(train_x.nbytes / (1024.0 ** 2))
                    gc.collect()
                    rss_before = _process_rss_mb()
                    rss_monitor = _PeakRSSMonitor().start()
                    tracemalloc.start()
                    if torch is not None and str(cfg.device).startswith("cuda") and torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    train_t0 = time.perf_counter()
                    neural_model, _ = fit_neural_model(
                        model_name,
                        input_size=train_x.shape[-1],
                        train_x=train_x,
                        train_y=train_y,
                        val_x=val_x,
                        val_y=val_y,
                        cfg=cfg,
                        seed_offset=seed_offset,
                    )
                    if torch is not None and str(cfg.device).startswith("cuda") and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    train_time_sec = time.perf_counter() - train_t0
                    peak_host_memory_mb = rss_monitor.stop()
                    peak_gpu_memory_mb = (
                        float(torch.cuda.max_memory_allocated() / (1024.0 ** 2))
                        if torch is not None and str(cfg.device).startswith("cuda") and torch.cuda.is_available()
                        else np.nan
                    )
                    _, peak_bytes = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    rss_after = _process_rss_mb()
                    peak_python_memory_mb = float(peak_bytes / (1024.0 ** 2))
                    rss_delta_mb = max(0.0, rss_after - rss_before) if np.isfinite(rss_before) and np.isfinite(rss_after) else np.nan

                    score_by_split = {}
                    local_by_split = {}
                    target_by_split = {}
                    test_inference_time_sec = np.nan
                    for split_name in ["validation", "threshold_calibration", "test"]:
                        seq_x, seq_y, local_pos = sequence_data[split_name]
                        if torch is not None and str(cfg.device).startswith("cuda") and torch.cuda.is_available():
                            torch.cuda.synchronize()
                        pred_t0 = time.perf_counter()
                        score_by_split[split_name] = predict_neural(neural_model, seq_x, cfg)
                        if torch is not None and str(cfg.device).startswith("cuda") and torch.cuda.is_available():
                            torch.cuda.synchronize()
                        pred_elapsed = time.perf_counter() - pred_t0
                        if split_name == "test":
                            test_inference_time_sec = pred_elapsed
                        local_by_split[split_name] = local_pos
                        target_by_split[split_name] = seq_y
                    if cfg.enable_efficiency_analysis:
                        complexity_rows.append(efficiency_record(
                            model_name=model_name,
                            feature_setting=feature_setting,
                            horizon=horizon,
                            kind="neural",
                            model=neural_model,
                            feature_count=train_x.shape[-1],
                            train_samples=len(train_y),
                            test_samples=len(target_by_split["test"]),
                            train_input_memory_mb=train_input_memory_mb,
                            train_time_sec=train_time_sec,
                            inference_time_sec=float(test_inference_time_sec),
                            peak_python_memory_mb=peak_python_memory_mb,
                            rss_delta_mb=rss_delta_mb,
                            sequence_length=cfg.sequence_length,
                            peak_host_memory_mb=peak_host_memory_mb,
                            peak_gpu_memory_mb=peak_gpu_memory_mb,
                        ))

                # Primary threshold: operational utility under the alarm-time budget.
                cal_indices = splits["threshold_calibration"]
                threshold, calibration_event_metrics = choose_utility_threshold(
                    score_by_split["threshold_calibration"],
                    local_by_split["threshold_calibration"],
                    y_original[cal_indices],
                    times.iloc[cal_indices].reset_index(drop=True),
                    horizon,
                    cfg,
                )

                if cfg.enable_threshold_ablation:
                    threshold_ablation_records.extend(threshold_policy_ablation_rows(
                        model_name=model_name,
                        feature_setting=feature_setting,
                        horizon=horizon,
                        score_by_split=score_by_split,
                        local_by_split=local_by_split,
                        target_by_split=target_by_split,
                        y_original=y_original,
                        times=times,
                        splits=splits,
                        cfg=cfg,
                    ))

                test_indices = splits["test"]
                test_alarm_full = np.zeros(len(test_indices), dtype=bool)
                test_positions = local_by_split["test"]
                test_scores = score_by_split["test"]
                test_alarm_full[test_positions] = test_scores >= threshold
                test_event = event_metrics(
                    y_original[test_indices],
                    test_alarm_full,
                    times.iloc[test_indices].reset_index(drop=True),
                    horizon,
                    cfg.event_merge_gap,
                )


                obs = observation_metrics(target_by_split["test"], test_scores, threshold)

                result_rows.append({
                    "Seed": int(cfg.random_seed),
                    "Model": model_name,
                    "FeatureSetting": feature_setting,
                    "ModelSetting": label,
                    "Horizon": horizon,
                    "ThresholdMethod": "UtilityMaximization_A<=10pct",
                    "Threshold": threshold,
                    **obs,
                    "CalibrationEventHitRate": calibration_event_metrics["EventHitRate"],
                    "CalibrationFalseAlarmEpisodesPerDay": calibration_event_metrics["FalseAlarmEpisodesPerDay"],
                    "CalibrationTimeUnderAlarm": calibration_event_metrics["TimeUnderAlarm"],
                    "CalibrationOperationalUtility": operational_utility(calibration_event_metrics, horizon, cfg),
                    "TestEvents": test_event["Events"],
                    "TestEventsDetected": test_event["EventsDetected"],
                    "TestEventHitRate": test_event["EventHitRate"],
                    "TestMeanLeadObservations": test_event["MeanLeadObservations"],
                    "TestFalseAlarmEpisodesPerDay": test_event["FalseAlarmEpisodesPerDay"],
                    "TestTimeUnderAlarm": test_event["TimeUnderAlarm"],
                    "TestOperationalUtility": operational_utility(test_event, horizon, cfg),
                })

                for local_pos, target, probability in zip(
                    test_positions,
                    target_by_split["test"],
                    test_scores,
                ):
                    prediction_rows.append({
                        "Seed": int(cfg.random_seed),
                        "Model": model_name,
                        "FeatureSetting": feature_setting,
                        "ModelSetting": label,
                        "Horizon": horizon,
                        "TestLocalPosition": int(local_pos),
                        "Target": int(target),
                        "Probability": float(probability),
                        "Threshold": float(threshold),
                        "Alarm": int(probability >= threshold),
                    })

                print(
                    f"Completed {label:24s} h={horizon:2d} | "
                    f"AUPRC={obs['AUPRC']:.4f} | Hit={test_event['EventHitRate']:.3f} | "
                    f"FA/day={test_event['FalseAlarmEpisodesPerDay']:.2f} | "
                    f"Alarm={100*test_event['TimeUnderAlarm']:.2f}%"
                )

    results = pd.DataFrame(result_rows)
    predictions = pd.DataFrame(prediction_rows)
    results.to_csv(cfg.output_dir / "model_ablation_metrics.csv", index=False)
    predictions.to_csv(cfg.output_dir / "test_predictions.csv", index=False)

    if cfg.enable_threshold_ablation and threshold_ablation_records:
        threshold_ablation = pd.DataFrame(threshold_ablation_records)
        threshold_ablation.insert(0, "Seed", int(cfg.random_seed))
        threshold_ablation.to_csv(cfg.output_dir / "threshold_policy_ablation.csv", index=False)

    if cfg.enable_efficiency_analysis and complexity_rows:
        efficiency = pd.DataFrame(complexity_rows)
        efficiency.insert(0, "Seed", int(cfg.random_seed))
        efficiency.to_csv(cfg.output_dir / "complexity_efficiency_metrics.csv", index=False)

    print(f"\nResults saved to: {cfg.output_dir}")
    return results, predictions


def run_repeated_seed_experiments(base_cfg: Config, seeds: tuple[int, ...]) -> None:
    """Run the complete experiment independently for each seed (output_dir/seed_<s>/)."""
    seeds = tuple(dict.fromkeys(int(s) for s in seeds)) or (int(base_cfg.random_seed),)
    root = Path(base_cfg.output_dir)
    for i, seed in enumerate(seeds, start=1):
        seed_dir = root / f"seed_{seed}"
        print(f"\n{'='*78}\nRepeated-seed run {i}/{len(seeds)}: seed={seed}\nOutput: {seed_dir}\n{'='*78}")
        run_experiment(replace(base_cfg, random_seed=int(seed), output_dir=seed_dir))
    print(f"\nAll seeds finished. Per-seed results are in: {root}/seed_*/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ProcessMiner raw / dynamics-only / dynamics-augmented comparison (results tables only)."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("processminer_uncertainty_run"))
    parser.add_argument("--horizons", type=parse_csv_ints, default=(1, 3, 5, 10))
    parser.add_argument("--feature-settings", type=parse_csv_strings,
                        default=("raw", "dynamics_only", "dynamics_augmented"),
                        help="Comma-separated subset of raw,dynamics_only,dynamics_augmented.")
    parser.add_argument("--models", type=parse_csv_strings, default=("HGB", "XGB", "LSTM", "TCN"))
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--vae-components", type=int, default=6)
    parser.add_argument("--vae-hidden-size", type=int, default=32)
    parser.add_argument("--vae-epochs", type=int, default=50)
    parser.add_argument("--vae-learning-rate", type=float, default=1e-3)
    parser.add_argument("--vae-beta", type=float, default=1e-3,
                        help="KL-divergence weight used by the compact tabular VAE.")
    parser.add_argument("--ftle-window", type=int, default=120)
    parser.add_argument("--ftle-horizon", type=int, default=5)
    parser.add_argument("--ftle-stride", type=int, default=5)
    parser.add_argument("--ftle-ridge", type=float, default=1e-3)
    parser.add_argument("--threshold-candidates", type=int, default=300)
    parser.add_argument("--utility-lead-weight", type=float, default=0.05)
    parser.add_argument("--utility-false-alarm-weight", type=float, default=0.02)
    parser.add_argument("--utility-alarm-time-weight", type=float, default=1.00)
    parser.add_argument("--max-time-under-alarm", type=float, default=0.10,
                        help="Maximum calibration time under alarm. Default 0.10 = 10%%.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Single-seed value used when --seeds is not supplied. Default: 42.")
    parser.add_argument("--seeds", type=parse_csv_ints, default=None,
                        help="Comma-separated repeated experiment seeds, e.g. 42,52,62,72,82.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fast", action="store_true",
                        help="Fast smoke-test mode: fewer epochs and reduced model sizes.")
    parser.add_argument("--skip-efficiency-analysis", action="store_true",
                        help="Do not write complexity_efficiency_metrics.csv.")
    parser.add_argument("--skip-threshold-ablation", action="store_true",
                        help="Do not write threshold_policy_ablation.csv.")
    parser.add_argument("--exclude-eventpress", action="store_true",
                        help="Exclude EventPress from the named ProcessMiner CSV (sensitivity mode).")
    parser.add_argument("--named-data-window", choices=("legacy", "full"), default="legacy",
                        help="Named CSV only: 'legacy' = 18,398-row benchmark window (default); "
                             "'full' = extended 20,458-row period.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        data_path=args.data,
        output_dir=args.output,
        horizons=args.horizons,
        feature_settings=args.feature_settings,
        models=args.models,
        sequence_length=args.sequence_length,
        neural_epochs=args.epochs,
        neural_batch_size=args.batch_size,
        neural_patience=args.patience,
        neural_learning_rate=args.learning_rate,
        neural_hidden_size=args.hidden_size,
        neural_dropout=args.dropout,
        vae_components=args.vae_components,
        vae_hidden_size=args.vae_hidden_size,
        vae_epochs=args.vae_epochs,
        vae_learning_rate=args.vae_learning_rate,
        vae_beta=args.vae_beta,
        ftle_window=args.ftle_window,
        ftle_horizon=args.ftle_horizon,
        ftle_stride=args.ftle_stride,
        ftle_ridge=args.ftle_ridge,
        threshold_candidates=args.threshold_candidates,
        utility_lead_weight=args.utility_lead_weight,
        utility_false_alarm_weight=args.utility_false_alarm_weight,
        utility_alarm_time_weight=args.utility_alarm_time_weight,
        max_time_under_alarm=args.max_time_under_alarm,
        random_seed=args.seed,
        device=args.device,
        fast=args.fast,
        enable_efficiency_analysis=not args.skip_efficiency_analysis,
        enable_threshold_ablation=not args.skip_threshold_ablation,
        include_eventpress=not args.exclude_eventpress,
        named_data_window=args.named_data_window,
    )
    seeds = tuple(int(v) for v in args.seeds) if args.seeds else (int(args.seed),)
    if len(seeds) > 1:
        run_repeated_seed_experiments(cfg, seeds)
    else:
        run_experiment(replace(cfg, random_seed=int(seeds[0])))


if __name__ == "__main__":
    main()
