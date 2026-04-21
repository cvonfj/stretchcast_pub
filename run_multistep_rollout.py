from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np

from stretchcast_runtime import predict_base, predict_step4
from stretchcast_runtime._io import load_input_payload, save_prediction_payload

_TIME_FMT = "%Y-%m-%dT%H:%M:%S"


def _to_str_array(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values)
    out = np.empty(arr.shape, dtype="U32")
    flat_in = arr.reshape(-1)
    flat_out = out.reshape(-1)
    for i, value in enumerate(flat_in.tolist()):
        if isinstance(value, (bytes, bytearray)):
            flat_out[i] = value.decode("utf-8")
        else:
            flat_out[i] = str(value)
    return out


def _parse_last_time(input_times_utc: np.ndarray | None) -> datetime:
    times = _to_str_array(input_times_utc)
    if times is None or times.size == 0:
        raise ValueError("input_times_utc is required for multistep rollout")
    return datetime.strptime(str(times.reshape(-1)[-1]), _TIME_FMT)


def _validate_positive_hours(forecast_hours: int) -> None:
    if int(forecast_hours) <= 0:
        raise ValueError("forecast_hours must be a positive integer")


def _validate_base_hours(forecast_hours: int) -> None:
    _validate_positive_hours(forecast_hours)
    if int(forecast_hours) % 6 != 0:
        raise ValueError("BASE forecast_hours must be a multiple of 6 hours")


def _validate_step4_hours(forecast_hours: int) -> None:
    _validate_positive_hours(forecast_hours)
    if int(forecast_hours) % 24 != 0:
        raise ValueError("4STEP forecast_hours must be a multiple of 24 hours")


def _validate_base_input(input_hist: np.ndarray) -> None:
    if input_hist.ndim != 5:
        raise ValueError(f"input_hist must be 5D [B,T,C,H,W], got shape {input_hist.shape}")
    if input_hist.shape[1] != 1:
        raise ValueError(
            f"BASE rollout requires exactly 1 input time, got time dimension {input_hist.shape[1]}"
        )


def _validate_step4_input(input_hist: np.ndarray) -> None:
    if input_hist.ndim != 5:
        raise ValueError(f"input_hist must be 5D [B,T,C,H,W], got shape {input_hist.shape}")
    if input_hist.shape[1] != 2:
        raise ValueError(
            f"4STEP rollout requires exactly 2 input times, got time dimension {input_hist.shape[1]}"
        )


def rollout_base_autoreg(
    *,
    input_hist: np.ndarray,
    input_times_utc: np.ndarray | None,
    forecast_hours: int,
    device: str = "cpu",
    predict_fn: Callable[[np.ndarray, str], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Autoregressive rollout using the BASE model (6h per call)."""

    _validate_base_hours(forecast_hours)
    _validate_base_input(input_hist)

    steps_needed = int(forecast_hours) // 6
    predict = predict_fn or predict_base

    current_hist = np.asarray(input_hist, dtype=np.float32)
    current_time = _parse_last_time(input_times_utc)

    outputs: list[np.ndarray] = []
    output_times: list[str] = []

    for _ in range(steps_needed):
        chunk = np.asarray(predict(current_hist, device=device), dtype=np.float32)
        if chunk.ndim != 5 or chunk.shape[1] != 1:
            raise ValueError(f"predict_base output must be shape [B,1,C,H,W], got shape {chunk.shape}")

        outputs.append(chunk)
        current_time = current_time + timedelta(hours=6)
        output_times.append(current_time.strftime(_TIME_FMT))
        current_hist = chunk

    pred = np.concatenate(outputs, axis=1)
    return pred, np.asarray(output_times, dtype="U19")


def rollout_step4_autoreg(
    *,
    input_hist: np.ndarray,
    input_times_utc: np.ndarray | None,
    forecast_hours: int,
    device: str = "cpu",
    predict_fn: Callable[[np.ndarray, str], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Autoregressive rollout using the 4STEP model.

    The model produces 4 future 6-hour steps each call. This function repeats calls
    until forecast_hours is covered, truncating the final chunk if needed.
    """

    _validate_step4_hours(forecast_hours)
    _validate_step4_input(input_hist)

    steps_needed = int(forecast_hours) // 6
    predict = predict_fn or predict_step4

    current_hist = np.asarray(input_hist, dtype=np.float32)
    current_time = _parse_last_time(input_times_utc)

    outputs: list[np.ndarray] = []
    output_times: list[str] = []

    produced_steps = 0
    while produced_steps < steps_needed:
        chunk = np.asarray(predict(current_hist, device=device), dtype=np.float32)
        if chunk.ndim != 5 or chunk.shape[1] < 1:
            raise ValueError(f"predict_step4 output must be 5D with time>0, got shape {chunk.shape}")

        take = min(4, steps_needed - produced_steps)
        outputs.append(chunk[:, :take, ...])

        for _ in range(take):
            current_time = current_time + timedelta(hours=6)
            output_times.append(current_time.strftime(_TIME_FMT))

        produced_steps += take
        current_hist = np.asarray(chunk[:, -2:, ...], dtype=np.float32)

    pred = np.concatenate(outputs, axis=1)
    return pred, np.asarray(output_times, dtype="U19")


def run_rollout(
    *,
    model: str,
    input_path: Path,
    output_path: Path,
    forecast_hours: int,
    device: str,
) -> Path:
    input_hist, meta = load_input_payload(input_path)
    if model == "base":
        pred, output_times = rollout_base_autoreg(
            input_hist=input_hist,
            input_times_utc=meta.get("input_times_utc"),
            forecast_hours=forecast_hours,
            device=device,
        )
        model_name = "scs0875_BASE"
    elif model == "4step":
        pred, output_times = rollout_step4_autoreg(
            input_hist=input_hist,
            input_times_utc=meta.get("input_times_utc"),
            forecast_hours=forecast_hours,
            device=device,
        )
        model_name = "scs0875_4STEP"
    else:
        raise ValueError(f"unsupported model: {model}")

    save_prediction_payload(
        output_path,
        pred_phys=pred,
        model_name=model_name,
        device=device,
        meta={
            "lat": meta.get("lat"),
            "lon": meta.get("lon"),
            "var_names": meta.get("var_names"),
            "output_times_utc": output_times,
        },
    )
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run autoregressive multistep rollout with StretchCast BASE/4STEP model.")
    parser.add_argument("--model", choices=("base", "4step"), default="4step", help="Model used for autoregressive rollout")
    parser.add_argument("--input", required=True, help="Path to model input .nc (BASE uses 1 time, 4STEP uses 2 times)")
    parser.add_argument("--output", required=True, help="Path to output prediction .nc")
    parser.add_argument(
        "--hours",
        type=int,
        required=True,
        help="Forecast horizon in hours (base: multiple of 6, 4step: multiple of 24)",
    )
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    run_rollout(
        model=str(args.model),
        input_path=Path(args.input).resolve(),
        output_path=Path(args.output).resolve(),
        forecast_hours=int(args.hours),
        device=str(args.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
