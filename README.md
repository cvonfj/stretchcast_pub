# StretchCast (Wheel Runtime Bundle)
The open inference libaray for "StretchCast: Global-Regional AI Weather Forecasting on Stretched Cubed-Sphere Mesh" [https://arxiv.org/abs/2603.27288]

This folder contains the wheel-based inference runtime and scripts for case `case_origin_0635_20220608T180000`.
One can also get the Baidu Disk version with all codes and inference data at https://pan.baidu.com/s/19Ls7W6EyKXrCwO3vvoLiXg?pwd=ide7

## Quick Start (5 minutes)

### 1) Install

CPU:

```bash
python -m pip install -r requirements-cpu.txt
python -m pip install ./stretchcast_runtime-0.1.0-cp312-cp312-linux_x86_64.whl --no-deps
```

GPU:

```bash
python -m pip install -r requirements-gpu.txt
python -m pip install ./stretchcast_runtime-0.1.0-cp312-cp312-linux_x86_64.whl --no-deps
```

### 2) Run 24h rollout for both models

```bash
python run_multistep_rollout.py \
  --model base \
  --input case_origin_0635_20220608T180000/scs0875_BASE_reference_io.nc \
  --output case_origin_0635_20220608T180000/scs0875_BASE_rollout_24h.nc \
  --hours 24 \
  --device cpu

python run_multistep_rollout.py \
  --model 4step \
  --input case_origin_0635_20220608T180000/scs0875_4STEP_reference_io.nc \
  --output case_origin_0635_20220608T180000/scs0875_4STEP_rollout_24h.nc \
  --hours 24 \
  --device cpu
```

### 3) Re-draw prediction figures from outputs

Open and run notebook:

- `case635_plots.ipynb` (or `reviewer_case635_plots.ipynb` if you keep the old filename locally)

Recommended run order in notebook:

1. Cell 2 (config)
2. Cell 3 (inference)
3. Cell 4 (BASE prediction re-draw: +6h/+12h/+24h)
4. Cell 5 (4STEP prediction re-draw: +6h/+12h/+24h)

Notes:

- The plotting cells now only visualize wheel output predictions.
- The notebook no longer imports helper modules from `notebooks/`.

Generated figures:

- `case_origin_0635_20220608T180000/pred_vs_gt_wheel_case635_base.png` (filename kept for compatibility; content is prediction-only)
- `case_origin_0635_20220608T180000/pred_vs_gt_wheel_case635_4step.png` (filename kept for compatibility; content is prediction-only)

## Folder Contents

- `stretchcast_runtime-0.1.0-cp312-cp312-linux_x86_64.whl`
- `requirements-cpu.txt`
- `requirements-gpu.txt`
- `run_multistep_rollout.py`
- `reviewer_case635_plots.ipynb`
- `case_origin_0635_20220608T180000/scs0875_BASE_reference_io.nc`
- `case_origin_0635_20220608T180000/scs0875_4STEP_reference_io.nc`

May also include generated outputs (after running rollout/notebook):

- `case_origin_0635_20220608T180000/scs0875_BASE_rollout_24h.nc`
- `case_origin_0635_20220608T180000/scs0875_4STEP_rollout_24h.nc`
- `case_origin_0635_20220608T180000/pred_vs_gt_wheel_case635_base.png`
- `case_origin_0635_20220608T180000/pred_vs_gt_wheel_case635_4step.png`

## Input / Output Data Format

- `scs0875_XXX_reference_io.nc`: input data + input metadata.
- Rollout output `.nc`: variable `pred_phys` and coordinate `output_times_utc`.
- Output `.nc` also includes geolocation and variable metadata (`lat`, `lon`, `var_names`).

## CLI Inference Example (single-shot, optional)

```bash
stretchcast-runtime predict \
  --model base \
  --input case_origin_0635_20220608T180000/scs0875_BASE_reference_io.nc \
  --output /tmp/base_pred.nc \
  --device cpu

stretchcast-runtime predict \
  --model 4step \
  --input case_origin_0635_20220608T180000/scs0875_4STEP_reference_io.nc \
  --output /tmp/step4_pred.nc \
  --device cpu
```

## Verify Installation

```bash
stretchcast-runtime verify --case case_origin_0635_20220608T180000 --device cpu
```

The command prints JSON metrics: `max_abs`, `mean_abs`, `rmse`, `max_rel`.

## Constraints and Notes

- OS target: Linux x86_64.
- Wheel ABI target: CPython 3.12.
- `--hours` must be positive.
- `--model base`: `--hours` must be a multiple of 6.
- `--model 4step`: `--hours` must be a multiple of 24.
- `--model base` input must contain 1 history step.
- `--model 4step` input must contain 2 history steps.

## Offline Installation

If reviewers are offline, provide dependency wheels under `deps/`:

```bash
python -m pip install --no-index --find-links ./deps -r requirements-cpu.txt
python -m pip install --no-index ./stretchcast_runtime-0.1.0-cp312-cp312-linux_x86_64.whl --no-deps
```

For GPU, replace `requirements-cpu.txt` with `requirements-gpu.txt`.
