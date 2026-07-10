# Fraud Detection System

ML project: detect fraudulent transactions (IEEE-CIS dataset) and serve predictions via an API.

## Status
- [x] Week 1: Scaffold, data loading, EDA, evaluation harness
- [x] Week 2: Modeling (baseline + LightGBM)
- [x] Week 3: FastAPI service
- [x] Week 4: Docker
- [ ] Week 5: Cloud deploy
- [ ] Week 6: Polish + writeup

## Setup
1. Create a virtual environment and install `requirements.txt`.
2. Download IEEE-CIS data from Kaggle into `data/raw/`.
3. Run `notebooks/01_eda.ipynb`.

## Key decisions

## Week 1 findings (EDA)
- **Dataset:** IEEE-CIS, merged transaction + identity. Shape: 590,540 rows × 434 columns.
- **Class imbalance:** Fraud is 3.50% of transactions (20,663 fraud vs 569,877 legit).
  Accuracy is meaningless here — a "never fraud" model scores 96.5%. **Primary metric: PR-AUC**,
  reported alongside precision, recall, and F1.
- **Missingness:** 12 columns >90% missing, 214 columns >50% missing. Worst offenders are
  `id_*` identity columns (~99% missing, expected — identity data is sparse) and several `D*`
  timedelta columns. No imputation planned: LightGBM handles NaNs natively, and missingness
  itself may be predictive.
- **Transaction amount:** Fraud slightly higher (median 75 vs 68.5) but distributions overlap
  heavily; weak standalone signal. Very large transactions (>5K) are almost entirely legitimate.
- **Time span:** TransactionDT covers 182 days (~6 months). This justifies a **time-based
  train/test split** (train on earlier, test on later) rather than a random split, to respect
  the temporal, adversarial nature of fraud and avoid leakage.

## Modeling plan (Week 2)
- Time-based split (~5 months train / ~1 month test).
- Baseline: logistic regression. Then LightGBM with `scale_pos_weight ≈ 27`.
- Compare class-weighting vs SMOTE (apply resampling to training fold only).
- Select operating threshold from the precision/recall tradeoff, not the default 0.5.

## Week 2 findings (Modeling)

### Train/test split
- **Time-based split** on `TransactionDT` (earliest 80% train / latest 20% test), not random.
  Respects the temporal, adversarial nature of fraud and prevents leakage from future to past.
- Train: 472,432 rows (fraud rate 3.51%). Test: 118,108 rows (fraud rate 3.44%).
- Time ranges do not overlap — confirmed clean split.

### Models compared (primary metric: PR-AUC)
| Model | PR-AUC | Notes |
|---|---|---|
| Logistic Regression (baseline) | 0.186 | Full pipeline: impute + scale + one-hot. Weak — linear, can't model feature interactions. |
| **LightGBM (`scale_pos_weight`)** | **0.548** | **Selected model.** Native NaN + categorical handling, no scaling/imputation needed. |
| LightGBM + SMOTE | 0.538 | Approximate comparison. More complex, no gain. |

- LightGBM improved PR-AUC ~3x over the linear baseline, confirming fraud signal lives in
  feature interactions that trees capture and linear models cannot.
- **SMOTE vs class-weighting:** class-weighting achieved marginally higher PR-AUC (0.548 vs
  0.538) with far less complexity and no synthetic data. Selected as final approach. The two
  differed mainly in threshold calibration, not underlying quality.

### Operating threshold
- Selected **0.50**, prioritizing recall (catching fraud) over precision.
- Rationale: in fraud, a missed fraud (false negative) is typically a direct financial loss,
  while a false alarm (false positive) is recoverable customer friction.
- At 0.50: recall 0.657, precision 0.330 — catches ~66% of fraud, accepting more false alarms.
- Production note: threshold should be tuned to the business's actual false-negative vs
  false-positive cost ratio.

### Saved artifact
- `models/fraud_model.joblib` — bundles model, feature order, categorical column list, and
  threshold, to guarantee train/serve preprocessing consistency for the Week 3 API.

## Modeling decisions log
- No imputation/scaling for LightGBM — handled natively; only needed for the LR baseline.
- Dropped `TransactionID` (identifier) and `TransactionDT` (raw time offset) as features.
- Categoricals converted to pandas `category` dtype for LightGBM's native handling.

## Week 3 findings (API service)

Built a FastAPI service (`src/api.py`) that serves the trained LightGBM model over HTTP.

### Endpoints
- `GET /health` — liveness check; returns model feature count and threshold.
- `POST /predict` — scores a single transaction; returns fraud probability, decision, and
  feature-count.
- `POST /predict_batch` — scores up to 1000 transactions in one request.

### Design: tolerant hybrid
- Accepts any subset of the model's 431 features; omitted features are treated as missing (NaN),
  which LightGBM handles natively.
- Enables both realistic full-record scoring and lightweight demo requests.
- **Known limitation:** sparse inputs (few features) produce inflated, low-confidence scores
  because the model runs on mostly-missing data. Full records give reliable predictions.

### Preprocessing consistency (validated)
- The API reconstructs each request into a DataFrame matching the model's exact training format:
  same column order, categorical dtype restored, numerics coerced.
- Verified against the notebook model on real test rows — API and notebook probabilities matched
  to 4 decimal places on all sampled transactions. This confirms no train/serve skew.

### Hardening
- Model loaded once at startup with clear failure on missing/corrupt artifact.
- Graceful HTTP errors (422 empty input, 413 oversized batch, 500 prediction failure) instead
  of raw tracebacks.
- Batch size capped at 1000 to protect against memory exhaustion.

### Run locally
- uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
- Interactive docs at `http://127.0.0.1:8000/docs`.

## Week 4 findings (Containerization)

Packaged the API into a Docker image for portable, reproducible deployment.

### Files
- `Dockerfile` — builds from `python:3.12-slim`, installs `libgomp1` (LightGBM runtime
  dependency), installs Python requirements, copies `src/` and `models/`, runs uvicorn
  bound to `0.0.0.0:8000`.
- `.dockerignore` — excludes `.venv/`, `data/`, notebooks, and caches to keep the image lean.

### Key decisions & gotchas resolved
- **`libgomp1` system library:** `python:3.12-slim` omits it, but LightGBM needs it at runtime.
  Installed via `apt-get` in the Dockerfile. (Build succeeded without it; failure only surfaced
  at container startup — a runtime, not build-time, dependency.)
- **`0.0.0.0` bind + `-p 8000:8000` port mapping:** required for the API to be reachable from
  outside the container. Binding to `127.0.0.1` would make it unreachable.
- **Excluded `data/` but kept `models/`:** the container needs the trained model, not the
  training data.

### Image
- Size: ~1.37 GB on disk (~312 MB compressed). Within cloud free-tier limits.

### Validation
- Containerized API validated against the notebook model on real test rows — predictions matched
  to 4 decimal places on all sampled transactions, identical to the local (Week 3) results.
  Confirms the container reproduces the local environment with no drift.

### Run
```
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```