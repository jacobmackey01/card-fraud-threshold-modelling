# Credit Card Fraud: Thresholding to Production Inference

This project follows one fraud model from business-aware threshold selection into a deployable inference service. It trains imbalance-aware candidates, selects a review threshold from explicit cost or alert-capacity assumptions, packages the chosen model as an immutable release, and serves versioned predictions through a tested API.

```text
train candidate -> validate threshold -> package immutable release -> promote pointer
                                                        |
                                                        v
                                         FastAPI -> metrics -> drift check
```

The output is `review_recommended`, not a declaration that a transaction is fraudulent. The data has anonymised PCA features, the cost assumptions are illustrative, and the model has not been approved for use in a real payment system.

## What this project demonstrates

- Chronological model evaluation and threshold selection tied to explicit review-cost and capacity assumptions.
- A stable preprocessing/inference contract packaged as an immutable, integrity-checked model release.
- Strictly validated single and batch scoring through FastAPI.
- Container, CI, health checks, structured logs, and scrapeable operational metrics.
- Offline input-drift checks and guarded model promotion or rollback without automatic retraining.
- A single-artifact synthetic deployment rehearsal: tested container, Artifact Registry digest, immutable Cloud Run revision, live provenance checks, and approval-gated application rollback.

## Dataset

The project uses the Kaggle/ULB credit-card fraud dataset mirrored by TensorFlow for its imbalanced-data tutorial:

`https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`

Expected schema:

- `Time`
- `V1` through `V28`
- `Amount`
- `Class`, where `1` is fraud

The dataset is extremely imbalanced: the commonly cited version has 492 frauds out of 284,807 transactions.

For this workspace's execution status, see `RUN_STATUS.md`.

## Train the candidate model

From this folder:

```powershell
python src/fraud_pipeline.py
```

If network access is blocked, place `creditcard.csv` at `data/raw/creditcard.csv`, then run:

```powershell
python src/fraud_pipeline.py --no-download
```

For a local code sanity check that does not use the real dataset:

```powershell
python src/smoke_test.py
```

## Optional ML Packages

The checked-in pipeline is runnable with NumPy and Pandas only, then automatically adds the boosted-tree challenger when scikit-learn is available. On a normal machine, install the usual ML stack with:

```powershell
python -m pip install -r requirements-full.txt
```

When scikit-learn is installed, the pipeline automatically adds a `HistGradientBoostingClassifier` challenger. The logistic models remain as interpretable baselines.

## Run the inference API

Install the API dependencies and start the service from the repository root:

```powershell
python -m pip install -r requirements-api.txt
python -m uvicorn service.app:app --host 127.0.0.1 --port 8080
```

The committed production pointer resolves to model release `1.0.0`. At startup, the service verifies the release manifest and model SHA-256 before accepting traffic. A changed artifact, a changed manifest, an unsupported feature order, or an invalid threshold prevents startup.

Send the explicitly synthetic request example:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8080/v1/predictions `
  -ContentType application/json `
  -InFile examples/synthetic_transaction.json
```

Useful endpoints:

- `GET /health/live`: process liveness.
- `GET /health/ready`: loaded model version and artifact digest.
- `GET /v1/model`: input contract, threshold, bounded evaluation metadata, and the decision boundary.
- `POST /v1/predictions`: one score and review recommendation.
- `POST /v1/predictions/batch`: up to 1,000 transactions.
- `GET /metrics`: process-lifetime prediction, error, score, and alert-rate metrics in Prometheus text format.

The service accepts `Time`, `Amount`, and `V1` through `V28`. It rejects missing fields, extra fields, negative time or amount, and non-finite values. Request logs contain the model version, score, decision, latency, and an internal request ID; they do not log the input feature vector or caller-supplied transaction ID.

## Rehearse the deployment lifecycle

The public Cloud Run configuration is intentionally narrower than the local inference API. It sets `SYNTHETIC_DEMO_ONLY=true`, removes the arbitrary-vector prediction and batch routes, and exposes one bodyless route:

```text
POST /v1/demo-prediction
```

That route loads the versioned, checked-in `examples/synthetic_demo_v1.json` fixture inside the container. It returns the fixture ID, model release, threshold, score and review recommendation. It never accepts transaction features from a caller.

The approval-protected deployment workflow tests the repository, builds and publishes one container to Google Artifact Registry, then deploys that exact `@sha256:...` digest to an immutable Cloud Run revision. It verifies `/health/ready`, `/v1/model`, `/v1/demo-prediction` and `/metrics`, recording the Git commit, container digest, model-file digest, model release, threshold, fixture version and verification time as a GitHub Actions artifact.

Application rollback is a separate approval-protected workflow. It requires an explicit target revision and an expected current revision, moves traffic only when that compare-and-swap guard passes, and reruns the same live verification. It does not create another model release. Live rollback evidence remains pending until a genuine later application revision exists.

See `deployment/README.md` for the bounded-cost Cloud Run configuration and one-time OIDC/WIF setup.

## Build the container

```powershell
docker build -t fraud-review-api:1.0.0 .
docker run --rm -p 8080:8080 fraud-review-api:1.0.0
```

The image runs as a non-root user and includes only the service code and promoted model artifacts. GitHub Actions runs the unit/API tests, the synthetic training smoke test, release verification, and a container build.

## Version and promote a model

Training writes a candidate to `artifacts/fraud_model.json`; it never changes the production pointer. Release packaging and promotion are separate commands:

```powershell
python -m scripts.package_model --version 1.0.1
python -m unittest discover -s tests -v
python -m scripts.promote_model --version 1.0.1 --expected-current 1.0.0
python -m scripts.promote_model --version 1.0.1 --check
```

Release directories are immutable. Promotion checks that the production version is still the version the operator reviewed, then atomically replaces the small pointer file. Rollback uses the same promotion command with a previously validated release and the current version supplied through `--expected-current`.

## Check unlabelled production inputs

The offline monitor scores a CSV with the same raw feature contract, compares feature means and scales with the training reference, and checks alert rate against the held-out test reference plus a sampling band:

```powershell
python -m service.monitor path/to/recent_transactions.csv --output monitoring/latest.json
```

This is an early-warning check, not performance measurement. Unlabelled data cannot establish recall, missed-fraud losses, or calibration. Those require delayed confirmed outcomes, as described in `MONITORING.md`.

## What The Pipeline Does

1. Downloads or loads the public transaction CSV.
2. Validates the expected fraud schema.
3. Engineers lightweight transaction features:
   - `log1p(Amount)`
   - zero-amount flag
   - hour-of-day sine/cosine
4. Uses a train/validation/test split. The default is chronological, which better resembles production monitoring than a random split.
5. Trains two logistic models:
   - unweighted baseline
   - class-weighted model for the extreme imbalance
6. Adds a boosted-tree challenger when scikit-learn is available.
7. Selects the fraud alert threshold on validation data using a cost policy by default:
   - false positive cost: analyst review / customer friction
   - false negative cost: transaction `Amount` plus an illustrative handling cost
   - optional capacity policy: cap alerts to the review budget
8. Reports test precision, recall, F1, F2, PR-AUC / average precision, ROC-AUC, false positive rate, flagged rate, and expected cost.
9. Writes interpretation outputs:
   - global feature drivers
   - local linear log-odds contributions for high-risk transactions

## Why Accuracy Is Not The Main Metric

Fraud is rare enough that an "always legitimate" classifier can look excellent by accuracy while catching no fraud. This project reports accuracy only as a cautionary baseline and uses precision, recall, F-scores, PR-AUC, and false positive rate for model selection.

## Verified operating point

[![Validation precision-recall curve with the selected fraud-alert threshold and its resulting precision, recall and review-queue composition on the untouched chronological test set](reports/fraud_operating_point.svg)](reports/model_report.md)

*The threshold was selected on chronological validation data and applied once to the untouched chronological test set: 75 frauds across 56,962 transactions. Costs remain illustrative assumptions. No real fraud-review capacity was supplied. Final production threshold ownership belongs jointly to fraud operations, risk/finance, product and data science. V1–V28 are anonymised PCA components, not customer-facing reason codes; this is a portfolio case study, not a deployed bank fraud system.*

## Outputs

After a real run, inspect:

- `reports/model_report.md`
- `reports/metrics_summary.csv`
- `reports/fraud_operating_point.svg`
- `reports/precision_recall_curve.svg`
- `reports/feature_drivers.svg`
- `artifacts/fraud_model.json`
- `interpretation/global_feature_drivers.csv`
- `interpretation/linear_log_odds_contributions_top_flags.csv`
- `interpretation/high_value_false_positive_examples.csv`
- `MONITORING.md`

The operating-point figure is regenerated from the committed report artifacts with:

```powershell
python src/generate_operating_point_figure.py
```

## Interpretation Note

The `V1`-`V28` fields are PCA-anonymized, so they are valid statistical drivers but not direct business concepts. The local contribution file is therefore not a customer-facing reason-code system. In a production bank setting, the same structure would be enriched with explainable fields such as merchant category, merchant country, cardholder velocity, account age, device reputation, and historical chargeback patterns.

## Repository Note

The raw CSV is included in the delivered zip for convenience, but `data/raw/` is ignored by Git because `creditcard.csv` is larger than GitHub's normal file-size limit. A portfolio repo should rely on the download command or local placement instructions rather than committing the raw dataset.
