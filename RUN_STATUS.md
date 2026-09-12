# Run Status

What has been completed for this repository:

- Built the end-to-end fraud detection project.
- An earlier verified run used the full public credit-card fraud CSV and produced the committed model, reports, and test metrics. The raw CSV is intentionally absent from the Git checkout.
- Added cost/capacity thresholding, monitoring notes, leakage cleanup, and a boosted-tree challenger.
- Ran the real dataset with amount-weighted cost thresholding. The validation costs are a near-tie, so the operating model is `unweighted_logistic`; test precision `0.8261`, test recall `0.7600`, test PR-AUC `0.8075`. The report includes the model-selection caveat and near-optimal threshold band.
- Verified the code path with `src/smoke_test.py`.

Production-layer verification on 12 September 2026:

- Packaged the unchanged audited JSON artifact as immutable release `1.0.0` and promoted it through an integrity-checked pointer. Artifact SHA-256: `9800efa3737531fdd0df78bcbb782dd0108cbfca593a76d33f0f534a1842645d`.
- Added the FastAPI service, strict input contract, single and batch prediction routes, liveness/readiness checks, bounded model metadata, structured prediction logs, and Prometheus-format process metrics.
- Added guarded release packaging and compare-and-swap promotion. Training still writes only a candidate artifact; it does not promote automatically.
- Added an unlabelled batch monitor for feature and alert-rate shifts. The synthetic smoke dataset correctly raised a drift failure against the real-data training reference.
- Passed Python compilation, the existing synthetic pipeline smoke test, release verification, a real localhost HTTP request, and all 17 unit/API/integrity/freshness tests.
- Added a non-root Dockerfile and a GitHub Actions container-build job. The image was not built locally because Docker is not installed on this Windows host; the CI build remains the container verification boundary.

Smoke-test artifacts, if generated, are only code-validation artifacts. They are not portfolio model results because the smoke fixture is synthetic and deliberately easy.

The raw CSV is intentionally ignored by `.gitignore`. It is not required by the inference image because the promoted release contains only the fitted model contract and parameters.

Real run command after the CSV is available:

```powershell
python src/fraud_pipeline.py --no-download
```

Real run command when network access is available:

```powershell
python src/fraud_pipeline.py
```
