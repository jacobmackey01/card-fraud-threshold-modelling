# Monitoring Plan

Fraud models decay because transaction behavior changes and attackers adapt. The operating question is not "is accuracy still high?" but "are we still catching the right fraud at an affordable alert volume?"

## Label Latency

Fraud labels often arrive late through chargebacks, customer disputes, investigations, or network notifications. That means recall cannot be measured fully in real time. The dashboard should separate:

- Immediate proxy signals: score distribution, alert volume, manual-review hit rate, decline/step-up rate, customer complaint rate.
- Delayed truth signals: confirmed fraud rate, chargeback dollars, fraud captured by alerts, fraud missed by alerts.

## Score And Feature Drift

Track drift weekly and daily during incidents:

- Score distribution by channel, merchant region, and card-present/card-not-present segment.
- Population Stability Index or a similar binned distribution distance for `Amount`, `Amount_log1p`, hour-of-day features, and the highest-impact PCA features.
- Missingness, extreme values, and zero-amount rates.

Suggested first alert: PSI above `0.20` for a key feature or score distribution. Treat `0.10` to `0.20` as a warning band, not an automatic retrain trigger.

## Alert-Rate And Capacity Guardrails

The model should be operated with an explicit review budget:

- Daily alert count and flagged transaction rate.
- Precision of reviewed alerts once analyst outcomes are available.
- Queue age and analyst capacity utilization.
- Segment-level false-positive spikes, especially for high-value customers or specific merchant categories.

If the queue exceeds capacity, threshold selection should move to the `capacity` policy and cap alerts at the review budget. If fraud losses spike while alert volume is stable, investigate recall decay and adversarial behavior.

## Adversarial Drift

Fraud rings adapt to controls. Watch for:

- Sudden drops in alert-confirmed fraud rate while chargeback losses rise.
- New clusters of high-risk merchants, devices, geographies, or transaction amounts.
- A rising share of fraud just below the current threshold.

The response should be operational before it is purely statistical: review recent misses, add rules or temporary watchlists where justified, then retrain once enough delayed labels arrive.

## Ownership

Threshold ownership should be explicit:

- Fraud operations owns alert capacity and review cost.
- Risk/finance owns missed-fraud loss assumptions.
- Product/customer teams own customer-friction tolerance.
- Data science owns model calibration, monitoring, and retraining recommendations.

The shipped pipeline defaults to a cost threshold, but production use should review the cost matrix and alert-capacity budget with those owners before deployment.

The example cost policy is illustrative: false positives use a fixed review/friction cost, while missed fraud uses transaction amount plus a handling cost. Real values should come from chargeback loss history, analyst handling time, customer-contact cost, and product-approved friction tolerance.

## Implemented operational checks

The API exposes process-lifetime counters and gauges at `GET /metrics`: predictions, review recommendations, scoring errors, alert rate, and mean score. Prediction logs record the promoted model version and latency without writing input features or caller-supplied transaction IDs.

For batch checks, run:

```powershell
python -m service.monitor path/to/recent_transactions.csv --output monitoring/latest.json
```

The command rejects malformed model inputs, reports mean and scale shifts for all 33 engineered features, and compares the observed alert rate with the held-out test reference plus a three-standard-error sampling band. `--fail-on-alert` returns a non-zero exit code when a warning or failure is present, which allows a scheduled job to gate promotion or page an owner.

These controls are intentionally limited. The process counters reset when an instance restarts, the example service has no durable metrics backend, and mean/scale checks do not replace PSI or segment-level analysis. A real deployment should scrape `/metrics`, retain versioned batch reports, add channel and geography slices from explainable production fields, and join predictions to delayed confirmed outcomes under the organisation's privacy and retention rules.
