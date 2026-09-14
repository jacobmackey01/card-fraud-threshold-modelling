# Synthetic Deployment Rehearsal

This deployment is a portfolio rehearsal, not a production fraud-detection system. It deploys the already selected model release without retraining or changing its threshold.

The artifact path is deliberately singular:

```text
tested Git commit
  -> one container build
  -> Artifact Registry digest
  -> Cloud Run revision pinned to that digest
  -> tagged candidate verification
  -> guarded traffic promotion
  -> stable-URL verification record
```

The public container runs with `SYNTHETIC_DEMO_ONLY=true`. In that mode it does not expose the arbitrary-vector prediction or batch routes. `POST /v1/demo-prediction` accepts no request body and scores only `examples/synthetic_demo_v1.json`. Logs contain the fixture ID, revision, latency and outcome, never the transaction features.

## One-time cloud setup

Create these resources outside the deployment workflow:

- A Google Cloud project with billing enabled and a budget alert configured.
- An Artifact Registry Docker repository.
- A Cloud Run runtime service account with no application permissions; the demo calls no Google APIs.
- A GitHub deployment service account allowed to write to that Artifact Registry repository, deploy Cloud Run revisions, and act as the runtime service account.
- A Workload Identity Federation provider restricted to this repository. Grant its repository principal `roles/iam.workloadIdentityUser` on the deployment service account. Do not create a service-account key.
- A GitHub environment named `cloud-run-demo` with a required reviewer. Restrict its deployment branches to `main`.

The deployment workflow needs these `cloud-run-demo` environment variables:

| Variable | Meaning |
| --- | --- |
| `GCP_PROJECT_ID` | Google Cloud project ID |
| `GCP_REGION` | Artifact Registry and Cloud Run region |
| `GCP_WIF_PROVIDER` | Full Workload Identity Provider resource name |
| `GCP_DEPLOY_SERVICE_ACCOUNT` | Deployment service-account email |
| `CLOUD_RUN_RUNTIME_SERVICE_ACCOUNT` | Minimal runtime service-account email |
| `GAR_REPOSITORY` | Artifact Registry repository name |
| `CLOUD_RUN_SERVICE` | Cloud Run service name |
| `BILLING_ALERT_CONFIRMED` | Literal `true`, set only after the alert exists |

The deployment identity should have only the permissions needed for Artifact Registry upload, Cloud Run deployment, and use of the runtime identity. The first public service creation also needs permission to set the unauthenticated invoker policy; this can instead be a one-time administrator action if the deployment identity is intentionally narrower.

## Deployment controls

`.github/workflows/deploy-cloud-run.yml`:

1. Runs the complete test and release-integrity suite.
2. Builds the container once and publishes a commit-addressed image to Artifact Registry.
3. Captures the registry digest and deploys `IMAGE@sha256:...`, never a mutable tag.
4. Creates a zero-traffic tagged candidate when the service already exists.
5. Verifies the candidate's commit, container digest, model-file digest, release, threshold and fixture version.
6. Moves traffic only if the previously observed revision still owns 100% of traffic.
7. Repeats verification through the stable service URL and uploads the evidence JSON for 90 days.

Cloud Run is limited to zero minimum instances, one maximum instance, four concurrent requests and a ten-second timeout. The workflow refuses to deploy unless `BILLING_ALERT_CONFIRMED=true`.

## Explicit rollback

`.github/workflows/rollback-cloud-run.yml` requires both an explicit target revision and the revision expected to be currently serving. The `cloud-run-demo` approval gate is applied again. The workflow refuses the move if traffic changed, restores the target revision, verifies its own recorded provenance through the stable URL, and uploads rollback evidence.

No placeholder or intentionally inferior revision is created. Until a genuine later application revision exists, live rollback evidence is correctly recorded as pending.
