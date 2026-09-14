from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def serving_revision(service: dict[str, Any]) -> str:
    traffic = service.get("status", {}).get("traffic", [])
    serving = [
        item.get("revisionName")
        for item in traffic
        if item.get("percent") == 100 and isinstance(item.get("revisionName"), str)
    ]
    if len(serving) != 1:
        raise RuntimeError("Cloud Run service must have exactly one revision receiving 100% of traffic")
    return serving[0]


def require_transition(*, current: str, expected_current: str, target: str) -> None:
    if current != expected_current:
        raise RuntimeError(
            f"Traffic changed: expected {expected_current!r}, found {current!r}. Revalidate before continuing."
        )
    if target == current:
        raise RuntimeError("Target revision already receives 100% of traffic")


def _gcloud_json(*args: str) -> dict[str, Any]:
    result = subprocess.run(["gcloud", *args, "--format=json"], check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("gcloud did not return a JSON object")
    return payload


def _describe_service(*, project: str, region: str, service: str) -> dict[str, Any]:
    return _gcloud_json(
        "run",
        "services",
        "describe",
        service,
        "--project",
        project,
        "--region",
        region,
    )


def move_traffic(args: argparse.Namespace) -> dict[str, Any]:
    before = _describe_service(project=args.project, region=args.region, service=args.service)
    before_revision = serving_revision(before)
    require_transition(
        current=before_revision,
        expected_current=args.expected_current_revision,
        target=args.target_revision,
    )
    subprocess.run(
        [
            "gcloud",
            "run",
            "services",
            "update-traffic",
            args.service,
            "--project",
            args.project,
            "--region",
            args.region,
            f"--to-revisions={args.target_revision}=100",
            "--quiet",
        ],
        check=True,
    )
    after = _describe_service(project=args.project, region=args.region, service=args.service)
    after_revision = serving_revision(after)
    if after_revision != args.target_revision:
        raise RuntimeError(
            f"Traffic update completed but {after_revision!r}, not {args.target_revision!r}, is serving"
        )
    return {
        "schema_version": 1,
        "action": args.action,
        "status": "passed",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "service": args.service,
        "region": args.region,
        "project": args.project,
        "from_revision": before_revision,
        "to_revision": after_revision,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        json.dump(payload, temporary, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guard a Cloud Run promotion or rollback by current revision.")
    parser.add_argument("--action", choices=("promote", "rollback"), required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--service", required=True)
    parser.add_argument("--expected-current-revision", required=True)
    parser.add_argument("--target-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    evidence = move_traffic(args)
    _write_json_atomic(args.output, evidence)
    print(
        f"Guarded {args.action}: {evidence['from_revision']} -> {evidence['to_revision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
