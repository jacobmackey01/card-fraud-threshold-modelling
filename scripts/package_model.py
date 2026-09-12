from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from service.model import EXPECTED_MODEL_FEATURES, MODEL_VERSION_PATTERN, RAW_INPUT_FEATURES, sha256_bytes


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _git_commit(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def package_model(*, project_root: Path, candidate: Path, model_version: str) -> Path:
    if not MODEL_VERSION_PATTERN.fullmatch(model_version):
        raise ValueError("model_version must use MAJOR.MINOR.PATCH, for example 1.0.0")
    project_root = project_root.resolve()
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Candidate model not found: {candidate}")
    artifact_bytes = candidate.read_bytes()
    artifact = json.loads(artifact_bytes)
    if artifact.get("feature_names") != EXPECTED_MODEL_FEATURES:
        raise ValueError("Candidate feature contract is missing, reordered, or unsupported")
    if not isinstance(artifact.get("weights"), list):
        raise ValueError("The current JSON service can package only an audited linear model")

    releases_dir = project_root / "artifacts" / "releases"
    release_dir = releases_dir / model_version
    if release_dir.exists():
        raise FileExistsError(
            f"Release {model_version} already exists. Model releases are immutable; choose a new version."
        )
    releases_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{model_version}-", dir=releases_dir))
    try:
        artifact_path = temporary_dir / "fraud_model.json"
        artifact_path.write_bytes(artifact_bytes)
        manifest = {
            "schema_version": 1,
            "model_version": model_version,
            "packaged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_commit": _git_commit(project_root),
            "artifact_file": artifact_path.name,
            "artifact_sha256": sha256_bytes(artifact_bytes),
            "model_name": artifact.get("model_name"),
            "threshold": artifact.get("threshold"),
            "raw_input_features": RAW_INPUT_FEATURES,
            "model_features": EXPECTED_MODEL_FEATURES,
            "evaluation": artifact.get("metrics", {}).get("test", {}),
            "boundary": (
                "Portfolio case study using anonymised PCA inputs and illustrative costs; "
                "not approved for autonomous fraud decisions."
            ),
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_dir.rename(release_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return release_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package a candidate fraud model as an immutable release.")
    parser.add_argument("--version", required=True, help="Semantic model version, for example 1.0.0")
    parser.add_argument("--candidate", type=Path, default=PROJECT_ROOT / "artifacts" / "fraud_model.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    release_dir = package_model(
        project_root=PROJECT_ROOT,
        candidate=args.candidate,
        model_version=args.version,
    )
    print(f"Packaged immutable release: {release_dir}")
    print("This did not change the production pointer. Run scripts/promote_model.py after validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
