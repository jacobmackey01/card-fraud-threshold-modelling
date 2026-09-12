from __future__ import annotations

import argparse
import json
from pathlib import Path

from service.model import ModelBundle, PROJECT_ROOT


def current_version(project_root: Path) -> str | None:
    pointer_path = project_root / "artifacts" / "production.json"
    if not pointer_path.exists():
        return None
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    version = pointer.get("model_version")
    return str(version) if version is not None else None


def promote(
    *,
    project_root: Path,
    model_version: str,
    expected_current: str | None,
) -> Path:
    project_root = project_root.resolve()
    actual_current = current_version(project_root)
    if actual_current != expected_current:
        raise RuntimeError(
            f"Production pointer changed: expected {expected_current!r}, found {actual_current!r}. "
            "Revalidate before promoting."
        )
    bundle = ModelBundle.load_release(project_root=project_root, model_version=model_version)
    pointer = {
        "schema_version": 1,
        "model_version": bundle.model_version,
        "manifest_sha256": bundle.manifest_sha256,
    }
    pointer_path = project_root / "artifacts" / "production.json"
    temporary_path = pointer_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(pointer, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(pointer_path)
    return pointer_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Atomically promote a validated fraud-model release.")
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--expected-current",
        help="Current production version. Omit only for the first promotion.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the current production release without changing the pointer.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.check:
        bundle = ModelBundle.load(PROJECT_ROOT)
        print(f"Verified production release {bundle.model_version}: {bundle.artifact_sha256}")
        return 0
    pointer_path = promote(
        project_root=PROJECT_ROOT,
        model_version=args.version,
        expected_current=args.expected_current,
    )
    print(f"Promoted {args.version} via {pointer_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
