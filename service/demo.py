from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from service.schemas import Transaction


DEMO_FIXTURE_RELATIVE_PATH = Path("examples/synthetic_demo_v1.json")


class DemoFixtureError(RuntimeError):
    """Raised when the checked-in synthetic deployment fixture is invalid."""


@dataclass(frozen=True)
class DemoFixture:
    fixture_id: str
    transaction: Transaction


def load_demo_fixture(project_root: Path) -> DemoFixture:
    fixture_path = project_root.resolve() / DEMO_FIXTURE_RELATIVE_PATH
    try:
        payload: Any = json.loads(fixture_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DemoFixtureError(f"Could not load synthetic demo fixture: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise DemoFixtureError("Synthetic demo fixture has an unsupported schema_version")
    fixture_id = payload.get("fixture_id")
    if not isinstance(fixture_id, str) or not fixture_id.startswith("synthetic-"):
        raise DemoFixtureError("Synthetic demo fixture_id must start with 'synthetic-'")
    transaction_payload = payload.get("transaction")
    if not isinstance(transaction_payload, dict):
        raise DemoFixtureError("Synthetic demo fixture is missing its transaction")
    try:
        transaction = Transaction.model_validate(transaction_payload)
    except ValueError as exc:
        raise DemoFixtureError(f"Synthetic demo fixture has an invalid transaction: {exc}") from exc
    if transaction.transaction_id != fixture_id:
        raise DemoFixtureError("Synthetic demo fixture transaction_id must equal fixture_id")
    return DemoFixture(fixture_id=fixture_id, transaction=transaction)
