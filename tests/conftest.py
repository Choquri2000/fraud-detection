import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fraud_detection.data import join_tables  # noqa: E402
from src.fraud_detection.synthetic import make_transactions  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return yaml.safe_load(open(ROOT / "configs" / "config.yaml", encoding="utf8"))


@pytest.fixture(scope="session")
def tables():
    return make_transactions(n=8000, seed=1)


@pytest.fixture(scope="session")
def raw(tables):
    return join_tables(*tables)
