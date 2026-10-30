"""Shared pytest setup: tags every test with the Test Plan test case(s) it executes.

The IDs come from @pytest.mark.tc("TC-xx") on the test or, for older tests, from
traceability.EXISTING_TEST_MAP. They are written to the JUnit XML as a `test_case` property,
which tools/generate_test_reports.py turns into the System Validation Report.
"""

import os

import pytest

from traceability import EXISTING_TEST_MAP, TEST_CASES

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def pytest_configure(config):
    """Registers the tc marker."""
    config.addinivalue_line("markers", "tc(*ids): Test Plan test case IDs (TC-xx) this test executes")


def _test_case_ids(item):
    marker = item.get_closest_marker("tc")
    if marker is not None:
        return list(marker.args)
    return EXISTING_TEST_MAP.get(getattr(item, "originalname", item.name), [])


def pytest_collection_modifyitems(items):
    """Records each test's test case IDs as a JUnit property; rejects IDs missing from traceability.py."""
    for item in items:
        ids = _test_case_ids(item)
        unknown = [case_id for case_id in ids if case_id not in TEST_CASES]
        if unknown:
            raise pytest.UsageError(f"{item.nodeid}: unknown test case IDs {unknown} (see traceability.py)")
        item.user_properties.append(("test_case", ",".join(ids) if ids else "unmapped"))
