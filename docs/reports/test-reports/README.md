# Test Reports

Test reports for Deliverable Part 3, generated on 30 October 2026 from one run of the automated test suite with no failures.

| Report | Contents |
|---|---|
| [Unit Test Report](UNIT_TEST_REPORT.md) | Result of every automated test, by module and level |
| [System Validation Report](SYSTEM_VALIDATION_REPORT.md) | Test case execution record and requirement validation matrix |
| [Code Coverage Report](CODE_COVERAGE_REPORT.md) | Statement coverage of each application module |
| [Branch Coverage Report](BRANCH_COVERAGE_REPORT.md) | Branch coverage and the branches not taken |

Raw evidence is in [raw/](raw/): JUnit XML, Cobertura XML, coverage JSON, and the pytest and pylint output.

To regenerate: `python tools/generate_test_reports.py`
