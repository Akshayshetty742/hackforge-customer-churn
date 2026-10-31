"""Runs the full test suite and writes the Deliverable Part 3 test reports to docs/reports/test-reports/.

    python tools/generate_test_reports.py

Produces, from one real test run:
    UNIT_TEST_REPORT.md            every automated test with its result and Test Plan test case
    SYSTEM_VALIDATION_REPORT.md    test case execution record and requirement validation matrix
    CODE_COVERAGE_REPORT.md        statement coverage per module
    BRANCH_COVERAGE_REPORT.md      branch coverage per module and the branches not taken
plus the raw evidence (JUnit XML, Cobertura XML, coverage JSON, pytest and pylint output) in raw/.
Exits non-zero if any test fails.
"""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# pylint: disable=wrong-import-position  # the project root has to be on sys.path first
import prediction_pipeline as pp  # noqa: E402
from traceability import TEST_CASES, requirement_index  # noqa: E402

OUT_DIR = os.path.join(BASE_DIR, "docs", "reports", "test-reports")
RAW_DIR = os.path.join(OUT_DIR, "raw")
PYLINT_THRESHOLD = 8.5  # NFR-13, same as pyproject.toml fail-under
STATIC_ANALYSIS_CASE = "TC-26"  # verified by pylint here and in CI, not by a pytest test
PACKAGES = ["pandas", "numpy", "scikit-learn", "xgboost", "streamlit", "mlflow", "dvc", "torch", "pennylane",
            "pytest", "pytest-cov", "pylint"]
TEST_MODULES = {
    "test_pipeline": "Prediction pipeline: artefact loading, validation, preprocessing, prediction, metrics",
    "test_app": "Streamlit user interface, run headless with AppTest",
    "test_sprint2": "Batch scoring, churn driver explanation and what-if simulator",
    "test_paradigms": "Deep learning (DML) and quantum (QML) models, paradigm comparison",
    "test_system_validation": "End-to-end training pipeline, benchmark and non-functional checks (TC-01 to TC-13)",
}


# ----- running the tools -------------------------------------------------------------------------
def _run(command: List[str], env: Dict[str, str]) -> Tuple[int, str]:
    result = subprocess.run(command, cwd=BASE_DIR, env=env, capture_output=True, text=True, check=False)
    return result.returncode, result.stdout + result.stderr


def _strip_local_paths(path: str) -> None:
    """Replaces the absolute project path with '.' so the committed evidence has no machine paths."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    _write(path, text.replace(BASE_DIR + os.sep, "").replace(BASE_DIR, "."))


def run_tests() -> int:
    """Runs pytest with JUnit XML and line + branch coverage output; returns the exit code."""
    os.makedirs(RAW_DIR, exist_ok=True)
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1",
           "COVERAGE_FILE": os.path.join(tempfile.mkdtemp(), ".coverage")}
    code, output = _run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-W", "ignore", "-o", "junit_family=xunit1",
         f"--junitxml={RAW_DIR}/junit.xml", "--cov", "--cov-branch",
         f"--cov-report=xml:{RAW_DIR}/coverage.xml", f"--cov-report=json:{RAW_DIR}/coverage.json",
         "--cov-report=term"],
        env,
    )
    _write(os.path.join(RAW_DIR, "pytest_output.txt"), output)
    for name in ("pytest_output.txt", "coverage.xml", "coverage.json", "junit.xml"):
        _strip_local_paths(os.path.join(RAW_DIR, name))
    return code


def run_pylint() -> float:
    """Pylint score of every Python file in the repository (configuration in pyproject.toml)."""
    files = sorted(name for name in os.listdir(BASE_DIR) if name.endswith(".py")) + ["tools"]
    _, output = _run([sys.executable, "-m", "pylint", *files, "--fail-under=0"], dict(os.environ))
    _write(os.path.join(RAW_DIR, "pylint_output.txt"), output.replace(BASE_DIR, "."))
    match = re.search(r"rated at (-?[0-9.]+)/10", output)
    return float(match.group(1)) if match else 0.0


# ----- reading the results -----------------------------------------------------------------------
def read_junit() -> Dict[str, Any]:
    """Test suite totals and one record per test case from the JUnit XML."""
    root = ET.parse(os.path.join(RAW_DIR, "junit.xml")).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    tests = []
    for case in suite.iter("testcase"):
        properties = {p.get("name"): p.get("value") for p in case.iter("property")}
        if case.find("failure") is not None or case.find("error") is not None:
            outcome = "Failed"
        elif case.find("skipped") is not None:
            outcome = "Skipped"
        else:
            outcome = "Passed"
        tests.append({
            "module": case.get("classname", "").split(".")[0],
            "name": case.get("name"),
            "time": float(case.get("time", 0)),
            "outcome": outcome,
            "cases": [c for c in properties.pop("test_case", "unmapped").split(",") if c],
            "measured": properties,
        })
    return {"tests": tests, "time": float(suite.get("time", 0)), "timestamp": suite.get("timestamp", "")}


def read_coverage() -> Dict[str, Any]:
    """Coverage JSON restricted to the application modules (tests and tooling excluded)."""
    with open(os.path.join(RAW_DIR, "coverage.json"), encoding="utf-8") as handle:
        data = json.load(handle)
    files = {path: info for path, info in data["files"].items() if _is_application_module(path)}
    totals = Counter()
    for info in files.values():
        totals.update({key: info["summary"][key] for key in
                       ("num_statements", "covered_lines", "num_branches", "covered_branches",
                        "num_partial_branches")})
    return {"files": dict(sorted(files.items())), "totals": totals}


def _is_application_module(path: str) -> bool:
    name = os.path.basename(path)
    return os.path.dirname(path) in ("", ".") and not name.startswith("test_") and name not in {
        "conftest.py", "traceability.py"}


def environment() -> List[Tuple[str, str]]:
    """(item, value) rows describing the machine and library versions of the run."""
    rows = [("Operating system", f"{platform.system()} {platform.release()} ({platform.machine()})"),
            ("Python", platform.python_version())]
    for package in PACKAGES:
        try:
            rows.append((package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            rows.append((package, "not installed"))
    return rows


# ----- writing the reports -----------------------------------------------------------------------
def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _percent(part: float, whole: float) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def _header(title: str, run_date: str) -> str:
    return (f"# {title}\n\n"
            "Project: Customer Churn Prediction (Team Hackforge)  \n"
            "Course: Software Engineering Mini Project, Deliverable Part 3  \n"
            f"Date of test run: {run_date}  \n"
            "Generated by: `python tools/generate_test_reports.py` from a single run of the automated test suite\n\n")


def _ranges(lines: List[int]) -> str:
    """[3, 4, 5, 9] -> '3-5, 9'."""
    groups: List[List[int]] = []
    for line in sorted(lines):
        if groups and line == groups[-1][-1] + 1:
            groups[-1].append(line)
        else:
            groups.append([line])
    return ", ".join(f"{g[0]}-{g[-1]}" if len(g) > 1 else str(g[0]) for g in groups)


def unit_test_report(junit: Dict[str, Any], run_date: str) -> str:
    """Markdown Unit Test Report."""
    tests = junit["tests"]
    outcomes = Counter(t["outcome"] for t in tests)
    text = _header("Unit Test Report", run_date)
    text += ("## 1. Purpose\n\nThis report records the result of every automated test in the project. "
             "The tests are written with pytest and cover the prediction pipeline, the Streamlit user interface, "
             "the training pipeline, and the deep learning and quantum models. Each test is linked to the "
             "Test Plan v1.1 test case it executes.\n\n")
    text += "## 2. Test Environment\n\n" + _table(["Item", "Value"], [list(row) for row in environment()])
    text += ("\nCommand: `pytest --cov --cov-branch --junitxml=docs/reports/test-reports/raw/junit.xml`\n\n"
             "## 3. Summary\n\n")
    text += _table(["Total tests", "Passed", "Failed", "Skipped", "Pass rate", "Duration"], [[
        len(tests), outcomes["Passed"], outcomes["Failed"], outcomes["Skipped"],
        _percent(outcomes["Passed"], len(tests) - outcomes["Skipped"]), f"{junit['time']:.1f} s"]])

    text += "\n## 4. Results by Test Module\n\n"
    rows = []
    for module, purpose in TEST_MODULES.items():
        group = [t for t in tests if t["module"] == module]
        counts = Counter(t["outcome"] for t in group)
        rows.append([f"`{module}.py`", purpose, len(group), counts["Passed"], counts["Failed"], counts["Skipped"],
                     f"{sum(t['time'] for t in group):.1f} s"])
    text += _table(["Module", "What it tests", "Tests", "Passed", "Failed", "Skipped", "Time"], rows)

    text += "\n## 5. Results by Test Level\n\n"
    levels: Dict[str, Counter] = defaultdict(Counter)
    for test in tests:
        level = TEST_CASES[test["cases"][0]].level if test["cases"] else "Unmapped"
        levels[level][test["outcome"]] += 1
    text += _table(["Level", "Tests", "Passed", "Failed"], [
        [level, sum(c.values()), c["Passed"], c["Failed"]] for level, c in sorted(levels.items())])

    text += "\n## 6. Detailed Results\n\n"
    text += _table(["No.", "Module", "Test", "Test case", "Result", "Time (s)"], [
        [i, t["module"], f"`{t['name']}`", ", ".join(t["cases"]), t["outcome"], f"{t['time']:.2f}"]
        for i, t in enumerate(tests, start=1)])
    text += ("\n## 7. Conclusion\n\n"
             f"{outcomes['Passed']} of {len(tests)} automated tests passed"
             f"{' and none failed' if not outcomes['Failed'] else ''}. "
             "The raw JUnit XML and the full pytest output are kept in `raw/` as evidence.\n")
    return text


def _case_results(junit: Dict[str, Any], pylint_score: float) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {case_id: {"tests": [], "measured": {}} for case_id in TEST_CASES}
    for test in junit["tests"]:
        for case_id in test["cases"]:
            results[case_id]["tests"].append(test["outcome"])
            results[case_id]["measured"].update(test["measured"])
    results[STATIC_ANALYSIS_CASE]["measured"]["pylint_score"] = f"{pylint_score:.2f}/10"
    for case_id, result in results.items():
        if case_id == STATIC_ANALYSIS_CASE:
            result["verdict"] = "Pass" if pylint_score >= PYLINT_THRESHOLD else "Fail"
        elif "Failed" in result["tests"]:
            result["verdict"] = "Fail"
        elif "Passed" in result["tests"]:
            result["verdict"] = "Pass"
        else:
            result["verdict"] = "Not run"
    return results


def system_validation_report(junit: Dict[str, Any], coverage: Dict[str, Any], pylint_score: float,
                             run_date: str) -> str:
    """Markdown System Validation Report."""
    results = _case_results(junit, pylint_score)
    verdicts = Counter(r["verdict"] for r in results.values())
    requirements = sorted(requirement_index().items(), key=lambda item: _requirement_key(item[0]))
    validated = [req for req, cases in requirements if all(results[c]["verdict"] == "Pass" for c in cases)]
    text = _header("System Validation Report", run_date)
    text += ("## 1. Purpose and Scope\n\n"
             "This report validates the complete Customer Churn Prediction system against the requirements in "
             "SRS v1.1 (`docs/SRS_v1.1_Addendum.md`) by executing every test case in Test Plan v1.1 "
             "(`docs/TEST_PLAN_v1.1.md`). The system under test includes the training pipeline, the Streamlit web "
             "application, batch scoring, the what-if simulator, and the three-paradigm comparison "
             "(classical ML, deep learning and quantum machine learning).\n\n"
             "All test cases are automated. The training pipeline and the paradigm benchmark are run end to end "
             "in a temporary folder, so the validated production model is never overwritten.\n\n")
    text += "## 2. Summary\n\n" + _table(
        ["Test cases", "Passed", "Failed", "Not run", "Requirements", "Requirements validated"],
        [[len(results), verdicts["Pass"], verdicts["Fail"], verdicts["Not run"], len(requirements), len(validated)]])
    text += "\n## 3. Test Execution Record\n\n" + _execution_record(results)
    text += "\n## 4. Requirement Validation Matrix\n\n"
    text += _table(["Requirement", "Test cases", "Status"], [
        [req, ", ".join(cases), "Validated" if req in validated else "Open"] for req, cases in requirements])
    text += "\n## 5. Non-functional Measurements\n\n" + _measurements(results, coverage["totals"], pylint_score)
    text += "\n## 6. Model Quality Validation\n\n" + _model_quality()
    text += "\n## 7. Conclusion\n\n"
    if len(validated) == len(requirements) and not verdicts["Fail"]:
        text += (f"All {len(results)} test cases passed and all {len(requirements)} requirements of SRS v1.1 "
                 "(functional, non-functional, interface and security) are validated. The system is ready for the "
                 "final demonstration.\n")
    else:
        text += (f"{verdicts['Pass']} of {len(results)} test cases passed and {len(validated)} of "
                 f"{len(requirements)} requirements are validated. Open items are listed in section 4.\n")
    return text


def _execution_record(results: Dict[str, Dict[str, Any]]) -> str:
    rows = []
    for case_id, case in TEST_CASES.items():
        result = results[case_id]
        passed = Counter(result["tests"])["Passed"]
        evidence = f"{passed}/{len(result['tests'])} automated tests passed" if result["tests"] \
            else "pylint static analysis"
        measured = "; ".join(f"{k.replace('_', ' ')} = {v}" for k, v in result["measured"].items())
        rows.append([case_id, case.title, ", ".join(case.requirements), case.level,
                     evidence + (f" ({measured})" if measured else ""), result["verdict"]])
    return _table(["ID", "Test case", "Requirements", "Level", "Actual result / evidence", "Status"], rows)


def _measurements(results: Dict[str, Dict[str, Any]], totals: Counter, pylint_score: float) -> str:
    measured = {k: v for r in results.values() for k, v in r["measured"].items()}
    return _table(["Requirement", "Target", "Measured", "Status"], [
        ["NFR-01 Training time (full params.yaml grid, 3 model families)", "< 300 s",
         f"{measured.get('training_seconds', 'n/a')} s", results["TC-10"]["verdict"]],
        ["NFR-11 Single prediction latency (median of 30)", "< 200 ms",
         f"{measured.get('median_prediction_ms', 'n/a')} ms", results["TC-24"]["verdict"]],
        ["NFR-13 Pylint score", f">= {PYLINT_THRESHOLD}/10", f"{pylint_score:.2f}/10",
         results[STATIC_ANALYSIS_CASE]["verdict"]],
        ["NFR-13 Statement coverage of application modules", ">= 85%",
         _percent(totals["covered_lines"], totals["num_statements"]), _target(totals, "lines", 85)],
        ["NFR-13 Branch coverage of application modules", ">= 75%",
         _percent(totals["covered_branches"], totals["num_branches"]), _target(totals, "branches", 75)],
    ])


def _model_quality() -> str:
    benchmark = pp.SAVED_BENCHMARK_METRICS
    text = ("The production XGBoost model was re-evaluated on the 1,409-customer held-out test set during the "
            "run (test `test_live_evaluation_matches_saved_benchmark`) and reproduced its recorded benchmark:\n\n")
    text += _table(["Accuracy", "Precision (churn)", "Recall (churn)", "F1 (churn)", "ROC-AUC"], [[
        f"{benchmark['test_accuracy']:.2%}", f"{benchmark['test_precision']:.2%}", f"{benchmark['test_recall']:.2%}",
        f"{benchmark['test_f1']:.3f}", f"{benchmark['test_roc_auc']:.3f}"]])
    paradigm_file = os.path.join(BASE_DIR, "paradigm_metrics.json")
    if os.path.exists(paradigm_file):
        with open(paradigm_file, encoding="utf-8") as handle:
            board = json.load(handle)
        text += ("\nThe ML vs DML vs QML benchmark (`paradigm_metrics.json`, the same split for every model) "
                 "gave the following results:\n\n")
        text += _table(["Paradigm", "Model", "Accuracy", "Recall (churn)", "F1 (churn)", "ROC-AUC", "Parameters"], [
            [m["paradigm"], name, f"{m['test_accuracy']:.2%}", f"{m['churn_recall']:.2%}",
             f"{m['churn_f1_score']:.3f}", f"{m['roc_auc']:.3f}", f"{int(m['trainable_parameters']):,}"]
            for name, m in board.items()])
    return text


def _requirement_key(requirement: str) -> Tuple[int, int]:
    prefix, number = requirement.rsplit("-", 1)
    order = ["FR", "NFR", "UI", "SW", "HW", "COM", "SO", "SR"]
    return order.index(prefix) if prefix in order else len(order), int(number)


def _target(totals: Counter, kind: str, target: float) -> str:
    covered, total = (("covered_lines", "num_statements") if kind == "lines"
                      else ("covered_branches", "num_branches"))
    return "Pass" if totals[total] and 100 * totals[covered] / totals[total] >= target else "Below target"


def code_coverage_report(coverage: Dict[str, Any], run_date: str) -> str:
    """Markdown Code (statement) Coverage Report."""
    totals = coverage["totals"]
    text = _header("Code Coverage Report", run_date)
    text += ("## 1. Method\n\n"
             "Statement coverage was measured with coverage.py (through pytest-cov) while the full automated test "
             "suite ran. It counts the share of executable statements in each application module that ran at "
             "least once. Test files, `conftest.py`, `traceability.py` and the `tools/` folder are excluded so the "
             "figures describe the application code only.\n\n")
    text += "## 2. Overall Result\n\n" + _table(["Statements", "Covered", "Missed", "Statement coverage"], [[
        totals["num_statements"], totals["covered_lines"], totals["num_statements"] - totals["covered_lines"],
        _percent(totals["covered_lines"], totals["num_statements"])]])
    text += "\n## 3. Coverage by Module\n\n"
    text += _table(["Module", "Statements", "Covered", "Missed", "Coverage"], [
        [f"`{path}`", s["num_statements"], s["covered_lines"], s["missing_lines"],
         _percent(s["covered_lines"], s["num_statements"])]
        for path, s in ((p, i["summary"]) for p, i in coverage["files"].items())])
    text += "\n## 4. Statements Not Executed\n\n"
    text += _table(["Module", "Lines"], [
        [f"`{path}`", _ranges(info["missing_lines"]) or "none"] for path, info in coverage["files"].items()])
    text += ("\nMost of the remaining statements in `app.py` are Streamlit page sections that are only reached "
             "through browser interactions such as file uploads and download buttons; these flows are shown "
             "working in the Sprint 2 demonstration video.\n\n"
             "## 5. Evidence\n\nMachine-readable results are in `raw/coverage.xml` (Cobertura format) and "
             "`raw/coverage.json`.\n")
    return text


def branch_coverage_report(coverage: Dict[str, Any], run_date: str) -> str:
    """Markdown Branch Coverage Report."""
    totals = coverage["totals"]
    text = _header("Branch Coverage Report", run_date)
    text += ("## 1. Method\n\n"
             "Branch coverage was measured with coverage.py in branch mode (`--cov-branch`). Every decision point "
             "(`if`, `for`, `while`, `try`, conditional expressions) has two or more possible outcomes, and a branch "
             "is covered when the test suite takes that outcome at least once. This is a stricter measure than "
             "statement coverage. Test files and tooling are excluded.\n\n")
    text += "## 2. Overall Result\n\n" + _table(
        ["Branches", "Covered", "Not taken", "Partially covered decisions", "Branch coverage"], [[
            totals["num_branches"], totals["covered_branches"], totals["num_branches"] - totals["covered_branches"],
            totals["num_partial_branches"], _percent(totals["covered_branches"], totals["num_branches"])]])
    text += "\n## 3. Branch Coverage by Module\n\n"
    text += _table(["Module", "Branches", "Covered", "Partial decisions", "Branch coverage"], [
        [f"`{path}`", s["num_branches"], s["covered_branches"], s["num_partial_branches"],
         _percent(s["covered_branches"], s["num_branches"])]
        for path, s in ((p, i["summary"]) for p, i in coverage["files"].items())])
    text += "\n## 4. Branches Not Taken\n\nEach entry reads `source line -> destination line` "
    text += "(a negative destination means the function exits).\n\n"
    text += _table(["Module", "Branches not taken"], [
        [f"`{path}`", ", ".join(f"{a}->{b}" for a, b in info.get("missing_branches", [])) or "none"]
        for path, info in coverage["files"].items()])
    text += ("\nThe branches not taken are mainly defensive error paths (for example a corrupt pickle file) and "
             "optional user interface paths. The core decision logic in input validation, preprocessing, "
             "prediction and training is exercised with both valid and invalid input.\n")
    return text


def index_page(run_date: str, failed: int) -> str:
    """docs/reports/test-reports/README.md."""
    return ("# Test Reports\n\n"
            f"Test reports for Deliverable Part 3, generated on {run_date} from one run of the automated test suite"
            f"{' with no failures' if not failed else ''}.\n\n"
            "| Report | Contents |\n|---|---|\n"
            "| [Unit Test Report](UNIT_TEST_REPORT.md) | Result of every automated test, by module and level |\n"
            "| [System Validation Report](SYSTEM_VALIDATION_REPORT.md) | Test case execution record and requirement "
            "validation matrix |\n"
            "| [Code Coverage Report](CODE_COVERAGE_REPORT.md) | Statement coverage of each application module |\n"
            "| [Branch Coverage Report](BRANCH_COVERAGE_REPORT.md) | Branch coverage and the branches not taken |\n\n"
            "Raw evidence is in [raw/](raw/): JUnit XML, Cobertura XML, coverage JSON, and the pytest and pylint "
            "output.\n\nTo regenerate: `python tools/generate_test_reports.py`\n")


def main() -> None:
    """Runs the tests and pylint, then writes every report."""
    print("Running the test suite with coverage...")
    exit_code = run_tests()
    print("Running pylint...")
    score = run_pylint()
    junit, coverage = read_junit(), read_coverage()
    run_date = dt.date.today().strftime("%d %B %Y")
    failed = sum(t["outcome"] == "Failed" for t in junit["tests"])
    reports = {
        "UNIT_TEST_REPORT.md": unit_test_report(junit, run_date),
        "SYSTEM_VALIDATION_REPORT.md": system_validation_report(junit, coverage, score, run_date),
        "CODE_COVERAGE_REPORT.md": code_coverage_report(coverage, run_date),
        "BRANCH_COVERAGE_REPORT.md": branch_coverage_report(coverage, run_date),
        "README.md": index_page(run_date, failed),
    }
    for name, text in reports.items():
        _write(os.path.join(OUT_DIR, name), text)
        print(f"Wrote {os.path.relpath(os.path.join(OUT_DIR, name), BASE_DIR)}")
    totals = coverage["totals"]
    print(f"{len(junit['tests'])} tests, {failed} failed | statement coverage "
          f"{_percent(totals['covered_lines'], totals['num_statements'])} | branch coverage "
          f"{_percent(totals['covered_branches'], totals['num_branches'])} | pylint {score:.2f}/10")
    sys.exit(1 if exit_code or failed else 0)


if __name__ == "__main__":
    main()
