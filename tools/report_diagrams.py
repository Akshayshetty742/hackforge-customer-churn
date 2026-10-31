"""Black-and-grey UML diagrams for the specification document, drawn with matplotlib.

Sequence diagrams are drawn from Mermaid-style text; the structural diagrams (context, use case,
component, class, deployment) have fixed layouts defined here.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

# pylint: disable=wrong-import-position  # the backend has to be chosen before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

BLACK = "#000000"
GREY = "#6b6b6b"
FILL = "#f2f2f2"
FONT = ["Arial", "DejaVu Sans"]

Node = Tuple[float, float, float, float, str, str]  # x, y, width, height, label, shape
Edge = Tuple[str, str, str, str]  # start, end, label, style: solid | dashed | line

plt.rcParams.update({"font.family": FONT, "text.color": BLACK})


def _canvas(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def _save(fig, path: str) -> str:
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _actor(ax, x: float, y: float, label: str):
    ax.add_patch(Circle((x, y + 0.42), 0.12, fill=False, ec=BLACK, lw=1.2))
    ax.plot([x, x], [y + 0.3, y - 0.05], color=BLACK, lw=1.2)
    ax.plot([x - 0.22, x + 0.22], [y + 0.18, y + 0.18], color=BLACK, lw=1.2)
    ax.plot([x, x - 0.18], [y - 0.05, y - 0.32], color=BLACK, lw=1.2)
    ax.plot([x, x + 0.18], [y - 0.05, y - 0.32], color=BLACK, lw=1.2)
    ax.text(x, y - 0.45, label, ha="center", va="top", fontsize=8)
    return Rectangle((x - 0.3, y - 0.35), 0.6, 0.95, fill=False, lw=0)


def _node(ax, node: Node):
    x, y, w, h, label, shape = node
    if shape == "actor":
        patch = _actor(ax, x, y, label)
    elif shape == "ellipse":
        patch = Ellipse((x, y), w, h, fc="white", ec=BLACK, lw=1.1)
    else:
        patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                               fc=FILL if shape == "store" else "white", ec=BLACK, lw=1.1)
    ax.add_patch(patch)
    if shape != "actor":
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5)
    return patch


def flowchart(nodes: Dict[str, Node], edges: List[Edge], size: Tuple[float, float], path: str,
              frames: Optional[List[Tuple[float, float, float, float, str]]] = None) -> str:
    # pylint: disable=too-many-locals  # one local per drawing element keeps the layout readable
    """Boxes, ellipses, data stores and actors joined by arrows, inside optional labelled frames."""
    fig, ax = _canvas(*size)
    for x0, y0, x1, y1, title in frames or []:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=GREY, lw=1, linestyle="--"))
        ax.text(x0 + 0.1, y1 - 0.1, title, ha="left", va="top", fontsize=8, fontweight="bold")
    patches = {key: _node(ax, node) for key, node in nodes.items()}
    for start, end, label, style in edges:
        ax.add_patch(FancyArrowPatch(
            (nodes[start][0], nodes[start][1]), (nodes[end][0], nodes[end][1]),
            patchA=patches[start], patchB=patches[end], color=GREY if style == "dashed" else BLACK,
            arrowstyle="-" if style == "line" else "-|>", mutation_scale=10, lw=1,
            linestyle="--" if style == "dashed" else "-", shrinkA=2, shrinkB=2))
        if label:
            mx, my = (nodes[start][0] + nodes[end][0]) / 2, (nodes[start][1] + nodes[end][1]) / 2
            ax.text(mx, my + 0.06, label, ha="center", va="bottom", fontsize=6.5,
                    bbox={"fc": "white", "ec": "none", "pad": 0.4})
    return _save(fig, path)


# ----- sequence diagrams ----------------------------------------------------------------------
def _parse_sequence(text: str):
    participants: List[Tuple[str, str, bool]] = []
    steps: List[Tuple[str, ...]] = []
    for raw in text.splitlines():
        line = raw.strip()
        declared = re.match(r"(actor|participant)\s+(\w+)(?:\s+as\s+(.+))?$", line)
        message = re.match(r"(\w+)\s*(-->>|->>)\s*(\w+)\s*:\s*(.*)$", line)
        if declared:
            participants.append((declared.group(2), declared.group(3) or declared.group(2),
                                 declared.group(1) == "actor"))
        elif message:
            steps.append(("msg", message.group(1), message.group(3), message.group(4), message.group(2)))
        elif re.match(r"(alt|loop|else)\b", line):
            keyword, _, label = line.partition(" ")
            steps.append((keyword, label))
        elif line == "end":
            steps.append(("end",))
    return participants, steps


def sequence_diagram(text: str, path: str) -> str:  # pylint: disable=too-many-locals
    """UML sequence diagram with lifelines, calls, returns (dashed) and alt / else / loop fragments."""
    participants, steps = _parse_sequence(text)
    gap, row = 1.35, 0.42
    width = gap * len(participants) + 0.25
    height = row * (len(steps) + 2) + 1.0
    fig, ax = _canvas(width, height)
    xs = {pid: 0.8 + i * gap for i, (pid, _, _) in enumerate(participants)}
    top = height - 0.4
    for pid, label, is_actor in participants:
        ax.add_patch(FancyBboxPatch((xs[pid] - 0.62, top - 0.2), 1.24, 0.4, boxstyle="round,pad=0.02",
                                    fc=FILL if is_actor else "white", ec=BLACK, lw=1.1))
        ax.text(xs[pid], top, label, ha="center", va="center", fontsize=7.2, fontweight="bold")
    y = top - 0.62
    fragments: List[Tuple[float, str, int]] = []
    for step in steps:
        if step[0] == "msg":
            _, src, dst, label, arrow = step
            dashed = arrow == "-->>"
            style = {"arrowstyle": "-|>", "color": GREY if dashed else BLACK, "lw": 1,
                     "linestyle": "--" if dashed else "-", "shrinkA": 0, "shrinkB": 0}
            if src == dst:
                ax.annotate("", (xs[src], y - 0.22), (xs[src], y),
                            arrowprops={**style, "connectionstyle": "bar,fraction=-0.45"})
                ax.text(xs[src] + 0.14, y - 0.06, label, fontsize=8, va="center")
            else:
                ax.annotate("", (xs[dst], y), (xs[src], y), arrowprops=style)
                ax.text((xs[src] + xs[dst]) / 2, y + 0.04, label, ha="center", va="bottom", fontsize=8)
            y -= row
        elif step[0] in ("alt", "loop"):
            fragments.append((y + 0.24, f"{step[0]} [{step[1]}]", len(fragments)))
            y -= row * 0.62
        elif step[0] == "else":
            depth = len(fragments) - 1
            ax.plot([0.2 + 0.1 * depth, width - 0.2 - 0.1 * depth], [y + 0.2, y + 0.2], color=GREY, lw=0.8,
                    linestyle="--")
            ax.text(0.3 + 0.1 * depth, y + 0.15, f"[{step[1]}]", fontsize=7.8, va="top", style="italic")
            y -= row * 0.62
        elif step[0] == "end" and fragments:
            start, title, depth = fragments.pop()
            inset = 0.2 + 0.1 * depth
            ax.add_patch(Rectangle((inset, y + 0.1), width - 2 * inset, start - y - 0.1, fill=False, ec=BLACK,
                                   lw=0.9))
            ax.text(inset + 0.07, start - 0.04, title, fontsize=7.8, va="top", fontweight="bold")
            y -= row * 0.4
    bottom = y - 0.05
    for pid in xs:
        ax.plot([xs[pid]] * 2, [top - 0.2, bottom], color=GREY, lw=0.8, linestyle=(0, (4, 3)), zorder=0)
    ax.set_ylim(bottom - 0.1, height)
    return _save(fig, path)


# ----- structural diagrams --------------------------------------------------------------------
def context_diagram(path: str) -> str:
    """System context: users, the system boundary and external data stores."""
    nodes = {
        "U": (0.8, 4.3, 0, 0, "Retention analyst", "actor"),
        "S": (0.8, 1.6, 0, 0, "Data scientist", "actor"),
        "APP": (4.9, 4.0, 2.8, 0.6, "Web application\n(Streamlit)", "box"),
        "TRN": (4.9, 2.0, 2.8, 0.6, "Training and\nbenchmark pipelines", "box"),
        "D": (8.9, 4.8, 2.0, 0.55, "Telco customer\ndataset (CSV)", "store"),
        "M": (8.9, 3.4, 2.0, 0.55, "Model artifacts", "store"),
        "F": (8.9, 2.0, 2.0, 0.55, "Experiment store\n(MLflow)", "store"),
        "G": (8.9, 0.7, 2.0, 0.55, "Source repository\nand CI (GitHub)", "store"),
    }
    edges = [("U", "APP", "HTTP (localhost)", "solid"), ("S", "TRN", "CLI / dvc repro", "solid"),
             ("D", "TRN", "reads", "solid"), ("TRN", "M", "writes", "solid"), ("APP", "M", "reads", "dashed"),
             ("TRN", "F", "logs", "dashed"), ("S", "G", "", "dashed")]
    return flowchart(nodes, edges, (10.2, 5.6), path,
                     frames=[(3.0, 1.2, 6.8, 4.95, "Customer Churn Prediction System")])


def use_case_diagram(path: str) -> str:
    """UML use case diagram covering all twelve use cases."""
    nodes = {
        "A": (0.7, 4.0, 0, 0, "Retention analyst", "actor"),
        "S": (10.3, 3.2, 0, 0, "Data scientist", "actor"),
        "UC1": (7.4, 6.2, 2.6, 0.56, "UC-01 Load dataset", "ellipse"),
        "UC2": (7.4, 5.4, 2.6, 0.56, "UC-02 Prepare features", "ellipse"),
        "UC3": (7.4, 4.6, 2.6, 0.56, "UC-03 Train model", "ellipse"),
        "UC4": (7.4, 3.8, 2.6, 0.56, "UC-04 Evaluate model", "ellipse"),
        "UC5": (7.4, 3.0, 2.6, 0.56, "UC-05 Save artifacts", "ellipse"),
        "UC12": (7.4, 2.2, 2.6, 0.62, "UC-12 Benchmark\nparadigms", "ellipse"),
        "UC6": (5.4, 0.9, 2.4, 0.56, "UC-06 Handle bad input", "ellipse"),
        "UC7": (3.4, 6.0, 2.6, 0.62, "UC-07 Predict churn\nfor one customer", "ellipse"),
        "UC8": (3.4, 5.0, 2.6, 0.62, "UC-08 Compare\nML, DML, QML", "ellipse"),
        "UC9": (3.4, 4.0, 2.6, 0.62, "UC-09 Evaluate and\ntune threshold", "ellipse"),
        "UC10": (3.4, 3.0, 2.6, 0.62, "UC-10 Score a batch", "ellipse"),
        "UC11": (3.4, 2.0, 2.6, 0.62, "UC-11 Explain risk and\nsimulate retention", "ellipse"),
    }
    edges = [("A", f"UC{i}", "", "line") for i in (7, 9, 10, 11)]
    edges += [("S", f"UC{i}", "", "line") for i in (1, 3, 4, 5, 12)]
    edges += [("UC7", "UC8", "«include»", "dashed"), ("UC7", "UC6", "«include»", "dashed"),
              ("UC10", "UC6", "«include»", "dashed"), ("UC1", "UC6", "«include»", "dashed"),
              ("UC3", "UC2", "«include»", "dashed")]
    return flowchart(nodes, edges, (11.0, 7.0), path,
                     frames=[(1.75, 0.35, 9.2, 6.75, "Customer Churn Prediction System")])


def component_diagram(path: str) -> str:
    """Layered component diagram."""
    nodes = {
        "APP": (3.5, 7.1, 2.8, 0.62, "app.py\nWeb application", "box"),
        "PRE": (8.0, 7.1, 2.8, 0.62, "customer_presets.py\nSample customers", "box"),
        "PP": (3.5, 5.75, 3.6, 0.62, "prediction_pipeline.py\nValidation, prediction, evaluation", "box"),
        "CM": (1.8, 4.3, 2.5, 0.62, "classical_models.py\nLR, RF, XGBoost", "box"),
        "DML": (5.4, 4.3, 2.5, 0.62, "dml_model.py\nPyTorch MLP", "box"),
        "QML": (9.0, 4.3, 2.5, 0.62, "qml_model.py\nPennyLane VQC", "box"),
        "T1": (1.8, 2.85, 2.7, 0.62, "01_churn_prediction_\nxgboost.py", "box"),
        "TC": (5.4, 2.85, 2.7, 0.62, "training_common.py\nSplit, metrics, logging", "box"),
        "T2": (9.0, 2.85, 2.7, 0.62, "02_paradigm_\nbenchmark.py", "box"),
        "DATA": (1.5, 1.0, 2.0, 0.55, "Dataset\n(DVC tracked)", "store"),
        "PROD": (4.2, 1.0, 2.4, 0.55, "Production model\nand encoders", "store"),
        "PAR": (7.0, 1.0, 2.4, 0.55, "DML and QML\nmodels", "store"),
        "MLF": (9.8, 1.0, 1.8, 0.55, "MLflow store", "store"),
    }
    edges = [("APP", "PP", "", "solid"), ("APP", "PRE", "", "solid"), ("PP", "PROD", "", "dashed"),
             ("PP", "PAR", "", "dashed"), ("T1", "CM", "", "solid"), ("T2", "CM", "", "solid"),
             ("T2", "DML", "", "solid"), ("T2", "QML", "", "solid"), ("QML", "DML", "extends", "solid"),
             ("T1", "TC", "", "solid"), ("T2", "TC", "", "solid"), ("TC", "MLF", "", "dashed"),
             ("T1", "PROD", "writes", "dashed"), ("T2", "PAR", "writes", "dashed"), ("DATA", "T1", "", "solid")]
    frames = [(0.3, 6.6, 10.6, 7.75, "Presentation layer"), (0.3, 5.2, 10.6, 6.4, "Service layer"),
              (0.3, 3.8, 10.6, 5.0, "Model layer"), (0.3, 2.35, 10.6, 3.6, "Training layer"),
              (0.3, 0.45, 10.6, 2.15, "Data and artifact storage")]
    return flowchart(nodes, edges, (11.0, 8.0), path, frames)


def class_diagram(path: str) -> str:  # pylint: disable=too-many-locals
    """Class design of the model layer."""
    fig, ax = _canvas(10.4, 6.6)
    boxes = {
        "BaseEstimator": (2.3, 6.0, "BaseEstimator", ["(scikit-learn)"]),
        "ClassifierMixin": (5.5, 6.0, "ClassifierMixin", ["(scikit-learn)"]),
        "Torch": (3.9, 3.95, "TorchTabularClassifier",
                  ["+ training: TrainingConfig", "+ random_state: int", "+ fit(X, y)", "+ predict_proba(X)",
                   "+ predict(X)", "+ n_parameters: int", "# _build_preprocessor()", "# _build_network(n)"]),
        "MLP": (1.9, 1.0, "ChurnMLPClassifier", ["+ hidden_layers", "+ dropout", "# _build_network(n)"]),
        "VQC": (5.9, 1.0, "VariationalQuantumClassifier", ["+ n_qubits", "+ n_layers", "# _build_network(n)"]),
        "Config": (8.7, 3.95, "TrainingConfig", ["+ learning_rate", "+ weight_decay", "+ batch_size",
                                                  "+ max_epochs", "+ patience", "+ validation_fraction"]),
    }
    geometry = {}
    for key, (x, y, title, members) in boxes.items():
        h = 0.42 + 0.22 * len(members)
        ax.add_patch(Rectangle((x - 1.4, y - h / 2), 2.8, h, fc="white", ec=BLACK, lw=1.1))
        ax.text(x, y + h / 2 - 0.21, title, ha="center", va="center", fontsize=7.6, fontweight="bold")
        ax.plot([x - 1.4, x + 1.4], [y + h / 2 - 0.42, y + h / 2 - 0.42], color=BLACK, lw=0.8)
        for i, member in enumerate(members):
            ax.text(x - 1.3, y + h / 2 - 0.56 - 0.22 * i, member, fontsize=6.5, va="center", family="monospace")
        geometry[key] = (x, y, h)

    def link(child: str, parent: str, hollow: bool) -> None:
        cx, cy, ch = geometry[child]
        px, py, ph = geometry[parent]
        start, end = (cx, cy + ch / 2), (px, py - ph / 2)
        if parent == "Config":
            start, end = (cx + 1.4, cy), (px - 1.4, py)
        ax.annotate("", end, start, arrowprops={"arrowstyle": "-|>", "lw": 1, "fc": "white" if hollow else BLACK,
                                                "ec": BLACK, "mutation_scale": 14})

    for child, parent in [("Torch", "BaseEstimator"), ("Torch", "ClassifierMixin"), ("MLP", "Torch"),
                          ("VQC", "Torch")]:
        link(child, parent, hollow=True)
    link("Torch", "Config", hollow=False)
    ax.text(6.95, 4.12, "uses", fontsize=6.5)
    return _save(fig, path)


def deployment_diagram(path: str) -> str:
    """Deployment view: the user's workstation and the GitHub hosted services."""
    nodes = {
        "BR": (1.4, 4.6, 1.9, 0.55, "Web browser", "box"),
        "ST": (4.1, 4.6, 2.4, 0.62, "Streamlit server\nport 8501 (app.py)", "box"),
        "PY": (7.0, 4.6, 2.5, 0.62, "Python 3.12 runtime\nprediction service, models", "box"),
        "FS": (7.0, 3.1, 2.5, 0.62, "Local files: models,\ndataset, DVC cache", "store"),
        "ML": (3.4, 3.1, 2.6, 0.62, "MLflow UI port 5001\nmlflow.db", "box"),
        "DR": (9.7, 3.1, 1.9, 0.62, "DVC remote\n(local folder)", "store"),
        "GH": (3.4, 0.9, 2.4, 0.62, "GitHub repository", "store"),
        "CI": (7.4, 0.9, 2.8, 0.62, "GitHub Actions runner\nUbuntu, Python 3.12", "box"),
    }
    edges = [("BR", "ST", "HTTP", "solid"), ("ST", "PY", "", "solid"), ("PY", "FS", "", "solid"),
             ("ML", "FS", "", "dashed"), ("FS", "DR", "dvc push / pull", "dashed"),
             ("GH", "CI", "pull request / push", "solid")]
    return flowchart(nodes, edges, (10.9, 5.6), path,
                     frames=[(0.2, 2.45, 10.8, 5.45, "Workstation (macOS, Windows or Linux)"),
                             (0.2, 0.2, 10.8, 1.9, "GitHub (hosted)")])
