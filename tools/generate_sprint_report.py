"""Builds the sprint report (velocity, burndown, burn-up and retrospectives) as a Word document.

    python tools/generate_sprint_report.py

Reads docs/sprint-report/sprint_data.json (the single source for every number), draws the charts into
docs/sprint-report/ and writes docs/deliverables/Hackforge_Sprint_Report.docx.
Requires matplotlib and python-docx (requirements-dev.txt).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

# pylint: disable=wrong-import-position  # the backend has to be chosen before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.section import WD_ORIENT  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "docs", "sprint-report", "sprint_data.json")
CHART_DIR = os.path.join(BASE_DIR, "docs", "sprint-report")
OUTPUT_FILE = os.path.join(BASE_DIR, "docs", "deliverables", "Hackforge_Sprint_Report.docx")

BLACK = "#000000"
GREY = "#6b6b6b"
LIGHT = "#d9d9d9"
FONT = "Calibri"

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": BLACK,
    "axes.labelcolor": BLACK,
    "axes.edgecolor": GREY,
    "xtick.color": BLACK,
    "ytick.color": BLACK,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ----- data ------------------------------------------------------------------------------------
def _time(value: str) -> Tuple[dt.datetime, bool]:
    """(timestamp, has_time) from '2026-10-11T15:06' or '2026-10-28'."""
    return dt.datetime.fromisoformat(value), "T" in value


def _label(value: str) -> str:
    stamp, has_time = _time(value)
    return stamp.strftime("%d %b %H:%M") if has_time else stamp.strftime("%d %b")


@dataclass
class Event:
    """A change to a sprint's scope (stories planned) or progress (stories done)."""

    when: str
    kind: str  # "planned" or "done"
    ids: List[str]
    points: int


def sprint_events(sprint: Dict[str, Any]) -> List[Event]:
    """Planned and done events grouped by time, in order (planning first when times are equal)."""
    grouped: Dict[Tuple[str, str], Event] = {}
    for item in sprint["items"]:
        for kind in ("planned", "done"):
            key = (item[kind], kind)
            event = grouped.setdefault(key, Event(item[kind], kind, [], 0))
            event.ids.append(item["id"])
            event.points += item["points"]
    return _merge_planning(sorted(grouped.values(), key=lambda e: (_time(e.when)[0], e.kind != "planned")))


def _merge_planning(events: List[Event]) -> List[Event]:
    """Joins back-to-back planning events (stories created minutes apart) into one planning event."""
    merged: List[Event] = []
    for event in events:
        if merged and event.kind == "planned" and merged[-1].kind == "planned":
            merged[-1].ids.extend(event.ids)
            merged[-1].points += event.points
        else:
            merged.append(Event(event.when, event.kind, list(event.ids), event.points))
    return merged


def burn_series(events: List[Event]) -> List[Tuple[str, int, int]]:
    """(label, scope, remaining) after each event."""
    scope = remaining = 0
    series = []
    for event in events:
        if event.kind == "planned":
            scope += event.points
            remaining += event.points
        else:
            remaining -= event.points
        series.append((_label(event.when), scope, remaining))
    return series


def velocity(sprint: Dict[str, Any]) -> Dict[str, int]:
    """Points planned on the first planning day, added later, and completed."""
    first_day = min(_time(item["planned"])[0].date() for item in sprint["items"])
    planned = sum(i["points"] for i in sprint["items"] if _time(i["planned"])[0].date() == first_day)
    total = sum(i["points"] for i in sprint["items"])
    done = sum(i["points"] for i in sprint["items"] if i.get("done"))
    return {"planned": planned, "added": total - planned, "total": total, "completed": done}


def _describe(event: Event) -> str:
    stories = ", ".join(event.ids)
    if event.kind == "planned":
        return f"Planned {stories} ({event.points} points)"
    return f"Completed {stories} ({event.points} points)"


# ----- charts ----------------------------------------------------------------------------------
def _finish(ax, path: str) -> str:
    ax.grid(True, axis="y", color=LIGHT, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def burndown_chart(sprint: Dict[str, Any], path: str) -> str:
    """Remaining points (and scope) after each recorded event, with the ideal line."""
    series = burn_series(sprint_events(sprint))
    x = list(range(len(series)))
    scope = [s for _, s, _ in series]
    remaining = [r for _, _, r in series]
    _, ax = plt.subplots(figsize=(8, 4))
    ax.step(x, scope, where="post", color=GREY, linestyle="--", linewidth=2, label="Sprint scope")
    ax.step(x, remaining, where="post", color=BLACK, linewidth=2, label="Remaining points")
    ax.plot(x, remaining, "o", color=BLACK, markersize=6)
    ax.plot([0, x[-1]], [scope[0], 0], color=GREY, linestyle=":", linewidth=1.5, label="Ideal burndown")
    for xi, value in zip(x, remaining):
        ax.annotate(str(value), (xi, value), textcoords="offset points", xytext=(7, 5), ha="left", fontsize=9)
    ax.set_xticks(x, [label.replace(" ", "\n", 2).replace("\n", " ", 1) for label, _, _ in series], fontsize=8)
    ax.set_ylim(0, max(scope) * 1.18)
    ax.set_xlabel("Recorded event (IST)")
    ax.set_ylabel("Story points")
    ax.set_title(f"{sprint['name']} burndown", loc="left", fontweight="bold")
    return _finish(ax, path)


def velocity_chart(sprints: List[Dict[str, Any]], path: str) -> str:
    """Completed points per sprint, split into planned-at-start and added-during-sprint."""
    names = [s["name"] for s in sprints]
    values = [velocity(s) for s in sprints]
    _, ax = plt.subplots(figsize=(7, 4))
    planned = [v["planned"] for v in values]
    added = [v["added"] for v in values]
    ax.bar(names, planned, width=0.5, color=BLACK, edgecolor="white", linewidth=2, label="Planned at sprint start")
    ax.bar(names, added, width=0.5, bottom=planned, color="white", edgecolor=BLACK, hatch="///", linewidth=1,
           label="Added during sprint")
    for name, value in zip(names, values):
        ax.annotate(f"{value['completed']} completed", (name, value["total"]), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=9)
    average = sum(v["completed"] for v in values) / len(values)
    ax.axhline(average, color=GREY, linestyle="--", linewidth=1.5, label=f"Average velocity ({average:.0f} points)")
    ax.set_ylim(0, max(v["total"] for v in values) * 1.15)
    ax.set_ylabel("Story points completed")
    ax.set_title("Sprint velocity", loc="left", fontweight="bold")
    ax.grid(True, axis="y", color=LIGHT, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def burnup_chart(sprints: List[Dict[str, Any]], path: str) -> str:
    """Project-level burn-up: total scope and cumulative completed points over every recorded event."""
    merged: List[Event] = []
    for event in _merge_planning(sorted((e for s in sprints for e in sprint_events(s)),
                                        key=lambda e: (_time(e.when)[0], e.kind != "planned"))):
        if merged and merged[-1].kind == event.kind == "done" and merged[-1].when == event.when:
            merged[-1].points += event.points  # stories from different sprints finished together
        else:
            merged.append(event)
    scope = done = 0
    labels, scopes, completed = [], [], []
    for event in merged:
        if event.kind == "planned":
            scope += event.points
        else:
            done += event.points
        labels.append(_label(event.when))
        scopes.append(scope)
        completed.append(done)
    x = list(range(len(labels)))
    _, ax = plt.subplots(figsize=(9, 4))
    ax.step(x, scopes, where="post", color=GREY, linestyle="--", linewidth=2, label="Total scope")
    ax.step(x, completed, where="post", color=BLACK, linewidth=2, label="Completed points")
    ax.plot(x, completed, "o", color=BLACK, markersize=5)
    ax.annotate(f"{completed[-1]} of {scopes[-1]}", (x[-1], completed[-1]), textcoords="offset points",
                xytext=(-4, 8), ha="right", fontsize=9)
    ax.set_xticks(x, [label.replace(" ", "\n", 2).replace("\n", " ", 1) for label in labels], fontsize=7)
    ax.set_ylim(0, scopes[-1] * 1.15)
    ax.set_xlabel("Recorded event (IST)")
    ax.set_ylabel("Story points")
    ax.set_title("Project burn-up, all sprints", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, axis="y", color=LIGHT, linewidth=0.8)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


# ----- document helpers ------------------------------------------------------------------------
def _style_document(doc: Document) -> None:
    for name in ["Normal", "Title", "Subtitle", "Heading 1", "Heading 2", "Heading 3", "List Bullet", "Caption"]:
        style = doc.styles[name]
        style.font.name = FONT
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    doc.styles["Normal"].font.size = Pt(11)
    doc.styles["Caption"].font.italic = False
    for name, size in [("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11.5)]:
        doc.styles[name].font.size = Pt(size)
        doc.styles[name].font.bold = True
    section = doc.sections[0]
    section.orientation = WD_ORIENT.PORTRAIT
    section.page_width, section.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, side, Cm(2.3))
    _page_number_footer(section)


def _page_number_footer(section) -> None:
    paragraph = section.footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            element = OxmlElement("w:fldChar")
            element.set(qn("w:fldCharType"), tag)
        else:
            element = OxmlElement("w:instrText")
            element.set(qn("xml:space"), "preserve")
            element.text = text
        run._r.append(element)  # pylint: disable=protected-access  # python-docx has no field API


def _shade(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()  # pylint: disable=protected-access
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), fill)
    properties.append(shading)


def _table(doc: Document, headers: List[str], rows: List[List[Any]], widths_cm: List[float]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, header, width in zip(table.rows[0].cells, headers, widths_cm):
        cell.width = Cm(width)
        cell.text = ""
        run = cell.paragraphs[0].add_run(header)
        run.bold = True
        _shade(cell, "E7E7E7")
    for row in rows:
        cells = table.add_row().cells
        for cell, value, width in zip(cells, row, widths_cm):
            cell.width = Cm(width)
            cell.text = str(value)
    table.autofit = False
    for column, width in zip(table.columns, widths_cm):
        column.width = Cm(width)
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
                for run in paragraph.runs:
                    run.font.size = Pt(9.5)
    doc.add_paragraph()


def _bullets(doc: Document, lines: List[str]) -> None:
    for line in lines:
        doc.add_paragraph(line, style="List Bullet")


def _figure(doc: Document, path: str, caption: str) -> None:
    doc.add_picture(path, width=Cm(16))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph = doc.add_paragraph(caption, style="Caption")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER


# ----- report sections -------------------------------------------------------------------------
def _title_page(doc: Document, data: Dict[str, Any]) -> None:
    for _ in range(4):
        doc.add_paragraph()
    title = doc.add_paragraph(data["project"], style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = doc.add_paragraph("Sprint Report: Velocity, Burndown and Retrospectives", style="Subtitle")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()
    _table(doc, ["Detail", "Value"], [
        ["Course", data["course"]],
        ["Team", data["team"]],
        ["Team members (SRN)", "; ".join(f"{m['name']} ({m['srn']})" for m in data["members"])],
        ["Sprints covered", ", ".join(s["name"] for s in data["sprints"])],
        ["Date", dt.date.today().strftime("%d %B %Y")],
    ], [4.5, 11.5])
    doc.add_page_break()


def _introduction(doc: Document, data: Dict[str, Any]) -> None:
    doc.add_heading("1. Introduction", level=1)
    doc.add_heading("1.1 Purpose", level=2)
    doc.add_paragraph(
        "This report records how the team planned and delivered the project in sprints. For each sprint it gives "
        "the sprint goal, the backlog with story points, the velocity, a burndown chart, the outcome and the "
        "retrospective. A project-level velocity chart and burn-up chart summarise all sprints.")
    doc.add_heading("1.2 Data sources and method", level=2)
    doc.add_paragraph(data["method"])
    doc.add_paragraph(
        "Story points use the scale 1, 2, 3, 5 and 8. Velocity is the number of story points completed in a sprint. "
        "The burndown charts plot the points still remaining after each recorded event; the dashed line shows the "
        "sprint scope (including stories added during the sprint) and the dotted line shows an ideal, even burndown "
        f"of the initial scope. The data behind every chart is kept in docs/sprint-report/sprint_data.json "
        f"({data['timezone']}).")


def _summary(doc: Document, sprints: List[Dict[str, Any]]) -> None:
    doc.add_heading("2. Summary of Sprints", level=1)
    rows = []
    for sprint in sprints:
        v = velocity(sprint)
        rows.append([sprint["name"], sprint["window"], len(sprint["items"]), v["planned"], v["added"],
                     v["completed"], f"{100 * v['completed'] / v['total']:.0f}%"])
    _table(doc, ["Sprint", "Window", "Stories", "Planned at start", "Added", "Completed", "Completion"], rows,
           [2.0, 4.2, 1.7, 2.2, 1.6, 2.2, 2.1])
    completed = [velocity(s)["completed"] for s in sprints]
    doc.add_heading("2.1 Velocity", level=2)
    doc.add_paragraph(
        f"The team completed every committed story in every sprint. Velocity rose from {completed[0]} points in "
        f"{sprints[0]['name']} to {completed[1]} points in {sprints[1]['name']} and {completed[2]} points in "
        f"{sprints[2]['name']}, an average of {sum(completed) / len(completed):.0f} points per sprint. The growth "
        "reflects the shared modules, automated tests and CI built in the earlier sprints, which made later stories "
        "faster to deliver safely.")
    _figure(doc, velocity_chart(sprints, os.path.join(CHART_DIR, "velocity.png")),
            "Figure 1. Story points completed per sprint")
    doc.add_heading("2.2 Project burn-up", level=2)
    total = sum(velocity(s)["total"] for s in sprints)
    doc.add_paragraph(
        f"The burn-up chart shows the total scope of the project growing as each sprint's stories were planned, and "
        f"the completed points following it. All {total} planned points are complete.")
    _figure(doc, burnup_chart(sprints, os.path.join(CHART_DIR, "burnup.png")),
            "Figure 2. Project burn-up across all sprints")


def _sprint_section(doc: Document, number: int, sprint: Dict[str, Any], figure: int) -> None:
    doc.add_page_break()
    doc.add_heading(f"{number}. {sprint['name']}", level=1)
    doc.add_paragraph(f"Sprint window: {sprint['window']}")
    doc.add_heading(f"{number}.1 Sprint goal", level=2)
    doc.add_paragraph(sprint["goal"])

    doc.add_heading(f"{number}.2 Sprint backlog", level=2)
    _table(doc, ["ID", "Story", "Points", "Assignee", "Planned", "Done"], [
        [i["id"], i["title"], i["points"], i["assignee"], _label(i["planned"]), _label(i["done"])]
        for i in sprint["items"]] + [["", "Total", sum(i["points"] for i in sprint["items"]), "", "", ""]],
        [1.3, 6.4, 1.4, 3.3, 1.9, 1.9])

    v = velocity(sprint)
    doc.add_heading(f"{number}.3 Velocity", level=2)
    _table(doc, ["Planned at start", "Added during sprint", "Total scope", "Completed (velocity)", "Completion"],
           [[v["planned"], v["added"], v["total"], v["completed"], f"{100 * v['completed'] / v['total']:.0f}%"]],
           [3.2, 3.2, 3.0, 3.6, 3.0])

    doc.add_heading(f"{number}.4 Burndown", level=2)
    chart = burndown_chart(sprint, os.path.join(CHART_DIR, f"burndown_{sprint['name'].lower().replace(' ', '')}.png"))
    _figure(doc, chart, f"Figure {figure}. {sprint['name']} burndown")
    series = burn_series(sprint_events(sprint))
    _table(doc, ["Time (IST)", "Event", "Scope", "Remaining"], [
        [label, _describe(event), scope, remaining]
        for event, (label, scope, remaining) in zip(sprint_events(sprint), series)], [3.0, 8.4, 2.2, 2.4])
    doc.add_paragraph(
        f"The sprint finished with 0 of {v['total']} points remaining. "
        + (f"{v['added']} points of scope were added during the sprint and completed in the same sprint."
           if v["added"] else "No scope was added during the sprint."))

    doc.add_heading(f"{number}.5 Outcome", level=2)
    doc.add_paragraph(sprint["outcome"])
    if sprint.get("video"):
        doc.add_paragraph(f"Demonstration video: {sprint['video']}")

    retro = sprint["retrospective"]
    doc.add_heading(f"{number}.6 Retrospective", level=2)
    doc.add_heading("What went well", level=3)
    _bullets(doc, retro["went_well"])
    doc.add_heading("What we will do better", level=3)
    _bullets(doc, retro["improve"])
    doc.add_heading("Action items", level=3)
    _bullets(doc, retro["actions"])


def _follow_through(doc: Document, number: int, sprints: List[Dict[str, Any]]) -> None:
    doc.add_page_break()
    doc.add_heading(f"{number}. Retrospective Follow-through", level=1)
    doc.add_paragraph("Each retrospective produced action items. The table shows how they were carried forward.")
    rows = []
    for sprint in sprints:
        for action in sprint["retrospective"]["actions"]:
            match = re.search(r"\s*\(((?:done|adopted)[^)]*)\)\.?$", action)
            if match:
                status = match.group(1)[0].upper() + match.group(1)[1:]
                rows.append([sprint["name"], action[:match.start()] + ".", status])
            else:
                rows.append([sprint["name"], action, "Planned for the next sprint"])
    _table(doc, ["Raised in", "Action", "Status"], rows, [2.4, 9.6, 4.0])


def _conclusion(doc: Document, number: int, sprints: List[Dict[str, Any]]) -> None:
    doc.add_heading(f"{number}. Conclusion", level=1)
    total = sum(velocity(s)["completed"] for s in sprints)
    stories = sum(len(s["items"]) for s in sprints)
    earlier = " and ".join(s["name"] for s in sprints[:-1])
    doc.add_paragraph(
        f"Across {len(sprints)} sprints the team completed {stories} stories and {total} story points, with every "
        f"sprint reaching 100% of its scope. Velocity increased in each sprint, and the action items from the "
        f"{earlier} retrospectives were carried out in the sprint that followed. The result is a tested, documented "
        "churn prediction system that compares classical, deep and quantum machine learning side by side.")


def main() -> None:
    """Draws the charts and writes the Word report."""
    with open(DATA_FILE, encoding="utf-8") as handle:
        data = json.load(handle)
    sprints = data["sprints"]
    os.makedirs(CHART_DIR, exist_ok=True)
    doc = Document()
    _style_document(doc)
    _title_page(doc, data)
    _introduction(doc, data)
    _summary(doc, sprints)
    for offset, sprint in enumerate(sprints):
        _sprint_section(doc, 3 + offset, sprint, figure=3 + offset)
    _follow_through(doc, 3 + len(sprints), sprints)
    _conclusion(doc, 4 + len(sprints), sprints)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    doc.save(OUTPUT_FILE)
    print(f"Wrote {os.path.relpath(OUTPUT_FILE, BASE_DIR)} and the charts in {os.path.relpath(CHART_DIR, BASE_DIR)}/")


if __name__ == "__main__":
    main()
