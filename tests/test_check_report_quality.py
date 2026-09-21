"""check_report_quality.py 的结构检查行为测试。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_report_quality.py"
SLOTS = ["tldr", "glossary", "background", "inputs", "model",
         "results", "process", "diagnosis", "limits", "next", "repro"]
FIGURES = (["concept_overview.png", "concept_arms.png"]
           + [f"data_{n}.png" for n in ("images", "cohort", "distributions")]
           + [f"result_{n}.png" for n in ("levels", "deltas")]
           + [f"diag_{n}.png" for n in ("embedding", "covariates", "errors")])
GLOSSARY = "| 代号 | 全称 | 说明 |\n|---|---|---|\n| CellPA | cell pattern analysis | 空间图模型 |\n"
LEDGER_HEADER = "claim\tvalue\tsource\trecomputed\tstatus\n"
PROSE_NUMBERS = "融合臂的患者级 AUC 是 0.634，对照是 0.783，队列共 52 名患者。\n"
LEDGER_ROWS = ("fusion arm AUC\t0.634\tsources/result_arm_summary.tsv\t0.634\tok\n"
               "comparator AUC\t0.783\tsources/result_arm_summary.tsv\t0.783\tok\n"
               "cohort size\t52\tsources/data_patients.tsv\t52\tok\n")


def load_module():
    spec = importlib.util.spec_from_file_location("check_report_quality", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_package(root: Path, figures=FIGURES, slots=SLOTS, glossary=GLOSSARY,
                  extra_figure_slot="results", ledger=LEDGER_HEADER + LEDGER_ROWS):
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "plot_data").mkdir(parents=True, exist_ok=True)
    for name in figures:
        (root / "figures" / name).write_bytes(b"png")
        if not name.startswith("concept_"):
            (root / "plot_data" / f"{Path(name).stem}.csv").write_text("a,b\n1,2\n")
    parts, index = ["# 报告标题\n"], 0
    for slot in slots:
        parts.append(f"<!-- slot:{slot} -->\n## {slot}\n")
        if slot == "glossary":
            parts.append(glossary)
        elif slot != "repro" and index < len(figures):
            parts.append(f"![说明](figures/{figures[index]})\n")
            index += 1
            if slot == extra_figure_slot and index < len(figures):
                parts.append(f"![说明](figures/{figures[index]})\n")
                index += 1
    parts.append(PROSE_NUMBERS)
    (root / "report.md").write_text("\n".join(parts), encoding="utf-8")
    if ledger is not None:
        (root / "review").mkdir(parents=True, exist_ok=True)
        (root / "review" / "number_checks.tsv").write_text(ledger, encoding="utf-8")
    return root


def check(root: Path, module, capsys) -> tuple[int, str]:
    sys.argv = ["check_report_quality.py", str(root)]
    code = module.main()
    return code, capsys.readouterr().out


def test_complete_package_passes(tmp_path, capsys):
    module = load_module()
    build_package(tmp_path)
    code, out = check(tmp_path, module, capsys)
    assert code == 0, out
    assert "RESULT PASS" in out


def test_missing_diagnostic_figures_fail(tmp_path, capsys):
    module = load_module()
    figures = [f for f in FIGURES if not f.startswith("diag_")]
    build_package(tmp_path, figures=figures)
    code, out = check(tmp_path, module, capsys)
    assert code == 1
    assert "figure family 'diag_*'" in out


def test_missing_slot_fails(tmp_path, capsys):
    module = load_module()
    build_package(tmp_path, slots=[s for s in SLOTS if s != "process"])
    code, out = check(tmp_path, module, capsys)
    assert code == 1
    assert "missing section slots: process" in out


def test_section_without_visual_fails(tmp_path, capsys):
    module = load_module()
    build_package(tmp_path)
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines()
             if not (ln.startswith("![说明](figures/diag_errors.png)"))]
    (tmp_path / "report.md").write_text("\n".join(lines), encoding="utf-8")
    code, out = check(tmp_path, module, capsys)
    assert code == 1
    assert "neither a figure nor a table" in out


def test_measured_figure_needs_plot_data(tmp_path, capsys):
    module = load_module()
    build_package(tmp_path)
    (tmp_path / "plot_data" / "result_levels.csv").unlink()
    code, out = check(tmp_path, module, capsys)
    assert code == 1
    assert "without plot data" in out


def test_uncovered_code_is_a_note_not_a_failure(tmp_path, capsys):
    module = load_module()
    build_package(tmp_path)
    path = tmp_path / "report.md"
    path.write_text(path.read_text(encoding="utf-8") + "\n融合臂 B5 的结果见上。\n",
                    encoding="utf-8")
    code, out = check(tmp_path, module, capsys)
    assert code == 0
    assert "NOTE" in out and "B5" in out


def test_missing_number_ledger_fails(tmp_path, capsys):
    module = load_module()
    build_package(tmp_path, ledger=None)
    code, out = check(tmp_path, module, capsys)
    assert code == 1
    assert "review/number_checks.tsv" in out


def test_mismatched_number_fails(tmp_path, capsys):
    module = load_module()
    ledger = LEDGER_HEADER + LEDGER_ROWS.replace(
        "comparator AUC\t0.783\tsources/result_arm_summary.tsv\t0.783\tok",
        "comparator AUC\t0.783\tsources/result_arm_summary.tsv\t0.785\tmismatch")
    build_package(tmp_path, ledger=ledger)
    code, out = check(tmp_path, module, capsys)
    assert code == 1
    assert "unresolved rows" in out


def test_low_number_coverage_fails(tmp_path, capsys):
    module = load_module()
    build_package(tmp_path, ledger=LEDGER_HEADER + "cohort size\t52\tsrc\t52\tok\n")
    code, out = check(tmp_path, module, capsys)
    assert code == 1
    assert "of the numbers in the prose appear in" in out


def test_unchecked_rows_are_a_note(tmp_path, capsys):
    module = load_module()
    ledger = LEDGER_HEADER + LEDGER_ROWS.replace(
        "cohort size\t52\tsources/data_patients.tsv\t52\tok",
        "cohort size\t52\tsources/data_patients.tsv\t\tunchecked")
    build_package(tmp_path, ledger=ledger)
    code, out = check(tmp_path, module, capsys)
    assert code == 0
    assert "still marked unchecked" in out
