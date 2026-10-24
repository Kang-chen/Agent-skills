#!/usr/bin/env python3
"""检查报告包是否满足 section slot、figure slate 与名词表的结构性要求。

用法：
    python3 check_report_quality.py <output_root> [--report report.md]

只做结构检查：章节齐全与顺序、图的数量与家族覆盖、每节是否有图或表、
图文件与 plot_data 是否存在、名词表是否覆盖正文里出现的内部代号。
它不判断科学正确性，也不判断图好不好看。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REQUIRED_SLOTS = [
    "tldr", "glossary", "background", "inputs", "model",
    "results", "process", "diagnosis", "limits", "next", "repro",
]
# 不强制要求配图的 slot
VISUAL_EXEMPT_SLOTS = {"glossary", "repro"}

FAMILY_MINIMUM = {"concept_": 2, "data_": 3, "result_": 2, "diag_": 3}
TOTAL_FIGURE_MINIMUM = 10

SLOT_RE = re.compile(r"<!--\s*slot:([a-z_]+)\s*-->")
FIGURE_RE = re.compile(r"!\[[^\]]*\]\(\s*<?([^)>\s]+)")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
DATE_RE = re.compile(r"\b\d{4}[-/年]\d{1,2}[-/月]\d{1,2}\b|\b(19|20)\d{2}\b")
NUMBER_RE = re.compile(r"(?<![\w.])(\d+\.\d{2,}|\d{2,})(?![\w.])")
NUMBER_LEDGER = "review/number_checks.tsv"
LEDGER_COLUMNS = {"claim", "value", "source", "recomputed", "status"}
ALLOWED_STATUS = {"ok", "mismatch", "unchecked"}
COVERAGE_MINIMUM = 0.70
# 内部代号：字母+数字的短标识（B5、M4、F0、v7、T3），或全大写缩写（OOF、AUC、PCA）
CODE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_./-])([A-Za-z]{1,4}\d{1,3}(?:-[A-Za-z]+)?|[A-Z]{2,6})(?![A-Za-z0-9_./-])")
# 常见到不需要解释、或不是内部代号的词
TOKEN_ALLOWLIST = {
    "AUC", "ROC", "CI", "SD", "PD", "PR", "CR", "GPU", "CPU", "RAM", "EFS",
    "CSV", "TSV", "JSON", "YAML", "PNG", "SVG", "PDF", "HTML", "MD", "UTC",
    "ID", "IDS", "URL", "API", "NA", "OK", "TODO", "AND", "OR", "NOT",
}


def strip_code(text: str) -> str:
    out, in_fence = [], False
    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        out.append(re.sub(r"`[^`]*`", " ", line))
    return "\n".join(out)


def split_slots(text: str) -> dict[str, str]:
    marks = [(m.group(1), m.start(), m.end()) for m in SLOT_RE.finditer(text)]
    sections: dict[str, str] = {}
    for i, (name, _, end) in enumerate(marks):
        stop = marks[i + 1][1] if i + 1 < len(marks) else len(text)
        sections.setdefault(name, text[end:stop])
    return sections


def has_table(text: str) -> bool:
    return any(TABLE_SEP_RE.match(line) for line in text.splitlines())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--report", default="report.md")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    report_path = root / args.report
    if not report_path.is_file():
        print(f"FAIL  report not found: {report_path}")
        return 1
    raw = report_path.read_text(encoding="utf-8")
    body = strip_code(raw)

    failures: list[str] = []
    notes: list[str] = []

    # 1) slot 齐全且顺序正确
    found = [m.group(1) for m in SLOT_RE.finditer(raw)]
    seen, ordered = set(), []
    for name in found:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    missing = [s for s in REQUIRED_SLOTS if s not in seen]
    if missing:
        failures.append(f"missing section slots: {', '.join(missing)}")
    present_required = [s for s in ordered if s in REQUIRED_SLOTS]
    expected_order = [s for s in REQUIRED_SLOTS if s in present_required]
    if present_required != expected_order:
        failures.append(f"slot order is {present_required}, expected {expected_order}")
    print(f"slots      : {len(present_required)}/{len(REQUIRED_SLOTS)} present"
          f"{'' if not missing else ' (missing: ' + ', '.join(missing) + ')'}")

    # 2) 图：数量、家族、文件存在、plot_data 配套
    figures = [f for f in FIGURE_RE.findall(body) if not f.startswith(("http://", "https://"))]
    unique_figs: list[str] = []
    for f in figures:
        if f not in unique_figs:
            unique_figs.append(f)
    counts = {p: 0 for p in FAMILY_MINIMUM}
    unprefixed: list[str] = []
    for f in unique_figs:
        stem = Path(f).name
        for prefix in FAMILY_MINIMUM:
            if stem.startswith(prefix):
                counts[prefix] += 1
                break
        else:
            unprefixed.append(f)
    print("figures    : " + ", ".join(
        f"{p.rstrip('_')}={counts[p]}/{FAMILY_MINIMUM[p]}" for p in FAMILY_MINIMUM)
        + f", total={len(unique_figs)}/{TOTAL_FIGURE_MINIMUM}")
    for prefix, minimum in FAMILY_MINIMUM.items():
        if counts[prefix] < minimum:
            failures.append(
                f"figure family '{prefix}*' has {counts[prefix]}, needs {minimum}")
    if len(unique_figs) < TOTAL_FIGURE_MINIMUM:
        failures.append(
            f"only {len(unique_figs)} main figures, needs {TOTAL_FIGURE_MINIMUM}")
    if unprefixed:
        failures.append("figures without a family prefix (concept_/data_/result_/diag_): "
                        + ", ".join(unprefixed[:8]))

    missing_files, missing_plot_data = [], []
    for f in unique_figs:
        path = (report_path.parent / f).resolve()
        if not path.is_file():
            missing_files.append(f)
            continue
        if Path(f).name.startswith("concept_"):
            continue
        stem = path.stem
        if not any((root / "plot_data" / f"{stem}{ext}").is_file()
                   for ext in (".csv", ".tsv", ".json", ".parquet")):
            missing_plot_data.append(f"plot_data/{stem}.(csv|tsv|json|parquet)")
    if missing_files:
        failures.append("figure files not found: " + ", ".join(missing_files[:8]))
    if missing_plot_data:
        failures.append("measured figures without plot data: "
                        + ", ".join(missing_plot_data[:8]))

    # 3) 每个 slot 至少一图或一表
    sections = split_slots(strip_code(raw))
    bare = []
    for slot, text in sections.items():
        if slot in VISUAL_EXEMPT_SLOTS or slot not in REQUIRED_SLOTS:
            continue
        if not FIGURE_RE.search(text) and not has_table(text):
            bare.append(slot)
    if bare:
        failures.append("sections with neither a figure nor a table: " + ", ".join(bare))

    # 4) 名词表
    glossary = sections.get("glossary", "")
    if not has_table(glossary):
        failures.append("the glossary slot has no table")
    glossary_text = glossary
    body_tokens = {t for t in CODE_TOKEN_RE.findall(body) if t not in TOKEN_ALLOWLIST}
    uncovered = sorted(t for t in body_tokens if t not in glossary_text)
    if uncovered:
        notes.append("codes used in the body but absent from the glossary table: "
                     + ", ".join(uncovered[:20])
                     + (" ..." if len(uncovered) > 20 else ""))

    # 5) 数字台账：每个数字都回源重算过
    ledger_path = root / NUMBER_LEDGER
    checked_values: set[str] = set()
    if not ledger_path.is_file():
        failures.append(f"missing {NUMBER_LEDGER}: every number the prose states "
                        "must be re-derived from its source and logged there")
        print("numbers    : no ledger")
    else:
        rows = [line.split("\t") for line in
                ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        header = [c.strip().lower() for c in rows[0]] if rows else []
        body_rows = rows[1:]
        missing_cols = LEDGER_COLUMNS - set(header)
        if missing_cols:
            failures.append(f"{NUMBER_LEDGER} is missing columns: "
                            + ", ".join(sorted(missing_cols)))
        else:
            value_i, status_i = header.index("value"), header.index("status")
            mismatches, unchecked = [], 0
            for row in body_rows:
                if len(row) <= max(value_i, status_i):
                    continue
                status = row[status_i].strip().lower()
                checked_values.add(row[value_i].strip())
                if status == "mismatch":
                    mismatches.append(row[value_i].strip())
                elif status == "unchecked":
                    unchecked += 1
                elif status not in ALLOWED_STATUS:
                    mismatches.append(f"{row[value_i].strip()} (status '{status}')")
            if mismatches:
                failures.append(f"{NUMBER_LEDGER} has unresolved rows: "
                                + ", ".join(mismatches[:8]))
            if unchecked:
                notes.append(f"{NUMBER_LEDGER} has {unchecked} rows still marked unchecked")
        ledger_text = ledger_path.read_text(encoding="utf-8")
        prose_numbers = {m.group(1) for m in NUMBER_RE.finditer(DATE_RE.sub(" ", body))}
        covered = {n for n in prose_numbers if n in ledger_text}
        coverage = len(covered) / len(prose_numbers) if prose_numbers else 1.0
        print(f"numbers    : {len(covered)}/{len(prose_numbers)} prose numbers in the "
              f"ledger ({coverage:.0%}, need {COVERAGE_MINIMUM:.0%}), "
              f"{len(body_rows)} ledger rows")
        if coverage < COVERAGE_MINIMUM:
            uncovered_numbers = sorted(prose_numbers - covered)[:10]
            failures.append(f"only {coverage:.0%} of the numbers in the prose appear in "
                            f"{NUMBER_LEDGER}; unlogged examples: "
                            + ", ".join(uncovered_numbers))

    print(f"sections   : {len(sections)} slot bodies scanned, "
          f"{len(bare)} without a visual")
    print(f"glossary   : {'table found' if has_table(glossary) else 'NO TABLE'}, "
          f"{len(uncovered)} uncovered codes")

    for note in notes:
        print(f"NOTE  {note}")
    for failure in failures:
        print(f"FAIL  {failure}")
    if failures:
        print(f"\nRESULT FAIL ({len(failures)} structural problems)")
        return 1
    print("\nRESULT PASS (structure only; it says nothing about the science)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
