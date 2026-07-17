#!/usr/bin/env python3
"""Validate structural traceability for a canonical Markdown report package."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Iterable, Sequence

import yaml


EVIDENCE_COLUMNS = (
    "record_id",
    "record_type",
    "section_question",
    "analysis_role",
    "report_critical",
    "scope_roles",
    "scope_role_reasons",
    "statement_role",
    "message_or_claim",
    "supported_by_ids",
    "source_ids",
    "evidence_origin",
    "generation_entry",
    "artifact_path",
    "presentation_role",
    "claim_type",
    "claim_level",
    "support_status",
    "interpretation_status",
    "decision_status",
    "hypothesis_status",
    "criterion",
    "criterion_locator",
    "boundary",
    "alternative",
    "comparison",
    "analysis_unit",
    "sample_n",
    "uncertainty",
    "encoding",
    "final_size",
    "save_target",
    "report_locator",
    "qa_status",
)

REQUIRED_MANIFEST_BLOCKS = frozenset(
    {
        "core",
        "scope",
        "claim_policy",
        "preflight",
        "sources",
        "reproducibility",
        "traceability",
        "validation",
        "planned_vs_actual",
        "derived_format_handoff",
        "critical_decisions",
        "overrides",
        "status",
    }
)

TERMINAL_STATUSES = frozenset({"final", "complete_with_limitations"})
HIGH_RISK_CLAIMS = frozenset(
    {"causal", "mechanistic", "clinical", "generalization", "recommendation"}
)
EMPTY_VALUES = frozenset({"", "not_applicable", "not_available"})
EXPLICIT_REPORT_ANCHOR = re.compile(
    r'(?m)^[ \t]*<a[ \t]+id="([a-z0-9]+(?:-[a-z0-9]+)*)"></a>[ \t]*$'
)
REFERENCE_LINK = re.compile(
    r"\[(?P<number>[1-9][0-9]*)\]\(#(?P<anchor>ref-[a-z0-9]+(?:-[a-z0-9]+)*)\)"
)
REFERENCE_ENTRY = re.compile(
    r'(?m)^<a id="(?P<anchor>ref-[a-z0-9]+(?:-[a-z0-9]+)*)"></a>[ \t]*\n'
    r'(?P<number>[1-9][0-9]*)\.[ \t]+(?P<text>\S.*)$'
)
INTERNAL_EVIDENCE_MARKER = re.compile(r"\[E:[^\]\n]+\]")
SOURCE_REPORT_REFERENCE = re.compile(
    r"report\.md#ref-[a-z0-9]+(?:-[a-z0-9]+)*"
)
FENCED_CODE_START = re.compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})(?P<info>.*)$"
)
BLOCKQUOTE_CONTAINER = re.compile(r"^[ ]{0,3}>[ \t]?")
LIST_CONTAINER = re.compile(
    r"^[ ]{0,3}(?:[-+*]|[0-9]{1,9}[.)])(?:[ \t]+|$)"
)
MARKDOWN_HEADING = re.compile(r"(?m)^[ ]{0,3}#{1,6}[ \t]+\S.*$")
SETEXT_UNDERLINE = re.compile(r"^[ ]{0,3}(?:=+|-+)[ \t]*$")


@dataclass(frozen=True)
class ValidationIssue:
    """One deterministic validation failure."""

    code: str
    message: str
    locator: str
    blocking: bool = False


@dataclass(frozen=True)
class ValidationResult:
    """Prospective terminal outcome without a scientific-correctness claim."""

    passed: bool
    candidate_status: str
    next_status: str
    issues: tuple[ValidationIssue, ...]

    @property
    def issue_codes(self) -> tuple[str, ...]:
        """Return stable issue codes in validation order."""

        return tuple(issue.code for issue in self.issues)


def _issue(
    code: str,
    message: str,
    locator: str,
    *,
    blocking: bool = False,
) -> ValidationIssue:
    return ValidationIssue(code, message, locator, blocking)


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def _read_evidence(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        header = tuple(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    return header, rows


def _split_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, str):
        return ()
    return tuple(
        sorted(
            {
                item.strip()
                for item in value.split(";")
                if item.strip() and item.strip() != "not_applicable"
            }
        )
    )


def _manifest_ids(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(sorted({str(item) for item in value if str(item)}))
    if isinstance(value, str):
        return _split_ids(value)
    return ()


def _safe_relative(locator: str) -> bool:
    path = PurePosixPath(locator)
    return bool(locator) and not path.is_absolute() and ".." not in path.parts


def _resolved_path(package_root: Path, locator: str) -> Path | None:
    if not _safe_relative(locator):
        return None
    candidate = (package_root / Path(locator)).resolve()
    try:
        candidate.relative_to(package_root.resolve())
    except ValueError:
        return None
    return candidate


def _validate_manifest_shape(manifest: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if manifest.get("schema_version") != 2:
        issues.append(
            _issue(
                "manifest.schema_version_invalid",
                "Reader-facing reference reports require schema_version 2",
                "manifest:schema_version",
            )
        )
    for block in sorted(REQUIRED_MANIFEST_BLOCKS - set(manifest)):
        issues.append(
            _issue(
                "manifest.required_block_missing",
                f"Required manifest block is missing: {block}",
                f"manifest:{block}",
            )
        )

    core = manifest.get("core")
    language = core.get("report_language") if isinstance(core, dict) else None
    if not isinstance(language, str) or language in EMPTY_VALUES | {"undetermined"}:
        issues.append(
            _issue(
                "language.scalar_required",
                "core.report_language must be one resolved scalar",
                "manifest:core.report_language",
            )
        )
    if isinstance(core, dict) and "report_languages" in core:
        issues.append(
            _issue(
                "language.parallel_values_forbidden",
                "Parallel report languages are outside this run",
                "manifest:core.report_languages",
            )
        )

    sources = manifest.get("sources")
    if not isinstance(sources, list):
        issues.append(
            _issue(
                "source.list_required",
                "Manifest sources must be a list",
                "manifest:sources",
            )
        )
    traceability = manifest.get("traceability")
    expected_traceability = {
        "report_locator_format": "report.md#stable-anchor",
        "explicit_anchor_syntax": '<a id="stable-anchor"></a>',
        "require_explicit_anchors": True,
        "reference_link_format": "[N](#ref-stable-anchor)",
        "reference_anchor_syntax": '<a id="ref-stable-anchor"></a>',
        "reference_section_locator": "report.md#references",
        "require_source_references": True,
    }
    if traceability != expected_traceability:
        issues.append(
            _issue(
                "manifest.traceability_contract_invalid",
                "Manifest must require explicit stable report anchors",
                "manifest:traceability",
            )
        )
    return issues


def _validate_report_locator(
    row: dict[str, str],
    evidence_locator: str,
    report_text: str,
    report_name: str,
) -> list[ValidationIssue]:
    report_locator = row.get("report_locator", "").strip()
    if report_locator in EMPTY_VALUES:
        if row.get("report_critical") == "true":
            return [
                _issue(
                    "report.locator_required",
                    "Report-critical evidence needs a report locator",
                    evidence_locator,
                )
            ]
        return []

    prefix = f"{report_name}#"
    anchor = report_locator.removeprefix(prefix)
    if not report_locator.startswith(prefix) or not re.fullmatch(
        r"[a-z0-9]+(?:-[a-z0-9]+)*",
        anchor,
    ):
        return [
            _issue(
                "report.locator_invalid",
                f"Report locator must use {report_name}#stable-anchor",
                evidence_locator,
            )
        ]
    if anchor not in set(EXPLICIT_REPORT_ANCHOR.findall(report_text)):
        return [
            _issue(
                "report.locator_anchor_unresolved",
                f"Report locator anchor is not explicitly declared: {anchor}",
                evidence_locator,
            )
        ]
    return []


def _mask_markdown_source(source: str) -> str:
    """Mask Markdown source while preserving its line structure."""

    return re.sub(r"[^\r\n]", " ", source)


def _fenced_code_spec(source_line: str) -> tuple[str, int] | None:
    """Return the fence character and length for a valid opening line."""

    source_line = _markdown_container_content(source_line)
    opening = FENCED_CODE_START.fullmatch(source_line)
    if opening is None:
        return None
    fence = opening.group("fence")
    if fence.startswith("`") and "`" in opening.group("info"):
        return None
    return fence[0], len(fence)


def _markdown_container_content(source_line: str) -> str:
    """Remove blockquote and list markers before Markdown block parsing."""

    content = source_line
    while content:
        blockquote = BLOCKQUOTE_CONTAINER.match(content)
        if blockquote is not None:
            content = content[blockquote.end() :]
            continue
        list_container = LIST_CONTAINER.match(content)
        if list_container is not None:
            content = content[list_container.end() :]
            continue
        break
    return content


def _is_fenced_code_close(
    source_line: str,
    fence_character: str,
    fence_length: int,
) -> bool:
    """Return whether a container-aware line closes the active fence."""

    content = _markdown_container_content(source_line)
    closing = re.fullmatch(
        rf"[ \t]*{re.escape(fence_character)}{{{fence_length},}}[ \t]*",
        content,
    )
    return closing is not None


def _reader_visible_markdown(report_text: str) -> str:
    """Mask comments and fenced code where Markdown links are inactive."""

    visible_lines: list[str] = []
    in_html_comment = False
    fence_character: str | None = None
    fence_length = 0

    for line in report_text.splitlines(keepends=True):
        source_line = line.rstrip("\r\n")
        if fence_character is not None:
            visible_lines.append(_mask_markdown_source(line))
            if _is_fenced_code_close(
                source_line,
                fence_character,
                fence_length,
            ):
                fence_character = None
                fence_length = 0
            continue

        fence_spec = (
            _fenced_code_spec(source_line) if not in_html_comment else None
        )
        if fence_spec is not None:
            fence_character, fence_length = fence_spec
            visible_lines.append(_mask_markdown_source(line))
            continue

        visible_characters = list(line)
        cursor = 0
        while cursor < len(line):
            if in_html_comment:
                closing_position = line.find("-->", cursor)
                mask_end = (
                    len(line) if closing_position == -1 else closing_position + 3
                )
                for position in range(cursor, mask_end):
                    if visible_characters[position] not in "\r\n":
                        visible_characters[position] = " "
                cursor = mask_end
                if closing_position == -1:
                    break
                in_html_comment = False
                continue

            opening_position = line.find("<!--", cursor)
            if opening_position == -1:
                break
            cursor = opening_position
            in_html_comment = True

        visible_line = "".join(visible_characters)
        masked_source_line = visible_line.rstrip("\r\n")
        fence_spec = (
            _fenced_code_spec(masked_source_line)
            if not in_html_comment
            else None
        )
        if fence_spec is not None:
            fence_character, fence_length = fence_spec
            visible_line = _mask_markdown_source(visible_line)
        visible_lines.append(visible_line)

    return "".join(visible_lines)


def _markdown_heading_spans(visible_text: str) -> list[tuple[int, int]]:
    """Return visible ATX and one-line Setext heading source spans."""

    spans = [
        (match.start(), match.end())
        for match in MARKDOWN_HEADING.finditer(visible_text)
    ]
    lines = visible_text.splitlines(keepends=True)
    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)

    for position in range(len(lines) - 1):
        title = lines[position].rstrip("\r\n")
        underline = lines[position + 1].rstrip("\r\n")
        if not title.strip() or SETEXT_UNDERLINE.fullmatch(underline) is None:
            continue
        if MARKDOWN_HEADING.fullmatch(title) is not None:
            continue
        spans.append(
            (
                offsets[position],
                offsets[position + 1] + len(lines[position + 1]),
            )
        )
    return sorted(set(spans))


def _references_section_bounds(
    visible_text: str,
) -> tuple[tuple[int, int] | None, list[ValidationIssue]]:
    """Locate the single terminal References chapter in visible Markdown."""

    issues: list[ValidationIssue] = []
    references_anchors = [
        match
        for match in EXPLICIT_REPORT_ANCHOR.finditer(visible_text)
        if match.group(1) == "references"
    ]
    if not references_anchors:
        issues.append(
            _issue(
                "report.references_section_missing",
                "Report must declare one References section anchor",
                "report.md#references",
            )
        )
        return None, issues
    if len(references_anchors) > 1:
        issues.append(
            _issue(
                "report.references_section_duplicate",
                "References section anchor must be declared exactly once",
                "report.md#references",
            )
        )
        return None, issues

    references_anchor = references_anchors[0]
    headings = [
        span
        for span in _markdown_heading_spans(visible_text)
        if span[0] >= references_anchor.end()
    ]
    heading = headings[0] if headings else None
    if heading is None or visible_text[references_anchor.end() : heading[0]].strip():
        issues.append(
            _issue(
                "report.references_heading_missing",
                "References anchor must be followed immediately by its section heading",
                "report.md#references",
            )
        )
        return None, issues

    later_heading = next(
        (span for span in headings[1:] if span[0] >= heading[1]),
        None,
    )
    end = len(visible_text)
    if later_heading is not None:
        end = later_heading[0]
        issues.append(
            _issue(
                "report.references_section_not_terminal",
                "References must be the final formal report section",
                "report.md#references",
            )
        )
    return (heading[1], end), issues


def _reference_index(
    report_text: str,
) -> tuple[dict[str, int], list[ValidationIssue]]:
    """Index numbered reference entries and validate their report links."""

    visible_text = _reader_visible_markdown(report_text)
    index: dict[str, int] = {}
    numbers: dict[int, str] = {}
    bounds, issues = _references_section_bounds(visible_text)

    reference_anchor_counts: dict[str, int] = {}
    for anchor in EXPLICIT_REPORT_ANCHOR.findall(visible_text):
        if anchor.startswith("ref-"):
            reference_anchor_counts[anchor] = reference_anchor_counts.get(anchor, 0) + 1
    for anchor, count in reference_anchor_counts.items():
        if count > 1:
            issues.append(
                _issue(
                    "report.reference_anchor_duplicate",
                    f"Reference anchor is declared more than once: {anchor}",
                    f"report.md#{anchor}",
                )
            )

    entries = list(REFERENCE_ENTRY.finditer(visible_text))
    valid_entries: list[re.Match[str]] = []
    for entry in entries:
        inside_references = (
            bounds is not None
            and bounds[0] <= entry.start()
            and entry.end() <= bounds[1]
        )
        if not inside_references:
            issues.append(
                _issue(
                    "report.reference_entry_outside_references",
                    "Reference entry must be inside the terminal References section",
                    f"report.md#{entry.group('anchor')}",
                )
            )
            continue
        valid_entries.append(entry)

    ordered_numbers: list[int] = []
    for entry in valid_entries:
        anchor = entry.group("anchor")
        number = int(entry.group("number"))
        ordered_numbers.append(number)
        if anchor not in index:
            index[anchor] = number
        if number in numbers:
            issues.append(
                _issue(
                    "report.reference_number_duplicate",
                    f"Reference number is assigned more than once: {number}",
                    f"report.md#{anchor}",
                )
            )
        else:
            numbers[number] = anchor

    if ordered_numbers and ordered_numbers != list(range(1, len(ordered_numbers) + 1)):
        issues.append(
            _issue(
                "report.reference_number_sequence_invalid",
                "Reference entry numbers must form a contiguous sequence starting at 1",
                "report.md",
            )
        )

    for link in REFERENCE_LINK.finditer(visible_text):
        anchor = link.group("anchor")
        number = int(link.group("number"))
        entry_number = index.get(anchor)
        if entry_number is None:
            issues.append(
                _issue(
                    "report.reference_target_unknown",
                    f"Reference link targets an unknown anchor: {anchor}",
                    f"report.md#{anchor}",
                )
            )
        elif number != entry_number:
            issues.append(
                _issue(
                    "report.reference_number_mismatch",
                    f"Reference link number {number} does not match entry {entry_number}",
                    f"report.md#{anchor}",
                )
            )
    return index, issues


def _section_text(report_text: str, anchor: str) -> str | None:
    """Return one explicitly anchored report section, excluding later sections."""

    if anchor.startswith("ref-"):
        return None
    visible_text = _reader_visible_markdown(report_text)
    anchors = list(EXPLICIT_REPORT_ANCHOR.finditer(visible_text))
    start_position: int | None = None
    for position, match in enumerate(anchors):
        if match.group(1) == anchor:
            start_position = position
            break
    if start_position is None:
        return None

    start = anchors[start_position].start()
    end = len(visible_text)
    for match in anchors[start_position + 1 :]:
        if not match.group(1).startswith("ref-"):
            end = match.start()
            break
    return visible_text[start:end]


def _validate_report_references(
    rows: list[dict[str, str]],
    sources: dict[str, dict[str, Any]],
    report_text: str,
    report_name: str,
) -> list[ValidationIssue]:
    """Validate reference structure and report-critical source closure."""

    reference_index, issues = _reference_index(report_text)
    report_prefix = f"{report_name}#"
    reference_locator_pattern = re.compile(
        rf"{re.escape(report_name)}#(?P<anchor>ref-[a-z0-9]+(?:-[a-z0-9]+)*)"
    )

    for position, row in enumerate(rows, start=2):
        if row.get("report_critical") != "true":
            continue
        evidence_locator = f"evidence.tsv:{position}"
        report_locator = row.get("report_locator", "").strip()
        section_anchor = report_locator.removeprefix(report_prefix)
        section_text = (
            _section_text(report_text, section_anchor)
            if report_locator.startswith(report_prefix)
            else None
        )

        for source_id in _split_ids(row.get("source_ids")):
            source = sources.get(source_id)
            report_reference = source.get("report_reference") if source else None
            reference_match = (
                reference_locator_pattern.fullmatch(report_reference)
                if isinstance(report_reference, str)
                else None
            )
            reference_anchor = (
                reference_match.group("anchor") if reference_match is not None else None
            )
            reference_number = (
                reference_index.get(reference_anchor) if reference_anchor is not None else None
            )
            expected_link = (
                f"[{reference_number}](#{reference_anchor})"
                if reference_number is not None and reference_anchor is not None
                else None
            )
            if section_text is None or expected_link is None or expected_link not in section_text:
                issues.append(
                    _issue(
                        "report.critical_source_reference_missing",
                        f"Report-critical source is not referenced in its section: {source_id}",
                        evidence_locator,
                    )
                )
    return issues


def _source_index(
    manifest: dict[str, Any],
    package_root: Path,
) -> tuple[dict[str, dict[str, Any]], list[ValidationIssue]]:
    index: dict[str, dict[str, Any]] = {}
    report_reference_owners: dict[str, str] = {}
    issues: list[ValidationIssue] = []
    sources = manifest.get("sources", [])
    if not isinstance(sources, list):
        return index, issues
    for position, source in enumerate(sources, start=1):
        locator = f"manifest:sources[{position}]"
        if not isinstance(source, dict):
            issues.append(_issue("source.mapping_required", "Source must be a mapping", locator))
            continue
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            issues.append(_issue("source.id_required", "Source ID is required", locator))
            continue
        if source_id in index:
            issues.append(_issue("source.id_duplicate", f"Duplicate source ID: {source_id}", locator))
            continue
        index[source_id] = source

        availability = source.get("availability")
        identity = source.get("identity")
        required = _manifest_ids(source.get("required_for_record_ids"))
        source_locator = source.get("locator")
        report_reference = source.get("report_reference")
        reference_missing = report_reference is None or (
            isinstance(report_reference, str) and report_reference in EMPTY_VALUES
        )
        if reference_missing:
            if required:
                issues.append(
                    _issue(
                        "source.report_reference_required",
                        f"Report-critical source needs a report reference: {source_id}",
                        locator,
                    )
                )
        elif not isinstance(report_reference, str) or (
            SOURCE_REPORT_REFERENCE.fullmatch(report_reference) is None
        ):
            issues.append(
                _issue(
                    "source.report_reference_invalid",
                    "Source report reference must use report.md#ref-stable-anchor",
                    locator,
                )
            )
        else:
            previous_source_id = report_reference_owners.get(report_reference)
            if previous_source_id is not None and previous_source_id != source_id:
                issues.append(
                    _issue(
                        "source.report_reference_duplicate",
                        f"Source report reference is duplicated: {report_reference}",
                        locator,
                    )
                )
            else:
                report_reference_owners[report_reference] = source_id
        if availability not in {"available", "restricted", "missing"}:
            issues.append(_issue("source.availability_invalid", "Invalid source availability", locator))
        if availability == "missing" or identity in EMPTY_VALUES:
            if required:
                issues.append(
                    _issue(
                        "source.critical_missing",
                        f"Report-critical source is unavailable: {source_id}",
                        locator,
                        blocking=True,
                    )
                )
        elif availability == "available" and isinstance(source_locator, str):
            resolved = _resolved_path(package_root, source_locator)
            if resolved is not None and not resolved.exists():
                issues.append(
                    _issue(
                        "source.locator_missing",
                        f"Available source path does not exist: {source_locator}",
                        locator,
                        blocking=bool(required),
                    )
                )
    return index, issues


def _validate_evidence_rows(
    rows: list[dict[str, str]],
    sources: dict[str, dict[str, Any]],
    report_text: str,
    package_root: Path,
    report_name: str,
) -> tuple[dict[str, dict[str, str]], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    records: dict[str, dict[str, str]] = {}
    allowed_record_types = {"claim", "figure", "table", "hypothesis", "decision"}
    allowed_support = {"verified", "unverified", "insufficient", "disputed", "not_applicable"}

    for position, row in enumerate(rows, start=2):
        locator = f"evidence.tsv:{position}"
        record_id = row.get("record_id", "").strip()
        if not record_id:
            issues.append(_issue("evidence.record_id_required", "Record ID is required", locator))
            continue
        if record_id in records:
            issues.append(_issue("evidence.record_id_duplicate", f"Duplicate record ID: {record_id}", locator))
            continue
        records[record_id] = row

        if row.get("record_type") not in allowed_record_types:
            issues.append(_issue("evidence.record_type_invalid", "Invalid record_type", locator))
        if row.get("report_critical") not in {"true", "false"}:
            issues.append(_issue("evidence.critical_invalid", "report_critical must be true or false", locator))
        if row.get("support_status") not in allowed_support:
            issues.append(_issue("evidence.support_status_invalid", "Invalid support_status", locator))
        issues.extend(
            _validate_report_locator(
                row,
                locator,
                report_text,
                report_name,
            )
        )

        source_ids = _split_ids(row.get("source_ids"))
        if row.get("report_critical") == "true" and not source_ids:
            issues.append(_issue("evidence.critical_source_required", "Critical record needs a source ID", locator))
        for source_id in source_ids:
            if source_id not in sources:
                issues.append(
                    _issue(
                        "source.forward_orphan",
                        f"Evidence references unknown source: {source_id}",
                        locator,
                    )
                )

        record_type = row.get("record_type")
        if record_type in {"figure", "table"}:
            issues.extend(_validate_figure_or_table(row, locator, report_text, package_root))
        if row.get("statement_role") == "interpretation":
            if not _split_ids(row.get("supported_by_ids")):
                issues.append(_issue("claim.upstream_required", "Interpretation needs upstream evidence", locator))
            if row.get("boundary") in EMPTY_VALUES:
                issues.append(_issue("claim.boundary_required", "Interpretation boundary is required", locator))

    for record_id, row in records.items():
        for upstream_id in _split_ids(row.get("supported_by_ids")):
            if upstream_id == record_id:
                issues.append(_issue("graph.self_edge", "Evidence row cannot support itself", f"evidence:{record_id}"))
            elif upstream_id not in records:
                issues.append(
                    _issue(
                        "graph.upstream_orphan",
                        f"Unknown upstream evidence ID: {upstream_id}",
                        f"evidence:{record_id}",
                    )
                )
    issues.extend(_validate_acyclic(records))
    return records, issues


def _validate_figure_or_table(
    row: dict[str, str],
    locator: str,
    report_text: str,
    package_root: Path,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    required_fields = (
        "section_question",
        "message_or_claim",
        "generation_entry",
        "artifact_path",
        "comparison",
        "analysis_unit",
        "sample_n",
        "uncertainty",
        "encoding",
        "report_locator",
        "qa_status",
    )
    for field in required_fields:
        if row.get(field) in EMPTY_VALUES:
            issues.append(
                _issue(
                    "figure.card_incomplete",
                    f"Figure/Table card field is missing: {field}",
                    locator,
                )
            )
    question = row.get("section_question", "")
    if question and not question.endswith(("?", "？")):
        issues.append(_issue("story.question_required", "Section question must end with ?", locator))

    artifact_locator = row.get("artifact_path", "")
    artifact_path = _resolved_path(package_root, artifact_locator)
    if artifact_path is None or not artifact_path.is_file():
        issues.append(_issue("artifact.missing", f"Artifact is missing: {artifact_locator}", locator))
    elif artifact_locator not in report_text:
        issues.append(_issue("report.artifact_unlinked", f"Artifact is not linked in report: {artifact_locator}", locator))

    generation_entry = row.get("generation_entry", "")
    if generation_entry.startswith("computed:"):
        generation_locator = generation_entry.removeprefix("computed:")
        generation_path = _resolved_path(package_root, generation_locator)
        if generation_path is None or not generation_path.is_file():
            issues.append(
                _issue(
                    "generation.entry_missing",
                    f"Generation entry is missing: {generation_locator}",
                    locator,
                )
            )
    if row.get("record_type") == "figure" and artifact_locator:
        plot_data_locator = f"plot_data/{Path(artifact_locator).stem}.csv"
        plot_data_path = _resolved_path(package_root, plot_data_locator)
        if plot_data_path is None or not plot_data_path.is_file():
            issues.append(
                _issue(
                    "figure.plot_data_missing",
                    f"Measured-data Figure lacks paired plot data: {plot_data_locator}",
                    locator,
                )
            )
    if row.get("qa_status") != "passed":
        issues.append(_issue("artifact.qa_not_passed", "Figure/Table QA has not passed", locator))
    return issues


def _validate_acyclic(records: dict[str, dict[str, str]]) -> list[ValidationIssue]:
    state: dict[str, int] = {}

    def visit(record_id: str) -> bool:
        marker = state.get(record_id, 0)
        if marker == 1:
            return False
        if marker == 2:
            return True
        state[record_id] = 1
        for upstream in _split_ids(records[record_id].get("supported_by_ids")):
            if upstream in records and not visit(upstream):
                return False
        state[record_id] = 2
        return True

    for record_id in records:
        if not visit(record_id):
            return [_issue("graph.cycle", "Evidence dependency graph contains a cycle", f"evidence:{record_id}")]
    return []


def _validate_source_pair_sets(
    records: dict[str, dict[str, str]],
    sources: dict[str, dict[str, Any]],
) -> list[ValidationIssue]:
    forward = {
        (source_id, record_id)
        for record_id, row in records.items()
        for source_id in _split_ids(row.get("source_ids"))
        if source_id in sources
    }
    reverse = {
        (source_id, record_id)
        for source_id, source in sources.items()
        for record_id in _manifest_ids(source.get("required_for_record_ids"))
    }
    if forward == reverse:
        return []
    missing_reverse = sorted(forward - reverse)
    extra_reverse = sorted(reverse - forward)
    return [
        _issue(
            "source.pair_set_mismatch",
            f"Source/evidence reverse pairs differ; missing={missing_reverse}, extra={extra_reverse}",
            "manifest:sources",
        )
    ]


def _validate_report_story(report_text: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not re.search(r"(?m)^#\s+\S", report_text):
        issues.append(_issue("report.title_missing", "Report title is missing", "report.md"))
    questions = re.findall(r"(?m)^##\s+[^\n]+[?？]\s*$", report_text)
    if len(questions) < 3:
        issues.append(
            _issue(
                "story.concrete_questions_missing",
                "Report needs concrete question-led sections",
                "report.md",
            )
        )
    forbidden = ("[TODO", "<TODO", "PLACEHOLDER", "Question 1")
    for token in forbidden:
        if token.casefold() in report_text.casefold():
            issues.append(
                _issue(
                    "report.placeholder_present",
                    f"Unresolved template token remains: {token}",
                    "report.md",
                )
            )
    for _ in INTERNAL_EVIDENCE_MARKER.finditer(report_text):
        issues.append(
            _issue(
                "report.internal_evidence_marker_present",
                "Internal evidence markers are forbidden in reader-facing report text",
                "report.md",
            )
        )
    return issues


def _validate_handoff(manifest: dict[str, Any]) -> list[ValidationIssue]:
    handoff = manifest.get("derived_format_handoff")
    core = manifest.get("core")
    if not isinstance(handoff, dict) or not isinstance(core, dict):
        return []
    if handoff.get("requested") is not True:
        if handoff.get("acceptance_status") != "not_applicable":
            return [
                _issue(
                    "handoff.unrequested_status_invalid",
                    "Unrequested handoff must be not_applicable",
                    "manifest:derived_format_handoff",
                )
            ]
        return []
    language = core.get("report_language")
    if handoff.get("report_language") != language or handoff.get("derived_language") != language:
        return [
            _issue(
                "handoff.language_mismatch",
                "Derived-format handoff must inherit the report language",
                "manifest:derived_format_handoff",
            )
        ]
    if handoff.get("acceptance_status") != "accepted":
        return [
            _issue(
                "handoff.acceptance_invalid",
                "Same-language requested handoff must be explicitly accepted",
                "manifest:derived_format_handoff",
            )
        ]
    return []


def _required_review_names(
    records: dict[str, dict[str, str]],
) -> set[str]:
    required = {"claim_coverage", "fresh_reader"}
    if any(
        row.get("record_type") == "figure" and row.get("presentation_role") == "main"
        for row in records.values()
    ):
        required.add("visual")
    if any(
        row.get("report_critical") == "true"
        and (
            row.get("claim_type") in HIGH_RISK_CLAIMS
            or row.get("claim_level") == "causal"
            or row.get("support_status") == "disputed"
        )
        for row in records.values()
    ):
        required.add("science")
    return required


def _validate_reviews(
    manifest: dict[str, Any],
    records: dict[str, dict[str, str]],
    package_root: Path,
) -> list[ValidationIssue]:
    validation = manifest.get("validation")
    reviews = validation.get("reviews") if isinstance(validation, dict) else None
    if not isinstance(reviews, dict):
        return [_issue("review.block_missing", "Validation review block is missing", "manifest:validation.reviews")]
    issues: list[ValidationIssue] = []
    for name in sorted(_required_review_names(records)):
        review = reviews.get(name)
        locator = f"manifest:validation.reviews.{name}"
        if not isinstance(review, dict):
            issues.append(_issue("review.required_missing", f"Required review is missing: {name}", locator))
            continue
        if review.get("requirement") != "required" or review.get("status") != "completed":
            issues.append(_issue("review.required_incomplete", f"Required review is incomplete: {name}", locator))
            continue
        artifact_locator = review.get("locator")
        artifact_path = (
            _resolved_path(package_root, artifact_locator)
            if isinstance(artifact_locator, str)
            else None
        )
        if artifact_path is None or not artifact_path.is_file():
            issues.append(_issue("review.artifact_missing", f"Review artifact is missing: {name}", locator))
    return issues


def validate_report_package(
    manifest_path: Path,
    evidence_path: Path,
    report_path: Path,
    *,
    candidate_status: str,
) -> ValidationResult:
    """Validate report structure and traceability without changing files."""

    issues: list[ValidationIssue] = []
    if candidate_status not in TERMINAL_STATUSES:
        issues.append(
            _issue(
                "status.terminal_candidate_required",
                "Candidate status must be final or complete_with_limitations",
                "candidate_status",
            )
        )
    for path, code in (
        (manifest_path, "manifest.file_missing"),
        (evidence_path, "evidence.file_missing"),
        (report_path, "report.file_missing"),
    ):
        if not path.is_file():
            issues.append(_issue(code, f"Required file is missing: {path}", str(path), blocking=True))
    if issues:
        return _result(candidate_status, issues)

    try:
        manifest = _read_yaml_mapping(manifest_path)
        header, rows = _read_evidence(evidence_path)
        report_text = report_path.read_text(encoding="utf-8")
    except (OSError, TypeError, ValueError, yaml.YAMLError, UnicodeError) as error:
        return _result(
            candidate_status,
            [_issue("input.parse_failed", str(error), "report-package", blocking=True)],
        )

    package_root = report_path.parent
    issues.extend(_validate_manifest_shape(manifest))
    if header != EVIDENCE_COLUMNS:
        issues.append(
            _issue(
                "evidence.header_invalid",
                "Evidence TSV header does not match the frozen schema",
                str(evidence_path),
            )
        )
    if not rows:
        issues.append(
            _issue(
                "evidence.empty",
                "A terminal evidence-backed report needs at least one evidence row",
                str(evidence_path),
            )
        )
    source_index, source_issues = _source_index(manifest, package_root)
    issues.extend(source_issues)
    records, record_issues = _validate_evidence_rows(
        rows,
        source_index,
        report_text,
        package_root,
        report_path.name,
    )
    issues.extend(record_issues)
    issues.extend(
        _validate_report_references(
            rows,
            source_index,
            report_text,
            report_path.name,
        )
    )
    issues.extend(_validate_source_pair_sets(records, source_index))
    issues.extend(_validate_report_story(report_text))
    issues.extend(_validate_handoff(manifest))
    issues.extend(_validate_reviews(manifest, records, package_root))
    return _result(candidate_status, issues)


def _result(
    candidate_status: str,
    issues: Iterable[ValidationIssue],
) -> ValidationResult:
    issue_tuple = tuple(issues)
    if not issue_tuple:
        next_status = candidate_status
    elif any(issue.blocking for issue in issue_tuple):
        next_status = "blocked"
    else:
        next_status = "review"
    return ValidationResult(
        passed=not issue_tuple,
        candidate_status=candidate_status,
        next_status=next_status,
        issues=issue_tuple,
    )


def _atomic_write_yaml(path: Path, value: dict[str, Any]) -> None:
    serialized = yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def apply_terminal_validation(
    manifest_path: Path,
    evidence_path: Path,
    report_path: Path,
    *,
    candidate_status: str,
) -> ValidationResult:
    """Validate prospectively, then atomically write deterministic status."""

    result = validate_report_package(
        manifest_path,
        evidence_path,
        report_path,
        candidate_status=candidate_status,
    )
    manifest = _read_yaml_mapping(manifest_path)
    validation = manifest.setdefault("validation", {})
    if not isinstance(validation, dict):
        validation = {}
        manifest["validation"] = validation
    validation["deterministic"] = {
        "status": "passed" if result.passed else "failed",
        "candidate_status": candidate_status,
        "issue_codes": list(result.issue_codes),
    }
    manifest["status"] = result.next_status
    _atomic_write_yaml(manifest_path, manifest)
    return result


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--candidate-status",
        choices=sorted(TERMINAL_STATUSES),
        required=True,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_common_arguments(subparsers.add_parser("check", help="Validate without writing"))
    _add_common_arguments(subparsers.add_parser("terminal", help="Validate and write status"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic validator CLI."""

    args = _build_parser().parse_args(argv)
    function = apply_terminal_validation if args.command == "terminal" else validate_report_package
    result = function(
        args.manifest,
        args.evidence,
        args.report,
        candidate_status=args.candidate_status,
    )
    print(f"Validation status: {'passed' if result.passed else 'failed'}")
    print(f"Next report status: {result.next_status}")
    for issue in result.issues:
        print(f"{issue.code}: {issue.message} [{issue.locator}]")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
