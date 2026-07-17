#!/usr/bin/env python3
"""Inventory local report sources without mutating them."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Any, Sequence

import yaml


@dataclass(frozen=True)
class SourceRecord:
    """One stable source entry in the report manifest."""

    source_id: str
    kind: str
    locator: str
    identity: str
    availability: str
    report_reference: str
    required_for_record_ids: tuple[str, ...]


@dataclass(frozen=True)
class InventoryResult:
    """Prospective source and preflight blocks plus nonfatal issues."""

    preflight: dict[str, object]
    sources: tuple[SourceRecord, ...]
    issues: tuple[str, ...]


def _normalize_locator(locator: str) -> str:
    raw = locator.strip().replace("\\", "/")
    if not raw:
        raise ValueError("Source locator must not be empty")
    path = PurePosixPath(raw)
    normalized = str(path)
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized or "."


def stable_source_id(locator: str) -> str:
    """Return a stable ID derived only from a normalized locator."""

    normalized = _normalize_locator(locator)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return f"SRC-{digest}"


def stable_reference_anchor(source_id: str) -> str:
    """Return the stable reference anchor for one source ID."""

    value = source_id.casefold()
    if value.startswith("src-"):
        value = value[4:]
    slug = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    if not slug:
        raise ValueError("Source ID cannot produce a reference anchor")
    return f"ref-{slug}"


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def _split_ids(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    ids = {
        item.strip()
        for item in value.split(";")
        if item.strip() and item.strip() != "not_applicable"
    }
    return tuple(sorted(ids))


def _required_record_map(evidence_path: Path) -> dict[str, tuple[str, ...]]:
    required: dict[str, set[str]] = {}
    if not evidence_path.exists():
        return {}
    with evidence_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "source_ids" not in reader.fieldnames:
            raise ValueError("Evidence TSV must contain source_ids")
        if "record_id" not in reader.fieldnames:
            raise ValueError("Evidence TSV must contain record_id")
        for row in reader:
            record_id = (row.get("record_id") or "").strip()
            if not record_id:
                continue
            for source_id in _split_ids(row.get("source_ids")):
                required.setdefault(source_id, set()).add(record_id)
    return {
        source_id: tuple(sorted(record_ids))
        for source_id, record_ids in required.items()
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_identity(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(str(child.stat().st_size).encode("ascii"))
    return f"directory:{digest.hexdigest()}"


def _dvc_identity(path: Path) -> str | None:
    if path.suffix != ".dvc" or not path.is_file():
        return None
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        return None
    outs = value.get("outs")
    if not isinstance(outs, list):
        return None
    identities: list[str] = []
    for output in outs:
        if not isinstance(output, dict):
            continue
        for key in ("md5", "etag", "checksum"):
            checksum = output.get(key)
            if isinstance(checksum, str) and checksum:
                identities.append(f"{key}:{checksum}")
                break
    if not identities:
        return None
    return "dvc:" + ";".join(sorted(identities))


def _git_identity(target_checkout: Path, source_path: Path) -> str | None:
    try:
        head = subprocess.run(
            ["git", "-C", str(target_checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    try:
        relative = source_path.resolve().relative_to(target_checkout.resolve())
    except ValueError:
        return None
    status = subprocess.run(
        [
            "git",
            "-C",
            str(target_checkout),
            "status",
            "--short",
            "--",
            relative.as_posix(),
        ],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    state = "dirty" if status else "clean"
    return f"git:{head}:{state}"


def _resolve_source(target_checkout: Path, locator: str) -> Path:
    candidate = Path(locator)
    if candidate.is_absolute():
        return candidate
    return target_checkout / candidate


def _infer_kind(locator: str) -> str:
    suffix = Path(locator).suffix.casefold()
    if suffix in {".py", ".r", ".sh", ".ipynb"}:
        return "code"
    if suffix in {".md", ".txt", ".pdf"}:
        return "document"
    if suffix in {".png", ".svg", ".jpg", ".jpeg"}:
        return "artifact"
    return "data"


def _identity_for(target_checkout: Path, source_path: Path) -> str:
    dvc = _dvc_identity(source_path)
    if dvc is not None:
        return dvc
    git = _git_identity(target_checkout, source_path)
    content = (
        _directory_identity(source_path)
        if source_path.is_dir()
        else f"sha256:{_sha256_file(source_path)}"
    )
    return f"{git}|{content}" if git is not None else content


def inventory_sources(
    manifest_path: Path,
    evidence_path: Path,
    target_checkout: Path,
    additional_locators: Sequence[str] = (),
) -> InventoryResult:
    """Build prospective source/preflight blocks without writing the manifest."""

    manifest = _read_yaml_mapping(manifest_path)
    source_values = manifest.get("sources", [])
    if not isinstance(source_values, list):
        raise TypeError("Manifest sources must be a list")
    existing: dict[str, dict[str, Any]] = {}
    for raw_source in source_values:
        if not isinstance(raw_source, dict):
            raise TypeError("Each manifest source must be a mapping")
        locator = _normalize_locator(str(raw_source.get("locator", "")))
        if locator in existing:
            raise ValueError(f"Duplicate source locator: {locator}")
        existing[locator] = raw_source

    locators = set(existing)
    locators.update(_normalize_locator(item) for item in additional_locators)
    required_by_source = _required_record_map(evidence_path)
    records: list[SourceRecord] = []
    issues: list[str] = []

    for locator in sorted(locators):
        prior = existing.get(locator, {})
        source_id = str(prior.get("source_id") or stable_source_id(locator))
        report_reference = str(
            prior.get("report_reference")
            or f"report.md#{stable_reference_anchor(source_id)}"
        )
        source_path = _resolve_source(target_checkout, locator)
        if source_path.exists():
            availability = str(prior.get("availability") or "available")
            if availability not in {"available", "restricted"}:
                availability = "available"
            identity = _identity_for(target_checkout, source_path)
        else:
            availability = "missing"
            identity = "not_available"
            issues.append(f"source.missing:{source_id}:{locator}")
        records.append(
            SourceRecord(
                source_id=source_id,
                kind=str(prior.get("kind") or _infer_kind(locator)),
                locator=locator,
                identity=identity,
                availability=availability,
                report_reference=report_reference,
                required_for_record_ids=required_by_source.get(source_id, ()),
            )
        )

    seen_report_references: set[str] = set()
    for record in records:
        if record.report_reference in seen_report_references:
            raise ValueError(
                f"Duplicate report reference: {record.report_reference}"
            )
        seen_report_references.add(record.report_reference)

    registered_ids = {record.source_id for record in records}
    for orphan_id in sorted(set(required_by_source) - registered_ids):
        issues.append(f"source.orphan_id:{orphan_id}")

    old_preflight = manifest.get("preflight", {})
    if not isinstance(old_preflight, dict):
        raise TypeError("Manifest preflight must be a mapping")
    gaps = [
        {
            "gap_id": f"GAP-{record.source_id}",
            "kind": "missing_source",
            "source_id": record.source_id,
            "affected_record_ids": list(record.required_for_record_ids),
            "blocking": bool(record.required_for_record_ids),
        }
        for record in records
        if record.availability == "missing"
    ]
    preflight = dict(old_preflight)
    preflight["gaps"] = gaps
    preflight["inventory_status"] = "issues" if issues else "complete"
    return InventoryResult(
        preflight=preflight,
        sources=tuple(records),
        issues=tuple(sorted(issues)),
    )


def _atomic_write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def update_inventory_blocks(
    manifest_path: Path,
    result: InventoryResult,
) -> None:
    """Atomically replace only the manifest preflight and source blocks."""

    manifest = _read_yaml_mapping(manifest_path)
    manifest["preflight"] = result.preflight
    manifest["sources"] = [
        {
            **asdict(record),
            "required_for_record_ids": list(record.required_for_record_ids),
        }
        for record in result.sources
    ]
    _atomic_write_yaml(manifest_path, manifest)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--target-checkout", type=Path, required=True)
    parser.add_argument("--add", nargs="*", default=[])
    parser.add_argument("--write", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the source inventory CLI."""

    args = _build_parser().parse_args(argv)
    result = inventory_sources(
        args.manifest,
        args.evidence,
        args.target_checkout,
        additional_locators=args.add,
    )
    if args.write:
        update_inventory_blocks(args.manifest, result)
    print(f"Inventory sources: {len(result.sources)}")
    print(f"Inventory issues: {len(result.issues)}")
    for issue in result.issues:
        print(issue)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
