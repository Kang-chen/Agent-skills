"""Tests for stable source references in manifest inventory and validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = REPOSITORY_ROOT / "scripts" / "inventory_sources.py"
SPEC = importlib.util.spec_from_file_location("inventory_sources", INVENTORY_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load inventory: {INVENTORY_PATH}")
inventory = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inventory
SPEC.loader.exec_module(inventory)

VALIDATOR_PATH = REPOSITORY_ROOT / "scripts" / "validate_report.py"
VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_report_inventory_references",
    VALIDATOR_PATH,
)
if VALIDATOR_SPEC is None or VALIDATOR_SPEC.loader is None:
    raise RuntimeError(f"Could not load validator: {VALIDATOR_PATH}")
validator = importlib.util.module_from_spec(VALIDATOR_SPEC)
sys.modules[VALIDATOR_SPEC.name] = validator
VALIDATOR_SPEC.loader.exec_module(validator)


REFERENCE_TRACEABILITY = {
    "report_locator_format": "report.md#stable-anchor",
    "explicit_anchor_syntax": '<a id="stable-anchor"></a>',
    "require_explicit_anchors": True,
    "reference_link_format": "[N](#ref-stable-anchor)",
    "reference_anchor_syntax": '<a id="ref-stable-anchor"></a>',
    "reference_section_locator": "report.md#references",
    "require_source_references": True,
}


class InventorySourceReferenceTests(unittest.TestCase):
    """Exercise stable report-reference generation and preservation."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.checkout = Path(self.temporary_directory.name)
        self.manifest_path = self.checkout / "report_manifest.yaml"
        self.evidence_path = self.checkout / "evidence.tsv"
        (self.checkout / "sources").mkdir()
        (self.checkout / "sources" / "a.md").write_text(
            "Source A\n",
            encoding="utf-8",
        )
        (self.checkout / "sources" / "b.md").write_text(
            "Source B\n",
            encoding="utf-8",
        )
        self.manifest_path.write_text(
            "schema_version: 2\npreflight: {}\nsources: []\n",
            encoding="utf-8",
        )
        self.evidence_path.write_text(
            "record_id\tsource_ids\n",
            encoding="utf-8",
        )

    def test_inventory_generates_stable_report_reference(self):
        self.manifest_path.write_text(
            """schema_version: 2
preflight: {}
sources:
  - source_id: SRC-Study__ALPHA---V2
    locator: sources/a.md
""",
            encoding="utf-8",
        )

        result = inventory.inventory_sources(
            self.manifest_path,
            self.evidence_path,
            self.checkout,
        )

        source = result.sources[0]
        self.assertEqual(
            "report.md#ref-study-alpha-v2",
            source.report_reference,
        )

    def test_inventory_rejects_false_report_reference_without_rewriting(self):
        self.manifest_path.write_text(
            """schema_version: 2
preflight: {}
sources:
  - source_id: SRC-A
    locator: sources/a.md
    report_reference: false
""",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            TypeError,
            "Source report_reference must be a string: SRC-A",
        ):
            inventory.inventory_sources(
                self.manifest_path,
                self.evidence_path,
                self.checkout,
            )

        manifest = validator._read_yaml_mapping(self.manifest_path)
        self.assertIs(False, manifest["sources"][0]["report_reference"])

    def test_inventory_preserves_empty_reference_for_validator(self):
        self.manifest_path.write_text(
            """schema_version: 2
preflight: {}
sources:
  - source_id: SRC-A
    locator: sources/a.md
    report_reference: ''
""",
            encoding="utf-8",
        )
        self.evidence_path.write_text(
            "record_id\tsource_ids\nREC-A\tSRC-A\n",
            encoding="utf-8",
        )

        result = inventory.inventory_sources(
            self.manifest_path,
            self.evidence_path,
            self.checkout,
        )
        inventory.update_inventory_blocks(self.manifest_path, result)
        manifest = validator._read_yaml_mapping(self.manifest_path)
        _, issues = validator._source_index(manifest, self.checkout)

        self.assertEqual("", result.sources[0].report_reference)
        self.assertEqual("", manifest["sources"][0]["report_reference"])
        self.assertIn(
            "source.report_reference_required",
            {issue.code for issue in issues},
        )

    def test_inventory_preserves_existing_report_reference(self):
        self.manifest_path.write_text(
            """schema_version: 2
preflight: {}
sources:
  - source_id: SRC-A
    kind: document
    locator: sources/a.md
    identity: old-identity
    availability: available
    report_reference: report.md#ref-curated-source-a
    required_for_record_ids: []
""",
            encoding="utf-8",
        )

        first_result = inventory.inventory_sources(
            self.manifest_path,
            self.evidence_path,
            self.checkout,
        )
        inventory.update_inventory_blocks(self.manifest_path, first_result)
        second_result = inventory.inventory_sources(
            self.manifest_path,
            self.evidence_path,
            self.checkout,
        )

        self.assertEqual(
            "report.md#ref-curated-source-a",
            first_result.sources[0].report_reference,
        )
        self.assertEqual(
            first_result.sources[0].report_reference,
            second_result.sources[0].report_reference,
        )

    def test_inventory_rejects_colliding_report_references(self):
        self.manifest_path.write_text(
            """schema_version: 2
preflight: {}
sources:
  - source_id: SRC-A_B
    locator: sources/a.md
  - source_id: SRC-A-B
    locator: sources/b.md
""",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "Duplicate report reference"):
            inventory.inventory_sources(
                self.manifest_path,
                self.evidence_path,
                self.checkout,
            )

    def test_stable_reference_anchor_rejects_empty_slug(self):
        with self.assertRaisesRegex(
            ValueError,
            "Source ID cannot produce a reference anchor",
        ):
            inventory.stable_reference_anchor("SRC-___")


class ManifestSourceReferenceValidationTests(unittest.TestCase):
    """Exercise manifest v2 and source report-reference validation."""

    @staticmethod
    def valid_manifest(**overrides: object) -> dict[str, object]:
        manifest = {
            block: {}
            for block in validator.REQUIRED_MANIFEST_BLOCKS
        }
        manifest.update(
            {
                "schema_version": 2,
                "core": {"report_language": "English"},
                "sources": [],
                "traceability": dict(REFERENCE_TRACEABILITY),
            }
        )
        manifest.update(overrides)
        return manifest

    @staticmethod
    def source(**overrides: object) -> dict[str, object]:
        source = {
            "source_id": "SRC-A",
            "kind": "document",
            "locator": "sources/a.md",
            "identity": "restricted-identity",
            "availability": "restricted",
            "report_reference": "report.md#ref-source-a",
            "required_for_record_ids": ["REC-A"],
        }
        source.update(overrides)
        return source

    @staticmethod
    def issue_codes(issues: list[object]) -> set[str]:
        return {issue.code for issue in issues}

    def test_manifest_requires_schema_version_two(self):
        manifest = self.valid_manifest(schema_version=1)

        issues = validator._validate_manifest_shape(manifest)

        self.assertIn(
            "manifest.schema_version_invalid",
            self.issue_codes(issues),
        )

    def test_manifest_accepts_complete_reference_traceability(self):
        issues = validator._validate_manifest_shape(self.valid_manifest())

        self.assertNotIn(
            "manifest.traceability_contract_invalid",
            self.issue_codes(issues),
        )

    def test_report_critical_source_requires_report_reference(self):
        source = self.source()
        source.pop("report_reference")

        _, issues = validator._source_index(
            self.valid_manifest(sources=[source]),
            REPOSITORY_ROOT,
        )

        self.assertIn(
            "source.report_reference_required",
            self.issue_codes(issues),
        )

    def test_source_report_reference_requires_stable_locator_format(self):
        source = self.source(report_reference="report.md#ref-Source_A")

        _, issues = validator._source_index(
            self.valid_manifest(sources=[source]),
            REPOSITORY_ROOT,
        )

        self.assertIn(
            "source.report_reference_invalid",
            self.issue_codes(issues),
        )

    def test_non_string_source_report_reference_is_invalid(self):
        source = self.source(
            report_reference=7,
            required_for_record_ids=[],
        )

        _, issues = validator._source_index(
            self.valid_manifest(sources=[source]),
            REPOSITORY_ROOT,
        )

        self.assertIn(
            "source.report_reference_invalid",
            self.issue_codes(issues),
        )

    def test_different_sources_cannot_share_report_reference(self):
        sources = [
            self.source(),
            self.source(
                source_id="SRC-B",
                locator="sources/b.md",
                report_reference="report.md#ref-source-a",
            ),
        ]

        _, issues = validator._source_index(
            self.valid_manifest(sources=sources),
            REPOSITORY_ROOT,
        )

        self.assertIn(
            "source.report_reference_duplicate",
            self.issue_codes(issues),
        )

    def test_unique_source_report_references_are_valid(self):
        sources = [
            self.source(),
            self.source(
                source_id="SRC-B",
                locator="sources/b.md",
                report_reference="report.md#ref-source-b",
            ),
        ]

        _, issues = validator._source_index(
            self.valid_manifest(sources=sources),
            REPOSITORY_ROOT,
        )

        reference_issue_codes = {
            code
            for code in self.issue_codes(issues)
            if code.startswith("source.report_reference_")
        }
        self.assertEqual(set(), reference_issue_codes)

    def test_manifest_seed_uses_reference_contract_v2(self):
        seed_path = REPOSITORY_ROOT / "assets" / "report_manifest.yaml"
        manifest = validator._read_yaml_mapping(seed_path)

        issues = validator._validate_manifest_shape(manifest)

        codes = self.issue_codes(issues)
        self.assertNotIn("manifest.schema_version_invalid", codes)
        self.assertNotIn("manifest.traceability_contract_invalid", codes)


    def test_source_reference_follows_renamed_report(self):
        # 简版占用 report.md 时，完整版改名为 report_full.md，来源引用随之改名
        source = self.source(report_reference="report_full.md#ref-source-a")

        _, renamed = validator._source_index(
            self.valid_manifest(sources=[source]),
            REPOSITORY_ROOT,
            "report_full.md",
        )
        _, default = validator._source_index(
            self.valid_manifest(sources=[source]),
            REPOSITORY_ROOT,
        )

        self.assertNotIn("source.report_reference_invalid", self.issue_codes(renamed))
        self.assertIn("source.report_reference_invalid", self.issue_codes(default))

    def test_traceability_contract_follows_renamed_report(self):
        traceability = dict(REFERENCE_TRACEABILITY)
        traceability["report_locator_format"] = "report_full.md#stable-anchor"
        traceability["reference_section_locator"] = "report_full.md#references"
        manifest = self.valid_manifest(traceability=traceability)

        renamed = validator._validate_manifest_shape(manifest, "report_full.md")
        default = validator._validate_manifest_shape(manifest)

        self.assertNotIn("manifest.traceability_contract_invalid", self.issue_codes(renamed))
        self.assertIn("manifest.traceability_contract_invalid", self.issue_codes(default))


if __name__ == "__main__":
    unittest.main()
