"""Tests for reader-facing report reference validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPOSITORY_ROOT / "scripts" / "validate_report.py"
SPEC = importlib.util.spec_from_file_location("validate_report", VALIDATOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load validator: {VALIDATOR_PATH}")
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


class ValidateReportReferenceTests(unittest.TestCase):
    """Exercise numeric references and evidence-to-section closure."""

    package_root = REPOSITORY_ROOT

    @staticmethod
    def evidence_row(**overrides: str) -> dict[str, str]:
        row = {
            "record_id": "REC-A",
            "record_type": "claim",
            "report_critical": "true",
            "support_status": "verified",
            "report_locator": "report.md#findings",
            "source_ids": "SRC-A",
            "supported_by_ids": "",
            "statement_role": "finding",
        }
        row.update(overrides)
        return row

    @staticmethod
    def source(**overrides: str) -> dict[str, str]:
        source = {
            "source_id": "SRC-A",
            "report_reference": "report.md#ref-source-a",
        }
        source.update(overrides)
        return source

    @staticmethod
    def valid_report(*, findings_reference: bool = True) -> str:
        finding = (
            "The result is supported [1](#ref-source-a)."
            if findings_reference
            else "The result is supported."
        )
        other = (
            "A misplaced citation [1](#ref-source-a)."
            if not findings_reference
            else "No additional result."
        )
        return (
            "# Report\n\n"
            '<a id="findings"></a>\n'
            "## What did the analysis find?\n\n"
            f"{finding}\n\n"
            '<a id="other-section"></a>\n'
            "## What else was checked?\n\n"
            f"{other}\n\n"
            '<a id="limitations"></a>\n'
            "## What are the limitations?\n\n"
            "The scope is bounded.\n\n"
            '<a id="references"></a>\n'
            "## Which sources support the report?\n\n"
            '<a id="ref-source-a"></a>\n'
            "1. Source A\n"
        )

    @staticmethod
    def issue_codes(issues: list[object]) -> set[str]:
        return {issue.code for issue in issues}

    def reference_index(self, report_text: str):
        self.assertTrue(
            hasattr(validator, "_reference_index"),
            "validator._reference_index is missing",
        )
        return validator._reference_index(report_text)

    def validate_references(self, report_text: str):
        self.assertTrue(
            hasattr(validator, "_validate_report_references"),
            "validator._validate_report_references is missing",
        )
        return validator._validate_report_references(
            [self.evidence_row()],
            {"SRC-A": self.source()},
            report_text,
            "report.md",
        )

    def test_numeric_reference_replaces_internal_evidence_marker(self):
        records, issues = validator._validate_evidence_rows(
            [self.evidence_row()],
            {"SRC-A": self.source()},
            self.valid_report(),
            self.package_root,
            "report.md",
        )

        self.assertEqual({"REC-A"}, set(records))
        self.assertNotIn("report.critical_record_unlinked", self.issue_codes(issues))

    def test_internal_evidence_marker_is_rejected(self):
        issues = validator._validate_report_story("# Report\n\n[E:REC-A]\n")

        self.assertIn("report.internal_evidence_marker_present", self.issue_codes(issues))

    def test_valid_reference_index_has_no_issues(self):
        index, issues = self.reference_index(self.valid_report())

        self.assertEqual({"ref-source-a": 1}, index)
        self.assertEqual(set(), self.issue_codes(issues))

    def test_duplicate_reference_anchor_is_rejected(self):
        report_text = (
            '<a id="references"></a>\n'
            "## References\n\n"
            '<a id="ref-source-a"></a>\n'
            "1. Source A\n\n"
            '<a id="ref-source-a"></a>\n'
            "2. Source A duplicate\n"
        )

        _, issues = self.reference_index(report_text)

        self.assertIn("report.reference_anchor_duplicate", self.issue_codes(issues))

    def test_bare_duplicate_reference_anchor_is_rejected(self):
        report_text = (
            '<a id="references"></a>\n'
            "## References\n\n"
            '<a id="ref-source-a"></a>\n'
            "1. Source A\n\n"
            '<a id="ref-source-a"></a>\n'
        )

        _, issues = self.reference_index(report_text)

        self.assertIn("report.reference_anchor_duplicate", self.issue_codes(issues))

    def test_duplicate_reference_number_is_rejected(self):
        report_text = (
            '<a id="references"></a>\n'
            "## References\n\n"
            '<a id="ref-source-a"></a>\n'
            "1. Source A\n\n"
            '<a id="ref-source-b"></a>\n'
            "1. Source B\n"
        )

        _, issues = self.reference_index(report_text)

        self.assertIn("report.reference_number_duplicate", self.issue_codes(issues))

    def test_reference_number_sequence_must_be_contiguous(self):
        report_text = (
            '<a id="references"></a>\n'
            "## References\n\n"
            '<a id="ref-source-a"></a>\n'
            "1. Source A\n\n"
            '<a id="ref-source-b"></a>\n'
            "3. Source B\n"
        )

        _, issues = self.reference_index(report_text)

        self.assertIn("report.reference_number_sequence_invalid", self.issue_codes(issues))

    def test_reference_numbers_must_follow_references_order(self):
        report_text = (
            '<a id="references"></a>\n'
            "## References\n\n"
            '<a id="ref-source-b"></a>\n'
            "2. Source B\n\n"
            '<a id="ref-source-a"></a>\n'
            "1. Source A\n"
        )

        _, issues = self.reference_index(report_text)

        self.assertIn("report.reference_number_sequence_invalid", self.issue_codes(issues))

    def test_references_anchor_is_required(self):
        report_text = '<a id="ref-source-a"></a>\n1. Source A\n'

        index, issues = self.reference_index(report_text)

        self.assertEqual({}, index)
        self.assertIn("report.references_section_missing", self.issue_codes(issues))

    def test_references_anchor_must_be_unique(self):
        report_text = (
            '<a id="references"></a>\n'
            "## References\n\n"
            '<a id="ref-source-a"></a>\n'
            "1. Source A\n\n"
            '<a id="references"></a>\n'
            "## Duplicate References\n"
        )

        index, issues = self.reference_index(report_text)

        self.assertEqual({}, index)
        self.assertIn("report.references_section_duplicate", self.issue_codes(issues))

    def test_reference_entry_before_references_is_not_indexed(self):
        report_text = (
            '<a id="ref-source-a"></a>\n'
            "1. Source A\n\n"
            '<a id="references"></a>\n'
            "## References\n"
        )

        index, issues = self.reference_index(report_text)

        self.assertEqual({}, index)
        self.assertIn(
            "report.reference_entry_outside_references",
            self.issue_codes(issues),
        )

    def test_references_anchor_requires_a_section_heading(self):
        report_text = (
            '<a id="references"></a>\n\n'
            '<a id="ref-source-a"></a>\n'
            "1. Source A\n"
        )

        index, issues = self.reference_index(report_text)

        self.assertEqual({}, index)
        self.assertIn("report.references_heading_missing", self.issue_codes(issues))

    def test_references_must_be_the_terminal_report_section(self):
        report_text = self.valid_report() + (
            "\n<a id=\"appendix\"></a>\n"
            "## Appendix\n\n"
            "Supporting detail.\n"
        )

        _, issues = self.reference_index(report_text)

        self.assertIn("report.references_section_not_terminal", self.issue_codes(issues))

    def test_setext_heading_after_references_is_not_terminal(self):
        report_text = self.valid_report() + (
            "\nAppendix\n"
            "--------\n\n"
            '<a id="ref-source-b"></a>\n'
            "2. Source B\n"
        )

        index, issues = self.reference_index(report_text)

        self.assertEqual({"ref-source-a": 1}, index)
        self.assertIn("report.references_section_not_terminal", self.issue_codes(issues))
        self.assertIn(
            "report.reference_entry_outside_references",
            self.issue_codes(issues),
        )

    def test_references_section_accepts_setext_heading(self):
        report_text = (
            "# Report\n\n"
            '<a id="references"></a>\n'
            "References\n"
            "==========\n\n"
            '<a id="ref-source-a"></a>\n'
            "1. Source A\n"
        )

        index, issues = self.reference_index(report_text)

        self.assertEqual({"ref-source-a": 1}, index)
        self.assertEqual(set(), self.issue_codes(issues))

    def test_masked_setext_heading_after_references_is_not_a_section(self):
        report_text = self.valid_report() + (
            "\n```text\n"
            "Appendix\n"
            "--------\n"
            "```\n\n"
            "<!--\n"
            "Comment appendix\n"
            "---------------\n"
            "-->\n"
        )

        index, issues = self.reference_index(report_text)

        self.assertEqual({"ref-source-a": 1}, index)
        self.assertEqual(set(), self.issue_codes(issues))

    def test_reference_entry_after_later_section_is_not_indexed(self):
        report_text = self.valid_report() + (
            "\n## Appendix\n\n"
            '<a id="ref-source-b"></a>\n'
            "2. Source B\n"
        )

        index, issues = self.reference_index(report_text)

        self.assertEqual({"ref-source-a": 1}, index)
        self.assertIn(
            "report.reference_entry_outside_references",
            self.issue_codes(issues),
        )

    def test_reference_link_target_must_exist(self):
        report_text = "Claim [1](#ref-missing).\n"

        _, issues = self.reference_index(report_text)

        self.assertIn("report.reference_target_unknown", self.issue_codes(issues))

    def test_reference_link_number_must_match_entry(self):
        report_text = (
            "Claim [2](#ref-source-a).\n\n"
            '<a id="references"></a>\n'
            "## References\n\n"
            '<a id="ref-source-a"></a>\n'
            "1. Source A\n"
        )

        _, issues = self.reference_index(report_text)

        self.assertIn("report.reference_number_mismatch", self.issue_codes(issues))

    def test_critical_source_reference_must_be_in_its_section(self):
        issues = self.validate_references(self.valid_report(findings_reference=False))

        self.assertIn("report.critical_source_reference_missing", self.issue_codes(issues))
        missing = [
            issue
            for issue in issues
            if issue.code == "report.critical_source_reference_missing"
        ]
        self.assertEqual(["evidence.tsv:2"], [issue.locator for issue in missing])

    def test_reference_in_html_comment_does_not_close_critical_source(self):
        report_text = self.valid_report(findings_reference=False).replace(
            "The result is supported.",
            "The result is supported.\n\n<!-- [1](#ref-source-a) -->",
        )

        issues = self.validate_references(report_text)

        self.assertIn("report.critical_source_reference_missing", self.issue_codes(issues))

    def test_reference_in_fenced_code_does_not_close_critical_source(self):
        report_text = self.valid_report(findings_reference=False).replace(
            "The result is supported.",
            "The result is supported.\n\n```markdown\n[1](#ref-source-a)\n```",
        )

        issues = self.validate_references(report_text)

        self.assertIn("report.critical_source_reference_missing", self.issue_codes(issues))

    def test_reference_in_container_fence_does_not_close_critical_source(self):
        fenced_examples = (
            "> ```markdown\n> [1](#ref-source-a)\n> ```",
            "- ```markdown\n  [1](#ref-source-a)\n  ```",
            "- item\n    ```markdown\n    [1](#ref-source-a)\n    ```",
            "> - ```markdown\n>   [1](#ref-source-a)\n>   ```",
        )
        for fenced_example in fenced_examples:
            with self.subTest(fenced_example=fenced_example):
                report_text = self.valid_report(findings_reference=False).replace(
                    "The result is supported.",
                    f"The result is supported.\n\n{fenced_example}",
                )

                issues = self.validate_references(report_text)

                self.assertIn(
                    "report.critical_source_reference_missing",
                    self.issue_codes(issues),
                )

    def test_reference_in_wide_ordered_list_fence_does_not_close_source(self):
        fenced_examples = (
            "100. item\n     ```markdown\n     [1](#ref-source-a)\n     ```",
            (
                "123456789. item\n"
                "           ```markdown\n"
                "           [1](#ref-source-a)\n"
                "           ```"
            ),
            (
                "> 100. item\n"
                ">      ```markdown\n"
                ">      [1](#ref-source-a)\n"
                ">      ```"
            ),
            (
                "- outer\n"
                "  100. item\n"
                "       ```markdown\n"
                "       [1](#ref-source-a)\n"
                "       ```"
            ),
        )
        for fenced_example in fenced_examples:
            with self.subTest(fenced_example=fenced_example):
                report_text = self.valid_report(findings_reference=False).replace(
                    "The result is supported.",
                    f"The result is supported.\n\n{fenced_example}",
                )

                issues = self.validate_references(report_text)

                self.assertIn(
                    "report.critical_source_reference_missing",
                    self.issue_codes(issues),
                )

    def test_comment_syntax_in_fenced_code_does_not_hide_later_references(self):
        report_text = self.valid_report().replace(
            "# Report\n\n",
            "# Report\n\n```html\n<!--\n```\n\n",
        )

        issues = self.validate_references(report_text)

        self.assertEqual(set(), self.issue_codes(issues))

    def test_fence_syntax_in_html_comment_does_not_hide_later_references(self):
        report_text = self.valid_report().replace(
            "# Report\n\n",
            "# Report\n\n<!--\n```\n-->\n\n",
        )

        issues = self.validate_references(report_text)

        self.assertEqual(set(), self.issue_codes(issues))

    def test_critical_source_reference_closes_in_its_section(self):
        issues = self.validate_references(self.valid_report())

        self.assertNotIn("report.critical_source_reference_missing", self.issue_codes(issues))
        self.assertEqual(set(), self.issue_codes(issues))


if __name__ == "__main__":
    unittest.main()
