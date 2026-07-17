"""Static tests for the reader-facing report reference contract."""

from pathlib import Path
import re
import unittest

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_TRACEABILITY = {
    "report_locator_format": "report.md#stable-anchor",
    "explicit_anchor_syntax": '<a id="stable-anchor"></a>',
    "require_explicit_anchors": True,
    "reference_link_format": "[N](#ref-stable-anchor)",
    "reference_anchor_syntax": '<a id="ref-stable-anchor"></a>',
    "reference_section_locator": "report.md#references",
    "require_source_references": True,
}
ANCHOR_PATTERN = re.compile(
    r'<a id="(?P<anchor>[a-z0-9]+(?:-[a-z0-9]+)*)"></a>'
)
REFERENCE_LINK_PATTERN = re.compile(
    r"\[(?P<number>[1-9][0-9]*)\]\(#(?P<anchor>ref-[a-z0-9]+(?:-[a-z0-9]+)*)\)"
)
REFERENCE_ENTRY_PATTERN = re.compile(
    r'<a id="(?P<anchor>ref-[a-z0-9]+(?:-[a-z0-9]+)*)"></a>'
    r"[ \t]*\n(?P<number>[1-9][0-9]*)\.[ \t]+(?P<description>\S[^\n]*)"
)


def read_repository_text(relative_path: str) -> str:
    """Read one UTF-8 text artifact from the skill repository."""

    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


class SkillReferenceContractTests(unittest.TestCase):
    """Keep agent-facing guidance aligned with manifest schema v2."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.skill = read_repository_text("SKILL.md")
        cls.skeleton = read_repository_text("assets/report_skeleton.md")
        cls.contracts = read_repository_text("references/contracts-and-status.md")
        cls.claims = read_repository_text("references/claims-and-review.md")
        cls.story = read_repository_text("references/story-and-figures.md")
        cls.assertions = read_repository_text(
            "evals/assertions/report_assertions.yaml"
        )
        cls.rubric = read_repository_text("evals/rubric.md")
        cls.manifest = yaml.safe_load(
            read_repository_text("assets/report_manifest.yaml")
        )
        cls.report_assertions = yaml.safe_load(cls.assertions)

    def assert_fresh_reader_contract(self, review_text: str) -> None:
        """Assert the three reader-only checks form one review contract."""

        normalized = " ".join(review_text.lower().split())
        self.assertRegex(
            normalized,
            r"raw report.*no internal `record_id`",
        )
        self.assertRegex(
            normalized,
            r"numeric reference.*resolves? to .*non-empty.*"
            r"reader-understandable source description",
        )
        self.assertRegex(
            normalized,
            r"sources? and conclusions?.*without opening.*manifest.*evidence\.tsv",
        )

    def test_manifest_seed_declares_complete_reference_contract_v2(self):
        self.assertEqual(2, self.manifest["schema_version"])
        self.assertEqual(EXPECTED_TRACEABILITY, self.manifest["traceability"])

    def test_report_skeleton_demonstrates_reader_facing_references(self):
        self.assertNotIn("[E:", self.skeleton)
        anchors = [
            match.group("anchor")
            for match in ANCHOR_PATTERN.finditer(self.skeleton)
        ]
        self.assertEqual(1, anchors.count("references"))

        links = list(REFERENCE_LINK_PATTERN.finditer(self.skeleton))
        self.assertGreaterEqual(len(links), 1)
        self.assertIn("[1](#ref-source-example)", self.skeleton)

        entries = list(REFERENCE_ENTRY_PATTERN.finditer(self.skeleton))
        entry_anchors = [match.group("anchor") for match in entries]
        self.assertEqual(len(entry_anchors), len(set(entry_anchors)))
        entries_by_anchor = {match.group("anchor"): match for match in entries}

        for link in links:
            anchor = link.group("anchor")
            self.assertEqual(1, anchors.count(anchor))
            self.assertIn(anchor, entries_by_anchor)
            entry = entries_by_anchor[anchor]
            self.assertEqual(link.group("number"), entry.group("number"))
            self.assertTrue(entry.group("description").strip())

    def test_inventory_recipe_keeps_sources_and_evidence_records_distinct(self):
        inventory_match = re.search(
            r"2\. \*\*Inventory:\*\*(?P<body>.*?)\n3\. \*\*Plan the story:\*\*",
            self.skill,
            re.DOTALL,
        )
        self.assertIsNotNone(inventory_match)
        inventory_step = " ".join(inventory_match.group("body").lower().split())
        self.assertRegex(
            inventory_step,
            r"inventory sources in the manifest",
        )
        self.assertRegex(
            inventory_step,
            r"register report-critical evidence records in `evidence\.tsv`",
        )
        self.assertRegex(
            inventory_step,
            r"evidence row.*`source_ids`.*supporting manifest sources",
        )
        self.assertRegex(
            inventory_step,
            r"manifest source.*stable `report_reference`",
        )
        self.assertNotRegex(
            inventory_step,
            r"register.*sources?.*in `evidence\.tsv`",
        )

    def test_skill_and_contract_define_persistent_v2_reference_roles(self):
        self.assertIn("reader-facing", self.skill)
        self.assertIn("[N](#ref-stable-anchor)", self.skill)
        self.assertIn("schema_version: 2", self.skill)
        self.assertRegex(
            " ".join(self.skill.split()),
            r"`evidence\.tsv` remains the persistent review sidecar.*"
            r"`record_id` values stay in the sidecar",
        )
        self.assertRegex(
            " ".join(self.skill.split()),
            r"substantive revision of a v1 package must migrate.*"
            r"`schema_version: 2`.*before terminal validation",
        )
        self.assertIn("not an automatic rewrite", self.contracts)

    def test_fresh_reader_reviews_are_self_contained_and_reference_aware(self):
        claim_review_match = re.search(
            r"3\. \*\*Fresh-reader review:\*\*(?P<body>.*?)\n4\. \*\*Visual review:\*\*",
            self.claims,
            re.DOTALL,
        )
        self.assertIsNotNone(claim_review_match)
        self.assert_fresh_reader_contract(claim_review_match.group("body"))

        story_review = self.story.split("## Fresh-reader check", maxsplit=1)
        self.assertEqual(2, len(story_review))
        self.assert_fresh_reader_contract(story_review[1])

    def test_evaluation_contract_checks_references_and_sidecars(self):
        requirements = self.report_assertions["report_requirements"]
        reference_requirements = requirements["reader_facing_references"]
        self.assertEqual(
            r"\[E:[^]]+\]",
            reference_requirements["forbidden_internal_marker_pattern"],
        )
        self.assertTrue(
            reference_requirements[
                "require_numeric_reference_for_report_critical_sources"
            ]
        )
        self.assertEqual(
            "references",
            reference_requirements["reference_section_anchor"],
        )

        sidecar_requirements = requirements["persistent_sidecars"]
        self.assertTrue(sidecar_requirements["evidence_ledger_required"])
        self.assertEqual(
            "manifests/evidence.tsv",
            sidecar_requirements["evidence_ledger_path"],
        )
        self.assertEqual(
            2,
            sidecar_requirements["source_manifest_schema_version"],
        )
        self.assertEqual(3, len(sidecar_requirements["required_closure"]))
        self.assertIn("numeric anchor", self.rubric)
        self.assertIn("evidence.tsv", self.rubric)


if __name__ == "__main__":
    unittest.main()
