"""Static tests for the reader-facing report reference contract."""

from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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

    def test_report_skeleton_demonstrates_reader_facing_references(self):
        self.assertNotIn("[E:", self.skeleton)
        self.assertIn("[1](#ref-source-example)", self.skeleton)
        self.assertIn('id="references"', self.skeleton)
        self.assertIn('id="ref-source-example"', self.skeleton)

    def test_skill_and_contract_define_schema_v2_reference_roles(self):
        self.assertIn("reader-facing", self.skill)
        self.assertIn("[N](#ref-stable-anchor)", self.skill)
        self.assertIn("schema_version: 2", self.skill)
        self.assertIn("report_reference", self.contracts)
        self.assertIn("record_id", self.contracts)

    def test_review_guidance_closes_report_references_to_the_ledger(self):
        self.assertIn("numeric reference", self.claims)
        self.assertIn("evidence ledger", self.claims)
        self.assertIn("source reference", self.story)

    def test_evaluation_contract_checks_references_and_sidecars(self):
        self.assertIn("reader_facing_references", self.assertions)
        self.assertIn("evidence.tsv", self.assertions)
        self.assertIn("numeric anchor", self.rubric)
        self.assertIn("evidence.tsv", self.rubric)


if __name__ == "__main__":
    unittest.main()
