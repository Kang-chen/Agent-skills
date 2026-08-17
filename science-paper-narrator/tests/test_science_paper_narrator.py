from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SkillContractTests(unittest.TestCase):
    def test_skill_is_self_contained_and_discoverable(self) -> None:
        skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("name: science-paper-narrator", skill)
        self.assertIn("description: Use when", skill)
        self.assertNotIn("workspaces/science-paper-narrator", skill)
        for script in (
            "validate_article_assets.py",
            "validate_figure_crops.py",
            "validate_emphasis_style.py",
        ):
            self.assertTrue((SCRIPTS / script).is_file(), script)
        self.assertTrue((SKILL_ROOT / "requirements.txt").is_file())


class ArticleAssetValidatorTests(unittest.TestCase):
    def _run(self, article: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "validate_article_assets.py"),
                "--require-images",
                "--require-captions",
                str(article),
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

    def test_relative_image_and_caption_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = Path(temp_dir)
            Image.new("RGB", (24, 16), "white").save(fixture / "figure.png")
            article = fixture / "article.md"
            article.write_text("# 测试\n\n![图](figure.png)\n\n*图 1｜可追溯图注。*\n", encoding="utf-8")
            result = self._run(article)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_unresolved_placeholder_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = Path(temp_dir)
            Image.new("RGB", (24, 16), "white").save(fixture / "figure.png")
            article = fixture / "article.md"
            article.write_text(
                "# 测试\n\n【配图建议】稍后补图\n\n![图](figure.png)\n\n*图 1｜可追溯图注。*\n",
                encoding="utf-8",
            )
            result = self._run(article)
            self.assertNotEqual(result.returncode, 0)


class FigureCropValidatorTests(unittest.TestCase):
    def _run(self, required_bounds: list[int]) -> subprocess.CompletedProcess[str]:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        fixture = Path(temp_dir.name)
        source = fixture / "source.png"
        crop = fixture / "crop.png"
        image = Image.new("RGB", (40, 30), "white")
        for x in range(40):
            for y in range(30):
                image.putpixel((x, y), ((x * 7) % 256, (y * 11) % 256, ((x + y) * 13) % 256))
        image.save(source)
        box = [4, 5, 36, 26]
        image.crop(tuple(box)).save(crop)
        manifest = fixture / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "figures": [
                        {
                            "name": "portable crop fixture",
                            "source": source.name,
                            "crop": crop.name,
                            "box": box,
                            "required_bounds": required_bounds,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "validate_figure_crops.py"), str(manifest)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

    def test_source_verified_crop_passes(self) -> None:
        result = self._run([5, 6, 35, 25])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_crop_missing_required_content_fails(self) -> None:
        result = self._run([3, 5, 35, 25])
        self.assertNotEqual(result.returncode, 0)


class EmphasisStyleValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validator = load_module("validate_emphasis_style", SCRIPTS / "validate_emphasis_style.py")

    def _article(self, include_anchors: bool) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        article = Path(temp_dir.name) / "article.md"
        anchor = "**这项关键证据支持主要判断边界**" if include_anchors else "这项关键证据支持主要判断边界"
        filler = "研究设计需要同时说明样本来源比较对象验证范围与结论边界。" * 6
        paragraph = f"{anchor}{filler}{anchor}{filler}{anchor}{filler}"
        article.write_text(
            f"# 测试文章\n\n## 方法与设计\n\n{paragraph}\n\n## 结果与限制\n\n{paragraph}\n",
            encoding="utf-8",
        )
        return article

    def test_balanced_anchors_pass(self) -> None:
        errors, _warnings, metrics = self.validator.validate(self._article(True))
        self.assertEqual(errors, [])
        self.assertEqual(metrics["covered_sections"], 2)

    def test_missing_anchors_fail(self) -> None:
        errors, _warnings, _metrics = self.validator.validate(self._article(False))
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
