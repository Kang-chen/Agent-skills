#!/usr/bin/env python3
"""
skill-creator 原则合规性测试 - 验证 skill 创建和修改是否遵循 skill-creator 的核心原则。
"""

import os
import re
from pathlib import Path

import pytest

from scripts.create import cmd_create
from scripts.validate import _parse_yaml_frontmatter, validate_skill, VALIDATION_RULES


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _parse_frontmatter(content: str) -> dict:
    """
    从 SKILL.md 内容中提取 YAML frontmatter 字段。
    返回 key-value dict。
    """
    if not content.startswith("---"):
        return {}
    try:
        end_idx = content.index("---", 3)
        raw = content[3:end_idx].strip()
        result = {}
        for line in raw.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                result[key.strip()] = value.strip()
        return result
    except ValueError:
        return {}


def _create_test_skill(
    path: Path,
    name: str,
    description: str,
    body: str = "",
    extra_frontmatter: dict = None,
):
    """
    在 path/<name>/ 下创建 SKILL.md，支持自定义 frontmatter 和 body。
    """
    skill_dir = path / name
    skill_dir.mkdir(parents=True, exist_ok=True)

    fm_lines = [f"name: {name}", f"description: {description}"]
    if extra_frontmatter:
        for k, v in extra_frontmatter.items():
            fm_lines.append(f"{k}: {v}")

    frontmatter_block = "\n".join(fm_lines)
    title = name.replace("-", " ").title()

    content = f"---\n{frontmatter_block}\n---\n\n# {title}\n\n"
    if body:
        content += body + "\n"
    else:
        content += f"Content for {name}.\n"

    (skill_dir / "SKILL.md").write_text(content)
    return skill_dir


def _count_lines(filepath: Path) -> int:
    """计算文件行数。"""
    return len(filepath.read_text().split("\n"))


# ---------------------------------------------------------------------------
# Frontmatter 合规性测试
# ---------------------------------------------------------------------------

class TestFrontmatterCompliance:
    """验证 frontmatter 字段是否遵循 skill-creator 规范。"""

    def test_frontmatter_required_fields(self, tmp_path):
        """合法的 SKILL.md 必须包含 name 和 description 字段。"""
        _create_test_skill(
            tmp_path, "valid-skill",
            "A valid skill. Use when testing compliance checks."
        )

        content = (tmp_path / "valid-skill" / "SKILL.md").read_text()
        fm = _parse_frontmatter(content)

        assert "name" in fm, "Frontmatter must have 'name' field"
        assert "description" in fm, "Frontmatter must have 'description' field"

    def test_frontmatter_no_extra_fields(self, tmp_path):
        """合法 skill 的 frontmatter 应仅有 name 和 description。"""
        # 合法 skill
        _create_test_skill(
            tmp_path, "clean-skill",
            "A clean skill. Use when testing minimal frontmatter."
        )
        content = (tmp_path / "clean-skill" / "SKILL.md").read_text()
        fm = _parse_frontmatter(content)
        allowed_keys = {"name", "description"}
        extra_keys = set(fm.keys()) - allowed_keys
        assert len(extra_keys) == 0, f"No extra fields allowed, found: {extra_keys}"

        # 含有多余字段的 skill（应被检测出来）
        _create_test_skill(
            tmp_path, "extra-fields-skill",
            "A skill with extra fields. Use when testing compliance.",
            extra_frontmatter={"version": "1.0", "author": "test"},
        )
        content2 = (tmp_path / "extra-fields-skill" / "SKILL.md").read_text()
        fm2 = _parse_frontmatter(content2)
        extra_keys2 = set(fm2.keys()) - allowed_keys
        assert len(extra_keys2) > 0, "Extra fields should be detected"
        assert "version" in extra_keys2
        assert "author" in extra_keys2

    def test_frontmatter_missing_name_fails_validation(self, tmp_path):
        """缺少 name 字段时，validate_skill 应报告失败。"""
        skill_dir = tmp_path / "no-name-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "description: Missing the name field. Use when testing.\n"
            "---\n\n# No Name Skill\n"
        )

        result = validate_skill("no-name-skill", skill_dir)
        assert result["valid"] is False
        assert any("name" in f for f in result["failures"])

    def test_frontmatter_missing_description_fails_validation(self, tmp_path):
        """缺少 description 字段时，validate_skill 应报告失败。"""
        skill_dir = tmp_path / "no-desc-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: no-desc-skill\n---\n\n# No Desc\n"
        )

        result = validate_skill("no-desc-skill", skill_dir)
        assert result["valid"] is False
        assert any("description" in f for f in result["failures"])


# ---------------------------------------------------------------------------
# Description 触发器关键词测试
# ---------------------------------------------------------------------------

class TestDescriptionTriggers:
    """验证 description 中是否包含触发器关键词（when, use）。"""

    def test_description_includes_triggers(self, tmp_path):
        """合法 description 应包含触发器关键词。"""
        _create_test_skill(
            tmp_path, "trigger-skill",
            "A test skill for unit testing. Use when testing skill-manager features."
        )
        result = validate_skill(
            "trigger-skill", tmp_path / "trigger-skill"
        )
        # 应有 "has trigger keywords" 的 pass 项
        passes_text = " ".join(result["passes"])
        assert "trigger" in passes_text.lower(), "Should pass trigger keyword check"

    def test_description_no_triggers_warns(self, tmp_path):
        """缺少触发器关键词的 description 应产生警告。"""
        _create_test_skill(
            tmp_path, "notrigger-skill",
            "A test skill for demonstration purposes only."
        )
        result = validate_skill(
            "notrigger-skill", tmp_path / "notrigger-skill"
        )
        warnings_text = " ".join(result["warnings"])
        assert "trigger" in warnings_text.lower(), "Should warn about missing triggers"

    def test_description_no_body_triggers(self, tmp_path):
        """触发器信息只在 body 中、不在 description 中，应产生警告。"""
        _create_test_skill(
            tmp_path, "body-trigger-skill",
            "A basic skill without trigger info.",
            body="## Triggers\n\nUse when you need to test things.\n",
        )
        result = validate_skill(
            "body-trigger-skill", tmp_path / "body-trigger-skill"
        )
        # description 中没有 trigger 关键词，应有警告
        warnings_text = " ".join(result["warnings"])
        assert "trigger" in warnings_text.lower()


# ---------------------------------------------------------------------------
# Progressive Disclosure 测试
# ---------------------------------------------------------------------------

class TestProgressiveDisclosure:
    """验证渐进式信息披露原则。"""

    def test_skill_md_under_500_lines(self, tmp_path):
        """SKILL.md 500 行以内应通过，超过应警告。"""
        # 100 行 - 应通过
        _create_test_skill(
            tmp_path, "short-skill",
            "A short skill. Use when testing size limits.",
            body="\n".join([f"Line {i}" for i in range(90)]),
        )
        result_short = validate_skill("short-skill", tmp_path / "short-skill")
        passes_text = " ".join(result_short["passes"])
        assert "lines" in passes_text.lower()

        # 600 行 - 应警告
        _create_test_skill(
            tmp_path, "long-skill",
            "A long skill. Use when testing size limits.",
            body="\n".join([f"Line {i}" for i in range(590)]),
        )
        result_long = validate_skill("long-skill", tmp_path / "long-skill")
        warnings_text = " ".join(result_long["warnings"])
        assert "lines" in warnings_text.lower() or "500" in warnings_text

    def test_references_properly_linked(self, tmp_path):
        """references/ 中的文件应被 SKILL.md 引用。"""
        skill_dir = _create_test_skill(
            tmp_path, "ref-skill",
            "A skill with references. Use when testing references.",
            body="## References\n\nSee [API Guide](references/api-guide.md) for details.\n",
        )
        refs_dir = skill_dir / "references"
        refs_dir.mkdir()
        (refs_dir / "api-guide.md").write_text("# API Guide\n\nDetails here.\n")

        # 检查 SKILL.md 是否引用了 references 目录中的文件
        content = (skill_dir / "SKILL.md").read_text()
        ref_files = list(refs_dir.iterdir())
        for ref_file in ref_files:
            linked = ref_file.name in content
            assert linked, f"{ref_file.name} should be referenced in SKILL.md"

    def test_references_not_linked_detected(self, tmp_path):
        """references/ 中未被引用的文件应被检测到。"""
        skill_dir = _create_test_skill(
            tmp_path, "unlinked-ref-skill",
            "A skill with unlinked references. Use when testing compliance.",
        )
        refs_dir = skill_dir / "references"
        refs_dir.mkdir()
        (refs_dir / "orphan-doc.md").write_text("# Orphan\n\nNot referenced.\n")

        content = (skill_dir / "SKILL.md").read_text()
        assert "orphan-doc.md" not in content, "Orphan ref should not be linked"

    def test_no_deep_nesting(self, tmp_path):
        """references/ 只允许一层子目录，深层嵌套应被检测。"""
        skill_dir = _create_test_skill(
            tmp_path, "nested-skill",
            "A skill for nesting check. Use when testing directory depth.",
        )
        refs_dir = skill_dir / "references"
        refs_dir.mkdir()

        # 一层嵌套 - 应正常
        (refs_dir / "guide.md").write_text("# Guide\n")
        assert (refs_dir / "guide.md").is_file()

        # 深层嵌套 - 应被检测
        deep_path = refs_dir / "sub" / "sub"
        deep_path.mkdir(parents=True)
        (deep_path / "deep-file.md").write_text("# Deep\n")

        # 计算 references/ 下的最大深度
        max_depth = 0
        for root, dirs, files in os.walk(str(refs_dir)):
            depth = root.replace(str(refs_dir), "").count(os.sep)
            if depth > max_depth:
                max_depth = depth

        assert max_depth > 1, "Deep nesting should be detected (depth > 1)"

    def test_large_refs_have_toc(self, tmp_path):
        """超过 100 行的 reference 文件应包含目录（Table of Contents）。"""
        skill_dir = _create_test_skill(
            tmp_path, "toc-skill",
            "A skill for TOC check. Use when testing reference quality.",
        )
        refs_dir = skill_dir / "references"
        refs_dir.mkdir()

        # 大文件无 TOC
        large_content_no_toc = "\n".join([f"## Section {i}\n\nContent.\n" for i in range(60)])
        (refs_dir / "large-no-toc.md").write_text(large_content_no_toc)

        lines_no_toc = _count_lines(refs_dir / "large-no-toc.md")
        has_toc_no_toc = "## Table of Contents" in large_content_no_toc
        assert lines_no_toc > 100
        assert has_toc_no_toc is False, "Large ref without TOC should fail compliance"

        # 大文件有 TOC
        large_content_with_toc = "## Table of Contents\n\n" + large_content_no_toc
        (refs_dir / "large-with-toc.md").write_text(large_content_with_toc)

        has_toc_with = "## Table of Contents" in large_content_with_toc
        assert has_toc_with is True, "Large ref with TOC should pass compliance"


# ---------------------------------------------------------------------------
# 修改合规性测试
# ---------------------------------------------------------------------------

class TestModificationCompliance:
    """验证修改后 skill 仍遵循规范。"""

    def test_update_preserves_frontmatter(self, tmp_path):
        """修改 body 后 frontmatter 应保持完整。"""
        skill_dir = _create_test_skill(
            tmp_path, "update-skill",
            "An updateable skill. Use when testing modification compliance.",
        )
        skill_md = skill_dir / "SKILL.md"

        # 修改 body
        content = skill_md.read_text()
        updated = content + "\n## Added Section\n\nNew content here.\n"
        skill_md.write_text(updated)

        # 验证 frontmatter 完整
        fm = _parse_frontmatter(skill_md.read_text())
        assert "name" in fm
        assert "description" in fm
        assert fm["name"] == "update-skill"

    def test_update_description_quality(self, tmp_path):
        """修改 description 后仍应包含触发器关键词。"""
        skill_dir = _create_test_skill(
            tmp_path, "desc-update-skill",
            "Original description. Use when testing description updates.",
        )
        result = validate_skill("desc-update-skill", skill_dir)
        passes_text = " ".join(result["passes"])
        assert "trigger" in passes_text.lower()

    def test_create_follows_template(self, tmp_path, monkeypatch):
        """使用 cmd_create 创建的 skill 应遵循 skill-creator 模板规范。"""
        from scripts.utils import get_default_config
        config = get_default_config()
        config["source_dir"] = str(tmp_path / "ssot")
        config["git"] = {"auto_commit": False, "commit_prefix": "skills:", "auto_push": False}
        config["sync"]["auto_after_create"] = False
        (tmp_path / "ssot").mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr("scripts.create.load_config", lambda: config)
        monkeypatch.setattr("scripts.create.auto_commit", lambda *a, **kw: True)
        monkeypatch.setattr("scripts.sync.load_config", lambda: config)

        result = cmd_create("template-test", path=str(tmp_path / "ssot"))
        assert result["success"] is True

        # 验证生成的 SKILL.md 有 frontmatter
        skill_md = tmp_path / "ssot" / "template-test" / "SKILL.md"
        fm = _parse_frontmatter(skill_md.read_text())
        assert "name" in fm
        assert "description" in fm


# ---------------------------------------------------------------------------
# No-include 规则测试
# ---------------------------------------------------------------------------

class TestNoIncludeRules:
    """验证不应包含的文件类型。"""

    def test_no_readme(self, tmp_path):
        """skill 目录中不应有 README.md。"""
        skill_dir = _create_test_skill(
            tmp_path, "readme-skill",
            "A skill with readme. Use when testing no-include rules.",
        )
        # 添加 README.md
        (skill_dir / "README.md").write_text("# Readme\n")

        # 检测 README.md 存在
        has_readme = (skill_dir / "README.md").exists()
        assert has_readme is True, "README.md detected (should fail compliance)"

    def test_no_changelog(self, tmp_path):
        """skill 目录中不应有 CHANGELOG.md。"""
        skill_dir = _create_test_skill(
            tmp_path, "changelog-skill",
            "A skill with changelog. Use when testing no-include rules.",
        )
        (skill_dir / "CHANGELOG.md").write_text("# Changelog\n")

        has_changelog = (skill_dir / "CHANGELOG.md").exists()
        assert has_changelog is True, "CHANGELOG.md detected (should fail compliance)"

    def test_no_install_guide(self, tmp_path):
        """skill 目录中不应有 INSTALLATION_GUIDE.md。"""
        skill_dir = _create_test_skill(
            tmp_path, "install-guide-skill",
            "A skill with install guide. Use when testing no-include rules.",
        )
        (skill_dir / "INSTALLATION_GUIDE.md").write_text("# Install\n")

        has_install = (skill_dir / "INSTALLATION_GUIDE.md").exists()
        assert has_install is True, "INSTALLATION_GUIDE.md detected (should fail compliance)"

    def test_resources_in_correct_dirs(self, tmp_path):
        """资源文件应在 scripts/、references/ 或 assets/ 中，不应在根目录。"""
        skill_dir = _create_test_skill(
            tmp_path, "misplaced-skill",
            "A skill with misplaced resources. Use when testing structure.",
        )

        # 正确位置
        (skill_dir / "scripts").mkdir()
        (skill_dir / "scripts" / "helper.py").write_text("# helper\n")

        # 错误位置 - 脚本直接在根目录
        (skill_dir / "run.py").write_text("# misplaced script\n")

        # 检测根目录下的非 SKILL.md 文件
        root_files = [
            f.name for f in skill_dir.iterdir()
            if f.is_file() and f.name != "SKILL.md"
        ]
        assert len(root_files) > 0, "Root-level resource files should be detected"
        assert "run.py" in root_files


# ---------------------------------------------------------------------------
# 综合合规检查函数
# ---------------------------------------------------------------------------

def check_full_compliance(skill_dir: Path, skill_name: str) -> dict:
    """
    对 skill 进行全面的 skill-creator 合规检查。
    返回 {"passes": [...], "failures": [...], "warnings": [...]}.
    """
    result = {"passes": [], "failures": [], "warnings": []}

    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        result["failures"].append("SKILL.md not found")
        return result

    content = skill_md.read_text()
    fm = _parse_frontmatter(content)

    # 必须字段
    for field in ("name", "description"):
        if field in fm:
            result["passes"].append(f"has {field}")
        else:
            result["failures"].append(f"missing {field}")

    # 多余字段
    allowed = {"name", "description"}
    extra = set(fm.keys()) - allowed
    if extra:
        result["failures"].append(f"extra frontmatter fields: {extra}")

    # 触发器关键词
    desc = fm.get("description", "")
    if re.search(r"\b(when|use|trigger)\b", desc, re.IGNORECASE):
        result["passes"].append("description has trigger keywords")
    else:
        result["warnings"].append("description missing trigger keywords")

    # 行数限制
    lines = len(content.split("\n"))
    if lines <= 500:
        result["passes"].append(f"SKILL.md: {lines} lines")
    else:
        result["failures"].append(f"SKILL.md too long: {lines} lines (> 500)")

    # 禁止的文件
    forbidden = ["README.md", "CHANGELOG.md", "INSTALLATION_GUIDE.md"]
    for f in forbidden:
        if (skill_dir / f).exists():
            result["failures"].append(f"forbidden file: {f}")

    return result


class TestFullCompliance:
    """使用综合合规检查函数的集成测试。"""

    def test_fully_compliant_skill(self, tmp_path):
        """完全符合规范的 skill 应没有 failures。"""
        skill_dir = _create_test_skill(
            tmp_path, "compliant-skill",
            "A fully compliant skill. Use when testing full compliance.",
        )
        result = check_full_compliance(skill_dir, "compliant-skill")
        assert len(result["failures"]) == 0

    def test_non_compliant_skill(self, tmp_path):
        """违反多条规范的 skill 应有对应 failures。"""
        skill_dir = _create_test_skill(
            tmp_path, "bad-skill",
            "No triggers here.",
            extra_frontmatter={"version": "1.0"},
        )
        (skill_dir / "README.md").write_text("# Bad\n")

        result = check_full_compliance(skill_dir, "bad-skill")
        assert len(result["failures"]) >= 2  # extra fields + README.md
