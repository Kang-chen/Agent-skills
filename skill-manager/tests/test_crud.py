#!/usr/bin/env python3
"""
全局 skill 增删改查流程测试。
"""

import json
import os
import shutil
from pathlib import Path

import pytest

# 导入被测模块（conftest 已将 scripts 加入 sys.path）
from scripts.utils import get_skills_from_dir, get_skill_description
from scripts.create import cmd_create, normalize_skill_name
from scripts.sync import sync_skill, cmd_sync
from scripts.remove import remove_skill_from_target, cmd_remove
from scripts.validate import validate_skill, _parse_yaml_frontmatter


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _make_skill(source_dir: Path, name: str, description: str = "A test skill. Use when testing.") -> Path:
    """在 source_dir 中手动创建一个合法 skill 并返回路径。"""
    skill_dir = source_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"---\n\n"
        f"# {name}\n\n"
        f"Auto-generated test skill.\n"
    )
    return skill_dir


# ---------------------------------------------------------------------------
# 创建测试
# ---------------------------------------------------------------------------

class TestCreate:
    """测试 skill 创建流程。"""

    def test_create_skill(self, tmp_ai_skills, mock_config, monkeypatch_config):
        """创建一个 skill，验证目录和 SKILL.md 存在且 frontmatter 合法。"""
        result = cmd_create(
            "my-new-skill",
            path=str(tmp_ai_skills),
            json_output=False,
        )

        assert result["success"] is True
        assert result["skill"] == "my-new-skill"

        skill_dir = tmp_ai_skills / "my-new-skill"
        assert skill_dir.exists()
        assert (skill_dir / "SKILL.md").exists()

        # 验证 frontmatter 含 name 和 description
        content = (skill_dir / "SKILL.md").read_text()
        fm = _parse_yaml_frontmatter(content)
        assert "name" in fm
        assert "description" in fm

    def test_create_skill_with_resources(self, tmp_ai_skills, mock_config, monkeypatch_config):
        """创建 skill 时附带 scripts/references/assets 子目录。"""
        result = cmd_create(
            "resource-skill",
            path=str(tmp_ai_skills),
            resources="scripts,references,assets",
            json_output=False,
        )

        assert result["success"] is True
        skill_dir = tmp_ai_skills / "resource-skill"
        assert skill_dir.exists()

        # 验证资源目录（cmd_create fallback 分支会创建这些目录）
        for res in ("scripts", "references", "assets"):
            assert (skill_dir / res).exists(), f"{res}/ directory should exist"

    def test_create_duplicate_fails(self, tmp_ai_skills, mock_config, monkeypatch_config):
        """重复创建同名 skill 应返回错误。"""
        # test-skill 已存在于 tmp_ai_skills fixture
        result = cmd_create(
            "test-skill",
            path=str(tmp_ai_skills),
            json_output=False,
        )
        assert result["success"] is False
        assert result["error"] is not None

    def test_create_normalizes_name(self):
        """验证名称规范化逻辑：大写转小写、空格转连字符。"""
        assert normalize_skill_name("My Cool Skill") == "my-cool-skill"
        assert normalize_skill_name("  HELLO  WORLD  ") == "hello-world"
        assert normalize_skill_name("skill---name") == "skill-name"


# ---------------------------------------------------------------------------
# 列表 / 发现测试
# ---------------------------------------------------------------------------

class TestList:
    """测试 skill 列表和发现功能。"""

    def test_list_skills(self, tmp_ai_skills, mock_config):
        """向源目录添加 3 个 skill，验证全部被发现且按字母排序。"""
        _make_skill(tmp_ai_skills, "alpha-skill")
        _make_skill(tmp_ai_skills, "beta-skill")
        # test-skill 已由 fixture 创建，共 3 个

        skills = get_skills_from_dir(
            tmp_ai_skills,
            mock_config.get("exclude_skills", []),
        )

        assert len(skills) == 3
        assert skills == sorted(skills), "Skills should be sorted alphabetically"
        assert "alpha-skill" in skills
        assert "beta-skill" in skills
        assert "test-skill" in skills

    def test_list_excludes_skill_manager(self, tmp_ai_skills, mock_config):
        """验证 exclude_skills 配置生效，skill-manager 不在结果中。"""
        # skill-manager 目录由 fixture 创建但没有 SKILL.md
        # 手动加一个 SKILL.md 使其成为有效 skill
        sm_dir = tmp_ai_skills / "skill-manager"
        (sm_dir / "SKILL.md").write_text(
            "---\nname: skill-manager\ndescription: Manager.\n---\n"
        )

        skills = get_skills_from_dir(
            tmp_ai_skills,
            mock_config.get("exclude_skills", []),
        )
        assert "skill-manager" not in skills

    def test_get_skill_description(self, tmp_ai_skills):
        """验证从 SKILL.md 提取 description 的功能。"""
        desc = get_skill_description(tmp_ai_skills / "test-skill")
        assert "test skill" in desc.lower()


# ---------------------------------------------------------------------------
# 搜索测试
# ---------------------------------------------------------------------------

class TestSearch:
    """测试本地搜索功能。"""

    def test_search_local(self, tmp_ai_skills, mock_config, monkeypatch_config):
        """创建 skill 索引 JSON，验证搜索返回匹配结果。"""
        from scripts.search import _search_skills

        skills_db = [
            {"name": "pdf-reader", "description": "Read PDF files. Use when parsing documents."},
            {"name": "code-review", "description": "Review code quality. Use when doing code review."},
            {"name": "test-helper", "description": "Help with testing. Use when writing tests."},
        ]

        results = _search_skills("pdf", skills_db, limit=10)
        assert len(results) >= 1
        assert results[0]["name"] == "pdf-reader"

    def test_search_no_match(self):
        """搜索不匹配时返回空列表。"""
        from scripts.search import _search_skills

        skills_db = [
            {"name": "alpha", "description": "Something about alpha."},
        ]
        results = _search_skills("zzz-nonexistent", skills_db)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# 安装测试（使用本地 mock repo）
# ---------------------------------------------------------------------------

class TestInstall:
    """测试从 git 仓库安装 skill。"""

    def test_install_from_local_repo(self, tmp_ai_skills, mock_github_repo, mock_config, monkeypatch_config):
        """
        从 mock bare 仓库克隆 skill 到 source_dir，
        验证 skill 目录和 SKILL.md 存在。
        """
        from scripts.install import _git_sparse_checkout, _validate_skill

        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="install-test-")
        try:
            repo_root = _git_sparse_checkout(
                str(mock_github_repo),
                "master",
                ["skills/test-remote-skill"],
                tmp_dir,
            )

            skill_src = os.path.join(repo_root, "skills", "test-remote-skill")
            _validate_skill(skill_src)

            dest = tmp_ai_skills / "test-remote-skill"
            shutil.copytree(skill_src, dest)

            assert dest.exists()
            assert (dest / "SKILL.md").exists()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_parse_github_url(self):
        """验证 GitHub URL 解析逻辑。"""
        from scripts.install import _parse_github_url

        owner, repo, ref, subpath = _parse_github_url(
            "https://github.com/user/repo/tree/main/skills/my-skill"
        )
        assert owner == "user"
        assert repo == "repo"
        assert ref == "main"
        assert subpath == "skills/my-skill"

    def test_parse_github_url_invalid(self):
        """非 GitHub URL 应抛出 InstallError。"""
        from scripts.install import _parse_github_url, InstallError

        with pytest.raises(InstallError):
            _parse_github_url("https://gitlab.com/user/repo")


# ---------------------------------------------------------------------------
# 同步测试
# ---------------------------------------------------------------------------

class TestSync:
    """测试 skill 同步到 IDE 目标目录。"""

    def test_sync_single_skill(self, tmp_ai_skills, mock_config, monkeypatch_config):
        """同步单个 skill，验证其出现在所有目标 IDE 目录。"""
        result = cmd_sync(
            skill_name="test-skill",
            scope="global",
            json_output=False,
        )

        assert result["global"]["synced"] >= 1

        for ide, target_path in mock_config["targets"].items():
            target_dir = Path(target_path) / "test-skill"
            assert target_dir.exists(), f"Skill should be synced to {ide} target"
            assert (target_dir / "SKILL.md").exists()

    def test_sync_all(self, tmp_ai_skills, mock_config, monkeypatch_config):
        """创建多个 skill 后同步全部，验证均出现在所有目标目录。"""
        _make_skill(tmp_ai_skills, "skill-a")
        _make_skill(tmp_ai_skills, "skill-b")

        result = cmd_sync(scope="global", json_output=False)

        # 应至少同步 3 个 skill (test-skill + skill-a + skill-b) x 5 IDEs
        assert result["global"]["synced"] >= 3 * len(mock_config["enabled_ides"])


# ---------------------------------------------------------------------------
# 删除测试
# ---------------------------------------------------------------------------

class TestRemove:
    """测试 skill 删除流程。"""

    def test_remove_skill(self, tmp_ai_skills, mock_config, monkeypatch_config):
        """创建 skill、同步后删除，验证源和目标均被清理。"""
        # 先同步
        cmd_sync(skill_name="test-skill", scope="global", json_output=False)

        # 确认目标中存在
        for ide, target_path in mock_config["targets"].items():
            assert (Path(target_path) / "test-skill").exists()

        # 删除（force=True 跳过交互确认）
        result = cmd_remove(
            "test-skill",
            scope="global",
            force=True,
            json_output=False,
        )

        assert result["success"] is True

        # 源目录中应已删除
        assert not (tmp_ai_skills / "test-skill").exists()

        # 所有目标中也应已删除
        for ide, target_path in mock_config["targets"].items():
            assert not (Path(target_path) / "test-skill").exists()

    def test_remove_from_single_target(self, tmp_ai_skills, mock_config):
        """测试从单个目标目录移除 skill 的低层函数。"""
        target_dir = Path(mock_config["targets"]["claude"])
        skill_target = target_dir / "test-skill"
        shutil.copytree(tmp_ai_skills / "test-skill", skill_target)
        assert skill_target.exists()

        success, msg = remove_skill_from_target("test-skill", target_dir)
        assert success is True
        assert msg == "removed"
        assert not skill_target.exists()

    def test_remove_nonexistent(self, tmp_ai_skills, mock_config):
        """移除不存在的 skill 应返回 'not present'。"""
        target_dir = Path(mock_config["targets"]["claude"])
        success, msg = remove_skill_from_target("nonexistent-skill", target_dir)
        assert success is True
        assert msg == "not present"


# ---------------------------------------------------------------------------
# 验证测试
# ---------------------------------------------------------------------------

class TestValidate:
    """测试 skill 验证逻辑。"""

    def test_validate_valid_skill(self, tmp_ai_skills, mock_config):
        """验证合法 skill 通过校验。"""
        result = validate_skill("test-skill", tmp_ai_skills / "test-skill")
        assert result["valid"] is True
        assert len(result["failures"]) == 0

    def test_validate_invalid_skill_missing_frontmatter(self, tmp_ai_skills):
        """缺少 frontmatter 的 SKILL.md 应报告验证失败。"""
        bad_skill = tmp_ai_skills / "bad-skill"
        bad_skill.mkdir()
        (bad_skill / "SKILL.md").write_text(
            "# Bad Skill\n\nNo frontmatter here.\n"
        )

        result = validate_skill("bad-skill", bad_skill)
        assert result["valid"] is False
        assert len(result["failures"]) > 0

    def test_validate_missing_skill_md(self, tmp_ai_skills):
        """没有 SKILL.md 的目录应报告验证失败。"""
        empty_skill = tmp_ai_skills / "empty-skill"
        empty_skill.mkdir()

        result = validate_skill("empty-skill", empty_skill)
        assert result["valid"] is False
        assert any("SKILL.md not found" in f for f in result["failures"])

    def test_validate_missing_description(self, tmp_ai_skills):
        """frontmatter 中缺少 description 字段应报告验证失败。"""
        no_desc = tmp_ai_skills / "no-desc-skill"
        no_desc.mkdir()
        (no_desc / "SKILL.md").write_text(
            "---\nname: no-desc-skill\n---\n\n# No Description\n"
        )

        result = validate_skill("no-desc-skill", no_desc)
        assert result["valid"] is False

    def test_parse_yaml_frontmatter(self):
        """测试 YAML frontmatter 解析。"""
        content = "---\nname: my-skill\ndescription: Does things.\n---\n\n# Content"
        fm = _parse_yaml_frontmatter(content)
        assert fm["name"] == "my-skill"
        assert fm["description"] == "Does things."

    def test_parse_yaml_frontmatter_missing(self):
        """无 frontmatter 时返回空 dict。"""
        content = "# Just a heading\n\nSome text."
        fm = _parse_yaml_frontmatter(content)
        assert fm == {}
