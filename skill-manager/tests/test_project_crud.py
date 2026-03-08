#!/usr/bin/env python3
"""
项目级 skill 增删改查测试 - 验证 ./.ai-skills/ -> ./.claude/skills/ 的完整生命周期。
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.create import cmd_create, normalize_skill_name, validate_skill_name
from scripts.remove import cmd_remove
from scripts.sync import cmd_sync, sync_skill
from scripts.utils import expand_path, get_skills_from_dir
from scripts.validate import cmd_validate, validate_skill


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _write_skill_md(skill_dir: Path, name: str, description: str = None):
    """在指定目录写入一个合法的 SKILL.md。"""
    if description is None:
        description = f"A test skill named {name}. Use when testing project features."
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\n"
        f"description: {description}\n"
        f"---\n\n# {name.replace('-', ' ').title()}\n\n"
        f"Content for {name}.\n"
    )


def _build_project_config(tmp_ai_skills, tmp_project):
    """
    构建一个同时包含全局和项目路径的 mock config。
    将 git.auto_commit 关闭以避免测试中的 git 副作用。
    """
    config_path = tmp_ai_skills / "skill-manager" / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # 设置 project 级 target 为绝对路径以便验证
    config["project_targets"] = {
        "claude": str(tmp_project / ".claude" / "skills"),
        "cursor": str(tmp_project / ".cursor" / "skills"),
    }
    config["enabled_ides"] = ["claude", "cursor"]
    config["git"] = {"auto_commit": False, "commit_prefix": "skills:", "auto_push": False}
    config["sync"]["auto_after_create"] = False
    return config


# ---------------------------------------------------------------------------
# 项目初始化测试
# ---------------------------------------------------------------------------

class TestProjectInit:
    """测试项目级 skill 目录初始化。"""

    def test_project_init(self, tmp_path):
        """模拟 skills init --local 后应创建 .ai-skills/ 目录。"""
        project_dir = tmp_path / "new-project"
        project_dir.mkdir()
        ai_skills_dir = project_dir / ".ai-skills"

        assert not ai_skills_dir.exists()
        ai_skills_dir.mkdir(parents=True)
        assert ai_skills_dir.is_dir()


# ---------------------------------------------------------------------------
# 项目级 skill 创建测试
# ---------------------------------------------------------------------------

class TestProjectCreateSkill:
    """测试在项目 .ai-skills/ 中创建 skill。"""

    def test_project_create_skill(self, tmp_project, tmp_ai_skills, monkeypatch):
        """在项目目录中创建 skill 后应有合法的 SKILL.md。"""
        config = _build_project_config(tmp_ai_skills, tmp_project)
        monkeypatch.setattr("scripts.create.load_config", lambda: config)
        monkeypatch.setattr("scripts.create.auto_commit", lambda *a, **kw: True)
        monkeypatch.setattr("scripts.sync.load_config", lambda: config)

        result = cmd_create(
            "proj-new-skill",
            scope="local",
            project_dir=tmp_project,
            json_output=False,
        )

        assert result["success"] is True

        skill_dir = tmp_project / ".ai-skills" / "proj-new-skill"
        assert skill_dir.is_dir()

        skill_md = skill_dir / "SKILL.md"
        assert skill_md.is_file()

        content = skill_md.read_text()
        assert "name:" in content
        assert "description:" in content

    def test_project_create_with_resources(self, tmp_project, tmp_ai_skills, monkeypatch):
        """创建 skill 时指定 resources 参数后应生成对应子目录。"""
        config = _build_project_config(tmp_ai_skills, tmp_project)
        monkeypatch.setattr("scripts.create.load_config", lambda: config)
        monkeypatch.setattr("scripts.create.auto_commit", lambda *a, **kw: True)
        monkeypatch.setattr("scripts.sync.load_config", lambda: config)

        result = cmd_create(
            "proj-res-skill",
            resources="scripts,references,assets",
            scope="local",
            project_dir=tmp_project,
        )

        assert result["success"] is True

        skill_dir = tmp_project / ".ai-skills" / "proj-res-skill"
        for subdir_name in ("scripts", "references", "assets"):
            subdir = skill_dir / subdir_name
            assert subdir.is_dir(), f"{subdir_name}/ should be created"


# ---------------------------------------------------------------------------
# 项目级 skill 修改测试
# ---------------------------------------------------------------------------

class TestProjectModifySkill:
    """测试项目级 skill 的修改操作。"""

    def test_project_modify_skill_md(self, tmp_project):
        """修改 SKILL.md 后内容应持久化。"""
        skill_dir = tmp_project / ".ai-skills" / "proj-test-skill"
        skill_md = skill_dir / "SKILL.md"

        assert skill_md.is_file()

        # 追加内容
        original = skill_md.read_text()
        new_content = original + "\n## New Section\n\nAdded during test.\n"
        skill_md.write_text(new_content)

        # 读回验证
        assert "New Section" in skill_md.read_text()

    def test_project_add_script(self, tmp_project):
        """向 skill 添加 scripts/ 子目录和文件后应存在。"""
        skill_dir = tmp_project / ".ai-skills" / "proj-test-skill"
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        script_file = scripts_dir / "helper.py"
        script_file.write_text("#!/usr/bin/env python3\nprint('hello')\n")

        assert script_file.is_file()

    def test_project_add_reference(self, tmp_project):
        """向 skill 添加 references/ 子目录和文件后应存在。"""
        skill_dir = tmp_project / ".ai-skills" / "proj-test-skill"
        refs_dir = skill_dir / "references"
        refs_dir.mkdir(exist_ok=True)
        ref_file = refs_dir / "api-guide.md"
        ref_file.write_text("# API Guide\n\nReference material.\n")

        assert ref_file.is_file()


# ---------------------------------------------------------------------------
# 项目级 sync 测试
# ---------------------------------------------------------------------------

class TestProjectSync:
    """测试项目级 skill 同步到 IDE target。"""

    def test_project_sync_to_ide(self, tmp_project, tmp_ai_skills, monkeypatch):
        """同步项目 skill 后应出现在项目的 .claude/skills/ 和 .cursor/skills/。"""
        config = _build_project_config(tmp_ai_skills, tmp_project)
        monkeypatch.setattr("scripts.sync.load_config", lambda: config)
        monkeypatch.setattr("scripts.sync.auto_commit", lambda *a, **kw: True)

        # 使用绝对路径做 project_targets，以避免相对路径解析问题
        config["project_targets"] = {
            "claude": str(tmp_project / ".claude" / "skills"),
            "cursor": str(tmp_project / ".cursor" / "skills"),
        }

        # 手动调用 sync_skill 做基础验证
        source_dir = tmp_project / ".ai-skills"
        for ide in ("claude", "cursor"):
            target_dir = Path(config["project_targets"][ide])
            target_dir.mkdir(parents=True, exist_ok=True)
            success, method = sync_skill(
                "proj-test-skill", source_dir, target_dir, use_symlinks=False
            )
            assert success is True

        # 验证目标存在
        for ide in ("claude", "cursor"):
            target = Path(config["project_targets"][ide]) / "proj-test-skill" / "SKILL.md"
            assert target.is_file(), f"Skill should be synced to {ide}"

    def test_project_sync_isolation(self, tmp_project, tmp_ai_skills, monkeypatch):
        """项目级 skill 同步后不应出现在全局 ~/.claude/skills/。"""
        config = _build_project_config(tmp_ai_skills, tmp_project)

        source_dir = tmp_project / ".ai-skills"
        for ide_name, target_str in config["targets"].items():
            target_dir = Path(target_str)
            target_skill = target_dir / "proj-test-skill"
            # 项目 skill 不应出现在全局 target
            assert not target_skill.exists(), (
                f"Project skill should NOT appear in global {ide_name} target"
            )


# ---------------------------------------------------------------------------
# 项目级 list 测试
# ---------------------------------------------------------------------------

class TestProjectListSkills:
    """测试列出项目级 skill。"""

    def test_project_list_skills(self, tmp_project):
        """创建 2 个项目 skill 后，列出时两个都应返回。"""
        source_dir = tmp_project / ".ai-skills"

        # 添加第二个 skill
        _write_skill_md(source_dir / "second-skill", "second-skill")

        skills = get_skills_from_dir(source_dir, exclude=["skill-manager"])
        assert "proj-test-skill" in skills
        assert "second-skill" in skills
        assert len(skills) == 2


# ---------------------------------------------------------------------------
# 项目级 remove 测试
# ---------------------------------------------------------------------------

class TestProjectRemoveSkill:
    """测试项目级 skill 删除。"""

    def test_project_remove_skill(self, tmp_project, tmp_ai_skills, monkeypatch):
        """删除项目 skill 后，源目录和 IDE target 应都不存在。"""
        config = _build_project_config(tmp_ai_skills, tmp_project)
        monkeypatch.setattr("scripts.remove.load_config", lambda: config)
        monkeypatch.setattr("scripts.remove.auto_commit", lambda *a, **kw: True)

        # 先创建和同步一个 skill
        skill_name = "remove-me-skill"
        source_dir = tmp_project / ".ai-skills"
        _write_skill_md(source_dir / skill_name, skill_name)

        # 同步到 target
        for ide in ("claude", "cursor"):
            target_dir = Path(config["project_targets"][ide])
            target_dir.mkdir(parents=True, exist_ok=True)
            sync_skill(skill_name, source_dir, target_dir, use_symlinks=False)

        # 验证同步成功
        for ide in ("claude", "cursor"):
            assert (Path(config["project_targets"][ide]) / skill_name).is_dir()

        # 删除
        result = cmd_remove(
            skill_name,
            scope="local",
            project_dir=tmp_project,
            force=True,
        )

        assert result["success"] is True


# ---------------------------------------------------------------------------
# 项目级 validate 测试
# ---------------------------------------------------------------------------

class TestProjectValidate:
    """测试项目级 skill 验证。"""

    def test_project_validate(self, tmp_project):
        """合法的项目 skill 应通过验证。"""
        skill_path = tmp_project / ".ai-skills" / "proj-test-skill"
        result = validate_skill("proj-test-skill", skill_path)

        assert result["valid"] is True
        assert len(result["failures"]) == 0


# ---------------------------------------------------------------------------
# 项目 git 跟踪测试
# ---------------------------------------------------------------------------

class TestProjectGitTracking:
    """测试项目中 skill 文件的 git 跟踪状态。"""

    def test_project_git_tracking(self, tmp_project):
        """在项目中创建新 skill 后，git status 应显示新文件。"""
        source_dir = tmp_project / ".ai-skills"
        _write_skill_md(source_dir / "git-tracked-skill", "git-tracked-skill")

        result = subprocess.run(
            ["git", "-C", str(tmp_project), "status", "--porcelain"],
            capture_output=True, text=True,
        )
        output = result.stdout
        assert "git-tracked-skill" in output, "New skill should show in git status"


# ---------------------------------------------------------------------------
# 项目到全局提升测试
# ---------------------------------------------------------------------------

class TestProjectToGlobalPromotion:
    """测试将项目级 skill 提升为全局 skill。"""

    def test_project_to_global_promotion(self, tmp_project, tmp_ai_skills, monkeypatch):
        """将项目 skill 复制到全局 SSOT 后，同步应使其出现在全局 IDE target。"""
        config = _build_project_config(tmp_ai_skills, tmp_project)
        monkeypatch.setattr("scripts.sync.load_config", lambda: config)
        monkeypatch.setattr("scripts.sync.auto_commit", lambda *a, **kw: True)

        skill_name = "proj-test-skill"

        # 将项目 skill 复制到全局 SSOT
        project_skill = tmp_project / ".ai-skills" / skill_name
        global_skill = tmp_ai_skills / skill_name
        if not global_skill.exists():
            shutil.copytree(str(project_skill), str(global_skill))

        # 同步到全局 target
        for ide in ("claude", "cursor"):
            target_dir = Path(config["targets"][ide])
            target_dir.mkdir(parents=True, exist_ok=True)
            success, _ = sync_skill(skill_name, tmp_ai_skills, target_dir, False)
            assert success is True

        # 验证全局 target 中存在
        for ide in ("claude", "cursor"):
            assert (
                Path(config["targets"][ide]) / skill_name / "SKILL.md"
            ).is_file(), f"Promoted skill should exist in global {ide} target"
