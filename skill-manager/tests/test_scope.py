#!/usr/bin/env python3
"""
Global/Project 作用域测试 - 验证全局和项目级 skill 的发现、同步和隔离。
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.utils import get_skills_from_dir, get_skill_description
from scripts.sync import cmd_sync, sync_skill
from scripts.cli import get_scope_and_project


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _make_skill(parent_dir: Path, name: str, description: str = "A test skill. Use when testing.") -> Path:
    """在 parent_dir 中创建一个合法 skill。"""
    skill_dir = parent_dir / name
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
# 全局发现
# ---------------------------------------------------------------------------

class TestGlobalDiscovery:
    """全局 skill 发现测试。"""

    def test_global_skill_discovery(self, tmp_ai_skills, mock_config):
        """向全局源目录添加 skill 后，get_skills_from_dir 应全部发现。"""
        _make_skill(tmp_ai_skills, "discover-a")
        _make_skill(tmp_ai_skills, "discover-b")
        # test-skill 已存在 → 共 3 个

        skills = get_skills_from_dir(
            tmp_ai_skills,
            mock_config.get("exclude_skills", []),
        )
        assert len(skills) == 3
        assert "discover-a" in skills
        assert "discover-b" in skills
        assert "test-skill" in skills

    def test_global_discovery_ignores_non_skill_dirs(self, tmp_ai_skills, mock_config):
        """没有 SKILL.md 的目录不应被当作 skill。"""
        (tmp_ai_skills / "random-dir").mkdir()
        (tmp_ai_skills / "random-dir" / "README.md").write_text("Not a skill.")

        skills = get_skills_from_dir(
            tmp_ai_skills,
            mock_config.get("exclude_skills", []),
        )
        assert "random-dir" not in skills


# ---------------------------------------------------------------------------
# 项目级发现
# ---------------------------------------------------------------------------

class TestProjectDiscovery:
    """项目级 skill 发现测试。"""

    def test_project_skill_discovery(self, tmp_project, mock_config):
        """项目 .ai-skills/ 下的 skill 应被正确发现。"""
        proj_skills_dir = tmp_project / ".ai-skills"
        _make_skill(proj_skills_dir, "proj-extra-skill")
        # proj-test-skill 已存在 → 共 2 个

        skills = get_skills_from_dir(
            proj_skills_dir,
            mock_config.get("exclude_skills", []),
        )
        assert len(skills) == 2
        assert "proj-test-skill" in skills
        assert "proj-extra-skill" in skills

    def test_project_empty_dir(self, tmp_path, mock_config):
        """空的 .ai-skills/ 目录应返回空列表。"""
        empty_dir = tmp_path / "empty-project" / ".ai-skills"
        empty_dir.mkdir(parents=True)

        skills = get_skills_from_dir(
            empty_dir,
            mock_config.get("exclude_skills", []),
        )
        assert skills == []


# ---------------------------------------------------------------------------
# 全局同步目标
# ---------------------------------------------------------------------------

class TestGlobalSyncTargets:
    """全局同步目标验证。"""

    def test_global_sync_targets(self, tmp_ai_skills, mock_config, monkeypatch_config):
        """同步全局 skill 后，应出现在所有 5 个 IDE target 目录中。"""
        result = cmd_sync(
            skill_name="test-skill",
            scope="global",
            json_output=False,
        )

        assert result["global"]["synced"] >= len(mock_config["enabled_ides"])

        for ide in mock_config["enabled_ides"]:
            target_dir = Path(mock_config["targets"][ide]) / "test-skill"
            assert target_dir.exists(), f"test-skill should exist in {ide} target"

    def test_global_sync_multiple(self, tmp_ai_skills, mock_config, monkeypatch_config):
        """同步多个 skill 到全局目标。"""
        _make_skill(tmp_ai_skills, "extra-skill")

        result = cmd_sync(scope="global", json_output=False)

        # 至少 2 skills x 5 IDEs = 10 次同步
        assert result["global"]["synced"] >= 10


# ---------------------------------------------------------------------------
# 项目同步隔离
# ---------------------------------------------------------------------------

class TestProjectSyncIsolation:
    """项目级同步隔离测试 - 项目 skill 不应出现在全局目标中。"""

    def test_project_sync_isolation(self, tmp_project, tmp_ai_skills, mock_config, monkeypatch_config):
        """项目 skill 同步到项目目标，不应出现在全局目标中。"""
        result = cmd_sync(
            skill_name="proj-test-skill",
            scope="local",
            project_dir=tmp_project,
            json_output=False,
        )

        # 验证 skill 出现在项目 IDE 目标中
        for ide in mock_config["enabled_ides"]:
            proj_target = mock_config["project_targets"].get(ide)
            if proj_target:
                target_path = tmp_project / proj_target / "proj-test-skill"
                assert target_path.exists(), f"proj-test-skill should be in project {ide} target"

        # 验证 skill 不在全局目标中
        for ide in mock_config["enabled_ides"]:
            global_target = Path(mock_config["targets"][ide]) / "proj-test-skill"
            assert not global_target.exists(), f"proj-test-skill should NOT be in global {ide} target"


# ---------------------------------------------------------------------------
# 作用域标志解析
# ---------------------------------------------------------------------------

class TestScopeFlagResolution:
    """测试 get_scope_and_project 函数处理不同标志组合。"""

    def test_global_flag(self):
        """传入 -g 时应返回 'global'。"""
        args = SimpleNamespace(global_scope=True, local_scope=False, project_dir=None)
        scope, _ = get_scope_and_project(args)
        assert scope == "global"

    def test_local_flag(self):
        """传入 -l 时应返回 'local'。"""
        args = SimpleNamespace(global_scope=False, local_scope=True, project_dir=None)
        scope, project_dir = get_scope_and_project(args)
        assert scope == "local"
        # 当没有 project_dir 时，应尝试自动检测
        assert project_dir is not None

    def test_both_flags(self):
        """同时传入 -g 和 -l 时应返回 'all'。"""
        args = SimpleNamespace(global_scope=True, local_scope=True, project_dir=None)
        scope, _ = get_scope_and_project(args)
        assert scope == "all"

    def test_no_flags_defaults_global(self):
        """不传标志时默认 'global'。"""
        args = SimpleNamespace(global_scope=False, local_scope=False, project_dir=None)
        scope, _ = get_scope_and_project(args)
        assert scope == "global"

    def test_explicit_project_dir(self, tmp_project):
        """显式指定 project_dir 时应原样使用。"""
        args = SimpleNamespace(
            global_scope=False, local_scope=True,
            project_dir=tmp_project,
        )
        scope, project_dir = get_scope_and_project(args)
        assert scope == "local"
        assert project_dir == tmp_project


# ---------------------------------------------------------------------------
# 默认作用域配置
# ---------------------------------------------------------------------------

class TestDefaultScopeConfig:
    """测试 config 中 default_scope 字段。"""

    def test_default_scope_config(self, tmp_ai_skills):
        """验证 config 中可以设定 default_scope 值。"""
        config_path = tmp_ai_skills / "skill-manager" / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        # fixture 默认设置为 "global"
        assert config["sync"]["default_scope"] == "global"

        # 修改为 "project" 并验证可读回
        config["sync"]["default_scope"] = "project"
        with open(config_path, "w") as f:
            json.dump(config, f)

        with open(config_path) as f:
            reloaded = json.load(f)
        assert reloaded["sync"]["default_scope"] == "project"


# ---------------------------------------------------------------------------
# Skill 触发关键词
# ---------------------------------------------------------------------------

class TestSkillTriggering:
    """验证 skill description 中的触发关键词。"""

    def test_skill_triggering_global(self, tmp_ai_skills):
        """全局 skill 的 description 应包含触发关键词。"""
        _make_skill(
            tmp_ai_skills, "trigger-skill",
            description="Generate PDF reports. Use when converting markdown to PDF.",
        )

        desc = get_skill_description(tmp_ai_skills / "trigger-skill")
        assert "pdf" in desc.lower() or "markdown" in desc.lower()

    def test_skill_triggering_project(self, tmp_project):
        """项目级 skill 的 description 应包含触发关键词。"""
        proj_skills = tmp_project / ".ai-skills"
        _make_skill(
            proj_skills, "deploy-skill",
            description="Deploy to production. Use when releasing new versions.",
        )

        desc = get_skill_description(proj_skills / "deploy-skill")
        assert "deploy" in desc.lower() or "production" in desc.lower()
