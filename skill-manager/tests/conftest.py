#!/usr/bin/env python3
"""
测试夹具 - 为所有测试提供临时目录、mock config 等基础设施。
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# 将 scripts 包加入 sys.path，允许直接导入 skill-manager 模块
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _git_init(repo_dir: Path) -> None:
    """在指定目录初始化 git 仓库并创建一个初始提交。"""
    subprocess.run(
        ["git", "init", str(repo_dir)],
        capture_output=True, text=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "config", "user.email", "test@test.com"],
        capture_output=True, text=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "config", "user.name", "Test User"],
        capture_output=True, text=True, check=True,
    )
    # 创建 .gitkeep 以保证至少有一次提交
    (repo_dir / ".gitkeep").touch()
    subprocess.run(
        ["git", "-C", str(repo_dir), "add", "-A"],
        capture_output=True, text=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "commit", "-m", "initial commit"],
        capture_output=True, text=True, check=True,
    )


SAMPLE_SKILL_MD_CONTENT = """\
---
name: test-skill
description: A test skill for unit testing. Use when testing skill-manager features.
---

# Test Skill

This is a test skill used for automated testing.
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_skill_md():
    """返回一个合法的 SKILL.md 文件内容字符串。"""
    return SAMPLE_SKILL_MD_CONTENT


@pytest.fixture
def tmp_ai_skills(tmp_path):
    """
    创建一个临时的 ~/.ai-skills/ 等价目录，结构包含:
    - skill-manager/ 子目录（含 config.json）
    - 一个示例 skill (test-skill) 及合法 SKILL.md
    - 已初始化为 git 仓库并完成首次提交
    """
    # 根目录作为 source_dir
    source_dir = tmp_path / "ai-skills"
    source_dir.mkdir()

    # skill-manager 子目录和配置
    sm_dir = source_dir / "skill-manager"
    sm_dir.mkdir()

    # 为每个 IDE 创建全局 target 目录
    targets = {}
    for ide in ("claude", "cursor", "codex", "gemini", "antigravity"):
        ide_dir = tmp_path / "targets" / ide / "skills"
        ide_dir.mkdir(parents=True, exist_ok=True)
        targets[ide] = str(ide_dir)

    config = {
        "source_dir": str(source_dir),
        "project_source_dir": ".ai-skills",
        "targets": targets,
        "project_targets": {
            "claude": ".claude/skills",
            "cursor": ".cursor/skills",
            "codex": ".codex/skills",
            "gemini": ".gemini/skills",
            "antigravity": ".agent/skills",
        },
        "enabled_ides": ["claude", "cursor", "codex", "gemini", "antigravity"],
        "git": {
            "auto_commit": False,
            "commit_prefix": "skills:",
            "auto_push": False,
        },
        "sync": {
            "auto_after_install": False,
            "auto_after_create": False,
            "auto_after_remove": False,
            "use_symlinks": False,
            "default_scope": "global",
        },
        "exclude_skills": ["skill-manager"],
        "preserve_target_skills": {
            "codex": [".system"],
        },
        "search": {
            "index_path": str(sm_dir / "data" / "index.json"),
            "cache_ttl_hours": 24,
        },
        "profile": {
            "default_export_path": str(tmp_path / "skills-profile.json"),
            "gist_id": None,
        },
    }

    config_path = sm_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2))

    # 创建示例 skill
    skill_dir = source_dir / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(SAMPLE_SKILL_MD_CONTENT)

    # 初始化 git
    _git_init(source_dir)

    return source_dir


@pytest.fixture
def tmp_project(tmp_path):
    """
    创建一个临时项目目录，结构包含:
    - .ai-skills/ 子目录
    - .git/ (已初始化)
    - 一个示例项目 skill (proj-test-skill) 及合法 SKILL.md
    """
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()

    # 项目级 skill 源目录
    proj_skills = project_dir / ".ai-skills"
    proj_skills.mkdir()

    skill_dir = proj_skills / "proj-test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: proj-test-skill\n"
        "description: A project-level test skill. Use when testing project scope.\n"
        "---\n\n"
        "# Project Test Skill\n\n"
        "This is a project-level test skill.\n"
    )

    # 初始化 git
    _git_init(project_dir)

    return project_dir


@pytest.fixture
def mock_config(tmp_ai_skills):
    """
    返回一个指向 tmp_ai_skills 临时目录的完整 config dict。
    所有路径均指向临时目录。
    """
    config_path = tmp_ai_skills / "skill-manager" / "config.json"
    with open(config_path) as f:
        return json.load(f)


@pytest.fixture
def mock_github_repo(tmp_path):
    """
    创建一个本地 bare git 仓库模拟 GitHub 远程:
    - 含 skills/test-remote-skill/ 目录
    - 含合法 SKILL.md
    返回 bare 仓库路径（可用作 git clone URL）。
    """
    # 先在工作目录中创建内容
    work_dir = tmp_path / "github-work"
    work_dir.mkdir()
    _git_init(work_dir)

    skills_dir = work_dir / "skills" / "test-remote-skill"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(
        "---\n"
        "name: test-remote-skill\n"
        "description: A remote test skill from GitHub. Use when testing install.\n"
        "---\n\n"
        "# Remote Test Skill\n\n"
        "Installed from a mock GitHub repo.\n"
    )

    subprocess.run(
        ["git", "-C", str(work_dir), "add", "-A"],
        capture_output=True, text=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(work_dir), "commit", "-m", "add remote skill"],
        capture_output=True, text=True, check=True,
    )

    # 创建 bare 仓库作为远程
    bare_dir = tmp_path / "github-bare.git"
    subprocess.run(
        ["git", "clone", "--bare", str(work_dir), str(bare_dir)],
        capture_output=True, text=True, check=True,
    )

    return bare_dir


@pytest.fixture
def monkeypatch_config(monkeypatch, mock_config):
    """
    用 monkeypatch 将 load_config 替换为返回 mock_config 的函数，
    避免测试读取真实的 ~/.ai-skills/skill-manager/config.json。
    """
    monkeypatch.setattr("scripts.utils.load_config", lambda: mock_config)
    # 同时 patch 各子模块中可能缓存的 load_config 引用
    for mod in ("scripts.sync", "scripts.install", "scripts.create",
                "scripts.remove", "scripts.validate", "scripts.git_ops",
                "scripts.search", "scripts.cli"):
        try:
            monkeypatch.setattr(f"{mod}.load_config", lambda: mock_config)
        except AttributeError:
            pass
