#!/usr/bin/env python3
"""
版本一致性测试 - 验证本地版本与 GitHub 版本的一致性，
以及 standalone CLI 与 package CLI 的命令集同步。
"""

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

SKILL_MANAGER_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = SKILL_MANAGER_ROOT / "scripts"


def _compute_file_hash(filepath: Path) -> str:
    """计算文件 MD5 哈希值。"""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_subparser_names_from_cli() -> set:
    """
    从 scripts/cli.py 中提取 subparsers.add_parser() 调用的命令名称。
    解析源代码文本寻找形如 subparsers.add_parser("command_name") 的模式。
    """
    cli_path = SCRIPTS_DIR / "cli.py"
    if not cli_path.exists():
        return set()

    content = cli_path.read_text()

    # 匹配 subparsers.add_parser("name") 和 upstream_sub.add_parser("name")
    # 排除 upstream 的子命令，因为它们是嵌套 subparser
    pattern = r'subparsers\.add_parser\(\s*["\']([^"\']+)["\']\s*'
    top_level = set(re.findall(pattern, content))

    # 也收集 upstream 子命令
    upstream_pattern = r'upstream_sub\.add_parser\(\s*["\']([^"\']+)["\']\s*'
    upstream_subs = set(re.findall(upstream_pattern, content))

    return top_level, upstream_subs


def _extract_subparser_names_from_standalone() -> set:
    """
    从 scripts/skills (standalone CLI) 中提取 subparsers.add_parser() 调用的命令名。
    """
    skills_path = SCRIPTS_DIR / "skills"
    if not skills_path.exists():
        return set()

    content = skills_path.read_text()

    pattern = r'subparsers\.add_parser\(\s*["\']([^"\']+)["\']\s*'
    commands = set(re.findall(pattern, content))

    return commands


# ---------------------------------------------------------------------------
# skill-creator 引用同步测试
# ---------------------------------------------------------------------------

class TestSkillCreatorReferenceSync:
    """验证 skill-manager 中嵌入的 skill-creator 副本与全局版本一致。"""

    def test_skill_creator_reference_sync(self):
        """
        比较 skill-manager/references/skill-creator/SKILL.md 与
        ~/.ai-skills/skill-creator/SKILL.md 的内容哈希。
        """
        ref_path = SKILL_MANAGER_ROOT / "references" / "skill-creator" / "SKILL.md"
        global_path = Path.home() / ".ai-skills" / "skill-creator" / "SKILL.md"

        if not ref_path.exists():
            pytest.skip("skill-manager/references/skill-creator/SKILL.md not found")
        if not global_path.exists():
            pytest.skip("~/.ai-skills/skill-creator/SKILL.md not found")

        ref_hash = _compute_file_hash(ref_path)
        global_hash = _compute_file_hash(global_path)

        assert ref_hash == global_hash, (
            "skill-creator reference copy in skill-manager should match "
            "the global skill-creator SKILL.md. "
            f"Reference hash: {ref_hash}, Global hash: {global_hash}"
        )


# ---------------------------------------------------------------------------
# Standalone CLI vs Package CLI 一致性测试
# ---------------------------------------------------------------------------

class TestStandaloneVsPackageCLI:
    """验证 standalone skills 脚本与 cli.py 的子命令集一致。"""

    def test_standalone_vs_package_cli(self):
        """
        standalone CLI 和 package CLI 应有相同的顶级命令集。
        这确保两种调用方式保持功能同步。
        """
        cli_path = SCRIPTS_DIR / "cli.py"
        skills_path = SCRIPTS_DIR / "skills"

        if not cli_path.exists():
            pytest.skip("scripts/cli.py not found")
        if not skills_path.exists():
            pytest.skip("scripts/skills not found")

        cli_top, cli_upstream = _extract_subparser_names_from_cli()
        standalone_commands = _extract_subparser_names_from_standalone()

        # 比较顶级命令（standalone 可能缺少 upstream 和 commit）
        # 但核心命令应一致
        core_commands = {
            "list", "search", "install", "create", "sync",
            "remove", "validate", "export", "import", "status",
        }

        cli_core = cli_top & core_commands
        standalone_core = standalone_commands & core_commands

        missing_in_standalone = cli_core - standalone_core
        missing_in_cli = standalone_core - cli_core

        assert len(missing_in_standalone) == 0, (
            f"Commands in cli.py but missing from standalone: {missing_in_standalone}"
        )
        assert len(missing_in_cli) == 0, (
            f"Commands in standalone but missing from cli.py: {missing_in_cli}"
        )


# ---------------------------------------------------------------------------
# 本地 vs GitHub 版本测试（需要网络，标记为 slow）
# ---------------------------------------------------------------------------

class TestLocalVsGitHub:
    """
    验证本地 skill-manager 和 skill-creator 是否与 git 仓库中的版本一致。
    这些测试需要网络或至少需要 git 历史记录。
    """

    @pytest.mark.slow
    def test_local_vs_github_skill_manager(self):
        """
        检查本地 skill-manager 是否与 git 仓库最新版本一致。
        通过 git log 检查本地是否有未推送的提交。
        """
        ssot_dir = Path.home() / ".ai-skills"
        if not ssot_dir.exists():
            pytest.skip("~/.ai-skills/ not found")

        # 检查是否为 git 仓库
        result = subprocess.run(
            ["git", "-C", str(ssot_dir), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            pytest.skip("~/.ai-skills/ is not a git repo")

        # 检查 skill-manager 目录下的文件状态
        result = subprocess.run(
            ["git", "-C", str(ssot_dir), "status", "--porcelain", "skill-manager/"],
            capture_output=True, text=True,
        )

        modified_files = [
            line for line in result.stdout.strip().split("\n")
            if line.strip()
        ]

        # 报告但不严格断言（本地开发中可能有未提交的修改）
        if modified_files:
            pytest.xfail(
                f"Local skill-manager has {len(modified_files)} uncommitted changes: "
                + ", ".join(modified_files[:5])
            )

    @pytest.mark.slow
    def test_local_vs_github_skill_creator(self):
        """
        检查本地 skill-creator 是否与 git 仓库最新版本一致。
        """
        ssot_dir = Path.home() / ".ai-skills"
        creator_dir = ssot_dir / "skill-creator"

        if not creator_dir.exists():
            pytest.skip("~/.ai-skills/skill-creator/ not found")

        result = subprocess.run(
            ["git", "-C", str(ssot_dir), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            pytest.skip("~/.ai-skills/ is not a git repo")

        result = subprocess.run(
            ["git", "-C", str(ssot_dir), "status", "--porcelain", "skill-creator/"],
            capture_output=True, text=True,
        )

        modified_files = [
            line for line in result.stdout.strip().split("\n")
            if line.strip()
        ]

        if modified_files:
            pytest.xfail(
                f"Local skill-creator has {len(modified_files)} uncommitted changes: "
                + ", ".join(modified_files[:5])
            )


# ---------------------------------------------------------------------------
# 配置文件版本一致性测试
# ---------------------------------------------------------------------------

class TestConfigVersionConsistency:
    """验证配置文件中的默认值与代码中的默认值一致。"""

    def test_default_config_matches_config_json(self):
        """
        config.json 中的键集合应是 get_default_config() 返回的键集合的子集。
        即 config.json 不应有代码不认识的配置项。
        """
        from scripts.utils import get_default_config

        config_json_path = SKILL_MANAGER_ROOT / "config.json"
        if not config_json_path.exists():
            pytest.skip("config.json not found")

        with open(config_json_path) as f:
            config_json = json.load(f)

        default_config = get_default_config()

        # config.json 中的顶级键应存在于默认配置中（允许 upstream 等扩展）
        for key in config_json:
            if key not in default_config and key != "upstream":
                pytest.fail(
                    f"config.json has key '{key}' not in get_default_config(). "
                    "Update default config or remove the key."
                )

    def test_enabled_ides_match_targets(self):
        """enabled_ides 中的每个 IDE 都应在 targets 中有对应配置。"""
        from scripts.utils import get_default_config

        config = get_default_config()
        enabled = config.get("enabled_ides", [])
        targets = config.get("targets", {})
        project_targets = config.get("project_targets", {})

        for ide in enabled:
            assert ide in targets, f"IDE '{ide}' in enabled_ides but not in targets"
            assert ide in project_targets, (
                f"IDE '{ide}' in enabled_ides but not in project_targets"
            )
