#!/usr/bin/env python3
"""
同步完整性测试 - 验证 SSOT 和目标之间的文件一致性。
"""

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from scripts.sync import sync_skill, cleanup_orphaned_skills, cmd_sync
from scripts.validate import _compute_dir_hash
from scripts.utils import get_skills_from_dir


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _make_skill(parent_dir: Path, name: str) -> Path:
    """创建一个带有多文件的 skill 用于 hash 校验。"""
    skill_dir = parent_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Integrity test skill. Use when testing sync.\n---\n\n"
        f"# {name}\n"
    )
    # 添加额外文件以增加 hash 校验覆盖度
    refs_dir = skill_dir / "references"
    refs_dir.mkdir(exist_ok=True)
    (refs_dir / "notes.md").write_text("Reference notes for testing.\n")
    return skill_dir


# ---------------------------------------------------------------------------
# Hash 一致性
# ---------------------------------------------------------------------------

class TestHashConsistency:
    """源和目标之间文件 hash 必须一致。"""

    def test_hash_consistency(self, tmp_ai_skills, mock_config, monkeypatch_config):
        """创建 skill、同步后，比较源和各目标的 MD5 hash 应完全一致。"""
        _make_skill(tmp_ai_skills, "hash-test-skill")

        cmd_sync(skill_name="hash-test-skill", scope="global", json_output=False)

        source_hash = _compute_dir_hash(tmp_ai_skills / "hash-test-skill")

        for ide in mock_config["enabled_ides"]:
            target_dir = Path(mock_config["targets"][ide]) / "hash-test-skill"
            assert target_dir.exists(), f"Skill should exist in {ide} target"
            target_hash = _compute_dir_hash(target_dir)
            assert source_hash == target_hash, (
                f"Hash mismatch for {ide}: source={source_hash} target={target_hash}"
            )

    def test_hash_changes_after_update(self, tmp_ai_skills, mock_config, monkeypatch_config):
        """修改源文件后重新同步，目标的 hash 应同步更新。"""
        _make_skill(tmp_ai_skills, "update-skill")
        cmd_sync(skill_name="update-skill", scope="global", json_output=False)

        old_hash = _compute_dir_hash(tmp_ai_skills / "update-skill")

        # 修改源文件
        (tmp_ai_skills / "update-skill" / "SKILL.md").write_text(
            "---\nname: update-skill\ndescription: Updated content. Use when testing updates.\n---\n\n"
            "# Updated Skill\n\nContent was modified.\n"
        )

        new_source_hash = _compute_dir_hash(tmp_ai_skills / "update-skill")
        assert old_hash != new_source_hash, "Source hash should change after modification"

        # 重新同步
        cmd_sync(skill_name="update-skill", scope="global", json_output=False)

        for ide in mock_config["enabled_ides"]:
            target_dir = Path(mock_config["targets"][ide]) / "update-skill"
            target_hash = _compute_dir_hash(target_dir)
            assert target_hash == new_source_hash, f"Target {ide} should match updated source"


# ---------------------------------------------------------------------------
# 孤儿清理
# ---------------------------------------------------------------------------

class TestOrphanCleanup:
    """移除源中已删除 skill 在目标中的残留副本。"""

    def test_orphan_cleanup(self, tmp_ai_skills, mock_config, monkeypatch_config):
        """
        同步多个 skill 后，从源中移除一个，再次同步时
        应自动清理目标中的孤儿。
        """
        _make_skill(tmp_ai_skills, "keep-skill")
        _make_skill(tmp_ai_skills, "orphan-skill")

        # 同步全部
        cmd_sync(scope="global", json_output=False)

        # 验证均已同步
        for ide in mock_config["enabled_ides"]:
            assert (Path(mock_config["targets"][ide]) / "orphan-skill").exists()

        # 从源中移除 orphan-skill
        shutil.rmtree(tmp_ai_skills / "orphan-skill")

        # 再次同步 → 孤儿应被清理
        result = cmd_sync(scope="global", json_output=False)
        assert result["global"]["removed"] >= 1

        for ide in mock_config["enabled_ides"]:
            assert not (Path(mock_config["targets"][ide]) / "orphan-skill").exists(), (
                f"orphan-skill should be removed from {ide} target"
            )

    def test_cleanup_function_direct(self, tmp_ai_skills, mock_config):
        """直接测试 cleanup_orphaned_skills 函数。"""
        target_dir = Path(mock_config["targets"]["claude"])

        # 手动在目标中创建两个 skill
        orphan = target_dir / "should-remove"
        orphan.mkdir(parents=True, exist_ok=True)
        (orphan / "SKILL.md").write_text("---\nname: should-remove\ndescription: x.\n---\n")

        legit = target_dir / "should-keep"
        legit.mkdir(parents=True, exist_ok=True)
        (legit / "SKILL.md").write_text("---\nname: should-keep\ndescription: x.\n---\n")

        # source_skills 只有 should-keep
        removed = cleanup_orphaned_skills(
            source_skills=["should-keep"],
            target_dir=target_dir,
            exclude=mock_config["exclude_skills"],
        )

        assert "should-remove" in removed
        assert not orphan.exists()
        assert legit.exists()


# ---------------------------------------------------------------------------
# 保护目录
# ---------------------------------------------------------------------------

class TestPreserveProtected:
    """verify_target_skills 中配置的目录在清理时不应被删除。"""

    def test_preserve_protected(self, tmp_ai_skills, mock_config, monkeypatch_config):
        """codex 目标中的 .system 目录应在同步时被保留。"""
        codex_target = Path(mock_config["targets"]["codex"])

        # 创建受保护的 .system 目录
        system_dir = codex_target / ".system"
        system_dir.mkdir(parents=True, exist_ok=True)
        (system_dir / "config.txt").write_text("protected content")

        # 同步全局 skill
        cmd_sync(scope="global", json_output=False)

        # .system 应仍然存在
        assert system_dir.exists(), ".system directory should be preserved"
        assert (system_dir / "config.txt").read_text() == "protected content"


# ---------------------------------------------------------------------------
# Symlink 模式
# ---------------------------------------------------------------------------

class TestSyncModes:
    """验证 symlink 和 copytree 两种同步模式。"""

    def test_symlink_mode(self, tmp_ai_skills, mock_config):
        """use_symlinks=True 时，目标应为指向源的符号链接。"""
        target_dir = Path(mock_config["targets"]["claude"])

        success, method = sync_skill(
            "test-skill",
            tmp_ai_skills,
            target_dir,
            use_symlinks=True,
        )

        assert success is True
        assert method == "symlink"

        target = target_dir / "test-skill"
        assert target.is_symlink()
        assert target.resolve() == (tmp_ai_skills / "test-skill").resolve()

    def test_copytree_mode(self, tmp_ai_skills, mock_config):
        """use_symlinks=False 时，目标应为独立的目录副本（非符号链接）。"""
        target_dir = Path(mock_config["targets"]["cursor"])

        success, method = sync_skill(
            "test-skill",
            tmp_ai_skills,
            target_dir,
            use_symlinks=False,
        )

        assert success is True
        assert method == "copy"

        target = target_dir / "test-skill"
        assert target.is_dir()
        assert not target.is_symlink()
        assert (target / "SKILL.md").exists()

    def test_sync_replaces_existing(self, tmp_ai_skills, mock_config):
        """重复同步时应替换已有的目标目录。"""
        target_dir = Path(mock_config["targets"]["gemini"])

        # 第一次同步
        sync_skill("test-skill", tmp_ai_skills, target_dir, use_symlinks=False)
        target = target_dir / "test-skill"
        assert target.exists()

        # 修改源
        (tmp_ai_skills / "test-skill" / "extra.txt").write_text("new file")

        # 第二次同步
        success, method = sync_skill("test-skill", tmp_ai_skills, target_dir, use_symlinks=False)
        assert success is True

        # 新文件应出现在目标中
        assert (target / "extra.txt").exists()

    def test_sync_nonexistent_skill(self, tmp_ai_skills, mock_config):
        """同步不存在的 skill 应返回失败。"""
        target_dir = Path(mock_config["targets"]["claude"])

        success, method = sync_skill(
            "nonexistent-skill",
            tmp_ai_skills,
            target_dir,
            use_symlinks=False,
        )

        assert success is False
        assert method == "not found"
