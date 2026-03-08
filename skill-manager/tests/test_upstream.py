#!/usr/bin/env python3
"""
上游同步测试 - 验证上游源的添加、更新、导入和冲突检测。
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.upstream import (
    _compute_file_hash,
    _detect_local_modifications,
    _load_manifest,
    _save_manifest,
    cmd_upstream_add,
    cmd_upstream_diff,
    cmd_upstream_import,
    cmd_upstream_list,
    cmd_upstream_status,
    cmd_upstream_update,
)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _create_upstream_repo(tmp_path, skill_names, subdir="skills"):
    """
    创建一个本地 bare git 仓库模拟上游源。
    在 <subdir>/<skill_name>/SKILL.md 中为每个 skill 写入有效内容。
    返回 bare 仓库路径和工作目录路径。
    """
    work_dir = tmp_path / "upstream-work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # 初始化并配置 git
    subprocess.run(["git", "init", str(work_dir)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(work_dir), "config", "user.email", "test@test.com"],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(work_dir), "config", "user.name", "Test User"],
        capture_output=True, check=True,
    )

    # 为每个 skill 创建目录和 SKILL.md
    for name in skill_names:
        skill_dir = work_dir / subdir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\n"
            f"description: Upstream {name} skill. Use when testing upstream import.\n"
            f"---\n\n# {name.replace('-', ' ').title()}\n\n"
            f"Content for {name}.\n"
        )

    # 提交
    subprocess.run(
        ["git", "-C", str(work_dir), "add", "-A"],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(work_dir), "commit", "-m", "add skills"],
        capture_output=True, check=True,
    )

    # 创建 bare 仓库
    bare_dir = tmp_path / "upstream-bare.git"
    subprocess.run(
        ["git", "clone", "--bare", str(work_dir), str(bare_dir)],
        capture_output=True, check=True,
    )

    return bare_dir, work_dir


def _add_commit_to_upstream(work_dir, bare_dir, file_relpath, content, msg="update"):
    """向上游工作仓库追加提交并推送到 bare 仓库。"""
    filepath = work_dir / file_relpath
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content)
    subprocess.run(
        ["git", "-C", str(work_dir), "add", "-A"],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(work_dir), "commit", "-m", msg],
        capture_output=True, check=True,
    )
    # 推送到 bare 仓库
    subprocess.run(
        ["git", "-C", str(work_dir), "remote", "add", "origin", str(bare_dir)],
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(work_dir), "push", "origin", "master"],
        capture_output=True,
    )
    # 部分系统默认分支为 main
    subprocess.run(
        ["git", "-C", str(work_dir), "push", "origin", "main"],
        capture_output=True,
    )


def _patch_upstream_module(monkeypatch, ssot_dir, upstream_cfg=None):
    """
    用 monkeypatch 将 upstream.py 中的 _get_ssot_dir 和
    _get_upstream_config 替换为指向临时目录的版本。
    """
    monkeypatch.setattr("scripts.upstream._get_ssot_dir", lambda: ssot_dir)
    if upstream_cfg is None:
        upstream_cfg = {
            "auto_sync_on_update": False,
            "sources_dir": ".sources",
            "manifest_file": ".upstream-manifest.json",
        }
    monkeypatch.setattr("scripts.upstream._get_upstream_config", lambda: upstream_cfg)

    # 同时 patch load_config 和 auto_commit，避免副作用
    monkeypatch.setattr(
        "scripts.upstream.load_config",
        lambda: {"upstream": upstream_cfg, "git": {"auto_commit": False}},
    )
    monkeypatch.setattr("scripts.upstream.auto_commit", lambda *a, **kw: True)


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

class TestUpstreamAdd:
    """测试上游源的注册。"""

    def test_add_source(self, tmp_path, monkeypatch):
        """添加上游源后，.sources/ 目录和 manifest 应有对应记录。"""
        ssot_dir = tmp_path / "ssot"
        ssot_dir.mkdir()
        bare_dir, _ = _create_upstream_repo(tmp_path, ["alpha-skill"])
        _patch_upstream_module(monkeypatch, ssot_dir)

        # 获取默认分支名称
        result = subprocess.run(
            ["git", "-C", str(bare_dir), "branch"],
            capture_output=True, text=True,
        )
        branch = result.stdout.strip().lstrip("* ").split("\n")[0].strip()

        res = cmd_upstream_add(str(bare_dir), name="test-source", branch=branch)

        assert res["success"] is True
        assert res["name"] == "test-source"

        # .sources/test-source/ 目录应存在
        sources_clone = ssot_dir / ".sources" / "test-source"
        assert sources_clone.is_dir(), ".sources/test-source/ directory should exist"

        # manifest 应记录这个源
        manifest = _load_manifest(ssot_dir)
        assert "test-source" in manifest["sources"]
        assert manifest["sources"]["test-source"]["url"] == str(bare_dir)

    def test_add_duplicate_source_fails(self, tmp_path, monkeypatch):
        """重复添加同名上游源应该失败。"""
        ssot_dir = tmp_path / "ssot"
        ssot_dir.mkdir()
        bare_dir, _ = _create_upstream_repo(tmp_path, ["beta-skill"])
        _patch_upstream_module(monkeypatch, ssot_dir)

        # 获取默认分支
        result = subprocess.run(
            ["git", "-C", str(bare_dir), "branch"],
            capture_output=True, text=True,
        )
        branch = result.stdout.strip().lstrip("* ").split("\n")[0].strip()

        cmd_upstream_add(str(bare_dir), name="dup-source", branch=branch)
        res = cmd_upstream_add(str(bare_dir), name="dup-source", branch=branch)

        assert res["success"] is False


class TestUpstreamUpdate:
    """测试上游源的拉取更新。"""

    def test_update_pulls(self, tmp_path, monkeypatch):
        """在上游追加提交后，update 应更新 manifest 中的 commit hash。"""
        ssot_dir = tmp_path / "ssot"
        ssot_dir.mkdir()
        bare_dir, work_dir = _create_upstream_repo(tmp_path, ["gamma-skill"])
        _patch_upstream_module(monkeypatch, ssot_dir)

        # 检测默认分支
        result = subprocess.run(
            ["git", "-C", str(bare_dir), "branch"],
            capture_output=True, text=True,
        )
        branch = result.stdout.strip().lstrip("* ").split("\n")[0].strip()

        cmd_upstream_add(str(bare_dir), name="up-src", branch=branch)

        old_manifest = _load_manifest(ssot_dir)
        old_commit = old_manifest["sources"]["up-src"]["last_pulled_commit"]

        # 向上游追加一个提交
        _add_commit_to_upstream(
            work_dir, bare_dir,
            "skills/gamma-skill/extra.md",
            "new content",
            msg="add extra file",
        )

        cmd_upstream_update("up-src")

        new_manifest = _load_manifest(ssot_dir)
        new_commit = new_manifest["sources"]["up-src"]["last_pulled_commit"]

        assert new_commit != old_commit, "Commit hash should change after update"


class TestUpstreamImport:
    """测试从上游源导入 skill。"""

    def test_import_new_skill(self, tmp_path, monkeypatch):
        """导入 skill 后应在 SSOT 根目录和 manifest 中出现。"""
        ssot_dir = tmp_path / "ssot"
        ssot_dir.mkdir()
        bare_dir, _ = _create_upstream_repo(tmp_path, ["delta-skill"])
        _patch_upstream_module(monkeypatch, ssot_dir)

        result = subprocess.run(
            ["git", "-C", str(bare_dir), "branch"],
            capture_output=True, text=True,
        )
        branch = result.stdout.strip().lstrip("* ").split("\n")[0].strip()

        cmd_upstream_add(str(bare_dir), name="imp-src", branch=branch)
        res = cmd_upstream_import(["delta-skill"], "imp-src")

        assert res["success"] is True
        assert "delta-skill" in res["imported"]

        # SSOT 中应存在该 skill
        assert (ssot_dir / "delta-skill" / "SKILL.md").is_file()

        # manifest 应记录该 skill
        manifest = _load_manifest(ssot_dir)
        assert "delta-skill" in manifest["skills"]
        assert manifest["skills"]["delta-skill"]["source"] == "imp-src"

    def test_import_with_force(self, tmp_path, monkeypatch):
        """使用 --force 导入应覆盖本地修改。"""
        ssot_dir = tmp_path / "ssot"
        ssot_dir.mkdir()
        bare_dir, _ = _create_upstream_repo(tmp_path, ["force-skill"])
        _patch_upstream_module(monkeypatch, ssot_dir)

        result = subprocess.run(
            ["git", "-C", str(bare_dir), "branch"],
            capture_output=True, text=True,
        )
        branch = result.stdout.strip().lstrip("* ").split("\n")[0].strip()

        cmd_upstream_add(str(bare_dir), name="force-src", branch=branch)
        cmd_upstream_import(["force-skill"], "force-src")

        # 本地修改 SKILL.md
        local_skill_md = ssot_dir / "force-skill" / "SKILL.md"
        local_skill_md.write_text("locally modified content\n")

        # 用 --force 重新导入
        res = cmd_upstream_import(["force-skill"], "force-src", force=True)

        assert res["success"] is True
        assert "force-skill" in res["imported"]
        # 内容应恢复为上游版本
        content = local_skill_md.read_text()
        assert "locally modified" not in content
        assert "Upstream" in content or "force-skill" in content

    def test_import_with_adopt(self, tmp_path, monkeypatch):
        """使用 --adopt 导入不应覆盖本地文件，仅在 manifest 中记录。"""
        ssot_dir = tmp_path / "ssot"
        ssot_dir.mkdir()
        bare_dir, _ = _create_upstream_repo(tmp_path, ["adopt-skill"])
        _patch_upstream_module(monkeypatch, ssot_dir)

        result = subprocess.run(
            ["git", "-C", str(bare_dir), "branch"],
            capture_output=True, text=True,
        )
        branch = result.stdout.strip().lstrip("* ").split("\n")[0].strip()

        cmd_upstream_add(str(bare_dir), name="adopt-src", branch=branch)

        # 手动在 SSOT 创建同名 skill（模拟已存在的本地 skill）
        local_skill_dir = ssot_dir / "adopt-skill"
        local_skill_dir.mkdir(parents=True)
        marker_text = "this is the local version\n"
        (local_skill_dir / "SKILL.md").write_text(marker_text)

        # 用 --adopt 导入
        res = cmd_upstream_import(["adopt-skill"], "adopt-src", adopt=True)

        assert res["success"] is True
        assert "adopt-skill" in res["imported"]

        # 文件不应被覆盖
        content = (local_skill_dir / "SKILL.md").read_text()
        assert content == marker_text, "Local files should NOT be overwritten in adopt mode"

        # 但 manifest 应记录该 skill
        manifest = _load_manifest(ssot_dir)
        assert "adopt-skill" in manifest["skills"]
        assert manifest["skills"]["adopt-skill"].get("adopted") is True

    def test_import_without_force_skips_existing(self, tmp_path, monkeypatch):
        """不使用 --force 导入已存在的 skill 应跳过。"""
        ssot_dir = tmp_path / "ssot"
        ssot_dir.mkdir()
        bare_dir, _ = _create_upstream_repo(tmp_path, ["skip-skill"])
        _patch_upstream_module(monkeypatch, ssot_dir)

        result = subprocess.run(
            ["git", "-C", str(bare_dir), "branch"],
            capture_output=True, text=True,
        )
        branch = result.stdout.strip().lstrip("* ").split("\n")[0].strip()

        cmd_upstream_add(str(bare_dir), name="skip-src", branch=branch)
        cmd_upstream_import(["skip-skill"], "skip-src")

        # 再次导入（无 --force）
        res = cmd_upstream_import(["skip-skill"], "skip-src")
        assert "skip-skill" in res["skipped"]


class TestConflictDetection:
    """测试本地修改与上游更新的冲突检测。"""

    def test_conflict_detection(self, tmp_path, monkeypatch):
        """本地修改了 skill 文件后，_detect_local_modifications 应返回 True。"""
        ssot_dir = tmp_path / "ssot"
        ssot_dir.mkdir()
        bare_dir, _ = _create_upstream_repo(tmp_path, ["conflict-skill"])
        _patch_upstream_module(monkeypatch, ssot_dir)

        result = subprocess.run(
            ["git", "-C", str(bare_dir), "branch"],
            capture_output=True, text=True,
        )
        branch = result.stdout.strip().lstrip("* ").split("\n")[0].strip()

        cmd_upstream_add(str(bare_dir), name="cf-src", branch=branch)
        cmd_upstream_import(["conflict-skill"], "cf-src")

        # 修改本地文件
        (ssot_dir / "conflict-skill" / "SKILL.md").write_text("local change\n")

        # 检测修改
        source_clone_dir = ssot_dir / ".sources" / "cf-src"
        modified = _detect_local_modifications(
            ssot_dir, "conflict-skill", source_clone_dir, "skills"
        )

        assert modified is True, "Should detect local modifications"


class TestManifestTracking:
    """测试 manifest 文件的记录完整性。"""

    def test_manifest_tracking(self, tmp_path, monkeypatch):
        """添加源并导入 skill 后，manifest 应含正确的 URL、commit、时间戳。"""
        ssot_dir = tmp_path / "ssot"
        ssot_dir.mkdir()
        bare_dir, _ = _create_upstream_repo(
            tmp_path, ["track-alpha", "track-beta"]
        )
        _patch_upstream_module(monkeypatch, ssot_dir)

        result = subprocess.run(
            ["git", "-C", str(bare_dir), "branch"],
            capture_output=True, text=True,
        )
        branch = result.stdout.strip().lstrip("* ").split("\n")[0].strip()

        cmd_upstream_add(str(bare_dir), name="track-src", branch=branch)
        cmd_upstream_import(["track-alpha", "track-beta"], "track-src")

        manifest = _load_manifest(ssot_dir)

        # 验证 source 记录
        src = manifest["sources"]["track-src"]
        assert src["url"] == str(bare_dir)
        assert "last_pulled_commit" in src
        assert len(src["last_pulled_commit"]) >= 7
        assert "last_pulled_at" in src

        # 验证 skill 记录
        for name in ("track-alpha", "track-beta"):
            skill = manifest["skills"][name]
            assert skill["source"] == "track-src"
            assert "synced_commit" in skill
            assert "synced_at" in skill


class TestAutoSyncAfterUpdate:
    """测试 auto_sync_on_update 配置。"""

    def test_auto_sync_after_update(self, tmp_path, monkeypatch):
        """当 auto_sync_on_update=True 时，update 后应自动重新导入变更的 skill。"""
        ssot_dir = tmp_path / "ssot"
        ssot_dir.mkdir()
        bare_dir, work_dir = _create_upstream_repo(tmp_path, ["auto-skill"])

        # 启用 auto_sync_on_update
        upstream_cfg = {
            "auto_sync_on_update": True,
            "sources_dir": ".sources",
            "manifest_file": ".upstream-manifest.json",
        }
        _patch_upstream_module(monkeypatch, ssot_dir, upstream_cfg)

        result = subprocess.run(
            ["git", "-C", str(bare_dir), "branch"],
            capture_output=True, text=True,
        )
        branch = result.stdout.strip().lstrip("* ").split("\n")[0].strip()

        cmd_upstream_add(str(bare_dir), name="auto-src", branch=branch)
        cmd_upstream_import(["auto-skill"], "auto-src")

        original_content = (ssot_dir / "auto-skill" / "SKILL.md").read_text()

        # 上游更新 SKILL.md
        new_content = (
            "---\nname: auto-skill\n"
            "description: Updated auto skill. Use when testing auto sync.\n"
            "---\n\n# Auto Skill Updated\n\nNew upstream content.\n"
        )
        _add_commit_to_upstream(
            work_dir, bare_dir,
            "skills/auto-skill/SKILL.md",
            new_content,
            msg="update auto-skill",
        )

        cmd_upstream_update("auto-src")

        # 因为 auto_sync_on_update=True，本地应更新为上游新内容
        updated_content = (ssot_dir / "auto-skill" / "SKILL.md").read_text()
        assert updated_content != original_content, "Content should be updated after auto sync"
        assert "Updated" in updated_content or "New upstream" in updated_content


class TestUpstreamList:
    """测试上游源中 skill 的列表功能。"""

    def test_list_shows_available_and_imported(self, tmp_path, monkeypatch):
        """列出 skill 时应正确区分已导入和未导入。"""
        ssot_dir = tmp_path / "ssot"
        ssot_dir.mkdir()
        bare_dir, _ = _create_upstream_repo(
            tmp_path, ["list-alpha", "list-beta", "list-gamma"]
        )
        _patch_upstream_module(monkeypatch, ssot_dir)

        result = subprocess.run(
            ["git", "-C", str(bare_dir), "branch"],
            capture_output=True, text=True,
        )
        branch = result.stdout.strip().lstrip("* ").split("\n")[0].strip()

        cmd_upstream_add(str(bare_dir), name="list-src", branch=branch)
        # 只导入 2 个 skill 中的 1 个
        cmd_upstream_import(["list-alpha"], "list-src")

        res = cmd_upstream_list("list-src")

        assert res["success"] is True
        assert "list-alpha" in res["imported"]
        assert "list-beta" in res["available"]
        assert "list-gamma" in res["available"]


class TestUpstreamStatus:
    """测试上游状态显示功能。"""

    def test_status_shows_sources_and_skills(self, tmp_path, monkeypatch):
        """status 应返回正确的 sources 和 skills 列表。"""
        ssot_dir = tmp_path / "ssot"
        ssot_dir.mkdir()
        bare_dir, _ = _create_upstream_repo(tmp_path, ["status-skill"])
        _patch_upstream_module(monkeypatch, ssot_dir)

        result = subprocess.run(
            ["git", "-C", str(bare_dir), "branch"],
            capture_output=True, text=True,
        )
        branch = result.stdout.strip().lstrip("* ").split("\n")[0].strip()

        cmd_upstream_add(str(bare_dir), name="status-src", branch=branch)
        cmd_upstream_import(["status-skill"], "status-src")

        res = cmd_upstream_status()

        assert res["success"] is True
        assert "status-src" in res["sources"]
        assert "status-skill" in res["skills"]


class TestUpstreamDiff:
    """测试 skill 的本地 vs 上游差异对比。"""

    def test_diff_identical(self, tmp_path, monkeypatch):
        """刚导入的 skill 应无差异。"""
        ssot_dir = tmp_path / "ssot"
        ssot_dir.mkdir()
        bare_dir, _ = _create_upstream_repo(tmp_path, ["diff-skill"])
        _patch_upstream_module(monkeypatch, ssot_dir)

        result = subprocess.run(
            ["git", "-C", str(bare_dir), "branch"],
            capture_output=True, text=True,
        )
        branch = result.stdout.strip().lstrip("* ").split("\n")[0].strip()

        cmd_upstream_add(str(bare_dir), name="diff-src", branch=branch)
        cmd_upstream_import(["diff-skill"], "diff-src")

        res = cmd_upstream_diff("diff-skill")

        assert res["success"] is True
        assert len(res["differences"]) == 0
        assert len(res["identical"]) > 0

    def test_diff_with_local_changes(self, tmp_path, monkeypatch):
        """本地修改后应检测到差异。"""
        ssot_dir = tmp_path / "ssot"
        ssot_dir.mkdir()
        bare_dir, _ = _create_upstream_repo(tmp_path, ["diff2-skill"])
        _patch_upstream_module(monkeypatch, ssot_dir)

        result = subprocess.run(
            ["git", "-C", str(bare_dir), "branch"],
            capture_output=True, text=True,
        )
        branch = result.stdout.strip().lstrip("* ").split("\n")[0].strip()

        cmd_upstream_add(str(bare_dir), name="diff2-src", branch=branch)
        cmd_upstream_import(["diff2-skill"], "diff2-src")

        # 修改本地文件
        (ssot_dir / "diff2-skill" / "SKILL.md").write_text("local change\n")

        res = cmd_upstream_diff("diff2-skill")

        assert res["success"] is True
        assert len(res["differences"]) > 0
        changed_files = [d["file"] for d in res["differences"]]
        assert "SKILL.md" in changed_files

    def test_diff_untracked_skill(self, tmp_path, monkeypatch):
        """对未追踪的 skill 调用 diff 应返回失败。"""
        ssot_dir = tmp_path / "ssot"
        ssot_dir.mkdir()
        _patch_upstream_module(monkeypatch, ssot_dir)

        res = cmd_upstream_diff("nonexistent-skill")

        assert res["success"] is False
