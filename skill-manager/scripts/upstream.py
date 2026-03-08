#!/usr/bin/env python3
"""
上游源管理模块 - 管理从外部 GitHub 仓库导入的 skill。
支持添加上游源、更新、导入 skill、检查状态和差异比较。
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .utils import expand_path, load_config
from .git_ops import auto_commit


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------

def _get_ssot_dir() -> Path:
    """获取 SSOT 根目录路径。"""
    config = load_config()
    return expand_path(config["source_dir"])


def _validate_url(url: str) -> None:
    """验证 URL 格式，仅允许安全的 git 远程协议和本地路径。

    Allowed: https://, git@, and local filesystem paths.
    Blocked: file://, ftp://, http://, ext://, and other exotic schemes.
    Raises ValueError if the URL uses a disallowed scheme.
    """
    # 允许 https 和 git SSH
    if url.startswith("https://") or url.startswith("git@"):
        return
    # 允许本地文件系统路径（绝对或相对）
    if url.startswith("/") or url.startswith("./") or url.startswith("../"):
        return
    raise ValueError(
        f"Invalid URL: {url!r}. "
        "Only https://, git@, or local paths are allowed."
    )


def _get_upstream_config() -> dict:
    """从配置文件读取 upstream 相关设置。"""
    config = load_config()
    return config.get("upstream", {})


def _get_sources_dir(ssot_dir: Path) -> Path:
    """获取上游源克隆的存储目录。"""
    upstream_cfg = _get_upstream_config()
    sources_rel = upstream_cfg.get("sources_dir", ".sources")
    return ssot_dir / sources_rel


def _get_manifest_path(ssot_dir: Path) -> Path:
    """获取 manifest 文件的完整路径。"""
    upstream_cfg = _get_upstream_config()
    manifest_rel = upstream_cfg.get("manifest_file", ".upstream-manifest.json")
    return ssot_dir / manifest_rel


def _load_manifest(ssot_dir: Path) -> dict:
    """加载上游 manifest 文件，不存在时返回空结构。"""
    manifest_path = _get_manifest_path(ssot_dir)
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"version": "1.0", "sources": {}, "skills": {}}


def _save_manifest(ssot_dir: Path, manifest: dict) -> None:
    """原子写入 manifest 文件：先写临时文件再重命名。"""
    manifest_path = _get_manifest_path(ssot_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # 写入临时文件后原子替换
    fd, tmp_path = tempfile.mkstemp(
        dir=str(manifest_path.parent), suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.rename(tmp_path, str(manifest_path))
    except Exception:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _compute_file_hash(filepath: str) -> str:
    """计算文件 MD5 哈希值。"""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_head_commit(repo_dir: Path) -> str:
    """获取本地仓库当前 HEAD commit hash。"""
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return "unknown"


def _detect_skills_prefix(source_dir: Path) -> str:
    """自动检测上游仓库中 skill 所在的子目录前缀。

    按优先级扫描常见目录名，如果没有找到就返回空字符串（仓库根即 skill 根）。
    """
    candidates = ["skills", "skill", "ai-skills", ".ai-skills", "src/skills"]
    for candidate in candidates:
        candidate_path = source_dir / candidate
        if candidate_path.is_dir():
            # 确认目录下确实有包含 SKILL.md 的子目录
            for item in candidate_path.iterdir():
                if item.is_dir() and (item / "SKILL.md").exists():
                    return candidate
    # 没有子目录也检查根目录本身是否直接含 skill 目录
    return ""


def _detect_local_modifications(
    ssot_dir: Path,
    skill_name: str,
    source_dir: Path,
    skills_prefix: str,
) -> bool:
    """检测本地 skill 文件是否与上游源不同。

    对每个文件计算 MD5 并与上游对应文件比较，任意不同即返回 True。
    """
    local_skill_dir = ssot_dir / skill_name
    if skills_prefix:
        upstream_skill_dir = source_dir / skills_prefix / skill_name
    else:
        upstream_skill_dir = source_dir / skill_name

    if not local_skill_dir.exists() or not upstream_skill_dir.exists():
        return True

    # 遍历本地文件
    for root, _dirs, files in os.walk(str(local_skill_dir)):
        for fname in files:
            local_file = os.path.join(root, fname)
            rel_path = os.path.relpath(local_file, str(local_skill_dir))
            upstream_file = os.path.join(str(upstream_skill_dir), rel_path)

            if not os.path.exists(upstream_file):
                return True
            if _compute_file_hash(local_file) != _compute_file_hash(upstream_file):
                return True

    # 检查上游是否有本地不存在的新文件
    for root, _dirs, files in os.walk(str(upstream_skill_dir)):
        for fname in files:
            upstream_file = os.path.join(root, fname)
            rel_path = os.path.relpath(upstream_file, str(upstream_skill_dir))
            local_file = os.path.join(str(local_skill_dir), rel_path)
            if not os.path.exists(local_file):
                return True

    return False


# ---------------------------------------------------------------------------
# 公开命令函数
# ---------------------------------------------------------------------------

def cmd_upstream_add(url: str, name: Optional[str] = None, branch: str = "main") -> dict:
    """注册一个新的上游源并克隆到 .sources/ 目录。"""
    # 验证 URL 协议
    try:
        _validate_url(url)
    except ValueError as e:
        print(f"[UPSTREAM] Error: {e}")
        return {"success": False, "error": str(e)}

    ssot_dir = _get_ssot_dir()
    sources_dir = _get_sources_dir(ssot_dir)
    manifest = _load_manifest(ssot_dir)

    # 从 URL 推导名称
    if not name:
        name = url.rstrip("/").split("/")[-1]
        if name.endswith(".git"):
            name = name[:-4]

    # 检查是否已注册
    if name in manifest.get("sources", {}):
        print(f"[UPSTREAM] Error: source '{name}' already registered.")
        print(f"[UPSTREAM] Use 'upstream update {name}' to pull latest changes.")
        return {"success": False, "error": f"source '{name}' already exists"}

    # 准备目录
    source_clone_dir = sources_dir / name
    sources_dir.mkdir(parents=True, exist_ok=True)

    print(f"[UPSTREAM] Adding source: {name}")
    print(f"[UPSTREAM] URL: {url}")
    print(f"[UPSTREAM] Branch: {branch}")
    print(f"[UPSTREAM] Cloning into {source_clone_dir}...")

    # 克隆仓库
    clone_cmd = [
        "git", "clone", "--depth=1", "--branch", branch,
        url, str(source_clone_dir),
    ]
    result = subprocess.run(clone_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[UPSTREAM] Error: git clone failed.")
        print(f"[UPSTREAM] {result.stderr.strip()}")
        return {"success": False, "error": result.stderr.strip()}

    # 获取 HEAD commit
    head_commit = _get_head_commit(source_clone_dir)

    # 自动检测 skills 前缀
    skills_prefix = _detect_skills_prefix(source_clone_dir)

    # 更新 manifest
    now_iso = datetime.now(timezone.utc).isoformat()
    manifest.setdefault("sources", {})[name] = {
        "url": url,
        "branch": branch,
        "skills_prefix": skills_prefix,
        "last_pulled_commit": head_commit,
        "last_pulled_at": now_iso,
        "added_at": now_iso,
    }
    _save_manifest(ssot_dir, manifest)

    print(f"[UPSTREAM] Clone complete. HEAD: {head_commit[:12]}")
    if skills_prefix:
        print(f"[UPSTREAM] Detected skills prefix: {skills_prefix}/")
    print(f"[UPSTREAM] Source '{name}' registered successfully.")

    return {"success": True, "name": name, "commit": head_commit}


def cmd_upstream_update(name: Optional[str] = None) -> dict:
    """更新一个或所有上游源。如果有变更则可选自动重新导入受影响的 skill。"""
    ssot_dir = _get_ssot_dir()
    sources_dir = _get_sources_dir(ssot_dir)
    manifest = _load_manifest(ssot_dir)
    upstream_cfg = _get_upstream_config()

    sources = manifest.get("sources", {})
    if not sources:
        print("[UPSTREAM] No upstream sources registered.")
        return {"success": True, "updated": []}

    # 确定更新范围
    if name:
        if name not in sources:
            print(f"[UPSTREAM] Error: source '{name}' not found.")
            return {"success": False, "error": f"source '{name}' not found"}
        targets = {name: sources[name]}
    else:
        targets = dict(sources)

    updated = []
    for src_name, src_info in targets.items():
        source_clone_dir = sources_dir / src_name
        if not source_clone_dir.exists():
            print(f"[UPSTREAM] Warning: source directory missing for '{src_name}', skipping.")
            continue

        old_commit = src_info.get("last_pulled_commit", "unknown")
        print(f"[UPSTREAM] Updating '{src_name}'...")

        # 拉取最新代码
        pull_cmd = ["git", "-C", str(source_clone_dir), "pull", "--ff-only"]
        result = subprocess.run(pull_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[UPSTREAM] Warning: pull failed for '{src_name}': {result.stderr.strip()}")
            continue

        new_commit = _get_head_commit(source_clone_dir)

        if new_commit == old_commit:
            print(f"[UPSTREAM] '{src_name}' already up to date. ({old_commit[:12]})")
            continue

        print(f"[UPSTREAM] '{src_name}' updated: {old_commit[:12]} -> {new_commit[:12]}")

        # 更新 manifest 中的 commit 记录
        now_iso = datetime.now(timezone.utc).isoformat()
        manifest["sources"][src_name]["last_pulled_commit"] = new_commit
        manifest["sources"][src_name]["last_pulled_at"] = now_iso
        updated.append(src_name)

        # 检查已导入的 skill 是否受影响
        skills_prefix = src_info.get("skills_prefix", "")
        changed_skills = []
        for skill_name, skill_info in manifest.get("skills", {}).items():
            if skill_info.get("source") != src_name:
                continue
            if _detect_local_modifications(
                ssot_dir, skill_name, source_clone_dir, skills_prefix
            ):
                changed_skills.append(skill_name)

        if changed_skills:
            print(f"[UPSTREAM] Skills with upstream changes: {', '.join(changed_skills)}")
            # 如果配置了自动同步则重新导入
            if upstream_cfg.get("auto_sync_on_update", False):
                print(f"[UPSTREAM] Auto-syncing changed skills...")
                cmd_upstream_import(
                    changed_skills, src_name, force=True, _skip_commit=True,
                )

    _save_manifest(ssot_dir, manifest)

    # 自动提交
    if updated:
        auto_commit(ssot_dir, "upstream update", ", ".join(updated))

    return {"success": True, "updated": updated}


def cmd_upstream_import(
    skills: list[str],
    source: str,
    force: bool = False,
    adopt: bool = False,
    _skip_commit: bool = False,
) -> dict:
    """从指定上游源导入一个或多个 skill 到 SSOT 根目录。"""
    ssot_dir = _get_ssot_dir()
    sources_dir = _get_sources_dir(ssot_dir)
    manifest = _load_manifest(ssot_dir)

    # 验证源存在
    src_info = manifest.get("sources", {}).get(source)
    if not src_info:
        print(f"[UPSTREAM] Error: source '{source}' not registered.")
        return {"success": False, "error": f"source '{source}' not found"}

    source_clone_dir = sources_dir / source
    skills_prefix = src_info.get("skills_prefix", "")
    head_commit = _get_head_commit(source_clone_dir)
    now_iso = datetime.now(timezone.utc).isoformat()

    imported = []
    skipped = []

    for skill_name in skills:
        # 定位上游 skill 目录
        if skills_prefix:
            upstream_skill = source_clone_dir / skills_prefix / skill_name
        else:
            upstream_skill = source_clone_dir / skill_name

        if not upstream_skill.is_dir():
            print(f"[UPSTREAM] Warning: skill '{skill_name}' not found in source '{source}'.")
            skipped.append(skill_name)
            continue

        # 检查是否有 SKILL.md
        if not (upstream_skill / "SKILL.md").exists():
            print(f"[UPSTREAM] Warning: '{skill_name}' has no SKILL.md, skipping.")
            skipped.append(skill_name)
            continue

        local_skill = ssot_dir / skill_name

        if adopt and local_skill.exists():
            # Adopt 模式：仅在 manifest 中记录，不复制文件
            print(f"[UPSTREAM] Adopting existing skill: {skill_name}")
            manifest.setdefault("skills", {})[skill_name] = {
                "source": source,
                "synced_commit": head_commit,
                "synced_at": now_iso,
                "adopted": True,
            }
            imported.append(skill_name)
            continue

        if local_skill.exists() and not force:
            # 本地已存在且未使用 --force，跳过
            print(f"[UPSTREAM] Skill '{skill_name}' already exists locally.")
            print(f"[UPSTREAM]   Use --force to overwrite or --adopt to track without copying.")
            skipped.append(skill_name)
            continue

        # 复制文件：如果 force 则先删除旧目录
        if local_skill.exists() and force:
            print(f"[UPSTREAM] Overwriting: {skill_name}")
            shutil.rmtree(local_skill)
        else:
            print(f"[UPSTREAM] Importing: {skill_name}")

        shutil.copytree(str(upstream_skill), str(local_skill))

        # 记录到 manifest
        manifest.setdefault("skills", {})[skill_name] = {
            "source": source,
            "synced_commit": head_commit,
            "synced_at": now_iso,
            "adopted": False,
        }
        imported.append(skill_name)
        print(f"[UPSTREAM] Imported: {skill_name}")

    _save_manifest(ssot_dir, manifest)

    # 自动提交
    if imported and not _skip_commit:
        auto_commit(ssot_dir, "upstream import", ", ".join(imported))

    summary_parts = []
    if imported:
        summary_parts.append(f"{len(imported)} imported")
    if skipped:
        summary_parts.append(f"{len(skipped)} skipped")
    print(f"[UPSTREAM] Import complete: {', '.join(summary_parts)}.")

    return {"success": True, "imported": imported, "skipped": skipped}


def cmd_upstream_list(source: Optional[str] = None) -> dict:
    """列出上游源中可用的 skill（区分已导入和未导入）。"""
    ssot_dir = _get_ssot_dir()
    sources_dir = _get_sources_dir(ssot_dir)
    manifest = _load_manifest(ssot_dir)

    sources = manifest.get("sources", {})
    imported_skills = manifest.get("skills", {})

    if not sources:
        print("[UPSTREAM] No upstream sources registered.")
        return {"success": True, "available": [], "imported": []}

    # 确定列出范围
    if source:
        if source not in sources:
            print(f"[UPSTREAM] Error: source '{source}' not found.")
            return {"success": False, "error": f"source '{source}' not found"}
        targets = {source: sources[source]}
    else:
        targets = dict(sources)

    all_available = []
    all_imported = []

    for src_name, src_info in targets.items():
        source_clone_dir = sources_dir / src_name
        skills_prefix = src_info.get("skills_prefix", "")

        if skills_prefix:
            skills_root = source_clone_dir / skills_prefix
        else:
            skills_root = source_clone_dir

        print(f"\n[UPSTREAM] Source: {src_name} ({src_info.get('url', 'N/A')})")
        print(f"[UPSTREAM] Branch: {src_info.get('branch', 'N/A')}")
        print(f"[UPSTREAM] Last pulled: {src_info.get('last_pulled_at', 'N/A')}")
        print()

        if not skills_root.exists():
            print(f"  (source directory missing)")
            continue

        # 扫描 skill 目录
        available = []
        imported = []
        for item in sorted(skills_root.iterdir()):
            if not item.is_dir():
                continue
            if not (item / "SKILL.md").exists():
                continue

            skill_name = item.name
            is_imported = (
                skill_name in imported_skills
                and imported_skills[skill_name].get("source") == src_name
            )

            if is_imported:
                imported.append(skill_name)
                marker = "[imported]"
            else:
                available.append(skill_name)
                marker = "[available]"

            print(f"  {marker} {skill_name}")

        if not available and not imported:
            print("  (no skills found)")

        all_available.extend(available)
        all_imported.extend(imported)

    print(f"\n[UPSTREAM] Summary: {len(all_imported)} imported, {len(all_available)} available.")

    return {"success": True, "available": all_available, "imported": all_imported}


def cmd_upstream_status() -> dict:
    """显示所有注册的上游源和已导入 skill 的状态。"""
    ssot_dir = _get_ssot_dir()
    sources_dir = _get_sources_dir(ssot_dir)
    manifest = _load_manifest(ssot_dir)

    sources = manifest.get("sources", {})
    imported_skills = manifest.get("skills", {})

    print("=" * 60)
    print("Upstream Status")
    print("=" * 60)

    # 显示上游源
    if sources:
        print(f"\n[SOURCES] ({len(sources)} registered)")
        for src_name, src_info in sources.items():
            source_clone_dir = sources_dir / src_name
            exists = source_clone_dir.exists()
            print(f"\n  {src_name}:")
            print(f"    URL:    {src_info.get('url', 'N/A')}")
            print(f"    Branch: {src_info.get('branch', 'N/A')}")
            print(f"    Commit: {src_info.get('last_pulled_commit', 'N/A')[:12]}")
            print(f"    Pulled: {src_info.get('last_pulled_at', 'N/A')}")
            print(f"    Local:  {'exists' if exists else 'MISSING'}")
    else:
        print("\n[SOURCES] No upstream sources registered.")

    # 显示已导入的 skill
    if imported_skills:
        print(f"\n[SKILLS] ({len(imported_skills)} imported)")
        for skill_name, skill_info in imported_skills.items():
            src_name = skill_info.get("source", "unknown")
            src_info = sources.get(src_name, {})
            skills_prefix = src_info.get("skills_prefix", "")
            source_clone_dir = sources_dir / src_name

            # 检测本地修改
            modified = False
            if source_clone_dir.exists():
                modified = _detect_local_modifications(
                    ssot_dir, skill_name, source_clone_dir, skills_prefix,
                )

            mod_marker = " [modified]" if modified else ""
            adopted_marker = " [adopted]" if skill_info.get("adopted") else ""

            print(f"\n  {skill_name}:{mod_marker}{adopted_marker}")
            print(f"    Source:  {src_name}")
            print(f"    Synced:  {skill_info.get('synced_commit', 'N/A')[:12]}")
            print(f"    Date:    {skill_info.get('synced_at', 'N/A')}")
    else:
        print("\n[SKILLS] No skills imported from upstream.")

    print()
    return {
        "success": True,
        "sources": list(sources.keys()),
        "skills": list(imported_skills.keys()),
    }


def cmd_upstream_diff(skill: str) -> dict:
    """比较本地 skill 文件与上游源文件的差异。"""
    ssot_dir = _get_ssot_dir()
    sources_dir = _get_sources_dir(ssot_dir)
    manifest = _load_manifest(ssot_dir)

    # 查找 skill 的上游信息
    skill_info = manifest.get("skills", {}).get(skill)
    if not skill_info:
        print(f"[UPSTREAM] Error: skill '{skill}' is not tracked as upstream-imported.")
        return {"success": False, "error": f"skill '{skill}' not tracked"}

    src_name = skill_info.get("source", "")
    src_info = manifest.get("sources", {}).get(src_name, {})
    skills_prefix = src_info.get("skills_prefix", "")
    source_clone_dir = sources_dir / src_name

    if not source_clone_dir.exists():
        print(f"[UPSTREAM] Error: source directory for '{src_name}' not found.")
        return {"success": False, "error": "source directory missing"}

    local_skill_dir = ssot_dir / skill
    if skills_prefix:
        upstream_skill_dir = source_clone_dir / skills_prefix / skill
    else:
        upstream_skill_dir = source_clone_dir / skill

    if not upstream_skill_dir.exists():
        print(f"[UPSTREAM] Error: skill '{skill}' not found in upstream source.")
        return {"success": False, "error": "skill not in upstream"}

    print(f"[UPSTREAM] Comparing: {skill}")
    print(f"[UPSTREAM] Local:    {local_skill_dir}")
    print(f"[UPSTREAM] Upstream: {upstream_skill_dir}")
    print()

    differences = []
    identical = []

    # 收集所有相关文件
    all_files = set()
    if local_skill_dir.exists():
        for root, _dirs, files in os.walk(str(local_skill_dir)):
            for fname in files:
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, str(local_skill_dir))
                all_files.add(rel)

    for root, _dirs, files in os.walk(str(upstream_skill_dir)):
        for fname in files:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, str(upstream_skill_dir))
            all_files.add(rel)

    for rel_path in sorted(all_files):
        local_file = os.path.join(str(local_skill_dir), rel_path)
        upstream_file = os.path.join(str(upstream_skill_dir), rel_path)

        local_exists = os.path.exists(local_file)
        upstream_exists = os.path.exists(upstream_file)

        if local_exists and upstream_exists:
            local_hash = _compute_file_hash(local_file)
            upstream_hash = _compute_file_hash(upstream_file)
            if local_hash != upstream_hash:
                print(f"  [CHANGED] {rel_path}")
                print(f"             local:    {local_hash}")
                print(f"             upstream: {upstream_hash}")
                differences.append({"file": rel_path, "status": "changed"})
            else:
                identical.append(rel_path)
        elif local_exists and not upstream_exists:
            print(f"  [LOCAL ONLY] {rel_path}")
            differences.append({"file": rel_path, "status": "local_only"})
        elif not local_exists and upstream_exists:
            print(f"  [UPSTREAM ONLY] {rel_path}")
            differences.append({"file": rel_path, "status": "upstream_only"})

    if not differences:
        print("  No differences found. Files are identical.")
    else:
        print(f"\n[UPSTREAM] {len(differences)} file(s) differ, {len(identical)} identical.")

    return {"success": True, "differences": differences, "identical": identical}
