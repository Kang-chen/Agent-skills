#!/usr/bin/env python3
"""
Sync skills to all enabled IDEs.
"""

import shutil
from pathlib import Path
from typing import Optional

from .utils import expand_path, load_config, get_skills_from_dir
from .git_ops import auto_commit


def sync_skill(skill_name: str, source_dir: Path, target_dir: Path, use_symlinks: bool) -> tuple[bool, str]:
    """Sync a single skill to a target directory."""
    source = source_dir / skill_name
    target = target_dir / skill_name
    
    if not source.exists():
        return False, "not found"
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    if target.exists() or target.is_symlink():
        if target.is_symlink():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    
    if use_symlinks:
        try:
            target.symlink_to(source.resolve())
            return True, "symlink"
        except OSError:
            shutil.copytree(source, target)
            return True, "copy (symlink failed)"
    else:
        shutil.copytree(source, target)
        return True, "copy"


def cleanup_orphaned_skills(source_skills: list[str], target_dir: Path, exclude: list[str],
                           preserve_dirs: list[str] = None) -> list[str]:
    """Remove skills in target that no longer exist in source."""
    removed = []
    if not target_dir.exists():
        return removed
    
    preserve_dirs = preserve_dirs or []
    
    for item in target_dir.iterdir():
        if item.is_dir() or item.is_symlink():
            skill_name = item.name
            if skill_name in source_skills or skill_name in exclude:
                continue
            if skill_name in preserve_dirs:
                continue
            is_skill = (item / "SKILL.md").exists() if item.is_dir() and not item.is_symlink() else True
            if is_skill:
                if item.is_symlink():
                    item.unlink()
                else:
                    shutil.rmtree(item)
                removed.append(skill_name)
    
    return removed


def cmd_sync(skill_name: Optional[str] = None, scope: str = "global", 
             project_dir: Optional[Path] = None, dry_run: bool = False,
             json_output: bool = False) -> dict:
    """Sync skills based on scope."""
    config = load_config()
    sync_config = config.get("sync", {})
    use_symlinks = sync_config.get("use_symlinks", False)
    exclude = config.get("exclude_skills", [])
    
    result = {
        "global": {"synced": 0, "removed": 0, "errors": []},
        "local": {"synced": 0, "removed": 0, "errors": []},
    }
    
    if scope in ("global", "all"):
        result["global"] = _sync_global(config, skill_name, use_symlinks, exclude, dry_run)
    
    if scope in ("project", "local", "all"):
        result["local"] = _sync_project(config, skill_name, project_dir, use_symlinks, exclude, dry_run)
    
    if json_output:
        from .utils import print_json
        print_json(result)
    
    return result


def _sync_global(config: dict, skill_name: Optional[str], use_symlinks: bool, 
                exclude: list[str], dry_run: bool = False) -> dict:
    """Sync global skills."""
    source_dir = expand_path(config.get("source_dir", "~/.ai-skills"))
    preserve_target_skills = config.get("preserve_target_skills", {})
    
    result = {"synced": 0, "removed": 0, "errors": [], "details": []}
    
    if skill_name:
        skills = [skill_name]
    else:
        skills = get_skills_from_dir(source_dir, exclude)
    
    if not skills:
        print(f"[GLOBAL] No skills found in {source_dir}")
        return result
    
    print(f"[GLOBAL] Source: {source_dir}")
    print(f"[GLOBAL] Syncing {len(skills)} skill(s)...\n")
    
    for ide_name in config.get("enabled_ides", []):
        target_path = config.get("targets", {}).get(ide_name)
        if not target_path:
            continue
        
        target_dir = expand_path(target_path)
        print(f"  [{ide_name.upper()}] {target_dir}")
        
        for skill in skills:
            if dry_run:
                print(f"    [DRY] {skill} (would sync)")
            else:
                success, method = sync_skill(skill, source_dir, target_dir, use_symlinks)
                status = "OK" if success else "FAIL"
                print(f"    [{status}] {skill} ({method})")
                if success:
                    result["synced"] += 1
                else:
                    result["errors"].append(f"{ide_name}:{skill}")
        
        if not skill_name and not dry_run:
            preserve_dirs = preserve_target_skills.get(ide_name, [])
            removed = cleanup_orphaned_skills(skills, target_dir, exclude, preserve_dirs)
            for r in removed:
                print(f"    [DEL] {r} (orphaned)")
                result["removed"] += 1
    
    print("\n[GLOBAL] Sync complete!")
    
    if not dry_run:
        auto_commit(source_dir, "sync global", skill_name)
    
    return result


def _sync_project(config: dict, skill_name: Optional[str], project_dir: Optional[Path],
                 use_symlinks: bool, exclude: list[str], dry_run: bool = False) -> dict:
    """Sync project skills."""
    result = {"synced": 0, "removed": 0, "errors": [], "details": []}
    
    if not project_dir:
        print("[PROJECT] No project directory specified")
        return result
    
    source_dir = expand_path(config.get("project_source_dir", ".ai-skills"), project_dir)
    
    if not source_dir.exists():
        print(f"[PROJECT] Source directory not found: {source_dir}")
        return result
    
    if skill_name:
        skills = [skill_name]
    else:
        skills = get_skills_from_dir(source_dir, exclude)
    
    if not skills:
        print(f"[PROJECT] No skills found in {source_dir}")
        return result
    
    print(f"[PROJECT] Project: {project_dir}")
    print(f"[PROJECT] Source: {source_dir}")
    print(f"[PROJECT] Syncing {len(skills)} skill(s)...\n")
    
    for ide_name in config.get("enabled_ides", []):
        target_path = config.get("project_targets", {}).get(ide_name)
        if not target_path:
            continue
        
        target_dir = expand_path(target_path, project_dir)
        print(f"  [{ide_name.upper()}] {target_dir}")
        
        for skill in skills:
            if dry_run:
                print(f"    [DRY] {skill} (would sync)")
            else:
                success, method = sync_skill(skill, source_dir, target_dir, use_symlinks)
                status = "OK" if success else "FAIL"
                print(f"    [{status}] {skill} ({method})")
                if success:
                    result["synced"] += 1
                else:
                    result["errors"].append(f"{ide_name}:{skill}")
        
        if not skill_name and not dry_run:
            removed = cleanup_orphaned_skills(skills, target_dir, exclude)
            for r in removed:
                print(f"    [DEL] {r} (orphaned)")
                result["removed"] += 1
    
    print("\n[PROJECT] Sync complete!")
    
    if not dry_run:
        auto_commit(project_dir, "sync local", skill_name)
    
    return result
