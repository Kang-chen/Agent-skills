#!/usr/bin/env python3
"""
Remove skills from source and all IDE targets.
"""

import shutil
from pathlib import Path
from typing import Optional

from .utils import expand_path, load_config
from .git_ops import auto_commit


def remove_skill_from_target(skill_name: str, target_dir: Path) -> tuple[bool, str]:
    """Remove a skill from a target directory."""
    target = target_dir / skill_name
    
    if not target.exists() and not target.is_symlink():
        return True, "not present"
    
    if target.is_symlink():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    
    return True, "removed"


def cmd_remove(skill_name: str, scope: str = "global", 
               project_dir: Optional[Path] = None, keep_source: bool = False,
               force: bool = False, json_output: bool = False) -> dict:
    """Remove a skill from source and all IDE targets."""
    config = load_config()
    exclude = config.get("exclude_skills", [])
    result = {"success": False, "skill": skill_name, "removed_from": [], "error": None}
    
    if skill_name in exclude and not force:
        result["error"] = f"Cannot remove protected skill: {skill_name}. Use --force to override."
        print(f"Error: {result['error']}")
        return result
    
    if not force:
        confirm = input(f"Remove skill '{skill_name}'? [y/N] ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return result
    
    print(f"Removing skill: {skill_name}\n")
    
    removed_count = 0
    
    if scope in ("global", "all"):
        source_dir = expand_path(config.get("source_dir", "~/.ai-skills"))
        
        # Remove from IDE targets
        for ide_name in config.get("enabled_ides", []):
            target_path = config.get("targets", {}).get(ide_name)
            if target_path:
                target_dir = expand_path(target_path)
                success, msg = remove_skill_from_target(skill_name, target_dir)
                print(f"  [{ide_name.upper()}:G] {msg}")
                if msg == "removed":
                    removed_count += 1
                    result["removed_from"].append(f"global:{ide_name}")
        
        # Remove from source
        if not keep_source:
            source = source_dir / skill_name
            if source.exists():
                shutil.rmtree(source)
                print(f"\n  [SOURCE:G] Removed: {source}")
                result["removed_from"].append("global:source")
                removed_count += 1
        
        # Git commit
        auto_commit(source_dir, "remove", skill_name)
    
    if scope in ("project", "local", "all") and project_dir:
        source_dir = expand_path(config.get("project_source_dir", ".ai-skills"), project_dir)
        
        # Remove from IDE targets
        for ide_name in config.get("enabled_ides", []):
            target_path = config.get("project_targets", {}).get(ide_name)
            if target_path:
                target_dir = expand_path(target_path, project_dir)
                success, msg = remove_skill_from_target(skill_name, target_dir)
                print(f"  [{ide_name.upper()}:L] {msg}")
                if msg == "removed":
                    removed_count += 1
                    result["removed_from"].append(f"local:{ide_name}")
        
        # Remove from source
        if not keep_source:
            source = source_dir / skill_name
            if source.exists():
                shutil.rmtree(source)
                print(f"\n  [SOURCE:L] Removed: {source}")
                result["removed_from"].append("local:source")
                removed_count += 1
        
        # Git commit
        auto_commit(project_dir, "remove", skill_name)
    
    result["success"] = True
    print(f"\nRemoval complete! Removed from {removed_count} location(s).")
    
    if json_output:
        from .utils import print_json
        print_json(result)
    
    return result
