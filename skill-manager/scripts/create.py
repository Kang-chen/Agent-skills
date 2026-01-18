#!/usr/bin/env python3
"""
Create new skills from template.

This module wraps the skill-creator scripts in references/skill-creator/.
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .utils import expand_path, load_config
from .sync import cmd_sync
from .git_ops import auto_commit


MAX_SKILL_NAME_LENGTH = 64
ALLOWED_RESOURCES = {"scripts", "references", "assets"}


def normalize_skill_name(skill_name: str) -> str:
    """Normalize a skill name to lowercase hyphen-case."""
    normalized = skill_name.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    normalized = normalized.strip("-")
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized


def validate_skill_name(skill_name: str) -> tuple[bool, str]:
    """Validate skill name."""
    if not skill_name:
        return False, "Skill name is required"
    
    normalized = normalize_skill_name(skill_name)
    if not normalized:
        return False, "Skill name must include at least one letter or digit"
    
    if len(normalized) > MAX_SKILL_NAME_LENGTH:
        return False, f"Skill name too long ({len(normalized)} > {MAX_SKILL_NAME_LENGTH})"
    
    return True, normalized


def cmd_create(skill_name: str, path: Optional[str] = None, 
               resources: Optional[str] = None, examples: bool = False,
               scope: str = "global", project_dir: Optional[Path] = None,
               auto_sync: bool = False, json_output: bool = False) -> dict:
    """Create a new skill from template."""
    config = load_config()
    result = {"success": False, "skill": None, "path": None, "error": None}
    
    # Validate name
    valid, normalized_or_error = validate_skill_name(skill_name)
    if not valid:
        result["error"] = normalized_or_error
        print(f"Error: {normalized_or_error}")
        return result
    
    skill_name = normalized_or_error
    if skill_name != skill_name:
        print(f"Note: Normalized skill name to '{skill_name}'")
    
    # Determine destination
    if path:
        dest_dir = Path(path)
    elif scope == "local" and project_dir:
        dest_dir = expand_path(config.get("project_source_dir", ".ai-skills"), project_dir)
    else:
        dest_dir = expand_path(config.get("source_dir", "~/.ai-skills"))
    
    skill_dir = dest_dir / skill_name
    
    if skill_dir.exists():
        result["error"] = f"Skill already exists: {skill_dir}"
        print(f"Error: {result['error']}")
        return result
    
    print(f"Creating new skill: {skill_name}")
    print(f"Using guidelines from: references/skill-creator/SKILL.md\n")
    
    # Try to use the embedded skill-creator script
    skill_creator_script = Path(__file__).parent.parent / "references" / "skill-creator" / "scripts" / "init_skill.py"
    
    if skill_creator_script.exists():
        # Use the embedded script
        cmd = [sys.executable, str(skill_creator_script), skill_name, "--path", str(dest_dir)]
        
        if resources:
            cmd.extend(["--resources", resources])
        if examples:
            cmd.append("--examples")
        
        proc = subprocess.run(cmd, capture_output=False, text=True)
        
        if proc.returncode == 0:
            result["success"] = True
            result["skill"] = skill_name
            result["path"] = str(skill_dir)
        else:
            result["error"] = "Skill creation failed"
    else:
        # Fallback: create minimal skill manually
        print("[1/3] Creating directory...")
        skill_dir.mkdir(parents=True, exist_ok=True)
        
        print("[2/3] Generating SKILL.md...")
        skill_md = skill_dir / "SKILL.md"
        skill_title = " ".join(word.capitalize() for word in skill_name.split("-"))
        skill_md.write_text(f"""---
name: {skill_name}
description: [TODO: Add description for when this skill should be used]
---

# {skill_title}

## Overview

[TODO: Describe what this skill does]

## Usage

[TODO: Add usage instructions]
""")
        
        # Create resource directories if requested
        if resources:
            resource_list = [r.strip() for r in resources.split(",") if r.strip()]
            for resource in resource_list:
                if resource in ALLOWED_RESOURCES:
                    (skill_dir / resource).mkdir(exist_ok=True)
                    print(f"[OK] Created {resource}/")
        
        print("[3/3] done")
        
        result["success"] = True
        result["skill"] = skill_name
        result["path"] = str(skill_dir)
    
    if result["success"]:
        # Auto sync if configured
        if auto_sync or config.get("sync", {}).get("auto_after_create", False):
            print("\n[4/4] Syncing to all IDEs...")
            cmd_sync(skill_name=skill_name, scope=scope, project_dir=project_dir)
        
        # Git commit
        auto_commit(dest_dir, "create", skill_name)
        
        print(f"\nSkill '{skill_name}' created at {skill_dir}")
        print("\nNext steps:")
        print(f"  1. Edit {skill_dir}/SKILL.md")
        print("  2. Add scripts/references if needed")
        print(f"  3. Run: skills validate {skill_name}")
        print(f"  4. Run: skills sync {skill_name}")
    
    if json_output:
        from .utils import print_json
        print_json(result)
    
    return result
