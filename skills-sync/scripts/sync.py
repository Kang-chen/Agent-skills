#!/usr/bin/env python3
"""
Skills Sync - Unified skill synchronization across AI IDEs.

Supports: Claude Code, Cursor, Codex, Gemini CLI, Antigravity

Modes:
  - Global: ~/.ai-skills/ -> ~/.xxx/skills/
  - Project: <project>/.ai-skills/ -> <project>/.xxx/skills/
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional


DEFAULT_CONFIG = {
    "global_source_dir": "~/.ai-skills",
    "project_source_dir": ".ai-skills",
    "global_targets": {
        "claude": "~/.claude/skills",
        "cursor": "~/.cursor/skills",
        "codex": "~/.codex/skills",
        "gemini": "~/.gemini/skills",
        "antigravity": "~/.gemini/antigravity/skills",
    },
    "project_targets": {
        "claude": ".claude/skills",
        "cursor": ".cursor/skills",
        "codex": ".codex/skills",
        "gemini": ".gemini/skills",
        "antigravity": ".agent/skills",
    },
    "enabled": ["claude", "cursor", "codex", "gemini", "antigravity"],
    "exclude_skills": ["skills-sync"],
    "use_symlinks": True,
}


def expand_path(path: str, base_dir: Optional[Path] = None) -> Path:
    """Expand ~ and environment variables in path."""
    expanded = os.path.expandvars(os.path.expanduser(path))
    p = Path(expanded)
    if not p.is_absolute() and base_dir:
        return base_dir / p
    return p


def load_config() -> dict:
    """Load configuration from config.json or use defaults."""
    config_path = expand_path("~/.ai-skills/skills-sync/config.json")
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
            # Handle legacy keys
            if "source_dir" in config and "global_source_dir" not in config:
                config["global_source_dir"] = config.pop("source_dir")
            if "targets" in config and "global_targets" not in config:
                config["global_targets"] = config.pop("targets")
            return config
    return DEFAULT_CONFIG.copy()


def get_source_dir(config: dict, scope: str, project_dir: Optional[Path] = None) -> Path:
    """Get source directory based on scope."""
    if scope == "project" and project_dir:
        return expand_path(config.get("project_source_dir", ".ai-skills"), project_dir)
    return expand_path(config.get("global_source_dir", "~/.ai-skills"))


def get_skills_from_dir(source_dir: Path, exclude: list[str]) -> list[str]:
    """Get list of skills in a source directory."""
    if not source_dir.exists():
        return []
    
    skills = []
    for item in source_dir.iterdir():
        if item.is_dir() and item.name not in exclude:
            skill_md = item / "SKILL.md"
            if skill_md.exists():
                skills.append(item.name)
    return sorted(skills)


def sync_skill(skill_name: str, source_dir: Path, target_dir: Path, use_symlinks: bool) -> tuple[bool, str]:
    """Sync a single skill to a target directory."""
    source = source_dir / skill_name
    target = target_dir / skill_name
    
    if not source.exists():
        return False, f"not found"
    
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


def cmd_sync(config: dict, skill_name: Optional[str], scope: str, project_dir: Optional[Path]):
    """Sync skills based on scope."""
    use_symlinks = config.get("use_symlinks", True)
    exclude = config.get("exclude_skills", [])
    
    if scope == "global":
        _sync_global(config, skill_name, use_symlinks, exclude)
    elif scope == "project":
        _sync_project(config, skill_name, project_dir, use_symlinks, exclude)
    elif scope == "all":
        _sync_global(config, skill_name, use_symlinks, exclude)
        print()
        _sync_project(config, skill_name, project_dir, use_symlinks, exclude)


def _sync_global(config: dict, skill_name: Optional[str], use_symlinks: bool, exclude: list[str]):
    """Sync global skills."""
    source_dir = expand_path(config.get("global_source_dir", "~/.ai-skills"))
    
    if skill_name:
        skills = [skill_name]
    else:
        skills = get_skills_from_dir(source_dir, exclude)
    
    if not skills:
        print("[GLOBAL] No skills found in", source_dir)
        return
    
    print(f"[GLOBAL] Source: {source_dir}")
    print(f"[GLOBAL] Syncing {len(skills)} skill(s)...\n")
    
    for ide_name in config["enabled"]:
        target_path = config.get("global_targets", {}).get(ide_name)
        if not target_path:
            continue
        
        target_dir = expand_path(target_path)
        print(f"  [{ide_name.upper()}] {target_dir}")
        
        for skill in skills:
            success, method = sync_skill(skill, source_dir, target_dir, use_symlinks)
            status = "OK" if success else "FAIL"
            print(f"    [{status}] {skill} ({method})")
    
    print("\n[GLOBAL] Sync complete!")


def _sync_project(config: dict, skill_name: Optional[str], project_dir: Optional[Path], 
                  use_symlinks: bool, exclude: list[str]):
    """Sync project skills."""
    if not project_dir:
        print("[PROJECT] No project directory specified")
        return
    
    source_dir = expand_path(config.get("project_source_dir", ".ai-skills"), project_dir)
    
    if not source_dir.exists():
        print(f"[PROJECT] Source directory not found: {source_dir}")
        print(f"[PROJECT] Create it with: mkdir -p {source_dir}")
        return
    
    if skill_name:
        skills = [skill_name]
    else:
        skills = get_skills_from_dir(source_dir, exclude)
    
    if not skills:
        print(f"[PROJECT] No skills found in {source_dir}")
        return
    
    print(f"[PROJECT] Project: {project_dir}")
    print(f"[PROJECT] Source: {source_dir}")
    print(f"[PROJECT] Syncing {len(skills)} skill(s)...\n")
    
    for ide_name in config["enabled"]:
        target_path = config.get("project_targets", {}).get(ide_name)
        if not target_path:
            continue
        
        target_dir = expand_path(target_path, project_dir)
        print(f"  [{ide_name.upper()}] {target_dir}")
        
        for skill in skills:
            success, method = sync_skill(skill, source_dir, target_dir, use_symlinks)
            status = "OK" if success else "FAIL"
            print(f"    [{status}] {skill} ({method})")
    
    print("\n[PROJECT] Sync complete!")


def cmd_install(config: dict, skill_name: str, source: Optional[str],
                scope: str, project_dir: Optional[Path]):
    """Install a skill."""
    if scope == "project" and project_dir:
        source_dir = expand_path(config.get("project_source_dir", ".ai-skills"), project_dir)
    else:
        source_dir = expand_path(config.get("global_source_dir", "~/.ai-skills"))
    
    target = source_dir / skill_name
    
    if target.exists():
        print(f"Skill '{skill_name}' already exists at {target}")
        return
    
    # Ensure source directory exists
    source_dir.mkdir(parents=True, exist_ok=True)
    
    if source:
        source_path = expand_path(source)
        if source_path.exists():
            shutil.copytree(source_path, target)
            print(f"Installed '{skill_name}' from {source_path}")
        elif source.startswith(("http://", "https://", "git@")):
            import subprocess
            try:
                subprocess.run(["git", "clone", source, str(target)], check=True)
                print(f"Cloned '{skill_name}' from {source}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to clone: {e}")
                return
        else:
            print(f"Source not found: {source}")
            return
    else:
        target.mkdir(parents=True)
        skill_md = target / "SKILL.md"
        skill_md.write_text(f"""---
name: {skill_name}
description: TODO - Add description for when this skill should be used.
---

# {skill_name.replace('-', ' ').title()}

TODO: Add skill instructions here.
""")
        print(f"Created new skill template at {target}")
    
    # Sync
    if scope == "project":
        _sync_project(config, skill_name, project_dir, config.get("use_symlinks", True), 
                      config.get("exclude_skills", []))
    else:
        _sync_global(config, skill_name, config.get("use_symlinks", True),
                     config.get("exclude_skills", []))


def cmd_remove(config: dict, skill_name: str, scope: str, 
               project_dir: Optional[Path], keep_source: bool):
    """Remove a skill."""
    exclude = config.get("exclude_skills", [])
    
    if skill_name in exclude:
        print(f"Cannot remove protected skill: {skill_name}")
        return
    
    print(f"Removing skill '{skill_name}'...\n")
    
    if scope in ("global", "all"):
        source_dir = expand_path(config.get("global_source_dir", "~/.ai-skills"))
        for ide_name in config["enabled"]:
            target_path = config.get("global_targets", {}).get(ide_name)
            if target_path:
                target_dir = expand_path(target_path)
                success, msg = remove_skill_from_target(skill_name, target_dir)
                print(f"  [{ide_name.upper()}:G] {msg}")
        
        if not keep_source:
            source = source_dir / skill_name
            if source.exists():
                shutil.rmtree(source)
                print(f"\n  [GLOBAL SOURCE] Removed: {source}")
    
    if scope in ("project", "all") and project_dir:
        source_dir = expand_path(config.get("project_source_dir", ".ai-skills"), project_dir)
        for ide_name in config["enabled"]:
            target_path = config.get("project_targets", {}).get(ide_name)
            if target_path:
                target_dir = expand_path(target_path, project_dir)
                success, msg = remove_skill_from_target(skill_name, target_dir)
                print(f"  [{ide_name.upper()}:P] {msg}")
        
        if not keep_source:
            source = source_dir / skill_name
            if source.exists():
                shutil.rmtree(source)
                print(f"\n  [PROJECT SOURCE] Removed: {source}")
    
    print("\nRemoval complete!")


def cmd_list(config: dict, scope: str, project_dir: Optional[Path]):
    """List skills."""
    exclude = config.get("exclude_skills", [])
    
    if scope in ("global", "all"):
        source_dir = expand_path(config.get("global_source_dir", "~/.ai-skills"))
        skills = get_skills_from_dir(source_dir, exclude)
        
        print(f"[GLOBAL] {source_dir}\n")
        if skills:
            for skill in skills:
                skill_path = source_dir / skill
                desc = _get_skill_description(skill_path)
                print(f"  - {skill}")
                if desc:
                    print(f"    {desc}")
            print(f"\n  Total: {len(skills)} skill(s)")
        else:
            print("  (no skills)")
    
    if scope in ("project", "all") and project_dir:
        source_dir = expand_path(config.get("project_source_dir", ".ai-skills"), project_dir)
        
        if scope == "all":
            print()
        
        print(f"[PROJECT] {source_dir}\n")
        if source_dir.exists():
            skills = get_skills_from_dir(source_dir, exclude)
            if skills:
                for skill in skills:
                    skill_path = source_dir / skill
                    desc = _get_skill_description(skill_path)
                    print(f"  - {skill}")
                    if desc:
                        print(f"    {desc}")
                print(f"\n  Total: {len(skills)} skill(s)")
            else:
                print("  (no skills)")
        else:
            print("  (directory not found)")


def _get_skill_description(skill_path: Path) -> str:
    """Extract description from SKILL.md."""
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        return ""
    
    content = skill_md.read_text()
    for line in content.split("\n"):
        if line.startswith("description:"):
            desc = line.replace("description:", "").strip()[:60]
            if len(desc) == 60:
                desc += "..."
            return desc
    return ""


def cmd_status(config: dict, scope: str, project_dir: Optional[Path]):
    """Check sync status."""
    exclude = config.get("exclude_skills", [])
    
    print("Sync Status:\n")
    
    if scope in ("global", "all"):
        source_dir = expand_path(config.get("global_source_dir", "~/.ai-skills"))
        skills = get_skills_from_dir(source_dir, exclude)
        
        print(f"[GLOBAL] Source: {source_dir}")
        print(f"[GLOBAL] Skills: {len(skills)}\n")
        
        for ide_name in config["enabled"]:
            target_path = config.get("global_targets", {}).get(ide_name)
            if not target_path:
                continue
            
            target_dir = expand_path(target_path)
            print(f"  [{ide_name.upper()}] {target_dir}")
            
            if not target_dir.exists():
                print("    (not exists)")
                continue
            
            for skill in skills:
                _print_skill_status(skill, source_dir, target_dir)
    
    if scope in ("project", "all") and project_dir:
        source_dir = expand_path(config.get("project_source_dir", ".ai-skills"), project_dir)
        
        if scope == "all":
            print()
        
        print(f"[PROJECT] Project: {project_dir}")
        print(f"[PROJECT] Source: {source_dir}")
        
        if not source_dir.exists():
            print("[PROJECT] (source not found)\n")
        else:
            skills = get_skills_from_dir(source_dir, exclude)
            print(f"[PROJECT] Skills: {len(skills)}\n")
            
            for ide_name in config["enabled"]:
                target_path = config.get("project_targets", {}).get(ide_name)
                if not target_path:
                    continue
                
                target_dir = expand_path(target_path, project_dir)
                print(f"  [{ide_name.upper()}] {target_dir}")
                
                if not target_dir.exists():
                    print("    (not exists)")
                    continue
                
                for skill in skills:
                    _print_skill_status(skill, source_dir, target_dir)


def _print_skill_status(skill: str, source_dir: Path, target_dir: Path):
    """Print status of a single skill."""
    source = source_dir / skill
    target = target_dir / skill
    
    if not target.exists() and not target.is_symlink():
        print(f"    [MISSING] {skill}")
    elif target.is_symlink():
        try:
            link_target = target.resolve()
            if link_target == source.resolve():
                print(f"    [OK] {skill}")
            else:
                print(f"    [STALE] {skill} -> {link_target}")
        except Exception:
            print(f"    [BROKEN] {skill}")
    elif target.is_dir():
        print(f"    [COPY] {skill}")
    else:
        print(f"    [?] {skill}")


def cmd_init_project(config: dict, project_dir: Path):
    """Initialize project skills directory."""
    source_dir = expand_path(config.get("project_source_dir", ".ai-skills"), project_dir)
    
    if source_dir.exists():
        print(f"Project skills directory already exists: {source_dir}")
        return
    
    source_dir.mkdir(parents=True)
    print(f"Created project skills directory: {source_dir}")
    print(f"\nNext steps:")
    print(f"  1. Add skills to {source_dir}/<skill-name>/SKILL.md")
    print(f"  2. Run: skills sync -p")


def find_project_root(start_dir: Path) -> Optional[Path]:
    """Find project root by looking for common markers."""
    markers = [".git", ".claude", ".cursor", ".codex", ".agent", "package.json", "pyproject.toml", ".ai-skills"]
    
    current = start_dir.resolve()
    while current != current.parent:
        for marker in markers:
            if (current / marker).exists():
                return current
        current = current.parent
    
    return None


def add_scope_args(parser):
    """Add scope arguments to a parser."""
    parser.add_argument("-g", "--global", dest="global_scope", action="store_true",
                        help="Target global (~/.ai-skills/ -> ~/.xxx/skills/)")
    parser.add_argument("-p", "--project", dest="project_scope", action="store_true",
                        help="Target project (.ai-skills/ -> .xxx/skills/)")
    parser.add_argument("--project-dir", type=Path, default=None,
                        help="Project directory (default: auto-detect)")


def main():
    parser = argparse.ArgumentParser(
        description="Sync skills across AI IDEs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scopes:
  -g, --global    Sync from ~/.ai-skills/ to ~/.xxx/skills/
  -p, --project   Sync from <project>/.ai-skills/ to <project>/.xxx/skills/
  (default)       Global only

Examples:
  %(prog)s sync -g              # Sync global skills
  %(prog)s sync -p              # Sync project skills  
  %(prog)s install my-skill -p  # Create skill in project
  %(prog)s init -p              # Initialize project .ai-skills/
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # sync
    sync_parser = subparsers.add_parser("sync", help="Sync skills")
    sync_parser.add_argument("skill", nargs="?", help="Specific skill")
    add_scope_args(sync_parser)
    
    # install
    install_parser = subparsers.add_parser("install", help="Install a skill")
    install_parser.add_argument("skill", help="Skill name")
    install_parser.add_argument("--source", "-s", help="Source path or git URL")
    add_scope_args(install_parser)
    
    # remove
    remove_parser = subparsers.add_parser("remove", help="Remove a skill")
    remove_parser.add_argument("skill", help="Skill name")
    remove_parser.add_argument("--keep-source", action="store_true", help="Keep source")
    add_scope_args(remove_parser)
    
    # list
    list_parser = subparsers.add_parser("list", help="List skills")
    add_scope_args(list_parser)
    
    # status
    status_parser = subparsers.add_parser("status", help="Check status")
    add_scope_args(status_parser)
    
    # init
    init_parser = subparsers.add_parser("init", help="Initialize project .ai-skills/")
    init_parser.add_argument("--project-dir", type=Path, default=None)
    init_parser.add_argument("-p", "--project", dest="project_scope", action="store_true")
    
    args = parser.parse_args()
    config = load_config()
    
    # Determine scope
    global_scope = getattr(args, "global_scope", False)
    project_scope = getattr(args, "project_scope", False)
    
    if global_scope and project_scope:
        scope = "all"
    elif project_scope:
        scope = "project"
    else:
        scope = "global"
    
    # Determine project directory
    project_dir = getattr(args, "project_dir", None)
    if scope in ("project", "all") and not project_dir:
        project_dir = find_project_root(Path.cwd())
        if not project_dir:
            project_dir = Path.cwd()
    
    # Execute command
    if args.command == "sync":
        cmd_sync(config, getattr(args, "skill", None), scope, project_dir)
    elif args.command == "install":
        cmd_install(config, args.skill, getattr(args, "source", None), scope, project_dir)
    elif args.command == "remove":
        cmd_remove(config, args.skill, scope, project_dir, getattr(args, "keep_source", False))
    elif args.command == "list":
        cmd_list(config, scope, project_dir)
    elif args.command == "status":
        cmd_status(config, scope, project_dir)
    elif args.command == "init":
        if project_dir:
            cmd_init_project(config, project_dir)
        else:
            print("Use -p or --project-dir to specify project")
    else:
        cmd_sync(config, None, scope, project_dir)


if __name__ == "__main__":
    main()
