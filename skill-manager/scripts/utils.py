#!/usr/bin/env python3
"""
Utility functions for skill manager.
"""

import json
import os
from pathlib import Path
from typing import Optional


def expand_path(path: str, base_dir: Optional[Path] = None) -> Path:
    """Expand ~ and environment variables in path."""
    expanded = os.path.expandvars(os.path.expanduser(path))
    p = Path(expanded)
    if not p.is_absolute() and base_dir:
        return base_dir / p
    return p


def get_config_path() -> Path:
    """Get config file path."""
    return expand_path("~/.ai-skills/skill-manager/config.json")


def get_default_config() -> dict:
    """Get default configuration."""
    return {
        "source_dir": "~/.ai-skills",
        "project_source_dir": ".ai-skills",
        "targets": {
            "claude": "~/.claude/skills",
            "cursor": "~/.cursor/skills",
            "codex": "~/.codex/skills",
            "gemini": "~/.gemini/skills",
            "antigravity": "~/.gemini/antigravity/global_skills",
        },
        "project_targets": {
            "claude": ".claude/skills",
            "cursor": ".cursor/skills",
            "codex": ".codex/skills",
            "gemini": ".gemini/skills",
            "antigravity": ".agent/skills",
        },
        "enabled_ides": ["claude", "cursor", "codex", "gemini", "antigravity"],
        "git": {
            "auto_commit": True,
            "commit_prefix": "skills:",
            "auto_push": False,
        },
        "sync": {
            "auto_after_install": True,
            "auto_after_create": False,
            "auto_after_remove": True,
            "use_symlinks": False,
            "default_scope": "global",
        },
        "exclude_skills": ["skill-manager"],
        "preserve_target_skills": {
            "codex": [".system"],
        },
        "search": {
            "index_path": "~/.ai-skills/skill-manager/data/index.json",
            "cache_ttl_hours": 24,
        },
        "profile": {
            "default_export_path": "~/.grimoire-profile.json",
            "gist_id": None,
        },
    }


def load_config() -> dict:
    """Load configuration from config.json or use defaults."""
    config_path = get_config_path()
    default_config = get_default_config()
    
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            # Merge with defaults
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict) and isinstance(config.get(key), dict):
                    for k, v in value.items():
                        if k not in config[key]:
                            config[key][k] = v
            return config
        except (json.JSONDecodeError, OSError):
            pass
    
    return default_config


def save_config(config: dict) -> None:
    """Save configuration to config.json."""
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


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


def get_skill_description(skill_path: Path) -> str:
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


def print_json(data: dict) -> None:
    """Print data as JSON (for LLM token efficiency)."""
    print(json.dumps(data, indent=2, ensure_ascii=False))
