#!/usr/bin/env python3
"""
Profile export/import for cross-machine skill synchronization.

Inspired by grimoire's profile system.
"""

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from .utils import expand_path, load_config, save_config, get_skills_from_dir


def _get_skill_source(skill_path: Path) -> str:
    """Try to determine the source of a skill."""
    # Check if it's a git repo
    git_dir = skill_path / ".git"
    if git_dir.exists():
        # Try to get remote URL
        config_file = git_dir / "config"
        if config_file.exists():
            content = config_file.read_text()
            for line in content.split("\n"):
                if "url = " in line:
                    return line.split("= ", 1)[1].strip()
    
    # Check for .source file (custom metadata)
    source_file = skill_path / ".source"
    if source_file.exists():
        return source_file.read_text().strip()
    
    return "local"


def _create_gist(content: dict, token: str, gist_id: Optional[str] = None) -> tuple[str, str]:
    """Create or update a GitHub Gist."""
    url = f"https://api.github.com/gists/{gist_id}" if gist_id else "https://api.github.com/gists"
    method = "PATCH" if gist_id else "POST"
    
    data = json.dumps({
        "description": "AI Skills Profile - skill-manager export",
        "public": False,
        "files": {
            "skills-profile.json": {
                "content": json.dumps(content, indent=2)
            }
        }
    }).encode()
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "skill-manager",
    }
    
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            return result["id"], result["html_url"]
    except urllib.error.HTTPError as e:
        raise Exception(f"Gist API error: {e.code} {e.reason}")


def _get_gist(gist_id: str, token: Optional[str] = None) -> dict:
    """Get a Gist by ID."""
    url = f"https://api.github.com/gists/{gist_id}"
    headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "skill-manager"}
    if token:
        headers["Authorization"] = f"token {token}"
    
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            content = result["files"]["skills-profile.json"]["content"]
            return json.loads(content)
    except (urllib.error.HTTPError, KeyError, json.JSONDecodeError) as e:
        raise Exception(f"Failed to fetch Gist: {e}")


def cmd_export(output: Optional[str] = None, gist: bool = False,
               json_output: bool = False) -> dict:
    """Export skills profile."""
    config = load_config()
    source_dir = expand_path(config.get("source_dir", "~/.ai-skills"))
    exclude = config.get("exclude_skills", [])
    
    result = {"success": False, "path": None, "gist_url": None, "error": None}
    
    skills = get_skills_from_dir(source_dir, exclude)
    
    profile = {
        "version": "1.0.0",
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "skills": [],
        "settings": {
            "enabled_ides": config.get("enabled_ides", []),
            "auto_commit": config.get("git", {}).get("auto_commit", True),
        }
    }
    
    print(f"Exporting profile from {source_dir}...")
    print(f"Found {len(skills)} skills\n")
    
    for skill in skills:
        skill_path = source_dir / skill
        source = _get_skill_source(skill_path)
        
        profile["skills"].append({
            "name": skill,
            "source": source,
            "location": "global",
        })
        print(f"  - {skill} ({source})")
    
    if gist:
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            result["error"] = "GITHUB_TOKEN environment variable required for Gist export"
            print(f"\nError: {result['error']}")
            return result
        
        try:
            existing_gist_id = config.get("profile", {}).get("gist_id")
            gist_id, gist_url = _create_gist(profile, token, existing_gist_id)
            
            # Save gist_id to config
            if "profile" not in config:
                config["profile"] = {}
            config["profile"]["gist_id"] = gist_id
            save_config(config)
            
            result["success"] = True
            result["gist_url"] = gist_url
            result["gist_id"] = gist_id
            
            print(f"\nProfile exported to Gist!")
            print(f"  URL: {gist_url}")
            print(f"\nTo import on another machine:")
            print(f"  skills import --gist {gist_id}")
        except Exception as e:
            result["error"] = str(e)
            print(f"\nError: {e}")
    else:
        # Export to file
        output_path = output or expand_path(config.get("profile", {}).get("default_export_path", "~/.skills-profile.json"))
        
        with open(output_path, "w") as f:
            json.dump(profile, f, indent=2)
        
        result["success"] = True
        result["path"] = str(output_path)
        
        print(f"\nProfile exported to: {output_path}")
        print(f"\nTo import:")
        print(f"  skills import {output_path}")
    
    if json_output:
        from .utils import print_json
        print_json(result)
    
    return result


def cmd_import(source: Optional[str] = None, gist_id: Optional[str] = None,
               dry_run: bool = False, json_output: bool = False) -> dict:
    """Import skills profile."""
    config = load_config()
    source_dir = expand_path(config.get("source_dir", "~/.ai-skills"))
    exclude = config.get("exclude_skills", [])
    
    result = {"success": False, "installed": [], "skipped": [], "error": None}
    
    # Load profile
    if gist_id:
        print(f"Fetching profile from Gist: {gist_id}...")
        token = os.environ.get("GITHUB_TOKEN")
        try:
            profile = _get_gist(gist_id, token)
        except Exception as e:
            result["error"] = str(e)
            print(f"Error: {e}")
            return result
    elif source:
        source_path = Path(source)
        if not source_path.exists():
            result["error"] = f"Profile not found: {source}"
            print(f"Error: {result['error']}")
            return result
        
        with open(source_path) as f:
            profile = json.load(f)
    else:
        result["error"] = "Specify --gist <id> or provide profile path"
        print(f"Error: {result['error']}")
        return result
    
    print(f"Found {len(profile.get('skills', []))} skills in profile\n")
    
    # Get currently installed skills
    installed_skills = set(get_skills_from_dir(source_dir, exclude))
    
    # Compare
    to_install = []
    already_installed = []
    
    for skill in profile.get("skills", []):
        name = skill["name"]
        source_url = skill.get("source", "local")
        
        if name in installed_skills:
            already_installed.append(name)
            print(f"  ✓ {name} (already installed)")
            result["skipped"].append(name)
        elif source_url == "local":
            print(f"  ○ {name} (local-only, cannot install)")
            result["skipped"].append(name)
        else:
            to_install.append(skill)
            print(f"  ○ {name} (will install from {source_url})")
    
    if not to_install:
        print("\nAll skills are already installed.")
        result["success"] = True
        return result
    
    if dry_run:
        print(f"\n[DRY RUN] Would install {len(to_install)} skill(s)")
        result["success"] = True
        return result
    
    # Confirm
    confirm = input(f"\nInstall {len(to_install)} skill(s)? [Y/n] ")
    if confirm.lower() == "n":
        print("Cancelled.")
        return result
    
    # Install
    from .install import cmd_install
    
    for skill in to_install:
        name = skill["name"]
        source_url = skill.get("source", "")
        
        if source_url and source_url != "local":
            print(f"\nInstalling {name}...")
            install_result = cmd_install(source_url, name=name)
            if install_result.get("success"):
                result["installed"].append(name)
            else:
                print(f"  Failed: {install_result.get('error')}")
    
    result["success"] = True
    print(f"\nImport complete!")
    print(f"  Installed: {len(result['installed'])}")
    print(f"  Skipped: {len(result['skipped'])}")
    
    if json_output:
        from .utils import print_json
        print_json(result)
    
    return result
