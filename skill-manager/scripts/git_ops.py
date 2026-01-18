#!/usr/bin/env python3
"""
Git operations for skill manager.
"""

import subprocess
from pathlib import Path
from typing import Optional

from .utils import load_config


def is_git_repo(path: Path) -> bool:
    """Check if path is in a git repository."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def git_add(path: Path, files: list[str] = None) -> bool:
    """Add files to git staging."""
    try:
        if files:
            cmd = ["git", "-C", str(path), "add"] + files
        else:
            cmd = ["git", "-C", str(path), "add", "-A"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def git_commit(path: Path, message: str) -> tuple[bool, str]:
    """Commit staged changes."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "commit", "-m", message],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        # No changes to commit is also ok
        if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
            return True, "nothing to commit"
        return False, result.stderr.strip()
    except Exception as e:
        return False, str(e)


def git_push(path: Path) -> tuple[bool, str]:
    """Push to remote."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "push"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True, result.stdout.strip() or "pushed"
        return False, result.stderr.strip()
    except Exception as e:
        return False, str(e)


def auto_commit(path: Path, action: str, skill_name: Optional[str] = None) -> bool:
    """Auto commit if configured."""
    config = load_config()
    git_config = config.get("git", {})
    
    if not git_config.get("auto_commit", True):
        return True
    
    if not is_git_repo(path):
        return True
    
    prefix = git_config.get("commit_prefix", "skills:")
    if skill_name:
        message = f"{prefix} {action} {skill_name}"
    else:
        message = f"{prefix} {action}"
    
    git_add(path)
    success, msg = git_commit(path, message)
    
    if success and msg != "nothing to commit":
        print(f"[Git] Committed: {message}")
        
        if git_config.get("auto_push", False):
            push_success, push_msg = git_push(path)
            if push_success:
                print(f"[Git] Pushed to remote")
            else:
                print(f"[Git] Push failed: {push_msg}")
    
    return success


def manual_commit(path: Path, message: str, auto_message: bool = False) -> bool:
    """Create a manual commit."""
    if not is_git_repo(path):
        print(f"[Error] Not a git repository: {path}")
        return False
    
    config = load_config()
    git_config = config.get("git", {})
    prefix = git_config.get("commit_prefix", "skills:")
    
    if auto_message:
        # Generate commit message based on changes
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            capture_output=True,
            text=True,
        )
        changes = result.stdout.strip().split("\n")
        if not changes or changes == [""]:
            print("[Info] No changes to commit")
            return True
        
        # Analyze changes
        added = []
        modified = []
        deleted = []
        for line in changes:
            if line.startswith("A") or line.startswith("??"):
                added.append(line[3:].strip())
            elif line.startswith("M"):
                modified.append(line[3:].strip())
            elif line.startswith("D"):
                deleted.append(line[3:].strip())
        
        parts = []
        if added:
            parts.append(f"add {', '.join(added[:3])}")
        if modified:
            parts.append(f"update {', '.join(modified[:3])}")
        if deleted:
            parts.append(f"remove {', '.join(deleted[:3])}")
        
        message = f"{prefix} {'; '.join(parts)}"
    else:
        if not message.startswith(prefix):
            message = f"{prefix} {message}"
    
    git_add(path)
    success, msg = git_commit(path, message)
    
    if success:
        if msg == "nothing to commit":
            print("[Info] Nothing to commit")
        else:
            print(f"[Git] Committed: {message}")
    else:
        print(f"[Error] Commit failed: {msg}")
    
    return success
