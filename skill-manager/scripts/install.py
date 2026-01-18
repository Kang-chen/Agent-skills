#!/usr/bin/env python3
"""
Install skills from GitHub or other sources.
"""

import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

from .utils import expand_path, load_config
from .sync import cmd_sync
from .git_ops import auto_commit


class InstallError(Exception):
    pass


def _github_request(url: str) -> bytes:
    """Make a request to GitHub, with token support."""
    headers = {"User-Agent": "skill-manager"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        raise InstallError(f"HTTP {e.code}: {e.reason}")


def _parse_github_url(url: str, default_ref: str = "main") -> tuple[str, str, str, Optional[str]]:
    """Parse GitHub URL to extract owner, repo, ref, and path."""
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc != "github.com":
        raise InstallError("Only GitHub URLs are supported.")
    
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise InstallError("Invalid GitHub URL.")
    
    owner, repo = parts[0], parts[1]
    ref = default_ref
    subpath = ""
    
    if len(parts) > 2:
        if parts[2] in ("tree", "blob"):
            if len(parts) < 4:
                raise InstallError("GitHub URL missing ref or path.")
            ref = parts[3]
            subpath = "/".join(parts[4:])
        else:
            subpath = "/".join(parts[2:])
    
    return owner, repo, ref, subpath or None


def _download_repo_zip(owner: str, repo: str, ref: str, dest_dir: str) -> str:
    """Download and extract repository zip."""
    zip_url = f"https://codeload.github.com/{owner}/{repo}/zip/{ref}"
    zip_path = os.path.join(dest_dir, "repo.zip")
    
    try:
        payload = _github_request(zip_url)
    except InstallError:
        raise
    
    with open(zip_path, "wb") as f:
        f.write(payload)
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Security check
        for info in zf.infolist():
            if info.filename.startswith("/") or ".." in info.filename:
                raise InstallError("Archive contains unsafe paths.")
        zf.extractall(dest_dir)
        top_levels = {name.split("/")[0] for name in zf.namelist() if name}
    
    if len(top_levels) != 1:
        raise InstallError("Unexpected archive layout.")
    
    return os.path.join(dest_dir, next(iter(top_levels)))


def _git_sparse_checkout(repo_url: str, ref: str, paths: list[str], dest_dir: str) -> str:
    """Clone with sparse checkout."""
    repo_dir = os.path.join(dest_dir, "repo")
    
    clone_cmd = [
        "git", "clone",
        "--filter=blob:none", "--depth", "1", "--sparse",
        "--single-branch", "--branch", ref,
        repo_url, repo_dir,
    ]
    
    result = subprocess.run(clone_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Try without branch
        clone_cmd = [
            "git", "clone",
            "--filter=blob:none", "--depth", "1", "--sparse",
            "--single-branch",
            repo_url, repo_dir,
        ]
        result = subprocess.run(clone_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise InstallError(f"Git clone failed: {result.stderr}")
    
    # Set sparse checkout
    subprocess.run(
        ["git", "-C", repo_dir, "sparse-checkout", "set"] + paths,
        capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "-C", repo_dir, "checkout", ref],
        capture_output=True, text=True,
    )
    
    return repo_dir


def _validate_skill(path: str) -> None:
    """Validate that path contains a valid skill."""
    if not os.path.isdir(path):
        raise InstallError(f"Skill path not found: {path}")
    skill_md = os.path.join(path, "SKILL.md")
    if not os.path.isfile(skill_md):
        raise InstallError("SKILL.md not found in skill directory.")


def cmd_install(url: str, dest: Optional[str] = None, name: Optional[str] = None,
                scope: str = "global", project_dir: Optional[Path] = None,
                auto_sync: bool = True, json_output: bool = False) -> dict:
    """Install a skill from GitHub URL."""
    config = load_config()
    result = {"success": False, "skill": None, "path": None, "error": None}
    
    # Determine destination
    if scope == "local" and project_dir:
        dest_root = expand_path(config.get("project_source_dir", ".ai-skills"), project_dir)
    else:
        dest_root = expand_path(config.get("source_dir", "~/.ai-skills"))
    
    if dest:
        dest_root = Path(dest)
    
    try:
        owner, repo, ref, subpath = _parse_github_url(url)
        
        if not subpath:
            raise InstallError("URL must include path to skill (e.g., /tree/main/skills/docx)")
        
        skill_name = name or os.path.basename(subpath.rstrip("/"))
        dest_dir = dest_root / skill_name
        
        if dest_dir.exists():
            raise InstallError(f"Skill already exists: {dest_dir}")
        
        print(f"Installing skill from: {url}")
        print(f"Destination: {dest_dir}\n")
        
        # Create temp directory
        tmp_dir = tempfile.mkdtemp(prefix="skill-install-")
        
        try:
            # Try git first, fall back to download
            print("[1/4] Cloning repository...")
            try:
                repo_url = f"https://github.com/{owner}/{repo}.git"
                repo_root = _git_sparse_checkout(repo_url, ref, [subpath], tmp_dir)
            except InstallError:
                print("  Git failed, trying download...")
                repo_root = _download_repo_zip(owner, repo, ref, tmp_dir)
            print("  done")
            
            # Extract skill
            print(f"[2/4] Extracting skill to {dest_dir}...")
            skill_src = os.path.join(repo_root, subpath)
            _validate_skill(skill_src)
            
            dest_root.mkdir(parents=True, exist_ok=True)
            shutil.copytree(skill_src, dest_dir)
            print("  done")
            
            # Validate
            print("[3/4] Validating skill...")
            _validate_skill(str(dest_dir))
            print("  done")
            
            # Sync
            if auto_sync and config.get("sync", {}).get("auto_after_install", True):
                print("[4/4] Syncing to all IDEs...")
                cmd_sync(skill_name=skill_name, scope=scope, project_dir=project_dir)
            else:
                print("[4/4] Skipping auto-sync")
            
            # Git commit
            auto_commit(dest_root, "install", skill_name)
            
            result["success"] = True
            result["skill"] = skill_name
            result["path"] = str(dest_dir)
            
            print(f"\nInstallation complete: {skill_name}")
            
        finally:
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
    
    except InstallError as e:
        result["error"] = str(e)
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  1. Check the URL is correct")
        print("  2. For private repos: export GITHUB_TOKEN=<your-token>")
        print("  3. Ensure the skill has SKILL.md")
    
    if json_output:
        from .utils import print_json
        print_json(result)
    
    return result
