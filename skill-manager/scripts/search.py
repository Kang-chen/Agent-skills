#!/usr/bin/env python3
"""
Search for skills in the community database.
"""

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from .utils import expand_path, load_config


def _get_cache_path() -> Path:
    """Get the search index cache path."""
    config = load_config()
    return expand_path(config.get("search", {}).get("index_path", "~/.ai-skills/skill-manager/data/index.json"))


def _get_local_db_path() -> Path:
    """Get the local skills database path."""
    return Path(__file__).parent.parent / "data" / "all_skills_with_cn.json"


def _load_local_db() -> list[dict]:
    """Load the local skills database."""
    db_path = _get_local_db_path()
    if db_path.exists():
        try:
            with open(db_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _load_cache() -> tuple[list[dict], float]:
    """Load cached search index."""
    cache_path = _get_cache_path()
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                data = json.load(f)
                return data.get("skills", []), data.get("timestamp", 0)
        except (json.JSONDecodeError, OSError):
            pass
    return [], 0


def _save_cache(skills: list[dict]) -> None:
    """Save search index to cache."""
    cache_path = _get_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"skills": skills, "timestamp": time.time()}, f)


def _is_cache_valid() -> bool:
    """Check if cache is still valid."""
    config = load_config()
    ttl_hours = config.get("search", {}).get("cache_ttl_hours", 24)
    _, timestamp = _load_cache()
    return time.time() - timestamp < ttl_hours * 3600


def _search_skills(query: str, skills: list[dict], limit: int = 15) -> list[dict]:
    """Search skills by query."""
    query_lower = query.lower()
    results = []
    
    for skill in skills:
        name = skill.get("name", "").lower()
        desc = skill.get("description", "").lower()
        desc_cn = skill.get("description_cn", "").lower()
        
        score = 0
        if query_lower in name:
            score += 10
        if query_lower == name:
            score += 20
        if query_lower in desc:
            score += 5
        if query_lower in desc_cn:
            score += 5
        
        if score > 0:
            results.append({**skill, "_score": score})
    
    results.sort(key=lambda x: (-x.get("_score", 0), x.get("name", "")))
    return results[:limit]


def cmd_search(query: str, online: bool = False, limit: int = 15, 
               json_output: bool = False) -> dict:
    """Search for skills."""
    result = {"query": query, "total": 0, "results": [], "source": "local"}
    
    # Load skills database
    if online or not _is_cache_valid():
        # Try to load fresh data (placeholder for future online search)
        skills = _load_local_db()
        if skills:
            _save_cache(skills)
            result["source"] = "database"
    else:
        skills, _ = _load_cache()
        if not skills:
            skills = _load_local_db()
        result["source"] = "cache"
    
    if not skills:
        print("No skills database found.")
        print("Place all_skills_with_cn.json in skill-manager/data/")
        return result
    
    # Search
    print(f'Searching for "{query}" in {len(skills):,} skills...\n')
    
    matches = _search_skills(query, skills, limit)
    result["total"] = len(matches)
    
    if not matches:
        print("No matching skills found.")
        print("\nTips:")
        print("  - Try different keywords")
        print("  - Use broader search terms")
        return result
    
    print(f"Found {len(matches)} results:\n")
    
    for i, skill in enumerate(matches, 1):
        name = skill.get("name", "unknown")
        repo = skill.get("repo", "")
        desc = skill.get("description", "")[:80]
        if len(skill.get("description", "")) > 80:
            desc += "..."
        
        print(f"{i}. {repo}/{name}" if repo else f"{i}. {name}")
        if desc:
            print(f"   {desc}")
        
        # Show Chinese description if available
        desc_cn = skill.get("description_cn", "")
        if desc_cn and desc_cn != desc:
            desc_cn = desc_cn[:60]
            if len(skill.get("description_cn", "")) > 60:
                desc_cn += "..."
            print(f"   {desc_cn}")
        
        url = skill.get("url", "")
        if url:
            print(f"   URL: {url}")
        print()
        
        result["results"].append({
            "name": name,
            "repo": repo,
            "description": skill.get("description", ""),
            "url": url,
        })
    
    print("Install: skills install <url>")
    print("         skills install --query <query> --index <number>")
    
    if json_output:
        from .utils import print_json
        print_json(result)
    
    return result


def cmd_install_from_search(query: str, index: int, **kwargs) -> dict:
    """Install skill from search results."""
    result = cmd_search(query, json_output=False)
    
    if not result["results"]:
        return {"success": False, "error": "No search results"}
    
    if index < 1 or index > len(result["results"]):
        return {"success": False, "error": f"Invalid index. Choose 1-{len(result['results'])}"}
    
    skill = result["results"][index - 1]
    url = skill.get("url")
    
    if not url:
        return {"success": False, "error": "No URL for this skill"}
    
    from .install import cmd_install
    return cmd_install(url, **kwargs)
