#!/usr/bin/env python3
"""List curated skills from a GitHub repo path."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error

from github_utils import github_api_contents_url, github_request

# Default: anthropics/skills curated list (can also use openai/skills)
DEFAULT_REPO = "anthropics/skills"
DEFAULT_PATH = "skills"
DEFAULT_REF = "main"


class ListError(Exception):
    pass


class Args(argparse.Namespace):
    repo: str
    path: str
    ref: str
    format: str


def _request(url: str) -> bytes:
    return github_request(url, "ai-skill-list")


def _skills_home() -> str:
    """Get the skills home directory."""
    return os.environ.get("AI_SKILLS_DIR", os.path.expanduser("~/.ai-skills"))


def _installed_skills() -> set[str]:
    """Get set of installed skill names."""
    root = _skills_home()
    if not os.path.isdir(root):
        return set()
    entries = set()
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and not name.startswith('.'):
            # Check if it has SKILL.md
            if os.path.isfile(os.path.join(path, "SKILL.md")):
                entries.add(name)
    return entries


def _list_curated(repo: str, path: str, ref: str) -> list[str]:
    api_url = github_api_contents_url(repo, path, ref)
    try:
        payload = _request(api_url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ListError(
                "Curated skills path not found: "
                f"https://github.com/{repo}/tree/{ref}/{path}"
            ) from exc
        raise ListError(f"Failed to fetch curated skills: HTTP {exc.code}") from exc
    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, list):
        raise ListError("Unexpected curated listing response.")
    skills = [item["name"] for item in data if item.get("type") == "dir"]
    return sorted(skills)


def _parse_args(argv: list[str]) -> Args:
    parser = argparse.ArgumentParser(description="List curated skills.")
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help=f"GitHub repo (default: {DEFAULT_REPO})")
    parser.add_argument("--path", default=DEFAULT_PATH,
                        help=f"Path in repo (default: {DEFAULT_PATH})")
    parser.add_argument("--ref", default=DEFAULT_REF,
                        help=f"Git ref (default: {DEFAULT_REF})")
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    return parser.parse_args(argv, namespace=Args())


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    try:
        skills = _list_curated(args.repo, args.path, args.ref)
        installed = _installed_skills()
        
        print(f"Skills from {args.repo}/{args.path}:\n")
        
        if args.format == "json":
            payload = [
                {"name": name, "installed": name in installed} for name in skills
            ]
            print(json.dumps(payload, indent=2))
        else:
            for idx, name in enumerate(skills, start=1):
                suffix = " (already installed)" if name in installed else ""
                print(f"{idx}. {name}{suffix}")
            
            print(f"\nTotal: {len(skills)} skills")
            print(f"Installed: {len([s for s in skills if s in installed])}")
        return 0
    except ListError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
