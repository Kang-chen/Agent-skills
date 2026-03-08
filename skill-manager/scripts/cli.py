#!/usr/bin/env python3
"""
CLI entry point for skill manager.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .utils import load_config, get_skills_from_dir, expand_path, find_project_root, get_skill_description


def show_interactive_menu():
    """Show interactive menu when called without arguments."""
    config = load_config()
    source_dir = expand_path(config.get("source_dir", "~/.ai-skills"))
    exclude = config.get("exclude_skills", [])
    skills = get_skills_from_dir(source_dir, exclude)
    enabled_ides = config.get("enabled_ides", [])
    
    print()
    print("╔════════════════════════════════════════════════════╗")
    print("║         Skill Manager - Interactive Mode           ║")
    print("╠════════════════════════════════════════════════════╣")
    print(f"║ Installed Skills: {len(skills):<33}║")
    print(f"║ Enabled IDEs: {', '.join(enabled_ides[:4]):<37}║")
    print("╚════════════════════════════════════════════════════╝")
    print()
    print("1. List installed skills")
    print("2. Search community skills")
    print("3. Install a skill")
    print("4. Create new skill")
    print("5. Sync to all IDEs")
    print("6. Validate skills")
    print("7. Export profile")
    print("8. Import profile")
    print("9. Check status")
    print("10. Upstream management")
    print("0. Exit")
    print()

    try:
        choice = input("Enter choice [0-10]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        return

    if choice == "0" or not choice:
        return
    elif choice == "1":
        cmd_list()
    elif choice == "2":
        query = input("Search query: ").strip()
        if query:
            from .search import cmd_search
            cmd_search(query)
    elif choice == "3":
        url = input("GitHub URL: ").strip()
        if url:
            from .install import cmd_install
            cmd_install(url)
    elif choice == "4":
        name = input("Skill name: ").strip()
        if name:
            from .create import cmd_create
            cmd_create(name)
    elif choice == "5":
        from .sync import cmd_sync
        cmd_sync()
    elif choice == "6":
        from .validate import cmd_validate
        cmd_validate()
    elif choice == "7":
        from .profile import cmd_export
        cmd_export()
    elif choice == "8":
        path = input("Profile path or Gist ID: ").strip()
        if path:
            from .profile import cmd_import
            if path.isalnum() and len(path) > 20:
                cmd_import(gist_id=path)
            else:
                cmd_import(source=path)
    elif choice == "9":
        cmd_status()
    elif choice == "10":
        _upstream_interactive_menu()
    else:
        print("Invalid choice.")


def _upstream_interactive_menu():
    """上游管理交互式子菜单。"""
    from .upstream import (cmd_upstream_add, cmd_upstream_update,
                           cmd_upstream_import, cmd_upstream_list,
                           cmd_upstream_status, cmd_upstream_diff)

    print("\n=== Upstream Management ===")
    print("1. Add upstream source")
    print("2. Update upstream sources")
    print("3. Import skills from upstream")
    print("4. List available skills")
    print("5. Check status")
    print("6. Compare differences")
    print("0. Back")
    print()

    try:
        choice = input("Enter choice [0-6]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nBack to main menu.")
        return

    if choice == "0" or not choice:
        return
    elif choice == "1":
        url = input("Repository URL: ").strip()
        if url:
            name = input("Source name (Enter for auto): ").strip() or None
            branch = input("Branch (Enter for main): ").strip() or "main"
            cmd_upstream_add(url, name=name, branch=branch)
    elif choice == "2":
        name = input("Source name (Enter for all): ").strip() or None
        cmd_upstream_update(name)
    elif choice == "3":
        source = input("Source name: ").strip()
        skills_input = input("Skill names (space-separated): ").strip()
        if source and skills_input:
            skills = skills_input.split()
            force = input("Force overwrite? [y/N]: ").strip().lower() == "y"
            adopt = input("Adopt existing? [y/N]: ").strip().lower() == "y"
            cmd_upstream_import(skills, source, force=force, adopt=adopt)
    elif choice == "4":
        source = input("Source name (Enter for all): ").strip() or None
        cmd_upstream_list(source)
    elif choice == "5":
        cmd_upstream_status()
    elif choice == "6":
        skill = input("Skill name: ").strip()
        if skill:
            cmd_upstream_diff(skill)


def cmd_list(scope: str = "global", project_dir: Optional[Path] = None, 
             json_output: bool = False):
    """List installed skills."""
    config = load_config()
    exclude = config.get("exclude_skills", [])
    
    result = {"global": [], "local": []}
    
    if scope in ("global", "all"):
        source_dir = expand_path(config.get("source_dir", "~/.ai-skills"))
        skills = get_skills_from_dir(source_dir, exclude)
        
        if not json_output:
            print(f"[GLOBAL] {source_dir}\n")
            if skills:
                for skill in skills:
                    skill_path = source_dir / skill
                    desc = get_skill_description(skill_path)
                    print(f"  - {skill}")
                    if desc:
                        print(f"    {desc}")
                print(f"\n  Total: {len(skills)} skill(s)")
            else:
                print("  (no skills)")
        
        for skill in skills:
            skill_path = source_dir / skill
            result["global"].append({
                "name": skill,
                "path": str(skill_path),
                "description": get_skill_description(skill_path),
            })
    
    if scope in ("project", "local", "all") and project_dir:
        source_dir = expand_path(config.get("project_source_dir", ".ai-skills"), project_dir)
        
        if not json_output:
            if scope == "all":
                print()
            print(f"[PROJECT] {source_dir}\n")
        
        if source_dir.exists():
            skills = get_skills_from_dir(source_dir, exclude)
            if not json_output:
                if skills:
                    for skill in skills:
                        skill_path = source_dir / skill
                        desc = get_skill_description(skill_path)
                        print(f"  - {skill}")
                        if desc:
                            print(f"    {desc}")
                    print(f"\n  Total: {len(skills)} skill(s)")
                else:
                    print("  (no skills)")
            
            for skill in skills:
                skill_path = source_dir / skill
                result["local"].append({
                    "name": skill,
                    "path": str(skill_path),
                    "description": get_skill_description(skill_path),
                })
        elif not json_output:
            print("  (directory not found)")
    
    if json_output:
        from .utils import print_json
        print_json(result)
    
    return result


def cmd_status(scope: str = "global", project_dir: Optional[Path] = None):
    """Check sync status."""
    config = load_config()
    exclude = config.get("exclude_skills", [])
    
    print("Skill Manager Status\n")
    print("=" * 50)
    
    if scope in ("global", "all"):
        source_dir = expand_path(config.get("source_dir", "~/.ai-skills"))
        skills = get_skills_from_dir(source_dir, exclude)
        
        print(f"\n[GLOBAL]")
        print(f"  Source: {source_dir}")
        print(f"  Skills: {len(skills)}")
        
        print(f"\n  IDE Targets:")
        for ide_name in config.get("enabled_ides", []):
            target_path = config.get("targets", {}).get(ide_name)
            if target_path:
                target_dir = expand_path(target_path)
                exists = target_dir.exists()
                count = len(list(target_dir.iterdir())) if exists else 0
                status = f"{count} skills" if exists else "not created"
                print(f"    [{ide_name}] {target_dir} ({status})")
    
    if scope in ("project", "local", "all") and project_dir:
        source_dir = expand_path(config.get("project_source_dir", ".ai-skills"), project_dir)
        
        print(f"\n[PROJECT]")
        print(f"  Project: {project_dir}")
        print(f"  Source: {source_dir}")
        
        if source_dir.exists():
            skills = get_skills_from_dir(source_dir, exclude)
            print(f"  Skills: {len(skills)}")
        else:
            print("  Status: not initialized")
            print(f"  Create with: skills init --project")
    
    # Git status
    from .git_ops import is_git_repo
    source_dir = expand_path(config.get("source_dir", "~/.ai-skills"))
    if is_git_repo(source_dir):
        print(f"\n[GIT]")
        print(f"  Repository: {source_dir}")
        print(f"  Auto-commit: {config.get('git', {}).get('auto_commit', True)}")


def add_scope_args(parser):
    """Add scope arguments to a parser."""
    parser.add_argument("-g", "--global", dest="global_scope", action="store_true",
                        help="Target global skills (~/.ai-skills/)")
    parser.add_argument("-l", "--local", "-p", "--project", dest="local_scope", action="store_true",
                        help="Target project skills (.ai-skills/)")
    parser.add_argument("--project-dir", type=Path, default=None,
                        help="Project directory (default: auto-detect)")


def get_scope_and_project(args) -> tuple[str, Optional[Path]]:
    """Get scope and project directory from args."""
    global_scope = getattr(args, "global_scope", False)
    local_scope = getattr(args, "local_scope", False)
    
    if global_scope and local_scope:
        scope = "all"
    elif local_scope:
        scope = "local"
    else:
        scope = "global"
    
    project_dir = getattr(args, "project_dir", None)
    if scope in ("local", "all") and not project_dir:
        project_dir = find_project_root(Path.cwd())
        if not project_dir:
            project_dir = Path.cwd()
    
    return scope, project_dir


def main():
    parser = argparse.ArgumentParser(
        description="Unified skill management for AI IDEs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  list      List installed skills
  search    Search community skills
  install   Install a skill from GitHub
  create    Create a new skill
  sync      Sync skills to all IDEs
  remove    Remove a skill
  validate  Validate skills
  export    Export skills profile
  import    Import skills profile
  status    Check status
  commit    Git commit changes
  upstream  Manage upstream sources

Examples:
  skills                              # Interactive menu
  skills list                         # List global skills
  skills search "pdf"                 # Search for PDF skills
  skills install <github-url>         # Install from GitHub
  skills create my-skill              # Create new skill
  skills sync                         # Sync to all IDEs
  skills validate                     # Validate all skills
  skills upstream add <url>           # Add upstream source
  skills upstream list                # List upstream skills
  skills upstream import <s> --from x # Import from upstream
"""
    )
    
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--version", action="version", version="skill-manager 1.0.0")
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # list
    list_parser = subparsers.add_parser("list", help="List installed skills")
    add_scope_args(list_parser)
    
    # search
    search_parser = subparsers.add_parser("search", help="Search community skills")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--online", action="store_true", help="Force online search")
    search_parser.add_argument("--limit", type=int, default=15, help="Max results")
    
    # install
    install_parser = subparsers.add_parser("install", help="Install a skill")
    install_parser.add_argument("url", nargs="?", help="GitHub URL")
    install_parser.add_argument("--query", help="Install from search query")
    install_parser.add_argument("--index", type=int, help="Index from search results")
    install_parser.add_argument("--name", help="Override skill name")
    install_parser.add_argument("--dest", help="Destination directory")
    install_parser.add_argument("--no-sync", action="store_true", help="Skip auto-sync")
    add_scope_args(install_parser)
    
    # create
    create_parser = subparsers.add_parser("create", help="Create a new skill")
    create_parser.add_argument("name", help="Skill name")
    create_parser.add_argument("--path", help="Output path")
    create_parser.add_argument("--resources", help="Resource dirs (scripts,references,assets)")
    create_parser.add_argument("--examples", action="store_true", help="Create example files")
    create_parser.add_argument("--sync", action="store_true", help="Auto-sync after create")
    add_scope_args(create_parser)
    
    # sync
    sync_parser = subparsers.add_parser("sync", help="Sync skills to IDEs")
    sync_parser.add_argument("skill", nargs="?", help="Specific skill")
    sync_parser.add_argument("--dry-run", action="store_true", help="Preview only")
    add_scope_args(sync_parser)
    
    # remove
    remove_parser = subparsers.add_parser("remove", help="Remove a skill")
    remove_parser.add_argument("skill", help="Skill name")
    remove_parser.add_argument("--keep-source", action="store_true", help="Keep source")
    remove_parser.add_argument("--force", "-f", action="store_true", help="Force remove")
    add_scope_args(remove_parser)
    
    # validate
    validate_parser = subparsers.add_parser("validate", help="Validate skills")
    validate_parser.add_argument("skill", nargs="?", help="Specific skill")
    validate_parser.add_argument("--no-verify", action="store_true", help="Skip sync verification")
    add_scope_args(validate_parser)
    
    # export
    export_parser = subparsers.add_parser("export", help="Export profile")
    export_parser.add_argument("--output", "-o", help="Output file")
    export_parser.add_argument("--gist", action="store_true", help="Export to Gist")
    
    # import
    import_parser = subparsers.add_parser("import", help="Import profile")
    import_parser.add_argument("source", nargs="?", help="Profile file path")
    import_parser.add_argument("--gist", dest="gist_id", help="Import from Gist ID")
    import_parser.add_argument("--dry-run", action="store_true", help="Preview only")
    
    # status
    status_parser = subparsers.add_parser("status", help="Check status")
    add_scope_args(status_parser)
    
    # verify (alias for validate with sync check)
    verify_parser = subparsers.add_parser("verify", help="Verify sync consistency")
    verify_parser.add_argument("skill", nargs="?", help="Specific skill")
    add_scope_args(verify_parser)
    
    # commit
    commit_parser = subparsers.add_parser("commit", help="Git commit")
    commit_parser.add_argument("message", nargs="?", help="Commit message")
    commit_parser.add_argument("--auto", action="store_true", help="Auto-generate message")
    
    # init
    init_parser = subparsers.add_parser("init", help="Initialize project skills")
    add_scope_args(init_parser)

    # upstream
    upstream_parser = subparsers.add_parser("upstream", help="Manage upstream sources")
    upstream_sub = upstream_parser.add_subparsers(dest="upstream_command", help="Upstream command")

    # upstream add
    up_add = upstream_sub.add_parser("add", help="Add upstream source")
    up_add.add_argument("url", help="Git repository URL")
    up_add.add_argument("--name", help="Source name (default: derived from URL)")
    up_add.add_argument("--branch", default="main", help="Branch (default: main)")

    # upstream update
    up_update = upstream_sub.add_parser("update", help="Update upstream sources")
    up_update.add_argument("name", nargs="?", help="Source name (default: all)")

    # upstream import
    up_import = upstream_sub.add_parser("import", help="Import skills from upstream")
    up_import.add_argument("skills", nargs="+", help="Skill names to import")
    up_import.add_argument("--from", dest="source", required=True, help="Source name")
    up_import.add_argument("--force", action="store_true", help="Force overwrite")
    up_import.add_argument("--adopt", action="store_true", help="Adopt existing skills")

    # upstream list
    up_list = upstream_sub.add_parser("list", help="List available upstream skills")
    up_list.add_argument("source", nargs="?", help="Source name")

    # upstream status
    upstream_sub.add_parser("status", help="Show upstream status")

    # upstream diff
    up_diff = upstream_sub.add_parser("diff", help="Compare local vs upstream")
    up_diff.add_argument("skill", help="Skill name")

    args = parser.parse_args()
    json_output = getattr(args, "json", False)
    
    # No command = interactive menu
    if not args.command:
        show_interactive_menu()
        return
    
    # Execute command
    if args.command == "list":
        scope, project_dir = get_scope_and_project(args)
        cmd_list(scope, project_dir, json_output)
    
    elif args.command == "search":
        from .search import cmd_search
        cmd_search(args.query, getattr(args, "online", False), 
                  getattr(args, "limit", 15), json_output)
    
    elif args.command == "install":
        scope, project_dir = get_scope_and_project(args)
        if args.query and args.index:
            from .search import cmd_install_from_search
            cmd_install_from_search(args.query, args.index, scope=scope, 
                                   project_dir=project_dir, json_output=json_output)
        elif args.url:
            from .install import cmd_install
            cmd_install(args.url, dest=args.dest, name=args.name, scope=scope,
                       project_dir=project_dir, auto_sync=not args.no_sync,
                       json_output=json_output)
        else:
            print("Error: Provide URL or --query with --index")
            sys.exit(1)
    
    elif args.command == "create":
        scope, project_dir = get_scope_and_project(args)
        from .create import cmd_create
        cmd_create(args.name, path=args.path, resources=args.resources,
                  examples=args.examples, scope=scope, project_dir=project_dir,
                  auto_sync=args.sync, json_output=json_output)
    
    elif args.command == "sync":
        scope, project_dir = get_scope_and_project(args)
        from .sync import cmd_sync
        cmd_sync(skill_name=args.skill, scope=scope, project_dir=project_dir,
                dry_run=args.dry_run, json_output=json_output)
    
    elif args.command == "remove":
        scope, project_dir = get_scope_and_project(args)
        from .remove import cmd_remove
        cmd_remove(args.skill, scope=scope, project_dir=project_dir,
                  keep_source=args.keep_source, force=args.force,
                  json_output=json_output)
    
    elif args.command in ("validate", "verify"):
        scope, project_dir = get_scope_and_project(args)
        from .validate import cmd_validate
        verify = args.command == "verify" or not getattr(args, "no_verify", False)
        cmd_validate(skill_name=getattr(args, "skill", None), scope=scope,
                    project_dir=project_dir, verify=verify, json_output=json_output)
    
    elif args.command == "export":
        from .profile import cmd_export
        cmd_export(output=args.output, gist=args.gist, json_output=json_output)
    
    elif args.command == "import":
        from .profile import cmd_import
        cmd_import(source=args.source, gist_id=args.gist_id,
                  dry_run=args.dry_run, json_output=json_output)
    
    elif args.command == "status":
        scope, project_dir = get_scope_and_project(args)
        cmd_status(scope, project_dir)
    
    elif args.command == "commit":
        from .git_ops import manual_commit
        from .utils import expand_path, load_config
        config = load_config()
        source_dir = expand_path(config.get("source_dir", "~/.ai-skills"))
        manual_commit(source_dir, args.message or "", auto_message=args.auto)
    
    elif args.command == "init":
        scope, project_dir = get_scope_and_project(args)
        if scope not in ("local", "all") or not project_dir:
            print("Error: Use --local or --project to specify scope")
            sys.exit(1)
        config = load_config()
        source_dir = expand_path(config.get("project_source_dir", ".ai-skills"), project_dir)
        if source_dir.exists():
            print(f"Project skills directory already exists: {source_dir}")
        else:
            source_dir.mkdir(parents=True)
            print(f"Created project skills directory: {source_dir}")
            print(f"\nNext steps:")
            print(f"  1. Add skills to {source_dir}/<skill-name>/SKILL.md")
            print(f"  2. Run: skills sync --local")

    elif args.command == "upstream":
        from .upstream import (
            cmd_upstream_add, cmd_upstream_update,
            cmd_upstream_import, cmd_upstream_list,
            cmd_upstream_status, cmd_upstream_diff,
        )
        uc = getattr(args, "upstream_command", None)
        if uc == "add":
            cmd_upstream_add(args.url, name=args.name, branch=args.branch)
        elif uc == "update":
            cmd_upstream_update(args.name)
        elif uc == "import":
            cmd_upstream_import(args.skills, args.source,
                               force=args.force, adopt=args.adopt)
        elif uc == "list":
            cmd_upstream_list(args.source)
        elif uc == "status":
            cmd_upstream_status()
        elif uc == "diff":
            cmd_upstream_diff(args.skill)
        else:
            upstream_parser.print_help()


if __name__ == "__main__":
    main()
