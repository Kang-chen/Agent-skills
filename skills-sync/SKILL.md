---
name: skills-sync
description: Unified skills synchronization across multiple AI IDEs. Syncs skills from a single source to Claude Code, Cursor, Codex, Gemini CLI, and Antigravity. Supports both global (user-level) and project-level syncing with separate source directories. Automatically removes orphaned skills from targets. Use when managing, installing, or syncing skills across IDEs.
---

# Skills Sync

Synchronize skills across AI coding assistants from a single source of truth.

## Architecture

```
GLOBAL:  ~/.ai-skills/          →  ~/.xxx/skills/
PROJECT: <project>/.ai-skills/  →  <project>/.xxx/skills/
```

## Supported IDEs

| IDE | Global Target | Project Target |
|-----|---------------|----------------|
| Claude Code | `~/.claude/skills/` | `.claude/skills/` |
| Cursor | `~/.cursor/skills/` | `.cursor/skills/` |
| Codex | `~/.codex/skills/` | `.codex/skills/` |
| Gemini CLI | `~/.gemini/skills/` | `.gemini/skills/` |
| Antigravity | `~/.gemini/antigravity/skills/` | `.agent/skills/` |

## Commands

### Scope Flags

- `-g, --global` : Global scope (`~/.ai-skills/`)
- `-p, --project` : Project scope (`<project>/.ai-skills/`)
- Default: global only

### Sync Skills

```bash
skills sync -g                    # Sync global skills to all IDEs
skills sync -p                    # Sync project skills
skills sync -g -p                 # Sync both scopes
skills sync my-skill -g           # Sync single skill (no cleanup)
```

Sync performs:
1. Copy/symlink skills from source to all enabled IDE targets
2. **Auto-cleanup**: Remove orphaned skills (exist in target but not in source)

### Initialize Project

```bash
skills init -p                    # Create <project>/.ai-skills/
```

### Install Skill

```bash
skills install my-skill -g        # Install to ~/.ai-skills/
skills install my-skill -p        # Install to <project>/.ai-skills/
skills install my-skill -s /path  # Install from source
```

### Remove Skill

```bash
skills remove my-skill -g         # Remove from global (source + targets)
skills remove my-skill -p         # Remove from project
skills remove my-skill --keep-source  # Keep source, remove from targets only
```

### List & Status

```bash
skills list -g                    # List global skills
skills list -p                    # List project skills
skills status -g -p               # Check sync status for all
```

## Workflow

### Global Skills (shared across all projects)

1. Add/remove skills in `~/.ai-skills/<skill-name>/SKILL.md`
2. Run `skills sync -g`
   - New skills: copied to all IDE targets
   - Removed skills: automatically deleted from targets

### Project Skills (version-controlled, project-specific)

1. Run `skills init -p` to create `.ai-skills/`
2. Add skills to `<project>/.ai-skills/<skill-name>/SKILL.md`
3. Run `skills sync -p`
4. Commit `.ai-skills/` to version control

## Sync Output

```
[GLOBAL] Source: /home/user/.ai-skills
[GLOBAL] Syncing 5 skill(s)...

  [CLAUDE] /home/user/.claude/skills
    [OK] git (copy)
    [OK] jupyter-notebooks (copy)
    [DEL] jupyter-notebook-ops (orphaned)    # Auto-removed
    [DEL] jupytext-notebook-edits (orphaned) # Auto-removed
```

## Configuration

Edit `~/.ai-skills/skills-sync/config.json`:

```json
{
  "global_source_dir": "~/.ai-skills",
  "project_source_dir": ".ai-skills",
  "global_targets": { ... },
  "project_targets": { ... },
  "enabled": ["claude", "cursor", "codex", "gemini", "antigravity"],
  "exclude_skills": ["skills-sync"],
  "use_symlinks": false
}
```

- `exclude_skills`: Skills excluded from sync and cleanup
- `use_symlinks`: Use symlinks instead of copies (faster, but may not work on all systems)
