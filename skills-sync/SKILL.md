---
name: skills-sync
description: Unified skills synchronization across multiple AI IDEs. Syncs skills from a single source to Claude Code, Cursor, Codex, Gemini CLI, and Antigravity. Supports both global (user-level) and project-level syncing with separate source directories. Use when managing, installing, or syncing skills across IDEs.
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

### Initialize Project

```bash
skills init -p                    # Create <project>/.ai-skills/
```

### Sync Skills

```bash
skills sync -g                    # Sync global skills
skills sync -p                    # Sync project skills
skills sync -g -p                 # Sync both
```

### Install Skill

```bash
skills install my-skill -g        # Install to ~/.ai-skills/
skills install my-skill -p        # Install to <project>/.ai-skills/
skills install my-skill -s /path  # Install from source
```

### Remove Skill

```bash
skills remove my-skill -g         # Remove from global
skills remove my-skill -p         # Remove from project
skills remove my-skill --keep-source  # Keep source, remove links
```

### List & Status

```bash
skills list -g                    # List global skills
skills list -p                    # List project skills
skills status -g -p               # Check all sync status
```

## Workflow

### Global Skills (shared across all projects)

1. Add skills to `~/.ai-skills/<skill-name>/SKILL.md`
2. Run `skills sync -g`

### Project Skills (version-controlled, project-specific)

1. Run `skills init -p` to create `.ai-skills/`
2. Add skills to `<project>/.ai-skills/<skill-name>/SKILL.md`
3. Run `skills sync -p`
4. Commit `.ai-skills/` to version control

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
  "use_symlinks": true
}
```
