---
name: skills-sync
description: Unified skills synchronization across multiple AI IDEs. Syncs skills from ~/.ai-skills/ to Claude Code, Cursor, Codex, Gemini CLI, and Antigravity. Use when you need to sync local skills to IDE targets, NOT for installing skills from GitHub (use skill-manager or skill-installer instead).
---

# Skills Sync

Synchronize skills across AI coding assistants from a single source of truth.

## When to Use

- **Use skills-sync**: After adding/removing skills in `~/.ai-skills/`, to push changes to all IDEs
- **Do NOT use skills-sync**: To search or install skills from GitHub (use skill-manager/skill-installer)

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

```bash
python ~/.ai-skills/skills-sync/scripts/sync.py sync -g       # Sync global skills
python ~/.ai-skills/skills-sync/scripts/sync.py sync -p       # Sync project skills
python ~/.ai-skills/skills-sync/scripts/sync.py list -g       # List global skills
python ~/.ai-skills/skills-sync/scripts/sync.py status -g     # Check sync status
```

## Workflow

1. Install skills to `~/.ai-skills/` using skill-manager or skill-installer
2. Run `sync -g` to push to all IDEs
3. Restart IDEs to load new skills

## Configuration

Edit `~/.ai-skills/skills-sync/config.json`:

```json
{
  "global_source_dir": "~/.ai-skills",
  "global_targets": { "claude": "~/.claude/skills", ... },
  "enabled": ["claude", "cursor", "codex", "gemini", "antigravity"],
  "exclude_skills": [],
  "preserve_target_skills": {
    "codex": [".system"]
  }
}
```

- `exclude_skills`: Skills excluded from sync (empty = sync all)
- `preserve_target_skills`: Directories preserved in target (e.g., Codex `.system/`)

## Notes

- Does NOT modify `~/.codex/skills/.system/` (Codex pre-installed skills)
- Auto-removes orphaned skills from targets during full sync
