---
name: skills-sync
description: "DEPRECATED - Use skill-manager instead. This skill is kept for backward compatibility. For syncing skills: use 'skills sync' from skill-manager."
---

# Skills Sync (DEPRECATED)

> **Note:** This skill has been merged into **skill-manager**. Please use skill-manager for all skill operations.

## Migration Guide

Use these skill-manager commands instead:

| Old (skills-sync) | New (skill-manager) |
|-------------------|---------------------|
| `sync.py sync -g` | `skills sync` |
| `sync.py sync -p` | `skills sync --local` |
| `sync.py list -g` | `skills list` |
| `sync.py status -g` | `skills status` |

## New Commands

```bash
# Sync all skills to all IDEs
python ~/.ai-skills/skill-manager/scripts/skills sync

# Sync single skill
python ~/.ai-skills/skill-manager/scripts/skills sync my-skill

# Sync with dry-run preview
python ~/.ai-skills/skill-manager/scripts/skills sync --dry-run

# Check sync status
python ~/.ai-skills/skill-manager/scripts/skills status

# Verify sync consistency
python ~/.ai-skills/skill-manager/scripts/skills verify
```

---

## Legacy Scripts (Still Available)

The original scripts remain for backward compatibility:

### Sync Commands

```bash
python ~/.ai-skills/skills-sync/scripts/sync.py sync -g       # Sync global skills
python ~/.ai-skills/skills-sync/scripts/sync.py sync -p       # Sync project skills
python ~/.ai-skills/skills-sync/scripts/sync.py list -g       # List global skills
python ~/.ai-skills/skills-sync/scripts/sync.py status -g     # Check sync status
```

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

## Configuration (Legacy)

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

## Notes

- Does NOT modify `~/.codex/skills/.system/` (Codex pre-installed skills)
- Auto-removes orphaned skills from targets during full sync
