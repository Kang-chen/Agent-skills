---
name: skill-manager
description: >-
  Unified skill management for AI IDEs. Triggers when user wants to manage, 
  sync, install, remove, or organize skills across IDEs. Common triggers include:
  (1) list/show skills (列出技能, 查看技能, "what skills do I have"),
  (2) sync skills (同步技能, 将skill同步, "sync to all IDEs", "push skills"),
  (3) install skills (安装技能, "install from GitHub"),
  (4) remove/delete skills (删除技能, 移除技能),
  (5) create new skills (创建技能, 新建skill),
  (6) validate skills (验证技能, 检查skill格式),
  (7) export/import profiles (导出技能, 导入配置).
  Also triggers for: skill management, 技能管理, skill操作.
  Supports 5 IDEs: Claude Code, Cursor, Codex, Gemini CLI, Antigravity.
  IMPORTANT: When creating or modifying skills, always follow the guidelines
  in references/skill-creator/SKILL.md.
---

# Skill Manager

Unified CLI for managing AI skills across all IDEs.

## IMPORTANT: Skill Guidelines Reference

**When creating, installing, or modifying skills, ALWAYS read and follow:**

[references/skill-creator/SKILL.md](references/skill-creator/SKILL.md)

This is the authoritative guide for skill structure, including:
- Concise is Key principle
- Progressive Disclosure pattern
- YAML frontmatter requirements
- File organization rules
- Anti-patterns to avoid

## Important Paths

- **Source (SSOT)**: `~/.ai-skills/` - Single Source of Truth
- **Guidelines**: `~/.ai-skills/skill-manager/references/skill-creator/SKILL.md`
- **Claude Code**: `~/.claude/skills/`
- **Cursor**: `~/.cursor/skills/`
- **Codex**: `~/.codex/skills/`
- **Gemini CLI**: `~/.gemini/skills/`
- **Antigravity**: `~/.gemini/antigravity/skills/`
- **Config**: `~/.ai-skills/skill-manager/config.json`

## Quick Reference

| Task             | Command                       |
| ---------------- | ----------------------------- |
| Interactive menu | `skills`                      |
| Search skills    | `skills search "query"`       |
| Install from URL | `skills install <github-url>` |
| Create new skill | `skills create <name>`        |
| Sync to IDEs     | `skills sync`                 |
| List installed   | `skills list`                 |
| Remove skill     | `skills remove <name>`        |
| Validate skill   | `skills validate <name>`      |
| Export profile   | `skills export`               |
| Import profile   | `skills import <file>`        |
| Check status     | `skills status`               |
| Verify sync      | `skills verify`               |

## Usage Examples

**User:** "What skills do I have installed?"

```bash
python ~/.ai-skills/skill-manager/scripts/skills list --json
```

**User:** "Find me a skill for working with Docker"

```bash
python ~/.ai-skills/skill-manager/scripts/skills search "docker"
```

**User:** "Install the docx skill from anthropics"

```bash
python ~/.ai-skills/skill-manager/scripts/skills install https://github.com/anthropics/skills/tree/main/skills/docx
```

**User:** "Create a skill for formatting SQL"

```bash
python ~/.ai-skills/skill-manager/scripts/skills create sql-formatter
```

**User:** "Sync my skills to all IDEs"

```bash
python ~/.ai-skills/skill-manager/scripts/skills sync
```

**User:** "Export my skills to share with another machine"

```bash
python ~/.ai-skills/skill-manager/scripts/skills export --gist
```

## Commands

### skills (no args) - Interactive Menu

When called without arguments, displays interactive menu:

```
╔════════════════════════════════════════════════════╗
║         Skill Manager - Interactive Mode           ║
╠════════════════════════════════════════════════════╣
║ Installed Skills: 15                               ║
║ Enabled IDEs: claude, cursor, codex, gemini        ║
╚════════════════════════════════════════════════════╝

1. List installed skills
2. Search community skills
3. Install a skill
...
```

### skills search

Search community skills database.

```bash
python ~/.ai-skills/skill-manager/scripts/skills search "python testing"
```

### skills install

Install from GitHub URL or search results.

```bash
# From URL
python ~/.ai-skills/skill-manager/scripts/skills install https://github.com/anthropics/skills/tree/main/skills/docx

# From search
python ~/.ai-skills/skill-manager/scripts/skills install --query "pdf" --index 1
```

### skills create

Create new skill from template (follows skill-creator guidelines).

```bash
python ~/.ai-skills/skill-manager/scripts/skills create my-skill --resources scripts,references
```

### skills sync

Sync skills to all enabled IDEs.

```bash
# Sync all
python ~/.ai-skills/skill-manager/scripts/skills sync

# Sync single skill
python ~/.ai-skills/skill-manager/scripts/skills sync my-skill

# Dry run
python ~/.ai-skills/skill-manager/scripts/skills sync --dry-run
```

### skills validate

Validate skills against guidelines and verify sync.

```bash
python ~/.ai-skills/skill-manager/scripts/skills validate my-skill
```

### skills export / import

Export and import skill profiles for cross-machine sync.

```bash
# Export to Gist
python ~/.ai-skills/skill-manager/scripts/skills export --gist

# Import from Gist
python ~/.ai-skills/skill-manager/scripts/skills import --gist <gist-id>
```

## Scope Flags

- `-g, --global`: Global scope (`~/.ai-skills/`)
- `-l, --local`: Project scope (`.ai-skills/`)

## Error Handling

| Error                | Solution                                  |
| -------------------- | ----------------------------------------- |
| git clone fails      | Check URL, network, or if repo is private |
| No SKILL.md found    | Skill repo may be structured differently  |
| Invalid YAML         | Show syntax error line and suggest fix    |
| Permission denied    | Check directory permissions               |
| Skill already exists | Ask if user wants to update or reinstall  |
| Sync hash mismatch   | Run `skills sync --force` to overwrite    |

## Configuration

Edit `~/.ai-skills/skill-manager/config.json` to customize:

- `git.auto_commit`: Enable/disable auto commit (default: true)
- `git.auto_push`: Enable/disable auto push (default: false)
- `sync.auto_after_install`: Auto sync after install (default: true)
- `enabled_ides`: List of IDEs to sync to
- `exclude_skills`: Skills to exclude from sync
- `preserve_target_skills`: Directories to preserve in targets (e.g., Codex `.system/`)

## For Creating Skills

**Always refer to:** [references/skill-creator/SKILL.md](references/skill-creator/SKILL.md)
