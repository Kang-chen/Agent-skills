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

## CRITICAL: Global vs Project Skills

**ALWAYS confirm scope before any operation. Default assumptions cause errors.**

| Scope | Location | When to Use |
|-------|----------|-------------|
| **Global** | `~/.ai-skills/` → syncs to `~/.claude/skills/` | Personal skills, shared across all projects |
| **Project** | `./.ai-skills/` → syncs to `./.cursor/skills/` | Project-specific skills, version controlled |

### Mandatory Confirmation

Before install, create, or sync, **ALWAYS ask user:**

> "Should this skill be installed globally (`~/.ai-skills/`) or for this project only (`./.ai-skills/`)?"

**Default behavior:**
- If user says "project" or "local" → use `./.ai-skills/` in current working directory
- If user says "global" → use `~/.ai-skills/`
- If unclear → **ASK, do not assume**

### Common Mistake to Avoid

**WRONG:** User says "save to project" but agent syncs to `~/.claude/skills/`
**RIGHT:** User says "save to project" → save to `./.cursor/skills/` or `./.ai-skills/` in CWD

## Skill Guidelines Reference

When creating or modifying skills: [references/skill-creator/SKILL.md](references/skill-creator/SKILL.md)

## Important Paths

| Type | Path |
|------|------|
| **Global SSOT** | `~/.ai-skills/` |
| **Project SSOT** | `./.ai-skills/` (in project root) |
| **Claude Code (global)** | `~/.claude/skills/` |
| **Claude Code (project)** | `./.claude/skills/` |
| **Cursor (global)** | `~/.cursor/skills/` |
| **Cursor (project)** | `./.cursor/skills/` |

## Quick Reference

| Task | Command |
|------|---------|
| **Search skills** | Browse https://skills.sh or `npx skills --help` |
| **Install from skills.sh** | `npx skills add <owner>/<skill-name>` |
| **Install specific skill** | `npx skills add anthropics/skills/docx` |
| **List installed** | `ls ~/.ai-skills/` or `ls ./.ai-skills/` |
| **Create new skill** | Follow skill-creator guidelines |
| **Sync global→IDEs** | Copy from `~/.ai-skills/` to IDE paths |
| **Validate skill** | Check SKILL.md has name + description frontmatter |

## Installing Skills (via skills.sh)

### Search Skills

Browse the leaderboard at https://skills.sh to find community skills.

### Install from skills.sh

```bash
# Install a skill (goes to current directory by default)
npx skills add <owner>/<skill-name>

# Examples:
npx skills add anthropics/skills/docx
npx skills add vercel-labs/skills/find-skills
```

### Post-Install: Choose Scope

After `npx skills add`, the skill is downloaded. Then:

1. **For Global**: Move to `~/.ai-skills/<skill-name>/`
2. **For Project**: Move to `./.ai-skills/<skill-name>/`

Then sync to appropriate IDE paths.

## Manual Install (from GitHub)

```bash
# Clone skill repo
git clone <github-url> /tmp/skill-temp

# Copy to appropriate scope
# Global:
cp -r /tmp/skill-temp/<skill-name> ~/.ai-skills/
# Project:
cp -r /tmp/skill-temp/<skill-name> ./.ai-skills/
```

## Sync to IDEs

After installing to SSOT, sync to IDE-specific paths:

```bash
# Global sync (from ~/.ai-skills/ to ~/.claude/skills/, ~/.cursor/skills/, etc.)
cp -r ~/.ai-skills/<skill-name> ~/.claude/skills/
cp -r ~/.ai-skills/<skill-name> ~/.cursor/skills/

# Project sync (from ./.ai-skills/ to ./.cursor/skills/)
cp -r ./.ai-skills/<skill-name> ./.cursor/skills/
```

## Creating Skills

Follow [references/skill-creator/SKILL.md](references/skill-creator/SKILL.md) for:

- SKILL.md structure (frontmatter + body)
- Progressive disclosure pattern
- Resource organization (scripts/, references/, assets/)

### Quick Create Template

```markdown
---
name: my-skill
description: >-
  What the skill does. When to use it (triggers, contexts, examples).
---

# My Skill

Instructions for using the skill...
```

## Scope Flags (Legacy Scripts)

If using legacy Python scripts:

- `-g, --global`: Global scope (`~/.ai-skills/`)
- `-l, --local`: Project scope (`./.ai-skills/`)

## Validation Checklist

- [ ] SKILL.md exists with valid YAML frontmatter
- [ ] `name` and `description` fields present
- [ ] Description includes trigger phrases
- [ ] No extraneous files (README.md, CHANGELOG.md, etc.)
- [ ] Resources in correct folders (scripts/, references/, assets/)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Skill in wrong scope | Delete from wrong location, reinstall to correct path |
| Sync not working | Verify SSOT path matches intended scope |
| npx skills fails | Check Node.js installed, try `npm exec skills add` |
| Skill not triggering | Check description includes usage triggers |
