---
name: skill-manager
description: >-
  Unified skill management for AI IDEs. Use when users want to:
  list skills ("what skills do I have", "show installed skills"),
  search skills ("find skills for X", "search for PDF skill"),
  install skills ("install a skill", "add skill from GitHub"),
  create skills ("create a new skill", "make a skill"),
  sync skills ("sync to all IDEs", "push skills"),
  remove skills ("remove skill X", "uninstall skill"),
  validate skills ("validate my skills", "check skill format"),
  export/import profiles ("export my skills", "import profile", "sync across machines").
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

### Mandatory Rules

1. **Git tracking**: When creating or editing skills, ensure all changes are tracked by Git within the project repository.
2. **Project-first**: All skill files MUST be created within the current repository only. Only sync to global (`~/.ai-skills/` → `~/.claude/skills/`) **after testing AND explicit user request**. Never auto-sync to global.

### Common Mistake to Avoid

**WRONG:** User says "save to project" but agent syncs to `~/.claude/skills/`
**RIGHT:** User says "save to project" → save to `./.cursor/skills/` or `./.ai-skills/` in CWD

**WRONG:** Agent creates skill and immediately syncs to `~/.claude/skills/`
**RIGHT:** Agent creates skill in project repo → tests it → user explicitly says "sync to global" → then sync

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

## Search Workflow (MUST FOLLOW)

When user asks to search/find skills, follow this fallback chain **in order**:

### Step 1: Check Local Installed Skills

```bash
# Check global
ls ~/.ai-skills/
# Check project
ls ./.ai-skills/
```

If a matching skill is already installed, inform user and stop.

### Step 2: Search Custom Sources

Check the **Custom Skill Sources** table below. Match user's query against the listed categories and skill names.

- If user asks about biology, bioinformatics, genomics, drug discovery → search **K-Dense-AI/claude-scientific-skills**
- Browse the repo to find matching skills, read their SKILL.md for details

### Step 3: Search skills.sh

```bash
npx skills search "<query>"
```

Or browse https://skills.sh for the leaderboard.

### Step 4: Search GitHub

If nothing found above, search GitHub for skill repositories:
- Search query: `<topic> claude skill SKILL.md`
- Look for repos with proper skill structure (SKILL.md with frontmatter)

### Search Result Format

Always present results to user in this format:

```
Found: <skill-name>
Source: <skills.sh | Custom: K-Dense-AI | GitHub: owner/repo>
Description: <from SKILL.md description>
Install: <command>
```

## Quick Reference

| Task | Command |
|------|---------|
| **Search skills** | Follow Search Workflow above |
| **Install from skills.sh** | `npx skills add <owner>/<skill-name>` |
| **Install from custom source** | Clone repo + copy to SSOT |
| **List installed** | `ls ~/.ai-skills/` or `ls ./.ai-skills/` |
| **Create new skill** | Follow skill-creator guidelines |
| **Sync global→IDEs** | Copy from `~/.ai-skills/` to IDE paths |
| **Validate skill** | Check SKILL.md has name + description frontmatter |

## Custom Skill Sources

Beyond skills.sh, these curated repositories contain high-quality skills:

| Repository | Focus | Skills Count | Install Command |
|------------|-------|--------------|-----------------|
| **K-Dense-AI/claude-scientific-skills** | Biology, Bioinformatics, Scientific Research | 142+ | See below |

### Scientific Skills (Biology Focus)

Repository: https://github.com/K-Dense-AI/claude-scientific-skills

**Genomics & Sequencing:** alphafold-database, biopython, scanpy, pysam, ensembl-database, cellxgene-census, scvi-tools, pydeseq2

**Drug Discovery:** rdkit, chembl-database, deepchem, pubchem-database, drugbank-database, diffdock

**Clinical/Health:** clinical-decision-support, pyhealth, clinicaltrials-database, pydicom, pathml

**Lab Automation:** opentrons-integration, benchling-integration, lamindb, pylabrobot

**Install from this source:**

```bash
# Clone the scientific skills repo
git clone https://github.com/K-Dense-AI/claude-scientific-skills.git /tmp/scientific-skills

# Install a specific skill (e.g., biopython)
# Global:
cp -r /tmp/scientific-skills/scientific-skills/biopython ~/.ai-skills/
# Project:
cp -r /tmp/scientific-skills/scientific-skills/biopython ./.ai-skills/

# Then sync to IDE
cp -r ~/.ai-skills/biopython ~/.claude/skills/
```

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
