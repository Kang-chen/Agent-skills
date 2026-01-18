---
name: skill-installer
description: "DEPRECATED - Use skill-manager instead. This skill is kept for backward compatibility. For installing skills: use 'skills install <url>' from skill-manager."
---

# Skill Installer (DEPRECATED)

> **Note:** This skill has been merged into **skill-manager**. Please use skill-manager for all skill operations.

## Migration Guide

Use these skill-manager commands instead:

| Old (skill-installer) | New (skill-manager) |
|----------------------|---------------------|
| `list-curated-skills.py` | `skills search <query>` |
| `install-skill-from-github.py --url <url>` | `skills install <url>` |

## New Commands

```bash
# Search for skills
python ~/.ai-skills/skill-manager/scripts/skills search "pdf"

# Install from GitHub URL
python ~/.ai-skills/skill-manager/scripts/skills install https://github.com/anthropics/skills/tree/main/skills/docx

# Install and auto-sync
python ~/.ai-skills/skill-manager/scripts/skills install <url>
```

---

## Legacy Scripts (Still Available)

The original scripts remain for backward compatibility:

### List Curated Skills

```bash
python ~/.ai-skills/skill-installer/scripts/list-curated-skills.py
python ~/.ai-skills/skill-installer/scripts/list-curated-skills.py --repo openai/skills --path skills/.curated
```

### Install from GitHub URL

```bash
python ~/.ai-skills/skill-installer/scripts/install-skill-from-github.py \
  --url https://github.com/anthropics/skills/tree/main/skills/docx
```

## Options (Legacy)

- `--repo <owner/repo>`: GitHub repository
- `--path <path>`: Path(s) to skill(s) inside repo
- `--url <url>`: Full GitHub URL (tree URL with path)
- `--ref <ref>`: Git ref/branch (default: main)
- `--dest <path>`: Destination directory (default: ~/.ai-skills)
- `--name <name>`: Override skill name
- `--method auto|download|git`: Download method

## Notes

- Private repos: set `GITHUB_TOKEN` or `GH_TOKEN` environment variable
- After installing, run `skills sync` to push to all IDEs
