---
name: skill-installer
description: Install skills from curated lists or GitHub repos into ~/.ai-skills/. Use when user asks to list available skills, install from curated repos (anthropics/skills, openai/skills), or install from any GitHub URL. Supports private repos via GITHUB_TOKEN.
---

# Skill Installer

Install skills from curated lists or GitHub repositories.

## When to Use

- **Use skill-installer**: To list/install from curated skill repos or install by repo/path
- **Use skill-manager**: To search through 31k+ community skills database
- **Use skills-sync**: After installation, to sync to all IDEs

## Scripts

### List Curated Skills

```bash
# List from anthropics/skills (default)
python ~/.ai-skills/skill-installer/scripts/list-curated-skills.py

# List from openai/skills
python ~/.ai-skills/skill-installer/scripts/list-curated-skills.py --repo openai/skills --path skills/.curated

# JSON output
python ~/.ai-skills/skill-installer/scripts/list-curated-skills.py --format json
```

### Install from GitHub URL

```bash
# Install from URL
python ~/.ai-skills/skill-installer/scripts/install-skill-from-github.py \
  --url https://github.com/anthropics/skills/tree/main/skills/docx

# Install from repo + path
python ~/.ai-skills/skill-installer/scripts/install-skill-from-github.py \
  --repo anthropics/skills --path skills/docx skills/pdf
```

## Options

- `--repo <owner/repo>`: GitHub repository
- `--path <path>`: Path(s) to skill(s) inside repo
- `--url <url>`: Full GitHub URL (tree URL with path)
- `--ref <ref>`: Git ref/branch (default: main)
- `--dest <path>`: Destination directory (default: ~/.ai-skills)
- `--name <name>`: Override skill name
- `--method auto|download|git`: Download method

## Workflow

1. List curated skills to see what's available
2. Install selected skill(s)
3. Run `skills-sync` to push to all IDEs
4. Restart IDE to load new skill

## Notes

- Installs to `~/.ai-skills/<skill-name>/`
- Aborts if destination already exists (delete first to reinstall)
- Private repos: set `GITHUB_TOKEN` or `GH_TOKEN` environment variable
- Uses Git sparse checkout (fast, downloads only needed files)
