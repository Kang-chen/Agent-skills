# Agent Skills

167 个 AI IDE skill 的统一管理仓库，作为所有 IDE（Claude Code、Cursor、Codex、Gemini、Antigravity）的 Single Source of Truth (SSOT)。

## 架构

```
~/.ai-skills (本仓库, SSOT)
├── <skill-name>/SKILL.md        # 各 skill 目录
├── skill-manager/               # 管理工具 (Python)
├── .upstream-manifest.json      # 上游追踪信息
└── .sources/                    # 上游克隆 (gitignored)

        ↓ skills sync

~/.claude/skills/                # Claude Code
~/.cursor/skills/                # Cursor
~/.codex/skills/                 # Codex
~/.gemini/skills/                # Gemini
~/.gemini/antigravity/skills/    # Antigravity
```

## 快速开始

### 新机器部署

```bash
# 1. 克隆仓库
git clone git@github.com:Kang-chen/Agent-skills.git ~/.ai-skills

# 2. 同步到所有 IDE
cd ~/.ai-skills
python3 -m skill-manager.scripts.cli sync

# 3. 注册上游源（用于后续从 anthropics/skills 更新）
python3 -m skill-manager.scripts.cli upstream add https://github.com/anthropics/skills --name anthropic-skills

# 4. 将已有 skill 标记为上游追踪
python3 -m skill-manager.scripts.cli upstream import docx pdf pptx doc-coauthoring --from anthropic-skills --adopt
```

### 日常管理

```bash
cd ~/.ai-skills

# 查看已安装 skill
python3 -m skill-manager.scripts.cli list

# 同步到所有 IDE
python3 -m skill-manager.scripts.cli sync

# 验证同步一致性
python3 -m skill-manager.scripts.cli verify

# 查看状态
python3 -m skill-manager.scripts.cli status
```

## 上游管理

部分 skill 来自 [anthropics/skills](https://github.com/anthropics/skills)，通过 upstream 系统追踪和更新。

### 当前追踪的 skill

| Skill | 上游源 | 说明 |
|-------|--------|------|
| doc-coauthoring | anthropic-skills | 文档协作指导 |
| docx | anthropic-skills | Word 文档操作 |
| pdf | anthropic-skills | PDF 操作 |
| pptx | anthropic-skills | PowerPoint 操作 |

### 从上游更新

```bash
# 拉取上游最新代码
python3 -m skill-manager.scripts.cli upstream update anthropic-skills

# 查看哪些 skill 有变更
python3 -m skill-manager.scripts.cli upstream status

# 查看具体差异
python3 -m skill-manager.scripts.cli upstream diff docx

# 更新（覆盖本地）
python3 -m skill-manager.scripts.cli upstream import docx --from anthropic-skills --force

# 同步并提交
python3 -m skill-manager.scripts.cli sync
git add -A && git commit -m "chore: sync upstream skills"
git push
```

### 注意事项

- `.sources/` 目录被 gitignore，每台机器需单独 `upstream add` 注册上游源
- `.upstream-manifest.json` 会被提交，skill 追踪关系在 `git pull` 后可见
- `upstream update` 不会自动覆盖本地文件，需手动 `import --force`

## Skill Manager 命令参考

```
skills list                          # 列出已安装 skill
skills search <keyword>              # 搜索社区 skill
skills install <github-url>          # 从 GitHub 安装
skills create <name>                 # 创建新 skill
skills sync                          # 同步到所有 IDE
skills remove <name>                 # 移除 skill
skills validate                      # 验证 skill 格式
skills verify                        # 验证同步一致性
skills upstream add <url>            # 添加上游源
skills upstream update [name]        # 更新上游源
skills upstream list [name]          # 列出上游 skill
skills upstream status               # 查看上游状态
skills upstream diff <skill>         # 比较本地与上游差异
skills upstream import <s> --from x  # 从上游导入
```

> 注：在本仓库中运行时使用 `python3 -m skill-manager.scripts.cli` 代替 `skills`。
