---
name: vibe-kanban-manager
description: >-
  Manages Vibe Kanban tasks via MCP. Supports project/task listing, 
  creating tasks, parallel task orchestration, subtask decomposition,
  and dependency tracking. Triggers: "kanban", "vibe kanban", "list tasks",
  "create task", "task status", "parallel execution", "subtask".
---

# Vibe Kanban Task Manager

> 通过 MCP 管理 Vibe Kanban 任务，支持并行任务编排、子任务分解、依赖追踪

## MCP Configuration

```json
{
  "mcpServers": {
    "vibe_kanban": {
      "command": "npx",
      "args": ["-y", "vibe-kanban@latest", "--mcp"]
    }
  }
}
```

---

## 一、MCP 工具速查

### 项目与仓库

| Tool            | 用途         | 参数         |
| --------------- | ------------ | ------------ |
| `list_projects` | 列出所有项目 | 无           |
| `list_repos`    | 列出项目仓库 | `project_id` |
| `get_repo`      | 获取仓库详情 | `repo_id`    |

### 任务管理

| Tool          | 用途         | 必需参数              | 可选参数                         |
| ------------- | ------------ | --------------------- | -------------------------------- |
| `list_tasks`  | 列出任务     | `project_id`          | `status`, `limit`                |
| `create_task` | 创建任务     | `project_id`, `title` | `description`                    |
| `get_task`    | 获取任务详情 | `task_id`             | -                                |
| `update_task` | 更新任务     | `task_id`             | `title`, `description`, `status` |
| `delete_task` | 删除任务     | `task_id`             | -                                |

### 任务执行

| Tool                      | 用途                 | 参数                             |
| ------------------------- | -------------------- | -------------------------------- |
| `start_workspace_session` | 启动任务             | `task_id`, `executor`, `repos[]` |
| `get_context`             | 获取当前工作区上下文 | 无 (仅在活动会话中)              |

---

## 二、常用操作

### 查看项目任务

```python
# 1. 获取 project_id
projects = list_projects()  # 选择目标项目

# 2. 列出任务
tasks = list_tasks(project_id=PROJECT_ID)
```

### 创建并启动任务

```python
# 1. 获取 repo_id
repos = list_repos(project_id=PROJECT_ID)

# 2. 创建任务
task_id = create_task(
    project_id=PROJECT_ID,
    title="任务标题",
    description="任务描述 (Markdown)"
)

# 3. 启动 workspace
start_workspace_session(
    task_id=task_id,
    executor="CLAUDE_CODE",
    repos=[{"repo_id": REPO_ID, "base_branch": "master"}]  # 检查实际分支名!
)
```

### 更新任务状态

> [!TIP]
> **状态约定**: Agent 完成任务后设为 `inreview`，`done` 仅由人工审核后设定。

```python
update_task(task_id=TASK_ID, status="inreview")  # 状态: todo/inprogress/inreview/done/cancelled
```

---

## 三、并行任务编排

```
             ┌── T1 (分支1) ──┐
P0 → P1 ────┼── T3 (分支2) ──┼── T5 → P3
             └── T4 (分支3) ──┘
```

创建并行任务组后，使用独立分支启动各子任务。详见 [references/worktree.md](references/worktree.md)。

---

## 四、子任务管理

> [!NOTE]
> MCP Server 不提供显式子任务 API。使用 `create_task` + description 元数据模拟。

### 4.1 Parent 编排模式 (推荐)

> [!IMPORTANT]
> **只启动 Parent 任务**，由 Parent Agent 负责创建、执行、调度子任务。

**编排流程**:
```
外部 Agent                    Parent Agent (Workspace Session)
    │                                │
    ├── create_task(Parent) ─────────┤
    ├── start_workspace_session ─────┤
    │        (只启动 Parent)         ├── create_task(子任务)
    │                                ├── 执行子任务内容
    │                                ├── update_task(子任务, inreview)
    │                                ├── update_task(Parent, inreview)
    ├── list_tasks(查询状态) ◄────────┘
```

**外部 Agent 操作限制**:

| 操作                      | Parent 任务 | 子任务             |
| ------------------------- | ----------- | ------------------ |
| `create_task`             | ✅           | ✅                  |
| `get_task` / `list_tasks` | ✅           | ✅                  |
| `update_task`             | ✅           | ❌ (由 Parent 管理) |
| `start_workspace_session` | ✅           | ❌                  |

### 4.2 实例: 三级子任务调度

**任务结构**:
```
Parent: 子任务调度测试
├── A: 二级子任务 A (由 Parent 创建和执行)
└── B: 二级子任务 B (由 Parent 创建, B 内部调度 C)
    └── C: 三级子任务 C (由 B 创建和执行)
```

**外部 Agent 操作**:
```python
# 1. 只创建 Parent 任务
parent_id = create_task(
    project_id=PROJECT_ID,
    title="Parent: 子任务调度测试",
    description=PARENT_DESCRIPTION  # 包含完整执行步骤
)

# 2. 只启动 Parent workspace
start_workspace_session(
    task_id=parent_id,
    executor="CLAUDE_CODE",
    repos=[{"repo_id": REPO_ID, "base_branch": "master"}]
)

# 3. 等待执行后查询状态
list_tasks(project_id=PROJECT_ID)
```

**预期结果**:

| 任务   | 状态     | 说明           |
| ------ | -------- | -------------- |
| Parent | inreview | Workspace 完成 |
| A      | inreview | 由 Parent 创建 |
| B      | inreview | 由 Parent 创建 |
| C      | inreview | 由 B 创建      |

日志文件:
```
/tmp/kanban_test/
├── parent_schedule.log   # Parent started → Parent completed
├── subtask_a.log         # A started → A completed
├── subtask_b.log         # B started → creating C → B completed (after C)
└── subtask_c.log         # C started by B → C completed
```

### 4.3 子任务描述模板

```markdown
## 任务描述
[任务整体目标]

## Parent Task
- **Parent ID**: `PARENT_TASK_UUID`
- **Parent Title**: Parent 任务标题
- **Subtask Index**: 1.2

## 依赖
- **前置子任务**: T1.1 (必须完成)
- **数据依赖**: `results/t1.1/output.csv`
```

---

## 五、依赖追踪

在任务 description 中定义:

```markdown
## 依赖
- **前置任务**: T1 (必须完成)
- **数据依赖**: `results/t1/output.csv`
- **环境依赖**: conda activate cartaPA
```

**依赖图示例**:
```
P0 ─── P1 ────┬── T1 ── T2 ──┬── T5 ── P3
              ├── T3 ────────┤
              └── T4 ────────┘
```

---

## 六、故障处理

### Branch not found 错误

**错误**: `Branch not found: main`

**原因**: `base_branch` 与实际仓库分支不匹配

**解决**:
```bash
git branch -a  # 检查实际分支名 (通常是 master)
```

### 任务卡住

```python
# 重置超时任务
update_task(task_id=STUCK_TASK_ID, status="todo")
```

### 任务失败重试

1. `get_task` 检查失败原因
2. 创建新 attempt
3. `start_workspace_session` 重新启动

---

## 七、参考资料

按需加载以下详细文档:

| 主题         | 文件                                               |
| ------------ | -------------------------------------------------- |
| Executor配置 | [references/executor.md](references/executor.md)   |
| Worktree策略 | [references/worktree.md](references/worktree.md)   |
| 工作流模板   | [references/templates.md](references/templates.md) |
| 高级功能     | [references/advanced.md](references/advanced.md)   |

---

## 八、项目配置

> [!WARNING]
> ID 可能过期。启动前使用 `list_projects` 和 `list_repos` 验证。

```yaml
# 使用 MCP 工具动态获取:
# list_projects() → project_id
# list_repos(project_id) → repo_id

project_id: 92f9433f-ce91-4370-aeeb-651359227edd  # 2026-02-01 验证

repos:
  CartaPA_analysis:
    id: b07e0759-5bd2-45d3-9644-3bc4ea47f176
    default_branch: master  # 重要: 不是 main!

vibe_kanban_url: http://127.0.0.1:35823
```
