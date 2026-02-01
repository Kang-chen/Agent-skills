# Git Worktree 策略

> 用于并行任务的分支隔离

## 何时使用 Worktree

| 场景                  | 使用 Worktree | 原因               |
| --------------------- | ------------- | ------------------ |
| 并行任务 (T1, T3, T4) | ✅             | 避免代码冲突       |
| 顺序任务 (P0 → P1)    | ❌             | 直接在 master 分支 |
| 长时间实验            | ✅             | 隔离实验环境       |

## Worktree 命名规范

```bash
# 创建
git worktree add ../CartaPA_analysis_t1 task/t1-annotation
git worktree add ../CartaPA_analysis_t3 task/t3-node-prob

# 目录结构
~/project/
├── CartaPA_analysis/      # master 分支
├── CartaPA_analysis_t1/   # task/t1-annotation
├── CartaPA_analysis_t3/   # task/t3-node-prob
└── CartaPA_analysis_t4/   # task/t4-safe-coords
```

## 共享数据访问

```python
# 所有 worktree 共享同一数据源
DATA_ROOT = Path("../CartaPA/data")  # 相对于 worktree
MODEL_ROOT = Path("../CartaPA_model/read_only_repo")  # 模型只读访问
```

## Worktree 清理

```bash
# 任务完成后
git worktree remove ../CartaPA_analysis_t1 --force

# 清理孤立 worktree
git worktree prune
```
