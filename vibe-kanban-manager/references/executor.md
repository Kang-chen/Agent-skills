# Executor 配置

## 支持的 Executor

| Executor       | 用途     | Variant             |
| -------------- | -------- | ------------------- |
| `CLAUDE_CODE`  | 通用编码 | DEFAULT, PLAN, YOLO |
| `GEMINI`       | 快速任务 | DEFAULT, FLASH      |
| `CODEX`        | 代码生成 | DEFAULT             |
| `CURSOR_AGENT` | IDE 集成 | DEFAULT             |

## 选择 Executor

```yaml
任务类型匹配:
  环境检查: CLAUDE_CODE (DEFAULT)
  复杂分析: CLAUDE_CODE (PLAN)
  快速修复: CLAUDE_CODE (YOLO)
  代码生成: CODEX (DEFAULT)
  轻量任务: GEMINI (FLASH)
```

## 启动时指定 Variant

```
使用 PLAN 模式启动 T1 任务
```

**执行**:
```json
{
  "task_id": "T1_ID",
  "executor": "CLAUDE_CODE",
  "variant": "PLAN",
  "repos": [...]
}
```
