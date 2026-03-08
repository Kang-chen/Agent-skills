# 高级功能

## Checkpoint 机制

> 任务执行中的质量关卡

### Checkpoint 类型

| 类型    | 标识      | 描述       | 示例               |
| ------- | --------- | ---------- | ------------------ |
| 🤖 自动  | `[AUTO]`  | 脚本可验证 | 文件存在、行数 >0  |
| 🧠 Agent | `[AGENT]` | AI 判断    | 结果是否合理       |
| 👤 人工  | `[HUMAN]` | 需人工确认 | 参数选择、最终审核 |

### Checkpoint 实现

```python
CHECKPOINTS = {
    "CP1": {"type": "AUTO", "check": "data_loaded", "blocking": True},
    "CP2": {"type": "AGENT", "check": "results_reasonable", "blocking": False},
    "CP3": {"type": "HUMAN", "check": "params_confirmed", "blocking": True},
}

def run_checkpoint(cp_id: str) -> bool:
    cp = CHECKPOINTS[cp_id]
    if cp["type"] == "AUTO":
        return automated_check(cp["check"])
    elif cp["type"] == "AGENT":
        return agent_evaluation(cp["check"])
    else:
        return await_human_confirmation(cp["check"])
```

---

## 清理规范

### 任务完成后清理

```bash
# 删除临时文件
find . -name "*.tmp" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# 清理空目录
find results/ -type d -empty -delete

# 压缩日志
gzip results/exp_*/logs/*.log
```

---

## 命令速查

```bash
# 查看任务状态 (curl)
curl -s "http://127.0.0.1:35823/api/tasks?project_id=PROJECT_ID" | python3 -m json.tool

# 更新任务状态
curl -X PUT "http://127.0.0.1:35823/api/tasks/TASK_ID" \
  -H "Content-Type: application/json" \
  -d '{"status": "todo"}'

# MCP 配置
cat .mcp.json
{
  "mcpServers": {
    "vibe_kanban": {
      "command": "npx",
      "args": ["-y", "vibe-kanban@latest", "--mcp"]
    }
  }
}
```
