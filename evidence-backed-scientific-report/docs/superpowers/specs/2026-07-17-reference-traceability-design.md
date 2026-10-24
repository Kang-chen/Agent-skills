# 科学报告 Reference 追溯设计

## 目标

让 `report.md` 使用普通数字 reference，例如 `[1](#ref-openai-official)`，不再向读者暴露 `[E:DEC-OFFICIAL-FIRST]` 等内部证据记录 ID，同时保留 `evidence.tsv` 对关键主张、来源和报告位置的机器可读审查能力。

## 非目标

- 不把 `evidence.tsv` 合并进 `report.md`。
- 不删除 `evidence.tsv` 或 `report_manifest.yaml`。
- 不实现 DOCX、PDF、HTML 或 LaTeX 引用转换。
- 不自动重写已有 v1 报告包。
- 不引入参考文献管理器或新的 Python 依赖。

## 报告包职责

`report.md` 是完整、可独立阅读的读者文档。它包含结论、数字、解释、限制和可读 References。普通读者不需要打开 manifest、证据账本、代码或日志。

`manifests/evidence.tsv` 是持久化的机器审查账本，不是可丢弃的临时文件。它保留 `record_id`、`supported_by_ids`、`source_ids`、`support_status`、`report_locator` 等结构化关系，但内部 `record_id` 不进入读者正文。

`manifests/report_manifest.yaml` 管理来源身份、可用性、反向证据关系和每个来源在报告中的稳定 reference 位置。

## Reader-facing reference 语法

正文使用数字链接：

```markdown
优先使用官方插件是当前风险最低的方案。[1](#ref-openai-official)
```

报告末尾使用稳定锚点和数字列表：

```markdown
<a id="references"></a>
## References

<a id="ref-openai-official"></a>
1. OpenAI. *Codex documentation*. 官方产品文档。
```

数字是可变的展示顺序。`ref-openai-official` 是稳定机器锚点，重新排序 References 时不应改变。一个来源可以在正文多次引用，但在 References 中只能定义一次。

## Manifest v2

新报告包使用 `schema_version: 2`。`traceability` 增加 reader-facing reference contract：

```yaml
traceability:
  report_locator_format: report.md#stable-anchor
  explicit_anchor_syntax: '<a id="stable-anchor"></a>'
  require_explicit_anchors: true
  reference_link_format: '[N](#ref-stable-anchor)'
  reference_anchor_syntax: '<a id="ref-stable-anchor"></a>'
  reference_section_locator: report.md#references
  require_source_references: true
```

每个来源增加 `report_reference`：

```yaml
sources:
  - source_id: SRC-OPENAI-OFFICIAL
    kind: document
    locator: sources/openai_official.md
    identity: sha256:...
    availability: available
    report_reference: report.md#ref-openai-official
    required_for_record_ids: [DEC-OFFICIAL-FIRST]
```

`inventory_sources.py` 在已有 `report_reference` 时原样保留；缺失时根据 `source_id` 生成小写、连字符形式的稳定锚点。生成结果必须检测空锚点和锚点冲突。

## Evidence 到正文的数据流

对于一条 `report_critical=true` 的证据记录：

1. `evidence.tsv.report_locator` 指向包含该主张的报告小节。
2. `evidence.tsv.source_ids` 指向 manifest 来源。
3. 每个 manifest 来源通过 `report_reference` 指向 References 中的稳定锚点。
4. 对应报告小节至少包含一次指向该稳定锚点的数字引用。
5. References 定义该锚点及读者可理解的来源说明。

该闭环让读者看到普通 `[1]`，让校验器看到“主张 → 小节 → reference → source → evidence”的完整映射。

## 校验规则

`validate_report.py` 对 v2 报告执行以下检查：

- manifest 必须声明 `schema_version: 2` 和完整 reference contract。
- `report.md` 不得包含匹配 `\[E:[^]]+\]` 的内部证据标识；发现时报告 `report.internal_evidence_marker_present`。
- `reference_section_locator` 必须解析到显式 `<a id="references"></a>`。
- 每个 report-critical 来源必须具有合法且唯一的 `report_reference`。
- `report_reference` 必须使用 `report.md#ref-stable-anchor` 格式。
- 每个 reference 锚点在 References 中恰好定义一次，并紧邻非空的编号条目。
- reference 编号必须为正整数、唯一，并按 References 出现顺序从 1 连续递增。
- 正文数字链接的编号必须与目标 reference 条目的编号一致。
- 正文不得链接未知 reference 锚点。
- 每条 report-critical 证据记录所指向的小节必须引用其全部 `source_ids`。
- 现有 source/evidence 正反向关系、证据图、Figure/Table 和状态校验继续执行。

小节范围从 `report_locator` 的显式锚点开始，到下一个不以 `ref-` 开头的显式报告锚点之前结束。这样 References 内部的来源锚点不会截断正文小节。

## 失败和状态

reference 缺失、编号错配、未知锚点、重复锚点或内部 `[E:...]` 均属于确定性结构错误，使候选 terminal 状态退回 `review`。来源文件缺失仍按现有规则决定是否进一步进入 `blocked`。

错误信息必须同时指出稳定 issue code、相关 `source_id` 或 `record_id`，以及 manifest、`evidence.tsv` 或报告中的定位位置。

## v1 迁移策略

不提供静默双模式。v1 报告包在旧 validator 下仍保持原历史状态；一旦使用新版 skill 做实质性修订，就必须：

1. 将 manifest 升级为 `schema_version: 2`。
2. 为来源补充 `report_reference`。
3. 在正文和 References 中建立数字引用。
4. 删除全部 `[E:...]`。
5. 使用新版 validator 重新验证。

新版 validator 对 v1 manifest 返回明确的 schema 错误，不自动猜测或改写旧报告。

## Skill 指令与模板

`SKILL.md` 和 `references/` 应明确：报告是 self-contained 的读者文档，`evidence.tsv` 是无需普通读者打开的持久审查 sidecar。报告写作阶段使用普通数字 reference；禁止把 `record_id` 作为正文引用。

`assets/report_skeleton.md` 提供一个可直接替换的正文引用和 References 示例。`assets/report_manifest.yaml` 使用 v2 reference contract。

## 测试边界

自动测试至少覆盖：

- 无 `[E:...]`、数字 reference 闭环完整的报告通过。
- 删除关键小节中的来源引用后出现缺失引用错误。
- `[E:DEC-...]` 即使对应有效 evidence row 也被拒绝。
- 重复 reference 锚点被拒绝。
- 正文编号与 References 编号不一致被拒绝。
- 未知 reference 锚点被拒绝。
- 不连续编号被拒绝。
- `inventory_sources.py` 生成稳定 `report_reference` 并在再次运行时保持不变。
- 两个来源生成相同锚点时 fail closed。
- 现有 validator 与 inventory 回归测试通过。

Agent 行为评估使用同一份合成来源分别运行旧版 skill 和新版 skill。旧版作为 RED 基线；新版必须生成普通数字 reference、保留 `evidence.tsv`，并通过新版 validator。

## 交付与 Git 历史

仓库在 `feat/reference-traceability` 分支管理。提交保持原子性：初始基线、设计、validator/inventory 行为、skill/template 文档和最终 eval 证据分别提交。临时预览、agent 输出和 eval workspace 保存在 `/tmp`，不进入 skill 仓库。
