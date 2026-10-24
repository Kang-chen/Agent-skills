# Reader-facing Reference Traceability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让报告正文使用普通数字 reference 并禁止 `[E:...]`，同时由 manifest、`evidence.tsv` 和 validator 保持关键主张的机器可读追溯闭环。

**Architecture:** Manifest v2 为每个来源保存稳定 `report_reference`；正文使用 `[N](#ref-anchor)`，References 定义同一锚点。Validator 解析 reference 定义和正文链接，并按 `evidence.tsv.report_locator` 在对应小节检查全部关键来源。`inventory_sources.py` 负责生成或保留稳定来源锚点。

**Tech Stack:** Python 3 标准库、PyYAML、`unittest`、Markdown/YAML/TSV、Git。

## Global Constraints

- `report.md` 必须对普通读者自包含，正文不得出现 `[E:<record_id>]`。
- `evidence.tsv` 是持久审查 sidecar，不合并进报告，也不删除。
- 新报告包使用 `schema_version: 2`，不静默兼容 v1 terminal validation。
- 正文引用格式固定为 `[N](#ref-stable-anchor)`，References 使用 `<a id="ref-stable-anchor"></a>`。
- 不新增 Python 依赖，不实现派生文档格式转换，不自动迁移旧报告。
- 所有代码标识符、字符串和日志保持英文；面向用户的设计与计划文本使用中文。

---

### Task 1: 建立 validator 的 reference 解析与禁止内部标识行为

**Files:**
- Create: `tests/test_validate_report_references.py`
- Modify: `scripts/validate_report.py`

**Interfaces:**
- Consumes: `report_text: str`、manifest source mappings、`evidence.tsv` rows。
- Produces: `_reference_index(report_text: str) -> tuple[dict[str, int], list[ValidationIssue]]`。
- Produces: `_section_text(report_text: str, anchor: str) -> str | None`。
- Produces: `_validate_report_references(...) -> list[ValidationIssue]`。
- Preserves: `_validate_evidence_rows(...)` 返回 `(records, issues)`。

- [ ] **Step 1: 写入失败测试，证明普通 reference 当前不能替代 `[E:...]`**

在 `tests/test_validate_report_references.py` 使用 `importlib.util.spec_from_file_location` 加载 `scripts/validate_report.py`，构造一个 report-critical row、一个带 `report_reference` 的 source，以及包含 `[1](#ref-source-a)` 的报告。断言 issues 不包含 `report.critical_record_unlinked`。

```python
def test_numeric_reference_replaces_internal_evidence_marker(self):
    records, issues = validator._validate_evidence_rows(
        [self.evidence_row()],
        {"SRC-A": self.source()},
        self.valid_report(),
        self.package_root,
        "report.md",
    )
    self.assertEqual({"REC-A"}, set(records))
    self.assertNotIn("report.critical_record_unlinked", self.issue_codes(issues))
```

- [ ] **Step 2: 运行单测并确认 RED**

Run: `python -m unittest tests.test_validate_report_references.ValidateReportReferenceTests.test_numeric_reference_replaces_internal_evidence_marker -v`

Expected: FAIL，issues 仍包含 `report.critical_record_unlinked`，证明旧 validator 强制要求 `[E:REC-A]`。

- [ ] **Step 3: 写入失败测试，证明 `[E:...]` 当前未被拒绝**

```python
def test_internal_evidence_marker_is_rejected(self):
    issues = validator._validate_report_story("# Report\n\n[E:REC-A]\n")
    self.assertIn("report.internal_evidence_marker_present", self.issue_codes(issues))
```

Run: `python -m unittest tests.test_validate_report_references.ValidateReportReferenceTests.test_internal_evidence_marker_is_rejected -v`

Expected: FAIL，因为旧 validator 没有该 issue code。

- [ ] **Step 4: 实现 reference 正则、索引和内部标识拒绝**

在 `scripts/validate_report.py` 添加：

```python
REFERENCE_LINK = re.compile(
    r"\[(?P<number>[1-9][0-9]*)\]\(#(?P<anchor>ref-[a-z0-9]+(?:-[a-z0-9]+)*)\)"
)
REFERENCE_ENTRY = re.compile(
    r'(?m)^<a id="(?P<anchor>ref-[a-z0-9]+(?:-[a-z0-9]+)*)"></a>[ \t]*\n'
    r'(?P<number>[1-9][0-9]*)\.[ \t]+(?P<text>\S.*)$'
)
INTERNAL_EVIDENCE_MARKER = re.compile(r"\[E:[^\]\n]+\]")
```

删除 `_validate_evidence_rows` 中对 `f"[E:{record_id}]"` 的检查。在 `_validate_report_story` 中为每个内部标识返回 `report.internal_evidence_marker_present`。

- [ ] **Step 5: 实现 reference 定义的结构校验**

`_reference_index` 必须检测：重复锚点、重复编号、编号不连续、链接到未知锚点、链接编号与条目编号不一致。使用以下稳定 issue codes：

```text
report.reference_anchor_duplicate
report.reference_number_duplicate
report.reference_number_sequence_invalid
report.reference_target_unknown
report.reference_number_mismatch
```

- [ ] **Step 6: 实现小节范围和 report-critical 来源闭环**

`_section_text` 从指定非 `ref-` 锚点开始，截取到下一个非 `ref-` 显式锚点。`_validate_report_references` 对每条 report-critical row：

```python
for source_id in _split_ids(row.get("source_ids")):
    source = sources.get(source_id)
    report_reference = source.get("report_reference") if source else None
    # Validate report.md#ref-* and require the matching numeric link in section_text.
```

缺少小节引用时返回 `report.critical_source_reference_missing`，locator 使用对应 `evidence.tsv:<line>`。

- [ ] **Step 7: 补齐错误路径测试并运行 Task 1 全集**

测试必须覆盖重复锚点、编号错配、未知目标、不连续编号、缺少关键小节引用和合法闭环。

Run: `python -m unittest tests.test_validate_report_references -v`

Expected: PASS，且至少 7 个 reference 行为测试通过。

- [ ] **Step 8: 提交 Task 1**

```bash
git add tests/test_validate_report_references.py scripts/validate_report.py
git commit -m "feat: validate reader-facing report references"
```

---

### Task 2: 让 manifest v2 和 inventory 管理稳定 report_reference

**Files:**
- Create: `tests/test_inventory_source_references.py`
- Modify: `scripts/inventory_sources.py`
- Modify: `scripts/validate_report.py`
- Modify: `assets/report_manifest.yaml`

**Interfaces:**
- Produces: `stable_reference_anchor(source_id: str) -> str`。
- Extends: `SourceRecord.report_reference: str`。
- Requires: `manifest.schema_version == 2`。

- [ ] **Step 1: 写入稳定锚点 RED 测试**

```python
def test_inventory_generates_stable_report_reference(self):
    result = inventory.inventory_sources(
        self.manifest_path,
        self.evidence_path,
        self.checkout,
        additional_locators=["sources/a.md"],
    )
    source = result.sources[0]
    self.assertRegex(source.report_reference, r"^report\.md#ref-[a-z0-9-]+$")
```

Run: `python -m unittest tests.test_inventory_source_references.InventorySourceReferenceTests.test_inventory_generates_stable_report_reference -v`

Expected: FAIL，旧 `SourceRecord` 没有 `report_reference`。

- [ ] **Step 2: 实现稳定锚点生成和字段保留**

```python
def stable_reference_anchor(source_id: str) -> str:
    value = source_id.casefold()
    if value.startswith("src-"):
        value = value[4:]
    slug = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    if not slug:
        raise ValueError("Source ID cannot produce a reference anchor")
    return f"ref-{slug}"
```

`inventory_sources` 使用已有 `report_reference`，否则生成 `report.md#{stable_reference_anchor(source_id)}`。在返回前检测两个 source 是否拥有同一 `report_reference`，冲突时抛出 `ValueError`。

- [ ] **Step 3: 写入并通过保留与冲突测试**

覆盖已有 `report_reference` 二次 inventory 后不改变，以及 `SRC-A_B`、`SRC-A-B` 发生锚点冲突时 fail closed。

Run: `python -m unittest tests.test_inventory_source_references -v`

Expected: PASS。

- [ ] **Step 4: 升级 manifest shape validation**

`_validate_manifest_shape` 要求：

```python
if manifest.get("schema_version") != 2:
    issues.append(_issue(
        "manifest.schema_version_invalid",
        "Reader-facing reference reports require schema_version 2",
        "manifest:schema_version",
    ))
```

将 `expected_traceability` 扩展为设计文档中的完整 v2 mapping。

- [ ] **Step 5: 校验 source.report_reference**

`_source_index` 验证格式为 `report.md#ref-stable-anchor`，并检测不同 source 的重复 reference locator。issue codes：

```text
source.report_reference_required
source.report_reference_invalid
source.report_reference_duplicate
```

- [ ] **Step 6: 更新 manifest seed 并运行 Task 1–2 测试**

将 `assets/report_manifest.yaml` 改为 `schema_version: 2`，加入完整 reference traceability block。

Run: `python -m unittest discover -s tests -v`

Expected: PASS。

- [ ] **Step 7: 提交 Task 2**

```bash
git add tests/test_inventory_source_references.py scripts/inventory_sources.py scripts/validate_report.py assets/report_manifest.yaml
git commit -m "feat: register stable source reference anchors"
```

---

### Task 3: 更新 skill contract、报告模板和 eval 断言

**Files:**
- Create: `tests/test_skill_reference_contract.py`
- Modify: `SKILL.md`
- Modify: `assets/report_skeleton.md`
- Modify: `references/contracts-and-status.md`
- Modify: `references/claims-and-review.md`
- Modify: `references/story-and-figures.md`
- Modify: `evals/assertions/report_assertions.yaml`
- Modify: `evals/rubric.md`

**Interfaces:**
- Consumes: Task 1–2 的 v2 reference contract。
- Produces: agent 可直接遵循的普通数字 reference 写作规则。

- [ ] **Step 1: 写入静态 contract RED 测试**

在 `tests/test_skill_reference_contract.py` 断言：

```python
self.assertNotIn("[E:", skeleton)
self.assertIn("[1](#ref-source-example)", skeleton)
self.assertIn('id="references"', skeleton)
self.assertIn("report_reference", contracts)
self.assertIn("reader-facing", skill)
```

Run: `python -m unittest tests.test_skill_reference_contract -v`

Expected: FAIL，旧模板仍要求 evidence ID 且未提供 reference 示例。

- [ ] **Step 2: 更新 SKILL.md**

在 contract 和执行阶段写明：

- 报告语义必须自包含。
- 普通读者无需阅读 `evidence.tsv`。
- 正文使用 `[N](#ref-stable-anchor)`。
- `record_id` 只存在于 sidecar，不作为正文引用。
- v1 实质性修订必须迁移到 v2 后才能 terminal validation。

- [ ] **Step 3: 更新 report skeleton 与 references**

模板加入一个完整正文引用和 References 条目；`contracts-and-status.md` 描述三文件职责及 v2 字段；`claims-and-review.md` 把 claim coverage 改为检查普通 reference 与 evidence ledger 映射；`story-and-figures.md` 要求每个关键小节使用对应 source reference。

- [ ] **Step 4: 更新 eval assertions 与 rubric**

新增断言：reader-facing report 不含 `[E:]`；关键主张使用数字锚点 reference；`evidence.tsv` 仍存在并与 manifest/source 闭环。

- [ ] **Step 5: 运行静态 contract 与完整单测**

Run: `python -m unittest discover -s tests -v`

Expected: PASS。

- [ ] **Step 6: 提交 Task 3**

```bash
git add SKILL.md assets/report_skeleton.md references evals/assertions/report_assertions.yaml evals/rubric.md tests/test_skill_reference_contract.py
git commit -m "docs: adopt reader-facing reference contract"
```

---

### Task 4: 运行旧版/新版 agent eval、回归验证和交付检查

**Files:**
- External outputs: `/tmp/evidence-backed-scientific-report-workspace/iteration-1/`。

**Interfaces:**
- Consumes: old snapshot、当前 skill、同一 `reference-rendering` prompt。
- Produces: old/new output packages、grading JSON、benchmark、静态 review viewer。

- [ ] **Step 1: 记录旧版 RED 基线**

检查 old-skill agent 输出，保存 raw `[E:...]`、validator conflict 和原始 rationale。确认失败来自旧 contract，而不是输入或运行错误。

- [ ] **Step 2: 使用新版 skill 运行同一 prompt**

输出到 `/tmp/evidence-backed-scientific-report-workspace/iteration-1/reference-rendering/with_skill/outputs/`。断言：

```text
report.md contains no [E:...]
report.md contains [N](#ref-...)
References defines every cited ref-* anchor exactly once
evidence.tsv exists
validate_report.py check exits 0
```

- [ ] **Step 3: 生成 grading 与 benchmark**

为 old/new 输出写入使用 `text`、`passed`、`evidence` 字段的 `grading.json`，然后运行 skill-creator aggregation script。生成静态 HTML reviewer，不把 `/tmp` eval 产物提交进仓库。

- [ ] **Step 4: 运行完整本地验证**

```bash
python -m unittest discover -s tests -v
python -m py_compile scripts/inventory_sources.py scripts/validate_report.py
python scripts/inventory_sources.py --help
python scripts/validate_report.py --help
rg -n '\[E:[^]]+\]' SKILL.md assets references evals scripts tests
git diff --check
```

Expected: 单测全部通过；两个脚本可编译且 help exit 0；`rg` 只在禁止规则、错误测试或迁移说明中命中，不在 reader-facing 模板示例中命中；Git diff 无格式错误。

- [ ] **Step 5: 独立 reviewer 检查**

Reviewer 对照设计逐项检查 manifest v2、source reference、section-local citation、legacy marker rejection、错误码和迁移说明。任何 Important/Blocking 问题必须修复并重新运行 Step 4。

- [ ] **Step 6: 处理 eval 失败而不隐藏回归**

如果新版 eval 任一断言失败，停止交付并返回产生该行为的 Task 1、2 或 3：先补充失败测试，再修复对应实现并重新提交。若 eval 全部通过，不为 `/tmp` 输出创建空提交，也不把 eval workspace 加入仓库。

- [ ] **Step 7: 最终 Git 检查**

Run: `git status --short --branch`、`git log --oneline --decorate -5`。

Expected: 功能分支干净；提交历史包含基线、设计和原子实现提交。
