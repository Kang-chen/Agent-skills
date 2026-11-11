"""check_brief.py 的结构检查行为测试。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_brief.py"

CARD = ("**图 1｜端到端架构能否替代后处理？** 看哪里：星号。结论：差 +0.011。"
        "限制：区间跨 0。下一项证据：[完整表](report_full.md#res-a)。\n")
MAIN = ("# 示例项目进展汇报：读出能否替代？\n\n<!-- layer:main -->\n\n"
        "> 本简版汇总一份已核对的报告，完整版见 [完整版](report_full.md)。\n\n"
        "**一句话：** C_soft 的确认 AUC 0.753，参照 0.742，共 50 位患者。\n\n"
        "<!-- brief:decisions -->\n"
        "| 要作的判断 | 当前答案 | 证据级别 | 什么结果会改变答案 |\n|---|---|---|---|\n"
        "| 用哪种读出 | C_soft 留作候选 | 点估计非劣 | 独立患者上的检验 |\n\n"
        "![main result](figures/result_main.png)\n\n" + CARD + "\n")
LIMITS = "<!-- brief:limits -->\n**必须知道的限制**\n\n- 区间 −0.058 到 +0.081 跨 0。\n\n"
NEXT = "<!-- brief:next -->\n**下一步**\n\n1. 收齐折外预测。\n\n"
QUESTIONS = "<!-- layer:questions -->\n\n## 哪个架构合格？\n\n答案见上。\n\n"
LOOKUP = ("<!-- layer:lookup -->\n\n## 名词速查\n\n<!-- brief:glossary -->\n"
          "| 名词 | 含义 |\n|---|---|\n| C_soft | 软池化加可学比值头 |\n")
LEDGER = ("claim\tvalue\tsource\trecomputed\tstatus\n"
          "C_soft AUC\t0.753\tsrc\t0.753\tok\n"
          "reference AUC\t0.742\tsrc\t0.742\tok\n"
          "difference\t+0.011\tsrc\t+0.011\tok\n"
          "CI\t−0.058 to +0.081\tsrc\t−0.058 to +0.081\tok\n"
          "patients\t50\tsrc\t50\tok\n")


def build(root: Path, brief: str | None = None, ledger: str = LEDGER, figures=("result_main.png",)) -> Path:
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "plot_data").mkdir(exist_ok=True)
    (root / "review").mkdir(exist_ok=True)
    for name in figures:
        (root / "figures" / name).write_bytes(b"png")
        (root / "plot_data" / f"{Path(name).stem}.csv").write_text("a,b\n1,2\n")
    (root / "report_full.md").write_text('<a id="res-a"></a>\n## 结果是什么？\n', encoding="utf-8")
    (root / "review" / "number_checks.tsv").write_text(ledger, encoding="utf-8")
    text = brief if brief is not None else MAIN + LIMITS + NEXT + QUESTIONS + LOOKUP
    (root / "report.md").write_text(text, encoding="utf-8")
    return root


def run(root: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), str(root), *extra],
                          capture_output=True, text=True, check=False)


def test_valid_brief_passes(tmp_path):
    result = run(build(tmp_path))
    assert result.returncode == 0, result.stdout
    assert "RESULT PASS" in result.stdout


def test_limits_outside_main_layer_fail(tmp_path):
    brief = MAIN + NEXT + QUESTIONS + LIMITS + LOOKUP
    result = run(build(tmp_path, brief))
    assert result.returncode == 1
    assert "brief:limits must be in the main layer" in result.stdout


def test_layer_order_is_enforced(tmp_path):
    brief = MAIN + LIMITS + NEXT + LOOKUP + QUESTIONS
    result = run(build(tmp_path, brief))
    assert "layer order" in result.stdout


def test_main_layer_over_budget_fails(tmp_path):
    padding = "这是一段很长的说明文字。" * 400 + "\n\n"
    brief = MAIN.replace("<!-- brief:decisions -->", padding + "<!-- brief:decisions -->") + LIMITS + NEXT + QUESTIONS + LOOKUP
    result = run(build(tmp_path, brief))
    assert result.returncode == 1
    assert "over the 5 min budget" in result.stdout


def test_figure_without_card_fails(tmp_path):
    brief = (MAIN.replace(CARD, "**图 1｜主结果。** 只是一张图。\n") + LIMITS + NEXT + QUESTIONS + LOOKUP)
    result = run(build(tmp_path, brief))
    assert result.returncode == 1
    assert "Figure card for figures/result_main.png is missing" in result.stdout
    assert "看哪里" in result.stdout


def test_english_card_labels_are_accepted(tmp_path):
    english = ("**Figure 1 | Can the end-to-end arm replace the pipeline?** Look at: the star. "
               "Takeaway: +0.011. Limit: the interval crosses 0. Next evidence: [table](report_full.md#res-a).\n")
    brief = MAIN.replace(CARD, english) + LIMITS + NEXT + QUESTIONS + LOOKUP
    result = run(build(tmp_path, brief))
    assert result.returncode == 0, result.stdout


def test_number_missing_from_ledger_fails(tmp_path):
    brief = MAIN.replace("共 50 位患者", "共 57 位患者") + LIMITS + NEXT + QUESTIONS + LOOKUP
    result = run(build(tmp_path, brief))
    assert result.returncode == 1
    assert "not in any ok ledger row: 57" in result.stdout


def test_mismatched_ledger_row_fails(tmp_path):
    ledger = LEDGER.replace("patients\t50\tsrc\t50\tok", "patients\t50\tsrc\t49\tmismatch")
    result = run(build(tmp_path, ledger=ledger))
    assert result.returncode == 1
    assert "ledger rows not ok" in result.stdout


def test_dates_and_times_are_not_counted_as_numbers(tmp_path):
    brief = MAIN.replace("**一句话：**", "状态观测于 2026-10-24 10:21 UTC。\n\n**一句话：**") + LIMITS + NEXT + QUESTIONS + LOOKUP
    result = run(build(tmp_path, brief))
    assert result.returncode == 0, result.stdout


def test_broken_anchor_fails(tmp_path):
    brief = MAIN.replace("report_full.md#res-a", "report_full.md#res-missing") + LIMITS + NEXT + QUESTIONS + LOOKUP
    result = run(build(tmp_path, brief))
    assert result.returncode == 1
    assert "res-missing (anchor not found)" in result.stdout


def test_heading_slug_anchor_resolves(tmp_path):
    brief = MAIN.replace("report_full.md#res-a", "report_full.md#结果是什么") + LIMITS + NEXT + QUESTIONS + LOOKUP
    result = run(build(tmp_path, brief))
    assert result.returncode == 0, result.stdout


def test_too_many_main_figures_fail(tmp_path):
    extra = "".join(f"![extra](figures/result_x{i}.png)\n\n" + CARD.replace("图 1", f"图 {i + 2}") + "\n"
                    for i in range(2))
    brief = MAIN + extra + LIMITS + NEXT + QUESTIONS + LOOKUP
    root = build(tmp_path, brief, figures=("result_main.png", "result_x0.png", "result_x1.png"))
    result = run(root)
    assert result.returncode == 1
    assert "3 Figures in the main layer, cap is 2" in result.stdout


def test_main_layer_code_must_be_in_glossary(tmp_path):
    brief = MAIN.replace("C_soft 留作候选", "C_soft 留作候选，参照 N05") + LIMITS + NEXT + QUESTIONS + LOOKUP
    result = run(build(tmp_path, brief))
    assert result.returncode == 1
    assert "main-layer codes missing from brief:glossary: N05" in result.stdout


def test_too_many_limits_fail(tmp_path):
    limits = "<!-- brief:limits -->\n" + "".join(f"- 限制条目。\n" for _ in range(6)) + "\n"
    brief = MAIN + limits + NEXT + QUESTIONS + LOOKUP
    result = run(build(tmp_path, brief))
    assert result.returncode == 1
    assert "brief:limits has 6 items, cap is 5" in result.stdout
