#!/usr/bin/env python3
"""
测试运行入口 - 运行所有或特定类别的测试，输出结构化结果。

用法:
    python tests/run_all.py                 # 运行全部测试
    python tests/run_all.py --test crud     # 仅运行 CRUD 测试
    python tests/run_all.py --test scope    # 仅运行作用域测试
    python tests/run_all.py --test sync     # 仅运行同步完整性测试
    python tests/run_all.py --verbose       # 详细输出
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# 测试类别到文件的映射
CATEGORY_MAP = {
    "crud": "test_crud.py",
    "project_crud": "test_project_crud.py",
    "scope": "test_scope.py",
    "compliance": "test_skill_creator_compliance.py",
    "sync": "test_sync_integrity.py",
    "upstream": "test_upstream.py",
    "version": "test_version_consistency.py",
}

# 所有支持的类别列表
ALL_CATEGORIES = list(CATEGORY_MAP.keys())


def _get_tests_dir() -> Path:
    """返回 tests 目录的绝对路径。"""
    return Path(__file__).parent.resolve()


def _run_pytest(test_file: str = None, verbose: bool = False) -> dict:
    """
    调用 pytest 运行测试并解析输出。
    返回包含 passed/failed/error 计数的 dict。
    """
    tests_dir = _get_tests_dir()
    cmd = [sys.executable, "-m", "pytest"]

    if test_file:
        cmd.append(str(tests_dir / test_file))
    else:
        cmd.append(str(tests_dir))

    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    # 禁用缓存以获得干净结果
    cmd.append("-p")
    cmd.append("no:cacheprovider")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(tests_dir.parent),
    )

    return _parse_pytest_output(result.stdout, result.stderr, result.returncode)


def _parse_pytest_output(stdout: str, stderr: str, returncode: int) -> dict:
    """从 pytest 标准输出中解析测试结果。"""
    parsed = {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "warnings": 0,
        "returncode": returncode,
        "output": stdout,
    }

    # 匹配 pytest 摘要行，例如: "28 passed", "2 failed, 26 passed"
    combined = stdout + stderr

    passed_match = re.search(r"(\d+)\s+passed", combined)
    if passed_match:
        parsed["passed"] = int(passed_match.group(1))

    failed_match = re.search(r"(\d+)\s+failed", combined)
    if failed_match:
        parsed["failed"] = int(failed_match.group(1))

    error_match = re.search(r"(\d+)\s+error", combined)
    if error_match:
        parsed["errors"] = int(error_match.group(1))

    warning_match = re.search(r"(\d+)\s+warning", combined)
    if warning_match:
        parsed["warnings"] = int(warning_match.group(1))

    return parsed


def _build_summary(category_results: dict) -> dict:
    """将各类别的结果汇总为 grading.json 格式。"""
    total_passed = 0
    total_failed = 0
    categories = {}

    for cat_name, result in category_results.items():
        passed = result["passed"]
        failed = result["failed"] + result["errors"]
        total_passed += passed
        total_failed += failed
        categories[cat_name] = {
            "passed": passed,
            "failed": failed,
        }

    total = total_passed + total_failed
    pass_rate = round(total_passed / total, 2) if total > 0 else 0.0

    return {
        "summary": {
            "passed": total_passed,
            "failed": total_failed,
            "total": total,
            "pass_rate": pass_rate,
        },
        "categories": categories,
    }


def _print_table(summary: dict) -> None:
    """打印人类可读的结果表格。"""
    print()
    print("=" * 56)
    print(f"{'Category':<20} {'Passed':>8} {'Failed':>8} {'Status':>10}")
    print("-" * 56)

    for cat_name, cat_data in summary["categories"].items():
        passed = cat_data["passed"]
        failed = cat_data["failed"]
        status = "PASS" if failed == 0 else "FAIL"
        print(f"{cat_name:<20} {passed:>8} {failed:>8} {status:>10}")

    print("-" * 56)
    s = summary["summary"]
    overall = "PASS" if s["failed"] == 0 else "FAIL"
    print(f"{'TOTAL':<20} {s['passed']:>8} {s['failed']:>8} {overall:>10}")
    print(f"{'Pass Rate':<20} {s['pass_rate']:.0%}")
    print("=" * 56)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run skill-manager test suite.",
    )
    parser.add_argument(
        "--test",
        choices=ALL_CATEGORIES,
        default=None,
        help="Run specific test category only.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose pytest output.",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Print results as JSON to stdout.",
    )
    args = parser.parse_args()

    category_results = {}

    if args.test:
        # 运行单个类别
        categories_to_run = [args.test]
    else:
        # 运行全部类别
        categories_to_run = ALL_CATEGORIES

    for cat in categories_to_run:
        test_file = CATEGORY_MAP[cat]
        print(f"Running: {cat} ({test_file})")
        result = _run_pytest(test_file, verbose=args.verbose)
        category_results[cat] = result

        if args.verbose:
            print(result["output"])

    summary = _build_summary(category_results)

    if args.json_output:
        print(json.dumps(summary, indent=2))
    else:
        _print_table(summary)

    # 返回退出码
    sys.exit(0 if summary["summary"]["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
