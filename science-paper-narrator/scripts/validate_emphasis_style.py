from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
HEADING_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$")
CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
SOURCE_RE = re.compile(r"来源|参考文献|延伸阅读|一手资料")


@dataclass
class Section:
    level: int
    title: str
    lines: list[str]


def chinese_count(text: str) -> int:
    return len(CHINESE_RE.findall(text))


def visible_count(text: str) -> int:
    return len(re.sub(r"[\s`*_]", "", text))


def strip_frontmatter(markdown: str) -> str:
    return re.sub(r"\A---.*?---\s*", "", markdown, flags=re.DOTALL)


def parse_sections(markdown: str) -> list[Section]:
    sections: list[Section] = []
    current: Section | None = None
    for raw in strip_frontmatter(markdown).splitlines():
        match = HEADING_RE.match(raw.strip())
        if match:
            current = Section(len(match.group(1)), match.group(2), [])
            sections.append(current)
        elif current is not None:
            current.lines.append(raw)
    return sections


def included_lines(markdown: str) -> list[str]:
    lines: list[str] = []
    in_sources = False
    for raw in strip_frontmatter(markdown).splitlines():
        line = raw.strip()
        heading = HEADING_RE.match(line)
        if heading and heading.group(1) == "##":
            in_sources = bool(SOURCE_RE.search(heading.group(2)))
        if in_sources or not line or line.startswith("#") or line.startswith("!["):
            continue
        if line.startswith("*图") or line.startswith(">"):
            continue
        lines.append(line)
    return lines


def substantive_sections(markdown: str) -> list[Section]:
    sections = parse_sections(markdown)
    result: list[Section] = []
    for index, section in enumerate(sections):
        if SOURCE_RE.search(section.title):
            continue
        own_text = "\n".join(section.lines)
        # H2 containers with several H3 children are structural, not content sections.
        next_h2 = next((i for i in range(index + 1, len(sections)) if sections[i].level == 2), len(sections))
        has_h3_child = section.level == 2 and any(s.level == 3 for s in sections[index + 1 : next_h2])
        if not has_h3_child and chinese_count(own_text) >= 60:
            result.append(section)
    return result


def validate(article: Path) -> tuple[list[str], list[str], dict[str, float | int]]:
    markdown = article.read_text(encoding="utf-8")
    lines = included_lines(markdown)
    body = "\n".join(lines)
    marks = BOLD_RE.findall(body)
    body_cn = chinese_count(BOLD_RE.sub(r"\1", body))
    blue_cn = sum(chinese_count(mark) for mark in marks)
    density = len(marks) * 1000 / max(body_cn, 1)
    blue_share = blue_cn * 100 / max(body_cn, 1)

    sections = substantive_sections(markdown)
    covered = [bool(BOLD_RE.search("\n".join(section.lines))) for section in sections]
    coverage = sum(covered) / max(len(sections), 1)
    errors: list[str] = []
    warnings: list[str] = []

    if not 5 <= density <= 10:
        errors.append(f"inline emphasis density {density:.2f}/1k Chinese characters is outside 5-10")
    if not 6 <= blue_share <= 11:
        errors.append(f"inline emphasized Chinese-character share {blue_share:.2f}% is outside 6%-11%")
    if coverage < 0.75:
        errors.append(f"only {sum(covered)}/{len(sections)} substantive sections contain an inline anchor")
    if any(not covered[i] and not covered[i + 1] for i in range(max(0, len(covered) - 1))):
        errors.append("two consecutive substantive sections contain no inline emphasis anchor")

    paragraph_lines: list[str] = []
    for raw in strip_frontmatter(markdown).splitlines() + [""]:
        if raw.strip():
            paragraph_lines.append(raw.strip())
            continue
        if paragraph_lines:
            paragraph = " ".join(paragraph_lines)
            plain = BOLD_RE.sub(r"\1", paragraph)
            blue = sum(visible_count(value) for value in BOLD_RE.findall(paragraph))
            if visible_count(plain) >= 100 and blue / visible_count(plain) > 0.55:
                errors.append("a long paragraph has more than 55% of its visible text emphasized")
                break
            paragraph_lines = []

    lengths = [visible_count(mark) for mark in marks]
    short = sum(length <= 5 for length in lengths)
    long = sum(length > 45 for length in lengths)
    if marks and short / len(marks) > 0.20:
        warnings.append(f"{short}/{len(marks)} anchors are five visible characters or shorter")
    if long:
        warnings.append(f"{long} anchor(s) exceed 45 visible characters and require semantic review")
    if any(len(BOLD_RE.findall(line)) > 2 for line in lines):
        warnings.append("at least one sentence/line contains more than two separate anchors")

    metrics: dict[str, float | int] = {
        "chinese_chars": body_cn,
        "anchors": len(marks),
        "anchors_per_1k": round(density, 2),
        "blue_share_pct": round(blue_share, 2),
        "substantive_sections": len(sections),
        "covered_sections": sum(covered),
        "coverage": round(coverage, 2),
    }
    return errors, warnings, metrics


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Validate blue-bold reading anchors in a WeChat paper narrative.")
    parser.add_argument("article", type=Path)
    args = parser.parse_args()
    article = args.article.resolve()
    if not article.is_file():
        print(f"article not found: {article}", file=sys.stderr)
        return 2
    errors, warnings, metrics = validate(article)
    print("METRICS", " ".join(f"{key}={value}" for key, value in metrics.items()))
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        print(f"FAIL: {article}")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"PASS: {article}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
