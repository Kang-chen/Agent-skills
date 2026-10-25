#!/usr/bin/env python3
"""检查简版报告（brief）的结构：三层、阅读预算、图卡、数字台账、链接与名词表。

用法：
    python3 check_brief.py <package_root> [--report report.md]
        [--ledger review/number_checks.tsv ...] [--budget-minutes 5]

只做结构检查，不判断科学正确性，也不判断写得好不好读。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

LAYERS = ["main", "questions", "lookup"]
MAIN_BLOCKS = ["decisions", "limits", "next"]
LOOKUP_BLOCKS = ["glossary"]
LIST_CAP = {"limits": 5, "next": 5}

LAYER_RE = re.compile(r"<!--\s*layer:([a-z_]+)\s*-->")
BLOCK_RE = re.compile(r"<!--\s*brief:([a-z_]+)\s*-->")
MARKER_RE = re.compile(r"<!--\s*(?:layer|brief):[a-z_]+\s*-->")
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
FENCE_RE = re.compile(r"^\s*(```|~~~)")
FIGURE_RE = re.compile(r"!\[[^\]]*\]\(\s*<?([^)>\s]+)")
LINK_RE = re.compile(r"(?<!!)\[([^\]]*)\]\(\s*<?([^)>\s]+)>?\s*\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
ANCHOR_ID_RE = re.compile(r"""\bid\s*=\s*["']([^"']+)["']""")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
LIST_ITEM_RE = re.compile(r"^\s{0,3}(?:[-*+]|\d+[.)])\s+\S")
CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]*")
DATE_RE = re.compile(r"\b\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?|\b\d{1,2}[-/]\d{1,2}(?=\s*\d{1,2}:\d{2})")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
YEAR_RE = re.compile(r"(?<![\w.])(?:19|20)\d{2}(?![\w.])")
NUMBER_RE = re.compile(r"(?<![\w.])(\d+\.\d{2,}|\d{2,})(?![\w.])")
NUMERIC_REF_RE = re.compile(r"\[\d+\]\(#[^)]+\)")

CARD_LABELS = {
    "zh": ("看哪里", "结论", "限制", "下一项证据"),
    "en": ("Look at", "Takeaway", "Limit", "Next evidence"),
}
# 内部代号：字母+数字（可带一个小写后缀，如 H1c）、全大写缩写、带下划线的臂名（如 C_soft）
CODE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_./#-])("
    r"[A-Za-z]{1,4}\d{1,3}[a-z]?(?:-[A-Za-z]+)?"
    r"|[A-Z]{2,6}"
    r"|[A-Z][A-Za-z0-9]*_[A-Za-z0-9]+"
    r")(?![A-Za-z0-9_./-])")
TOKEN_ALLOWLIST = {
    "AUC", "ROC", "CI", "SD", "GPU", "CPU", "RAM", "CSV", "TSV", "JSON", "YAML",
    "PNG", "SVG", "PDF", "HTML", "MD", "UTC", "ID", "IDS", "URL", "API", "NA",
    "OK", "TODO", "AND", "OR", "NOT",
}


def strip_fences(text: str) -> str:
    """去掉代码块，保留行数不变以便定位。"""
    out, in_fence = [], False
    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            out.append("")
            continue
        out.append("" if in_fence else line)
    return "\n".join(out)


def split_layers(text: str) -> tuple[list[str], dict[str, str]]:
    """按 layer 标记切分；标题等出现在第一个标记之前的内容并入 main。"""
    marks = [(m.group(1), m.start(), m.end()) for m in LAYER_RE.finditer(text)]
    order = [name for name, _, _ in marks]
    layers: dict[str, str] = {}
    for i, (name, _, end) in enumerate(marks):
        stop = marks[i + 1][1] if i + 1 < len(marks) else len(text)
        body = text[end:stop]
        if name == "main" and i == 0:
            body = text[:marks[0][1]] + body
        layers.setdefault(name, body)
    return order, layers


def block_text(layer: str, name: str) -> str | None:
    """取 brief 块标记之后、下一个标记或同级以上标题之前的内容。"""
    m = re.search(rf"<!--\s*brief:{name}\s*-->", layer)
    if not m:
        return None
    rest = layer[m.end():]
    stop = MARKER_RE.search(rest)
    rest = rest[:stop.start()] if stop else rest
    heading = re.search(r"^#{1,2}\s", rest, re.M)
    return rest[:heading.start()] if heading else rest


def tables(text: str) -> list[list[str]]:
    """返回每张表的行（含表头，不含分隔行）。"""
    lines, found, i = text.splitlines(), [], 0
    while i < len(lines):
        if i + 1 < len(lines) and lines[i].strip().startswith("|") and TABLE_SEP_RE.match(lines[i + 1]):
            rows, j = [lines[i]], i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(lines[j])
                j += 1
            found.append(rows)
            i = j
        else:
            i += 1
    return found


def readable_text(text: str) -> str:
    """估算阅读量时只保留读者真正会读的文字。"""
    text = COMMENT_RE.sub(" ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"`[^`]*`", " x ", text)
    return text


def reading_minutes(text: str, cjk_per_min: float, words_per_min: float) -> tuple[float, int, int]:
    body = readable_text(text)
    cjk = len(CJK_RE.findall(body))
    words = len(WORD_RE.findall(CJK_RE.sub(" ", body)))
    return cjk / cjk_per_min + words / words_per_min, cjk, words


def figure_cards(text: str) -> list[tuple[str, list[str]]]:
    """对每张图，检查其后（到下一张图或标题前）的图卡是否齐全。"""
    lines = text.splitlines()
    problems: list[tuple[str, list[str]]] = []
    for i, line in enumerate(lines):
        for target in FIGURE_RE.findall(line):
            card_lines = []
            for nxt in lines[i + 1:]:
                if FIGURE_RE.search(nxt) or HEADING_RE.match(nxt) or MARKER_RE.search(nxt):
                    break
                card_lines.append(nxt)
            card = "\n".join(card_lines)
            missing = []
            title = re.search(r"\*\*([^*]+)\*\*", card)
            if not title or not re.search(r"[?？]", title.group(1)):
                missing.append("bold title phrased as a question")
            label_sets = [labels for labels in CARD_LABELS.values()
                          if all(re.search(rf"{re.escape(lab)}\s*[:：]", card) for lab in labels)]
            if not label_sets:
                best = max(CARD_LABELS.values(),
                           key=lambda labs: sum(bool(re.search(rf"{re.escape(lab)}\s*[:：]", card)) for lab in labs))
                missing += [lab for lab in best if not re.search(rf"{re.escape(lab)}\s*[:：]", card)]
            if missing:
                problems.append((target, missing))
    return problems


def slugify(heading: str) -> str:
    """近似 GitHub 的标题锚点：去格式符与标点，小写，空格换成连字符。"""
    text = re.sub(r"<[^>]+>", "", heading)
    text = re.sub(r"[`*_~]", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[^\w\- ]", "", text.strip().lower())
    return text.replace(" ", "-")


def anchors_of(path: Path, cache: dict[Path, set[str]]) -> set[str]:
    if path not in cache:
        raw = path.read_text(encoding="utf-8", errors="replace")
        ids = set(ANCHOR_ID_RE.findall(raw))
        seen: dict[str, int] = {}
        for line in strip_fences(raw).splitlines():
            m = HEADING_RE.match(line)
            if not m:
                continue
            slug = slugify(m.group(2))
            n = seen.get(slug, 0)
            ids.add(slug if n == 0 else f"{slug}-{n}")
            seen[slug] = n + 1
        cache[path] = ids
    return cache[path]


def ledger_numbers(paths: list[Path]) -> tuple[set[str], list[str], list[str]]:
    """从台账中状态为 ok 的行收集数字；返回（数字集合，未解决的行，缺失的台账）。"""
    numbers: set[str] = set()
    unresolved, missing = [], []
    for path in paths:
        if not path.is_file():
            missing.append(str(path))
            continue
        rows = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            continue
        header = [c.strip().lower() for c in rows[0]]
        if "value" not in header or "status" not in header:
            unresolved.append(f"{path.name}: needs 'value' and 'status' columns")
            continue
        vi, si = header.index("value"), header.index("status")
        for row in rows[1:]:
            if len(row) <= max(vi, si):
                continue
            status = row[si].strip().lower()
            if status == "ok":
                numbers.update(NUMBER_RE.findall(row[vi]))
            elif status == "mismatch":
                unresolved.append(f"{path.name}: {row[vi].strip()}")
    return numbers, unresolved, missing


def prose_numbers(text: str) -> set[str]:
    body = readable_text(text)
    body = NUMERIC_REF_RE.sub(" ", body)
    body = DATE_RE.sub(" ", body)
    body = TIME_RE.sub(" ", body)
    body = YEAR_RE.sub(" ", body)
    return set(NUMBER_RE.findall(body))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="package root that holds the brief")
    ap.add_argument("--report", default="report.md", help="brief file, relative to root")
    ap.add_argument("--ledger", action="append",
                    help="number ledger TSV relative to root; repeatable "
                         "(default: review/number_checks.tsv, plus review/brief_number_checks.tsv if present)")
    ap.add_argument("--budget-minutes", type=float, default=5.0, help="main-layer reading budget")
    ap.add_argument("--cjk-per-min", type=float, default=400.0, help="CJK characters read per minute")
    ap.add_argument("--words-per-min", type=float, default=200.0, help="Latin words or numbers read per minute")
    ap.add_argument("--max-main-figures", type=int, default=2)
    ap.add_argument("--max-figures", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    report = root / args.report
    if not report.is_file():
        print(f"FAIL  brief not found: {report}")
        return 1
    raw = report.read_text(encoding="utf-8")
    text = strip_fences(raw)
    failures: list[str] = []
    notes: list[str] = []

    # 1) 三层齐全且顺序正确
    order, layers = split_layers(text)
    missing_layers = [name for name in LAYERS if name not in layers]
    if missing_layers:
        failures.append("missing layers: " + ", ".join(missing_layers))
    present = [name for name in dict.fromkeys(order) if name in LAYERS]
    if present != [name for name in LAYERS if name in present]:
        failures.append(f"layer order is {present}, expected {LAYERS}")
    print(f"layers     : {', '.join(present) or 'none'}")

    main_layer = layers.get("main", "")
    lookup_layer = layers.get("lookup", "")

    # 2) 主层必须有判断表、限制与下一步；查阅层必须有名词表
    for name in MAIN_BLOCKS:
        body = block_text(main_layer, name)
        if body is None:
            where = "another layer" if any(block_text(v, name) is not None for v in layers.values()) else "nowhere"
            failures.append(f"brief:{name} must be in the main layer (found in {where})")
            continue
        if name == "decisions":
            tabs = tables(body)
            if not tabs or len(tabs[0]) < 2:
                failures.append("brief:decisions needs a table with at least one row")
            elif min(row.count("|") for row in tabs[0]) - 1 < 4:
                failures.append("brief:decisions table needs 4 columns: judgment, answer, evidence level, what would change it")
            else:
                print(f"decisions  : {len(tabs[0]) - 1} rows")
        else:
            items = [line for line in body.splitlines() if LIST_ITEM_RE.match(line)]
            if not items:
                failures.append(f"brief:{name} has no list items")
            elif len(items) > LIST_CAP[name]:
                failures.append(f"brief:{name} has {len(items)} items, cap is {LIST_CAP[name]}")
            else:
                print(f"{name:<11}: {len(items)} items")
    glossary = block_text(lookup_layer, "glossary")
    if glossary is None or not tables(glossary):
        failures.append("brief:glossary table missing from the lookup layer")
        glossary = ""

    # 3) 主层阅读时间
    minutes, cjk, words = reading_minutes(main_layer, args.cjk_per_min, args.words_per_min)
    total_minutes, _, _ = reading_minutes(text, args.cjk_per_min, args.words_per_min)
    print(f"reading    : main layer ~{minutes:.1f} min ({cjk} CJK chars, {words} words), "
          f"budget {args.budget_minutes:g} min; whole brief ~{total_minutes:.1f} min")
    if minutes > args.budget_minutes:
        failures.append(f"main layer needs ~{minutes:.1f} min, over the {args.budget_minutes:g} min budget")

    # 4) 图：数量、图卡、文件与 plot_data
    main_figs = [f for f in FIGURE_RE.findall(main_layer) if not f.startswith(("http://", "https://"))]
    all_figs = list(dict.fromkeys(f for f in FIGURE_RE.findall(text) if not f.startswith(("http://", "https://"))))
    print(f"figures    : {len(main_figs)} in main layer (cap {args.max_main_figures}), "
          f"{len(all_figs)} total (cap {args.max_figures})")
    if not main_figs:
        failures.append("the main layer has no Figure")
    if len(main_figs) > args.max_main_figures:
        failures.append(f"{len(main_figs)} Figures in the main layer, cap is {args.max_main_figures}")
    if len(all_figs) > args.max_figures:
        failures.append(f"{len(all_figs)} Figures in the brief, cap is {args.max_figures}")
    for target, missing in figure_cards(text):
        failures.append(f"Figure card for {target} is missing: {', '.join(missing)}")
    for fig in all_figs:
        path = (report.parent / fig).resolve()
        if not path.is_file():
            failures.append(f"Figure file not found: {fig}")
        elif not path.stem.startswith("concept_") and not any(
                (root / "plot_data" / f"{path.stem}{ext}").is_file() for ext in (".csv", ".tsv", ".json", ".parquet")):
            failures.append(f"measured Figure without plot data: plot_data/{path.stem}.*")

    # 5) 数字：每个数字都在台账里且状态为 ok
    ledger_paths = [root / p for p in args.ledger] if args.ledger else [root / "review/number_checks.tsv"]
    if not args.ledger and (root / "review/brief_number_checks.tsv").is_file():
        ledger_paths.append(root / "review/brief_number_checks.tsv")
    known, unresolved, missing_ledgers = ledger_numbers(ledger_paths)
    if missing_ledgers:
        failures.append("number ledger not found: " + ", ".join(missing_ledgers))
    if unresolved:
        failures.append("ledger rows not ok: " + "; ".join(unresolved[:8]))
    stated = prose_numbers(text)
    unlogged = sorted(stated - known)
    print(f"numbers    : {len(stated) - len(unlogged)}/{len(stated)} numbers in the brief found in the ledger")
    if unlogged:
        failures.append(f"{len(unlogged)} numbers not in any ok ledger row: " + ", ".join(unlogged[:12])
                        + (" ..." if len(unlogged) > 12 else ""))

    # 6) 相对链接与锚点都能解析
    cache: dict[Path, set[str]] = {}
    broken = []
    link_count = 0
    for _, target in LINK_RE.findall(text):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I):
            continue
        link_count += 1
        path_part, _, anchor = target.partition("#")
        dest = report if not path_part else (report.parent / path_part).resolve()
        if not dest.exists():
            broken.append(f"{target} (file not found)")
        elif anchor and dest.suffix.lower() == ".md" and anchor not in anchors_of(dest, cache):
            broken.append(f"{target} (anchor not found)")
    print(f"links      : {link_count} relative links, {len(broken)} broken")
    if broken:
        failures.append("broken links: " + "; ".join(broken[:8]) + (" ..." if len(broken) > 8 else ""))

    # 7) 主层出现的代号都在名词表里
    main_body = readable_text(main_layer)
    tokens = {t for t in CODE_TOKEN_RE.findall(main_body) if t not in TOKEN_ALLOWLIST}
    undefined = sorted(t for t in tokens if t not in glossary)
    if undefined:
        failures.append("main-layer codes missing from brief:glossary: " + ", ".join(undefined[:15]))
    rest_tokens = {t for t in CODE_TOKEN_RE.findall(readable_text(text)) if t not in TOKEN_ALLOWLIST}
    rest_undefined = sorted(t for t in rest_tokens - tokens if t not in glossary)
    if rest_undefined:
        notes.append("codes outside the main layer missing from the glossary: " + ", ".join(rest_undefined[:15]))

    for note in notes:
        print(f"NOTE  {note}")
    for failure in failures:
        print(f"FAIL  {failure}")
    if failures:
        print(f"\nRESULT FAIL ({len(failures)} problems)")
        return 1
    print("\nRESULT PASS (structure only; it says nothing about correctness or readability)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
