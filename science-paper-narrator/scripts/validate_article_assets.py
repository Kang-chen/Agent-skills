from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote


IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")
PLACEHOLDER_RE = re.compile(r"【\s*配图建议|待补图|TODO\s*[:：-]?\s*图", re.IGNORECASE)


def local_image_path(article: Path, raw_path: str) -> Path | None:
    value = unquote(raw_path.strip().strip("<>"))
    if re.match(r"^https?://", value, re.IGNORECASE):
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = article.parent / candidate
    return candidate.resolve()


def validate(article: Path, require_images: bool, require_captions: bool, min_images: int) -> list[str]:
    text = article.read_text(encoding="utf-8")
    lines = text.splitlines()
    errors: list[str] = []

    placeholders = [(i + 1, line.strip()) for i, line in enumerate(lines) if PLACEHOLDER_RE.search(line)]
    for line_number, line in placeholders:
        errors.append(f"line {line_number}: unresolved figure placeholder: {line}")

    matches = list(IMAGE_RE.finditer(text))
    required_count = max(min_images, 1 if require_images else 0)
    if len(matches) < required_count:
        errors.append(f"expected at least {required_count} image reference(s), found {len(matches)}")

    for match in matches:
        raw_path = match.group("path")
        resolved = local_image_path(article, raw_path)
        if resolved is None:
            errors.append(f"remote image is not a verified local asset: {raw_path}")
        elif not resolved.is_file():
            errors.append(f"missing image asset: {raw_path} -> {resolved}")

        if require_captions:
            line_number = text.count("\n", 0, match.start())
            following = [line.strip() for line in lines[line_number + 1 : line_number + 7] if line.strip()]
            caption_found = any(line.startswith("*图") and line.endswith("*") for line in following)
            if not caption_found:
                errors.append(f"line {line_number + 1}: image lacks an immediate italic Chinese caption")

    return errors


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Validate image assets in a science-paper-narrator article.")
    parser.add_argument("article", type=Path)
    parser.add_argument("--require-images", action="store_true")
    parser.add_argument("--require-captions", action="store_true")
    parser.add_argument("--min-images", type=int, default=0)
    args = parser.parse_args()

    article = args.article.resolve()
    if not article.is_file():
        print(f"article not found: {article}", file=sys.stderr)
        return 2

    errors = validate(article, args.require_images, args.require_captions, args.min_images)
    if errors:
        print(f"FAIL: {article}")
        for error in errors:
            print(f"- {error}")
        return 1

    count = len(IMAGE_RE.findall(article.read_text(encoding="utf-8")))
    print(f"PASS: {article} ({count} verified image reference(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
