from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def resolve_path(manifest: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest.parent / path
    return path.resolve()


def encloses(outer: list[int], inner: list[int]) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "figure"


def build_review_sheet(
    source: Image.Image,
    crop: Image.Image,
    box: list[int],
    required_bounds: list[int] | None,
    destination: Path,
) -> None:
    overlay = source.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(tuple(box), outline="#e53935", width=5)
    if required_bounds:
        draw.rectangle(tuple(required_bounds), outline="#00a65a", width=3)

    preview_width = 900
    scale = min(1.0, preview_width / overlay.width)
    overlay.thumbnail((int(overlay.width * scale), int(overlay.height * scale)))
    crop_preview = crop.convert("RGB").copy()
    crop_preview.thumbnail((preview_width, 850))

    sheet_width = max(overlay.width, crop_preview.width)
    sheet_height = overlay.height + crop_preview.height + 30
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    sheet.paste(overlay, ((sheet_width - overlay.width) // 2, 0))
    sheet.paste(crop_preview, ((sheet_width - crop_preview.width) // 2, overlay.height + 30))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, optimize=True)


def validate(manifest: Path, review_dir: Path | None) -> list[str]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    errors: list[str] = []
    for index, item in enumerate(data.get("figures", []), start=1):
        name = item.get("name", f"figure-{index}")
        source_path = resolve_path(manifest, item["source"])
        crop_path = resolve_path(manifest, item["crop"])
        box = [int(value) for value in item["box"]]
        required = item.get("required_bounds")
        required_bounds = [int(value) for value in required] if required else None

        if not source_path.is_file():
            errors.append(f"{name}: missing source image: {source_path}")
            continue
        if not crop_path.is_file():
            errors.append(f"{name}: missing crop image: {crop_path}")
            continue

        with Image.open(source_path) as source_raw, Image.open(crop_path) as crop_raw:
            source = source_raw.convert("RGB")
            crop = crop_raw.convert("RGB")
            source_bounds = [0, 0, source.width, source.height]
            if not encloses(source_bounds, box):
                errors.append(f"{name}: crop box {box} exceeds source bounds {source_bounds}")
                continue
            if required_bounds and not encloses(box, required_bounds):
                errors.append(
                    f"{name}: crop box {box} does not enclose required content {required_bounds}"
                )

            expected_size = (box[2] - box[0], box[3] - box[1])
            if crop.size != expected_size:
                errors.append(
                    f"{name}: crop size {crop.size} does not match box size {expected_size}"
                )
            else:
                expected = source.crop(tuple(box))
                if expected.tobytes() != crop.tobytes():
                    errors.append(f"{name}: crop pixels do not match the declared source box")

            if review_dir:
                build_review_sheet(
                    source,
                    crop,
                    box,
                    required_bounds,
                    review_dir / f"{slugify(name)}-review.png",
                )
    if not data.get("figures"):
        errors.append("manifest contains no figures")
    return errors


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Validate declared figure crops against rendered source pages."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--review-dir", type=Path)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    if not manifest.is_file():
        print(f"manifest not found: {manifest}", file=sys.stderr)
        return 2

    review_dir = args.review_dir.resolve() if args.review_dir else None
    errors = validate(manifest, review_dir)
    if errors:
        print(f"FAIL: {manifest}")
        for error in errors:
            print(f"- {error}")
        return 1
    count = len(json.loads(manifest.read_text(encoding="utf-8"))["figures"])
    print(f"PASS: {manifest} ({count} source-verified crop(s))")
    if review_dir:
        print(f"review sheets: {review_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
