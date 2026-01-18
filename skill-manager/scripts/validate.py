#!/usr/bin/env python3
"""
Validate skills against skill-creator guidelines.
"""

import hashlib
import re
from pathlib import Path
from typing import Optional

from .utils import expand_path, load_config, get_skills_from_dir


# Validation rules based on skill-creator guidelines
VALIDATION_RULES = {
    "name": {
        "max_length": 64,
        "pattern": r"^[a-z0-9-]+$",
        "reserved_words": ["anthropic", "claude"],
    },
    "description": {
        "max_length": 1024,
        "forbidden_patterns": [r"\bI will\b", r"\bI can\b", r"\bI am\b"],
        "recommended_patterns": ["when", "use"],
    },
    "file": {
        "max_lines": 500,
        "required_fields": ["name", "description"],
        "unsafe_patterns": [r"!\`"],
    },
}


def _compute_dir_hash(path: Path) -> str:
    """Compute hash of all files in directory."""
    hasher = hashlib.md5()
    
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file():
            hasher.update(str(file_path.relative_to(path)).encode())
            hasher.update(file_path.read_bytes())
    
    return hasher.hexdigest()


def _parse_yaml_frontmatter(content: str) -> dict:
    """Parse YAML frontmatter from SKILL.md."""
    if not content.startswith("---"):
        return {}
    
    try:
        end = content.index("---", 3)
        frontmatter = content[3:end].strip()
        
        result = {}
        for line in frontmatter.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                result[key.strip()] = value.strip()
        
        return result
    except ValueError:
        return {}


def validate_skill(skill_name: str, skill_path: Path) -> dict:
    """Validate a single skill."""
    result = {
        "skill": skill_name,
        "path": str(skill_path),
        "valid": True,
        "passes": [],
        "warnings": [],
        "failures": [],
    }
    
    skill_md = skill_path / "SKILL.md"
    
    # Check SKILL.md exists
    if not skill_md.exists():
        result["failures"].append("SKILL.md not found")
        result["valid"] = False
        return result
    
    content = skill_md.read_text()
    lines = content.split("\n")
    frontmatter = _parse_yaml_frontmatter(content)
    
    # Validate name
    rules = VALIDATION_RULES["name"]
    if len(skill_name) <= rules["max_length"]:
        result["passes"].append(f'name: "{skill_name}" ({len(skill_name)} chars)')
    else:
        result["failures"].append(f'name: "{skill_name}" exceeds {rules["max_length"]} chars')
        result["valid"] = False
    
    if re.match(rules["pattern"], skill_name):
        result["passes"].append("name: valid format (kebab-case)")
    else:
        result["failures"].append("name: must be lowercase kebab-case (a-z, 0-9, -)")
        result["valid"] = False
    
    for word in rules["reserved_words"]:
        if word in skill_name.lower():
            result["warnings"].append(f'name: contains reserved word "{word}"')
    
    # Validate frontmatter
    for field in VALIDATION_RULES["file"]["required_fields"]:
        if field in frontmatter:
            result["passes"].append(f"YAML: has required field '{field}'")
        else:
            result["failures"].append(f"YAML: missing required field '{field}'")
            result["valid"] = False
    
    # Validate description
    desc = frontmatter.get("description", "")
    if desc:
        if len(desc) <= VALIDATION_RULES["description"]["max_length"]:
            result["passes"].append(f"description: {len(desc)} chars")
        else:
            result["warnings"].append(f"description: {len(desc)} chars (> {VALIDATION_RULES['description']['max_length']} recommended)")
        
        # Check for first-person
        for pattern in VALIDATION_RULES["description"]["forbidden_patterns"]:
            if re.search(pattern, desc, re.IGNORECASE):
                result["warnings"].append(f"description: uses first-person ('{pattern}'), prefer third-person")
        
        # Check for triggers
        has_triggers = any(re.search(p, desc, re.IGNORECASE) for p in VALIDATION_RULES["description"]["recommended_patterns"])
        if has_triggers:
            result["passes"].append("description: has trigger keywords")
        else:
            result["warnings"].append("description: consider adding trigger keywords (when, use)")
    else:
        result["failures"].append("description: empty")
        result["valid"] = False
    
    # Validate file size
    if len(lines) <= VALIDATION_RULES["file"]["max_lines"]:
        result["passes"].append(f"SKILL.md: {len(lines)} lines")
    else:
        result["warnings"].append(f"SKILL.md: {len(lines)} lines (> {VALIDATION_RULES['file']['max_lines']} recommended)")
    
    # Check for unsafe patterns
    for pattern in VALIDATION_RULES["file"]["unsafe_patterns"]:
        if re.search(pattern, content):
            result["warnings"].append(f"SKILL.md: contains potentially unsafe pattern")
    
    return result


def verify_sync(skill_name: str, source_dir: Path, config: dict) -> dict:
    """Verify skill is synced to all IDE targets."""
    result = {"skill": skill_name, "synced": [], "missing": [], "mismatched": []}
    
    source_path = source_dir / skill_name
    if not source_path.exists():
        result["missing"].append(f"source: {source_path}")
        return result
    
    source_hash = _compute_dir_hash(source_path)
    
    for ide_name in config.get("enabled_ides", []):
        target_path = config.get("targets", {}).get(ide_name)
        if not target_path:
            continue
        
        target_dir = expand_path(target_path) / skill_name
        
        if not target_dir.exists():
            result["missing"].append(f"{ide_name}: {target_dir}")
        elif target_dir.is_symlink():
            # Symlinks are considered synced if they point to source
            try:
                if target_dir.resolve() == source_path.resolve():
                    result["synced"].append(f"{ide_name}: symlink")
                else:
                    result["mismatched"].append(f"{ide_name}: wrong symlink target")
            except Exception:
                result["mismatched"].append(f"{ide_name}: broken symlink")
        else:
            target_hash = _compute_dir_hash(target_dir)
            if source_hash == target_hash:
                result["synced"].append(f"{ide_name}: hash match")
            else:
                result["mismatched"].append(f"{ide_name}: hash mismatch")
    
    return result


def cmd_validate(skill_name: Optional[str] = None, scope: str = "global",
                project_dir: Optional[Path] = None, verify: bool = True,
                json_output: bool = False) -> dict:
    """Validate skills."""
    config = load_config()
    exclude = config.get("exclude_skills", [])
    
    result = {"validated": 0, "passed": 0, "warnings": 0, "failures": 0, "skills": []}
    
    if scope in ("global", "all"):
        source_dir = expand_path(config.get("source_dir", "~/.ai-skills"))
        
        if skill_name:
            skills = [skill_name]
        else:
            skills = get_skills_from_dir(source_dir, exclude)
        
        print(f"Validating {len(skills)} skill(s) from {source_dir}\n")
        print("Reference: references/skill-creator/SKILL.md\n")
        
        for skill in skills:
            skill_path = source_dir / skill
            print(f"Validating: {skill}")
            print("=" * 60)
            
            # Validate skill structure
            v_result = validate_skill(skill, skill_path)
            result["validated"] += 1
            
            for p in v_result["passes"]:
                print(f"[PASS] {p}")
            for w in v_result["warnings"]:
                print(f"[WARN] {w}")
                result["warnings"] += 1
            for f in v_result["failures"]:
                print(f"[FAIL] {f}")
                result["failures"] += 1
            
            if v_result["valid"]:
                result["passed"] += 1
            
            # Verify sync
            if verify:
                print("\nSync Verification:")
                sync_result = verify_sync(skill, source_dir, config)
                
                for s in sync_result["synced"]:
                    print(f"[PASS] {s}")
                for m in sync_result["missing"]:
                    print(f"[FAIL] {m} (missing)")
                    result["failures"] += 1
                for mm in sync_result["mismatched"]:
                    print(f"[FAIL] {mm}")
                    result["failures"] += 1
            
            result["skills"].append(v_result)
            print()
    
    # Summary
    print("=" * 60)
    if result["failures"] == 0 and result["warnings"] == 0:
        print(f"Result: All {result['validated']} skill(s) passed ✓")
    else:
        print(f"Result: {result['failures']} failure(s), {result['warnings']} warning(s)")
        if result["failures"] > 0:
            print("\nTo fix sync issues: skills sync --force")
    
    if json_output:
        from .utils import print_json
        print_json(result)
    
    return result
