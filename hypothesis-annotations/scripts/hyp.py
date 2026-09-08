#!/usr/bin/env python3
"""Query Hypothes.is annotations (comments) from the command line.

用途: 让 agent 快速读取某个文档上的 Hypothesis 批注/评论。支持三种模式(自动识别):
  - annotation: 传入单条批注链接或 id -> 只读那一条
  - doc:        传入被批注文档的 URL   -> 拉取该文档上所有可见批注(公开 + 用户所在私有组)
  - group:      传入 group 链接或 --mode group --group <pubid> -> 拉取整个组的全部批注

Auth: 从环境变量读取 developer token, 优先 H_TOKEN, 回退 HYPOTHESIS_API_TOKEN.
      默认 group 过滤可用 HYPOTHESIS_GROUP 环境变量(可选)。
Token 优先从 env 读取, 未设置时回退到 ~/.config/hypothesis/token.env, 从不打印明文。

Endpoints used (https://api.hypothes.is/api):
  GET /annotations/{id}                     单条读取
  GET /search?uri=<enc>[&group=<pubid>]     按文档 URI 搜索
  GET /search?group=<pubid>&search_after=.. 整组分页拉取
  GET /profile                              (--whoami) 验证 token 归属
  GET /profile/groups                       (--groups) 列出用户所在的组
"""
import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request

API_BASE = "https://api.hypothes.is/api"

# Fallback token file (user's own, chmod 600). Agents often run non-interactive
# shells where ~/.bashrc's interactive guard skips exported vars, so reading the
# file directly is the reliable path.
_ENV_FILE = os.path.join(
    os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
    "hypothesis", "token.env")
_WANTED_KEYS = ("H_TOKEN", "HYPOTHESIS_API_TOKEN", "HYPOTHESIS_GROUP")


def hydrate_env_from_file():
    """Populate missing H_TOKEN/HYPOTHESIS_* from the token.env file (if present).
    Only fills keys not already set in the real environment, so an explicit
    export always wins. Parses simple `export KEY='value'` / `KEY=value` lines."""
    if all(os.environ.get(k) for k in ("H_TOKEN", "HYPOTHESIS_API_TOKEN")):
        return
    try:
        with open(_ENV_FILE, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if val and val[0] in "\"'":
            # quoted value: take what's inside the quotes, ignore any trailing comment
            quote = val[0]
            end = val.find(quote, 1)
            val = val[1:end] if end != -1 else val[1:]
        else:
            # bare value: strip an inline # comment
            val = val.split("#", 1)[0].strip()
        if key in _WANTED_KEYS and not os.environ.get(key):
            os.environ[key] = val


def get_token():
    tok = os.environ.get("H_TOKEN") or os.environ.get("HYPOTHESIS_API_TOKEN")
    if not tok:
        sys.exit(
            "ERROR: no Hypothesis token found.\n"
            f"Looked in env (H_TOKEN / HYPOTHESIS_API_TOKEN) and {_ENV_FILE}.\n"
            "Generate a personal token at https://hypothes.is/account/developer, then either\n"
            "  export H_TOKEN='6879-...'\n"
            f"or write it to {_ENV_FILE} as:  export H_TOKEN='6879-...'\n"
        )
    return tok


def api_get(path, token, params=None):
    """GET a JSON endpoint with Bearer auth. Returns parsed JSON dict."""
    url = API_BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "Authorization": "Bearer " + token,
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        sys.exit(f"ERROR: HTTP {e.code} for {path}\n{body}")
    except urllib.error.URLError as e:
        sys.exit(f"ERROR: cannot reach api.hypothes.is ({e.reason})")


# --- input parsing -------------------------------------------------------

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{20,25}$")  # flake ids are ~22 url-safe b64 chars


def parse_input(text):
    """Classify a pasted string. Returns (kind, value) where kind in
    {'annotation','group','doc'} and value is the id / pubid / url."""
    s = text.strip()
    # bare annotation id (no scheme, no slash, no dot)
    if _ID_RE.match(s) and "/" not in s and "." not in s:
        return "annotation", s
    # hypothes.is/a/<id>  (share link)
    m = re.search(r"hypothes\.is/a/([A-Za-z0-9_-]+)", s)
    if m:
        return "annotation", m.group(1)
    # hyp.is/<id>/<target-url>  (incontext link)
    m = re.search(r"hyp\.is/([A-Za-z0-9_-]+)", s)
    if m:
        return "annotation", m.group(1)
    # hypothes.is/groups/<pubid>/<slug>  (group page)
    m = re.search(r"hypothes\.is/groups/([A-Za-z0-9_-]+)", s)
    if m:
        return "group", m.group(1)
    # otherwise: a document URL to search for comments on
    return "doc", s


# --- annotation formatting ----------------------------------------------

def quotes_of(ann):
    """Extract highlighted source text (TextQuoteSelector.exact) list."""
    out = []
    for t in ann.get("target", []) or []:
        for sel in t.get("selector", []) or []:
            if sel.get("type") == "TextQuoteSelector":
                out.append(sel.get("exact", ""))
    return out


def incontext_link(ann):
    return ann.get("links", {}).get("incontext") or ("https://hypothes.is/a/" + ann.get("id", ""))


def print_annotation(ann, idx=None):
    head = f"[{idx}] " if idx is not None else ""
    print(f"{head}id       : {ann.get('id')}")
    print(f"    group    : {ann.get('group')}")
    print(f"    author   : {ann.get('user')}")
    print(f"    created  : {ann.get('created')}")
    tags = ann.get("tags") or []
    if tags:
        print(f"    tags     : {', '.join(tags)}")
    body = (ann.get("text") or "").strip()
    print(f"    comment  : {body if body else '(no body / highlight-only)'}")
    for q in quotes_of(ann):
        q1 = q.replace("\n", " ").strip()
        print(f"    highlight: {q1}")
    print(f"    link     : {incontext_link(ann)}")
    print()


# --- search with cursor pagination --------------------------------------

def search_all(token, params, max_rows=1000):
    """Page through /search using search_after cursor (sort=created asc)."""
    rows = []
    p = dict(params)
    p.setdefault("sort", "created")
    p.setdefault("order", "asc")
    p["limit"] = 200
    while len(rows) < max_rows:
        data = api_get("/search", token, p)
        batch = data.get("rows", [])
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 200:
            break
        p["search_after"] = batch[-1].get("created")
    return rows


# --- main ----------------------------------------------------------------

def main():
    hydrate_env_from_file()  # load ~/.config/hypothesis/token.env if env is unset
    ap = argparse.ArgumentParser(
        description="Read Hypothes.is annotations (comments) for a doc, a group, or a single annotation.")
    ap.add_argument("input", nargs="?", default="",
                    help="pasted annotation link/id, document URL, or group link")
    ap.add_argument("--mode", choices=["auto", "annotation", "doc", "group"], default="auto")
    ap.add_argument("--group", default=os.environ.get("HYPOTHESIS_GROUP"),
                    help="group pubid: filter for doc mode, or target for group mode "
                         "(default: HYPOTHESIS_GROUP env if set)")
    ap.add_argument("--limit", type=int, default=1000, help="max annotations to fetch")
    ap.add_argument("--json", action="store_true", help="emit raw JSON rows instead of text")
    ap.add_argument("--whoami", action="store_true", help="print token owner (/profile) and exit")
    ap.add_argument("--groups", action="store_true", help="list the user's groups (/profile/groups) and exit")
    args = ap.parse_args()

    token = get_token()

    if args.whoami:
        prof = api_get("/profile", token)
        print("userid:", prof.get("userid"))
        print("groups:", ", ".join(f"{g.get('id')}({g.get('name')})" for g in prof.get("groups", [])))
        return
    if args.groups:
        for g in api_get("/profile/groups", token):
            print(f"{g.get('id'):10s}  {g.get('type','?'):10s}  {g.get('name')}")
        return

    if not args.input:
        ap.error("need an annotation link/id, document URL, or group link (or use --whoami/--groups)")

    kind, value = (args.mode, args.input) if args.mode != "auto" else parse_input(args.input)
    # in forced modes, still normalise the value from the raw input
    if args.mode == "annotation":
        _, value = ("annotation", parse_input(args.input)[1])
    elif args.mode == "group":
        value = args.group or parse_input(args.input)[1]

    if kind == "annotation":
        ann = api_get("/annotations/" + value, token)
        if args.json:
            print(json.dumps(ann, ensure_ascii=False, indent=2))
        else:
            print_annotation(ann)
        return

    if kind == "group":
        rows = search_all(token, {"group": value}, max_rows=args.limit)
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        else:
            print(f"# group {value}: {len(rows)} annotation(s)\n")
            for i, r in enumerate(rows, 1):
                print_annotation(r, i)
        return

    # kind == "doc": search that document's URI for all visible comments
    params = {"uri": value}
    if args.group:
        params["group"] = args.group
    rows = search_all(token, params, max_rows=args.limit)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        scope = f" (group={args.group})" if args.group else ""
        print(f"# {len(rows)} comment(s) on {value}{scope}\n")
        for i, r in enumerate(rows, 1):
            print_annotation(r, i)


if __name__ == "__main__":
    main()
