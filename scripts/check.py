"""Single constraint gate. Exit 0 + CHECK OK, else list violations. Run before every commit."""
import json, re, sqlite3, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ERRS = []
def err(m): ERRS.append(m)

SRC_EXT = {".py", ".sh"}
DATA_EXT = {".json", ".toml", ".lock"}
ALLOWED_MD = {"AGENTS.md", "GRAPH_GUIDE.md", "PROTOCOL.md"}
REQUIRED_MODEL = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
NONASCII = re.compile(r"[^\x00-\x7f]")
HEDGE = re.compile(r"\b(maybe|might|perhaps|possibly)\b", re.I)
HYPO_PATTERNS = [r"\bIF\b", r"\bIN\b", r"\bTHEN\b", r"\bBECAUSE\b", r"DISPROVED\s+IF"]
FALSIFIER_RE = re.compile(r"DISPROVED\s+IF(.*)$", re.S | re.I)
SWITCH_RE = re.compile(r"[a-z][a-z0-9_]+")
MODEL_ID_RE = re.compile(r"^N(\d{4})$")
ID_STATUS = {"open", "done", "failed", "timeout"}

# 1. extension whitelist: docs only at root, only the three whitelisted ones
for f in sorted(ROOT.rglob("*")):
    if ".git" in f.parts or f.is_dir() or f.name in ("neug.db", ".gitignore", "LICENSE"):
        continue
    rel = f.relative_to(ROOT)
    if f.suffix == ".md":
        if len(rel.parts) != 1 or f.name not in ALLOWED_MD:
            err(f"language: doc {rel} not allowed (only {sorted(ALLOWED_MD)} at repo root)")
    elif f.suffix not in SRC_EXT and f.suffix not in DATA_EXT:
        err(f"language: forbidden extension {rel} (source: python+bash, data: json/toml/lock)")

# 2. docs: pure ASCII English (catches CJK, accents, curly quotes, arrows)
for name in sorted(ALLOWED_MD):
    f = ROOT / name
    if not f.exists():
        err(f"doc: {name} missing")
        continue
    if NONASCII.search(f.read_text()):
        err(f"english: {name} contains non-ASCII (docs must be pure ASCII English)")

# 3. code: no CJK in .py/.sh (English prose everywhere)
for f in sorted(ROOT.rglob("*")):
    if ".git" in f.parts or f.is_dir() or f.suffix not in SRC_EXT:
        continue
    txt = f.read_text(errors="replace")
    if CJK.search(txt):
        err(f"language: CJK text in {f.relative_to(ROOT)} (code/comments must be English)")

# 4. required backbone must exist in model code, not just in docs
model_py = [p for p in (ROOT / "models").rglob("*.py")] if (ROOT / "models").exists() else []
if not any(REQUIRED_MODEL in p.read_text() for p in model_py):
    err(f"model: required backbone {REQUIRED_MODEL} not found in models/**.py")

args_py = ROOT / "models" / "cdino" / "tf_pipeline" / "args.py"
args_text = args_py.read_text() if args_py.exists() else ""

# 5. NeuG integrity
db = ROOT / "neug.db"
if not db.exists():
    err("neug: neug.db missing (run python3 scripts/seed_db.py)")
else:
    c = sqlite3.connect(db)
    try:
        ctr = {k: int(c.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()[0])
               for k in ("model_ctr", "hypo_ctr", "ban_ctr")}
        nodes = [(r[0], r[1], json.loads(r[2])) for r in
                 c.execute("SELECT id,type,data FROM nodes")]
        edges = set(c.execute("SELECT src,rel,dst FROM edges").fetchall())

        # 5a. ID format + sequence: gaps or invented IDs fail
        expected = {
            "model": set(range(29, ctr["model_ctr"] + 1)),
            "hypo": set(range(1, ctr["hypo_ctr"] + 1)),
            "ban": set(range(1, ctr["ban_ctr"] + 1)),
        }
        prefixes = {"model": "N", "hypo": "H", "ban": "B"}
        actual = {"model": set(), "hypo": set(), "ban": set()}
        for nid, ntype, _d in nodes:
            m = re.fullmatch(r"([NH])(\d{4})", nid) if ntype in ("model", "hypo") else \
                re.fullmatch(r"B(\d{4})", nid)
            if ntype == "ban":
                if not m:
                    err(f"neug: bad ban id {nid}")
                else:
                    actual["ban"].add(int(m.group(1)))
                continue
            if not m or m.group(1) != prefixes[ntype]:
                err(f"neug: bad id format {nid} for type {ntype}")
            else:
                actual[ntype].add(int(m.group(2)))
        for t in ("model", "hypo", "ban"):
            if actual[t] != expected[t]:
                err(f"neug: {t} ids out of sequence (missing={sorted(expected[t]-actual[t])}"
                    f" invented={sorted(actual[t]-expected[t])})")

        # 5b. hypothesis format + hedging + falsifier number
        for nid, ntype, d in nodes:
            if ntype != "hypo":
                continue
            t = d.get("text", "")
            pos = []
            for p in HYPO_PATTERNS:
                m = re.search(p, t, re.I)
                pos.append(m.start() if m else -1)
            if -1 in pos or pos != sorted(pos):
                err(f"neug: {nid} hypothesis marker order broken")
            if HEDGE.search(t):
                err(f"neug: {nid} hypothesis contains hedging word")
            fm = FALSIFIER_RE.search(t)
            if not fm or not re.search(r"\d", fm.group(1)):
                err(f"neug: {nid} falsifier lacks a number")

        # 5c. model node constraints (64M cap, switch=flag, status, lineage edges)
        # 5d. edge endpoints must all exist (ghost ids from hand-injected edges)
        node_ids = {n[0] for n in nodes}
        for s_, rel_, d_ in edges:
            if s_ not in node_ids or d_ not in node_ids:
                err(f"neug: edge ({s_},{rel_},{d_}) references unknown node")
        for nid, ntype, d in nodes:
            if ntype not in ("model", "hypo", "ban"):
                err(f"neug: unknown node type {ntype!r} on {nid}")
            if ntype != "model":
                continue
            if not MODEL_ID_RE.fullmatch(nid):
                err(f"neug: model id {nid} does not match Nxxxx")
            if d.get("status") not in ID_STATUS:
                err(f"neug: {nid} bad status {d.get('status')!r}")
            if int(d.get("params") or 0) > 64_000_000:
                err(f"neug: {nid} params {d.get('params')} exceeds 64M cap")
            sw = d.get("switch")
            if sw is not None:
                if not SWITCH_RE.fullmatch(sw):
                    err(f"neug: {nid} switch {sw!r} invalid")
                elif args_text and f"--{sw}" not in args_text:
                    err(f"neug: {nid} switch --{sw} not declared in args.py")
            if d.get("status") == "done" and not isinstance(d.get("subset_mae"), (int, float)):
                err(f"neug: {nid} status=done but subset_mae missing")
            parent = d.get("parent")
            if parent:
                if (nid, "CHILD_OF", parent) not in edges:
                    err(f"neug: {nid} missing CHILD_OF edge to {parent}")
                if not any(e[0] == nid and e[1] == "TESTS" for e in edges):
                    err(f"neug: {nid} has no TESTS edge (new-model creates it; do not delete)")
            for e in [e for e in edges if e[0] == nid]:
                if e[2] not in {n[0] for n in nodes}:
                    err(f"neug: edge {e} points to unknown node")
    except Exception as e:
        err(f"neug: unreadable ({e})")

# 6. protocol pins
prot = (ROOT / "PROTOCOL.md").read_text() if (ROOT / "PROTOCOL.md").exists() else ""
for needle, label in (("20260830", "fixed seed 20260830"), (REQUIRED_MODEL, "required model"),
                       ("64M", "64M cap"), ("subset286", "subset286 gate")):
    if needle not in prot:
        err(f"protocol: PROTOCOL.md missing {label}")

if ERRS:
    print("CHECK FAIL:")
    for e in ERRS:
        print(" -", e)
    sys.exit(1)
print("CHECK OK")
