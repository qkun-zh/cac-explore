"""Single constraint gate. Exit 0 + CHECK OK, else list violations. Run before every commit."""
import re, sys, sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ERRS = []
def err(m): ERRS.append(m)

ALLOWED = {".py", ".sh", ".toml", ".json", ".lock", ".gitignore"}
ALLOWED_MD = {"AGENTS.md", "GRAPH_GUIDE.md", "PROTOCOL.md"}
REQUIRED_MODEL = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\uf900-\ufaff]")

# 1. language whitelist
for f in sorted(ROOT.rglob("*")):
    if ".git" in f.parts or f.is_dir() or f.name in ("neug.db", ".gitignore", "LICENSE"):
        continue
    rel = f.relative_to(ROOT)
    if f.suffix == ".md":
        if f.name not in ALLOWED_MD:
            err(f"language: unexpected doc {rel} (only {sorted(ALLOWED_MD)} allowed)")
    elif f.suffix not in ALLOWED and f.name != ".gitignore":
        err(f"language: forbidden extension {rel} (only python+bash allowed)")

# 2. English-only docs
for name in ALLOWED_MD:
    f = ROOT / name
    if f.exists() and CJK.search(f.read_text()):
        err(f"english: {name} contains non-English prose")

# 3. NeuG integrity (fresh-zero rule: IDs sequential, no direct confidence edits detectable via schema)
db = ROOT / "neug.db"
if not db.exists():
    err("neug: neug.db missing (run python scripts/seed_db.py)")
else:
    c = sqlite3.connect(db)
    try:
        types = {r[0] for r in c.execute("SELECT DISTINCT type FROM nodes")}
        if not types.issuperset({"model", "ban"}):
            err(f"neug: node types incomplete: {types}")
        import json
        for hid, data in c.execute("SELECT id,data FROM nodes WHERE type='hypo'"):
            d = json.loads(data)
            t = d.get("text", "")
            up = t.upper()
            order = [up.find(k) for k in ("IF", "IN", "THEN", "BECAUSE", "DISPROVED IF")]
            if any(x < 0 for x in order) or order != sorted(order):
                err(f"neug: {hid} hypothesis marker order broken")
            if not re.search(r"\d", t.split("DISPROVED IF", 1)[-1] if "DISPROVED IF" in up else ""):
                err(f"neug: {hid} falsifier lacks a number")
    except Exception as e:
        err(f"neug: unreadable ({e})")

# 4. seed + required-model + size cap pinned in PROTOCOL.md
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
