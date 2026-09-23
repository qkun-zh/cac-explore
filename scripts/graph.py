"""NeuG: minimal SQLite graph. Node types: model|hypo|ban. Edges: CHILD_OF|TESTS|SUPPORTS|CONTRADICTS."""
import argparse, json, sqlite3, sys, re
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "neug.db"
ETA, CONFIRM, REFUTE = 0.20, 0.75, 0.25
HYPE_RE = re.compile(r"IF\b.*\bIN\b.*\bTHEN\b.*\bBECAUSE\b.*\bDISPROVED\s+IF\b", re.S | re.I)
HEDGE = re.compile(r"\b(maybe|might|perhaps|possibly)\b", re.I)
FALSIFIER_RE = re.compile(r"DISPROVED\s+IF(.*)$", re.S | re.I)
SWITCH_RE = re.compile(r"[a-z][a-z0-9_]+")

def con():
    db = sqlite3.connect(DB)
    db.execute("CREATE TABLE IF NOT EXISTS nodes(id TEXT PRIMARY KEY, type TEXT, data TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS edges(src TEXT, rel TEXT, dst TEXT, UNIQUE(src,rel,dst))")
    db.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
    for k, v in (("model_ctr", "29"), ("hypo_ctr", "0"), ("ban_ctr", "0")):
        db.execute("INSERT OR IGNORE INTO meta(k,v) VALUES(?,?)", (k, v))
    return db

def nxt(db, k, prefix, width=4):
    n = int(db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()[0]) + 1
    db.execute("UPDATE meta SET v=? WHERE k=?", (str(n), k))
    return f"{prefix}{n:0{width}d}"

def tok(s):
    return re.findall(r"[a-z]{3,}", s.lower())

def novelty(db, text):
    words = [tok(n[0]) for n in db.execute("SELECT data FROM nodes WHERE type='hypo'").fetchall()]
    ws, best = set(tok(text)), 0.0
    for w in words:
        w = set(w)
        if ws and w:
            best = max(best, len(ws & w) / len(ws | w))
    return best

def cmd_seed(_):
    db = con()
    if db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]:
        print("DB already seeded"); return
    m = {"id": "N0029", "parent": None, "switch": None, "params": 28000000,
         "subset_mae": 25.818, "full_mae": 26.485, "status": "done",
         "note": "TF champion: CountingDINO paper-best + fs=0.5 + MWEx + ADEI t1.5d1m8, ConvNeXt-T required"}
    db.execute("INSERT INTO nodes VALUES(?,?,?)", ("N0029", "model", json.dumps(m)))
    bans = [
        "never: retune closed ADEI/fs/gate constants as free hyper-parameters",
        "never: fit thresholds or scales on val GT for board numbers",
        "never: report subset286 MAE as a champion row (champion = full val 1286 only)",
        "never: drop facebook/dinov3-convnext-tiny-pretrain-lvd1689m from the inference graph",
        "never: exceed 64M total stack without user approval",
        "never: supervised head training or gradient updates (training-free track only)",
    ]
    for b in bans:
        bid = nxt(db, "ban_ctr", "B")
        db.execute("INSERT INTO nodes VALUES(?,?,?)", (bid, "ban", json.dumps({"rule": b})))
    db.commit()
    print("seeded N0029 + 6 bans")

def _depth(models, mid, limit=100):
    d, cur, seen = 0, mid, set()
    while cur not in seen and d < limit:
        seen.add(cur)
        parent = (models.get(cur) or {}).get("parent")
        if not parent:
            break
        cur, d = parent, d + 1
    return d

def cmd_live(_):
    db = con()
    rows = [(r[0], json.loads(r[1])) for r in db.execute("SELECT id,data FROM nodes WHERE type='model'")]
    done = [(i, d) for i, d in rows if d.get("status") == "done" and d.get("subset_mae") is not None]
    if not done:
        print("no live parent"); sys.exit(1)
    models = dict(rows)
    # tie-break at 3dp board precision: prefer shallower node (no gain => stay on parent)
    pid, p = min(done, key=lambda x: (round(x[1]["subset_mae"], 3), _depth(models, x[0])))
    bans = [json.loads(r[0])["rule"] for r in db.execute("SELECT data FROM nodes WHERE type='ban'").fetchall()]
    print(json.dumps({"live_parent": pid, "subset_mae": p["subset_mae"], "full_mae": p.get("full_mae"),
                       "next_bar_subset": round(p["subset_mae"] - 0.30, 4), "bans": bans}, indent=1))

def _descendants(db, root):
    kids = {}
    for src, dst in db.execute("SELECT src,dst FROM edges WHERE rel='CHILD_OF'"):
        kids.setdefault(dst, []).append(src)
    out, stack = set(), [root]
    while stack:
        n = stack.pop()
        if n in out:
            continue
        out.add(n)
        stack.extend(kids.get(n, []))
    return out

def cmd_untested(a):
    db = con()
    if a.parent:
        if not db.execute("SELECT 1 FROM nodes WHERE id=? AND type='model'",
                          (a.parent,)).fetchone():
            sys.exit(f"FAIL: unknown parent {a.parent}")
        scope = _descendants(db, a.parent)
    else:
        scope = None
    tested = {r[0] for r in db.execute(
        "SELECT dst FROM edges WHERE rel='TESTS' AND src IN (SELECT id FROM nodes WHERE type='model')")}
    if scope is not None:
        tested = {r[0] for r in db.execute(
            "SELECT dst FROM edges WHERE rel='TESTS' AND src IN (%s)" % ",".join("?" * len(scope)),
            tuple(scope))}
    out = []
    for hid, data in db.execute("SELECT id,data FROM nodes WHERE type='hypo'"):
        d = json.loads(data)
        c = d.get("confidence", 0.5)
        if hid not in tested and REFUTE < c < CONFIRM:
            out.append({"id": hid, "text": d["text"], "confidence": c})
    print(json.dumps(out, indent=1))

def cmd_bans(_):
    db = con()
    print(json.dumps([{"id": r[0], **json.loads(r[1])} for r in
                      db.execute("SELECT id,data FROM nodes WHERE type='ban'")], indent=1))

def cmd_new_hypo(a):
    t = a.text.strip()
    if not HYPE_RE.search(t):
        sys.exit("FAIL: hypothesis must match IF..IN..THEN..BECAUSE..DISPROVED IF in order (case-insensitive)")
    if HEDGE.search(t):
        sys.exit("FAIL: hedging word (maybe/might/perhaps/possibly) forbidden")
    m = FALSIFIER_RE.search(t)
    if not m or not re.search(r"\d", m.group(1)):
        sys.exit("FAIL: DISPROVED IF clause must contain a number/comparison")
    db = con()
    if novelty(db, t) >= 0.82:
        sys.exit("FAIL: novelty gate (jaccard >= 0.82 duplicate)")
    hid = nxt(db, "hypo_ctr", "H")
    db.execute("INSERT INTO nodes VALUES(?,?,?)", (hid, "hypo", json.dumps({"text": t, "confidence": 0.5})))
    db.commit()
    print(hid)

def cmd_new_model(a):
    db = con()
    if not db.execute("SELECT 1 FROM nodes WHERE id=? AND type='model'", (a.parent,)).fetchone():
        sys.exit(f"FAIL: unknown parent {a.parent}")
    if not db.execute("SELECT 1 FROM nodes WHERE id=? AND type='hypo'", (a.hypo,)).fetchone():
        sys.exit(f"FAIL: unknown hypo {a.hypo}")
    if not SWITCH_RE.fullmatch(a.switch):
        sys.exit("FAIL: switch must match [a-z][a-z0-9_]+ (it IS the CLI flag name, passed as --<switch>)")
    args_py = DB.parent / "models" / "cdino" / "tf_pipeline" / "args.py"
    if args_py.exists() and f"--{a.switch}" not in args_py.read_text():
        sys.exit(f"FAIL: flag --{a.switch} not declared in models/cdino/tf_pipeline/args.py")
    mid = nxt(db, "model_ctr", "N")
    db.execute("INSERT INTO nodes VALUES(?,?,?)", (mid, "model", json.dumps(
        {"id": mid, "parent": a.parent, "switch": a.switch, "params": a.params, "status": "open"})))
    db.execute("INSERT INTO edges VALUES(?,?,?)", (mid, "CHILD_OF", a.parent))
    db.execute("INSERT INTO edges VALUES(?,?,?)", (mid, "TESTS", a.hypo))
    db.commit()
    print(mid)

def cmd_evidence(a):
    db = con()
    row = db.execute("SELECT data FROM nodes WHERE id=? AND type='hypo'", (a.hypo,)).fetchone()
    if not row:
        sys.exit(f"FAIL: unknown hypo {a.hypo}")
    mrow = db.execute("SELECT data FROM nodes WHERE id=? AND type='model'", (a.model,)).fetchone()
    if not mrow:
        sys.exit(f"FAIL: unknown model {a.model}")
    if a.mae <= 0:
        sys.exit("FAIL: --mae must be > 0")
    d = json.loads(row[0])
    c = d.get("confidence", 0.5)
    if a.type == "support":
        c += ETA * a.weight * (1 - c)
        rel = "SUPPORTS"
    elif a.type == "contradict":
        c -= ETA * a.weight * c
        rel = "CONTRADICTS"
    else:
        rel = "TESTS"
    d["confidence"] = round(c, 4)
    db.execute("UPDATE nodes SET data=? WHERE id=?", (json.dumps(d), a.hypo))
    md = json.loads(mrow[0])
    md.update({"subset_mae": a.mae, "status": "done", "note": a.note})
    if a.full_mae is not None:
        md["full_mae"] = a.full_mae
    db.execute("UPDATE nodes SET data=? WHERE id=?", (json.dumps(md), a.model))
    db.execute("INSERT OR IGNORE INTO edges VALUES(?,?,?)", (a.model, rel, a.hypo))
    db.commit()
    print(f"{a.hypo} confidence -> {d['confidence']} | {a.model} subset_mae={a.mae} type={a.type}")

def cmd_ban(a):
    db = con()
    bid = nxt(db, "ban_ctr", "B")
    db.execute("INSERT INTO nodes VALUES(?,?,?)", (bid, "ban", json.dumps({"rule": a.rule, "ref": a.ref})))
    db.commit()
    print(bid)

def cmd_tree(_):
    db = con()
    models = {r[0]: json.loads(r[1]) for r in db.execute("SELECT id,data FROM nodes WHERE type='model'")}
    kids = {}
    for m, d in models.items():
        kids.setdefault(d.get("parent") or ".", []).append(m)
    def show(p, dep):
        for m in sorted(kids.get(p, [])):
            d = models[m]
            print("  " * dep + f"{m} switch={d.get('switch')} mae={d.get('subset_mae')} {d.get('status')}")
            show(m, dep + 1)
    show(".", 0)

p = argparse.ArgumentParser()
s = p.add_subparsers(dest="c", required=True)
s.add_parser("seed"); s.add_parser("live-parent"); s.add_parser("bans"); s.add_parser("tree")
u = s.add_parser("untested"); u.add_argument("--parent", default="")
h = s.add_parser("new-hypo"); h.add_argument("--text", required=True)
m = s.add_parser("new-model"); m.add_argument("--parent", required=True); m.add_argument("--hypo", required=True)
m.add_argument("--switch", required=True); m.add_argument("--params", type=int, default=0)
e = s.add_parser("evidence"); e.add_argument("--hypo", required=True); e.add_argument("--model", required=True)
e.add_argument("--mae", type=float, required=True); e.add_argument("--type", required=True,
    choices=["support", "contradict", "neutral"]); e.add_argument("--weight", type=float, default=0.85)
e.add_argument("--note", default=""); e.add_argument("--full-mae", type=float, default=None)
b = s.add_parser("ban"); b.add_argument("--rule", required=True); b.add_argument("--ref", default="")
a = p.parse_args()
{"seed": cmd_seed, "live-parent": cmd_live, "untested": cmd_untested, "bans": cmd_bans,
 "new-hypo": cmd_new_hypo, "new-model": cmd_new_model, "evidence": cmd_evidence,
 "ban": cmd_ban, "tree": cmd_tree}[a.c](a)
