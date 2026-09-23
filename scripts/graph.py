"""NeuG: minimal SQLite graph. Node types: model|hypo|ban. Edges: CHILD_OF|TESTS|SUPPORTS|CONTRADICTS."""
import argparse, json, sqlite3, sys, os, re
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "neug.db"
ETA, CONFIRM, REFUTE = 0.20, 0.75, 0.25
HYPE_RE = re.compile(r"IF\b.*\bIN\b.*\bTHEN\b.*\bBECAUSE\b.*\bDISPROVED\s+IF\b", re.S)
HEDGE = re.compile(r"\b(maybe|might|perhaps|possibly)\b", re.I)

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

def cmd_live(_):
    db = con()
    rows = [(r[0], json.loads(r[1])) for r in db.execute("SELECT id,data FROM nodes WHERE type='model'")]
    done = [(i, d) for i, d in rows if d.get("status") == "done" and d.get("subset_mae") is not None]
    if not done:
        print("no live parent"); return
    pid, p = min(done, key=lambda x: x[1]["subset_mae"])
    bans = [json.loads(r[0])["rule"] for r in db.execute("SELECT data FROM nodes WHERE type='ban'").fetchall()]
    print(json.dumps({"live_parent": pid, "subset_mae": p["subset_mae"], "full_mae": p.get("full_mae"),
                       "next_bar_subset": round(p["subset_mae"] - 0.30, 4), "bans": bans}, indent=1))

def cmd_untested(a):
    db = con()
    tested = {r[0] for r in db.execute(
        "SELECT dst FROM edges WHERE rel='TESTS' AND src IN (SELECT id FROM nodes WHERE type='model')")}
    out = []
    for hid, data in db.execute("SELECT id,data FROM nodes WHERE type='hypo'"):
        d = json.loads(data)
        if hid not in tested and d.get("confidence", 0.5) < CONFIRM and d.get("confidence", 0.5) > REFUTE:
            out.append({"id": hid, "text": d["text"], "confidence": d.get("confidence", 0.5)})
    print(json.dumps(out, indent=1))

def cmd_bans(_):
    db = con()
    print(json.dumps([{"id": r[0], **json.loads(r[1])} for r in
                      db.execute("SELECT id,data FROM nodes WHERE type='ban'")], indent=1))

def cmd_new_hypo(a):
    t = a.text.strip()
    if not HYPE_RE.search(t.upper().replace("DISPROVED IF", "DISPROVED IF")):
        sys.exit("FAIL: hypothesis must match IF..IN..THEN..BECAUSE..DISPROVED IF in order")
    if HEDGE.search(t):
        sys.exit("FAIL: hedging word (maybe/might/perhaps/possibly) forbidden")
    if not re.search(r"\d", t.split("DISPROVED IF", 1)[1]):
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
    if not re.fullmatch(r"use_[a-z0-9_]+", a.switch):
        sys.exit("FAIL: switch must match use_<lowercase>")
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
    mrow = db.execute("SELECT data FROM nodes WHERE id=? AND type='model'", (a.model,)).fetchone()
    if mrow:
        md = json.loads(mrow[0])
        md.update({"subset_mae": a.mae, "status": "done", "note": a.note})
        db.execute("UPDATE nodes SET data=? WHERE id=?", (json.dumps(md), a.model))
    db.execute("INSERT OR IGNORE INTO edges VALUES(?,?,?)", (a.model, rel, a.hypo))
    db.commit()
    print(f"{a.hypo} confidence -> {d['confidence']}")

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
e.add_argument("--mae", type=float, required=True); e.add_argument("--type", default="neutral",
    choices=["support", "contradict", "neutral"]); e.add_argument("--weight", type=float, default=0.85)
e.add_argument("--note", default="")
b = s.add_parser("ban"); b.add_argument("--rule", required=True); b.add_argument("--ref", default="")
a = p.parse_args()
{"seed": cmd_seed, "live-parent": cmd_live, "untested": cmd_untested, "bans": cmd_bans,
 "new-hypo": cmd_new_hypo, "new-model": cmd_new_model, "evidence": cmd_evidence,
 "ban": cmd_ban, "tree": cmd_tree}[a.c](a)
