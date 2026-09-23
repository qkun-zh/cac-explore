"""NeuG: minimal SQLite graph. Node types: model|hypo|ban. Edges: CHILD_OF|TESTS|SUPPORTS|CONTRADICTS."""
import argparse, ast, json, sqlite3, sys, re
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "neug.db"
CF_PATH = Path(__file__).resolve().parent / "closed_families.json"
ETA, CONFIRM, REFUTE = 0.20, 0.75, 0.25
BAR_STEP = 0.30
HYPE_RE = re.compile(r"IF\b.*\bIN\b.*\bTHEN\b.*\bBECAUSE\b.*\bDISPROVED\s+IF\b", re.S | re.I)
HEDGE = re.compile(r"\b(maybe|might|perhaps|possibly)\b", re.I)
FALSIFIER_RE = re.compile(r"DISPROVED\s+IF(.*)$", re.S | re.I)
SWITCH_RE = re.compile(r"[a-z][a-z0-9_]+")
EXT_RE = re.compile(r"EXTENDED_METRICS\s+(\{.*?\})")
METRICS_RE = re.compile(r"\{[^{}]*'MAE':\s*([0-9.eE+-]+)[^{}]*'n_images':\s*(\d+)[^{}]*\}")
METRICS_RE2 = re.compile(r"\{[^{}]*'n_images':\s*(\d+)[^{}]*'MAE':\s*([0-9.eE+-]+)[^{}]*\}")
SUBSET_N = 286


def load_cf():
    if not CF_PATH.exists():
        sys.exit(f"FAIL: missing {CF_PATH}")
    return json.loads(CF_PATH.read_text())


def _ensure_bans(db):
    cf = load_cf()
    rows = {json.loads(r[0])["rule"] for r in
            db.execute("SELECT data FROM nodes WHERE type='ban'").fetchall()}
    added = False
    for rule in cf["bans"]:
        if rule not in rows:
            n = int(db.execute("SELECT v FROM meta WHERE k=?", ("ban_ctr",)).fetchone()[0]) + 1
            db.execute("UPDATE meta SET v=? WHERE k=?", (str(n), "ban_ctr"))
            db.execute("INSERT INTO nodes VALUES(?,?,?)",
                       (f"B{n:04d}", "ban", json.dumps({"rule": rule})))
            added = True
    if added:
        db.commit()


def con():
    db = sqlite3.connect(DB)
    db.execute("CREATE TABLE IF NOT EXISTS nodes(id TEXT PRIMARY KEY, type TEXT, data TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS edges(src TEXT, rel TEXT, dst TEXT, UNIQUE(src,rel,dst))")
    db.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
    for k, v in (("model_ctr", "29"), ("hypo_ctr", "0"), ("ban_ctr", "0")):
        db.execute("INSERT OR IGNORE INTO meta(k,v) VALUES(?,?)", (k, v))
    _ensure_bans(db)
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


def parse_log(path):
    """Return (mae, n, crashed) from a run log. mae/n None if no valid metric."""
    text = Path(path).read_text(errors="replace")
    crashed = "Traceback" in text
    mae = n = None
    for m in EXT_RE.finditer(text):
        try:
            d = ast.literal_eval(m.group(1))
        except Exception:
            continue
        if isinstance(d, dict) and "MAE" in d:
            mae = float(d["MAE"])
            n = int(d.get("n") or d.get("n_images") or 0)
    if mae is None:
        m = METRICS_RE.search(text)
        if m:
            mae, n = float(m.group(1)), int(m.group(2))
    if mae is None:
        m = METRICS_RE2.search(text)
        if m:
            n, mae = int(m.group(1)), float(m.group(2))
    return mae, n, crashed


def cmd_seed(_):
    db = con()
    if db.execute("SELECT COUNT(*) FROM nodes WHERE type='model'").fetchone()[0]:
        print("DB already seeded"); return
    m = {"id": "N0029", "parent": None, "switch": None, "params": 28000000,
         "subset_mae": 25.818, "full_mae": 26.485, "status": "done",
         "note": "TF champion: CountingDINO paper-best + fs=0.5 + MWEx + ADEI t1.5d1m8, ConvNeXt-T required"}
    db.execute("INSERT INTO nodes VALUES(?,?,?)", ("N0029", "model", json.dumps(m)))
    db.commit()
    nb = db.execute("SELECT COUNT(*) FROM nodes WHERE type='ban'").fetchone()[0]
    print(f"seeded N0029 + {nb} bans")


def _depth(models, mid, limit=100):
    d, cur, seen = 0, mid, set()
    while cur not in seen and d < limit:
        seen.add(cur)
        parent = (models.get(cur) or {}).get("parent")
        if not parent:
            break
        cur, d = parent, d + 1
    return d


def _live(models):
    done = [(i, d) for i, d in models.items()
            if d.get("status") == "done" and d.get("subset_mae") is not None]
    if not done:
        return None, None
    return min(done, key=lambda x: (round(x[1]["subset_mae"], 3), _depth(models, x[0])))


def cmd_live(_):
    db = con()
    rows = [(r[0], json.loads(r[1])) for r in db.execute("SELECT id,data FROM nodes WHERE type='model'")]
    models = dict(rows)
    pid, p = _live(models)
    if pid is None:
        print("no live parent"); sys.exit(1)
    bans = [json.loads(r[0])["rule"] for r in db.execute("SELECT data FROM nodes WHERE type='ban'").fetchall()]
    print(json.dumps({"live_parent": pid, "subset_mae": p["subset_mae"], "full_mae": p.get("full_mae"),
                       "next_bar_subset": round(p["subset_mae"] - BAR_STEP, 4), "bans": bans}, indent=1))


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
        q = ",".join("?" * len(scope))
        evidenced = {r[0] for r in db.execute(
            f"SELECT DISTINCT dst FROM edges WHERE rel IN ('SUPPORTS','CONTRADICTS') AND src IN ({q})",
            tuple(scope))}
    else:
        evidenced = {r[0] for r in db.execute(
            "SELECT DISTINCT dst FROM edges WHERE rel IN ('SUPPORTS','CONTRADICTS')")}
    out = []
    for hid, data in db.execute("SELECT id,data FROM nodes WHERE type='hypo'"):
        d = json.loads(data)
        c = d.get("confidence", 0.5)
        if hid not in evidenced and REFUTE < c < CONFIRM:
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
    tl = t.lower()
    for kw in load_cf()["keywords"]:
        if kw.lower() in tl:
            sys.exit(f"FAIL: closed family keyword '{kw}' (see graph.py bans / closed_families.json)")
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
    if a.switch in load_cf()["switches"]:
        sys.exit(f"FAIL: switch --{a.switch} is a closed family (see graph.py bans)")
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
    """Mechanically verify a run log, auto-judge support/contradict from the bar, book once."""
    if not (0 < a.weight <= 1):
        sys.exit("FAIL: --weight must be in (0, 1]")
    logp = Path(a.log)
    if not logp.is_file():
        sys.exit(f"FAIL: log not found: {logp} (scp it from the server first)")
    mae, n, crashed = parse_log(logp)
    if crashed or mae is None:
        sys.exit("FAIL: log has no valid EXTENDED_METRICS (use `graph.py fail` for crashes)")
    if n != SUBSET_N:
        sys.exit(f"FAIL: log n_images={n}, expected {SUBSET_N} (subset286 only)")
    db = con()
    row = db.execute("SELECT data FROM nodes WHERE id=? AND type='hypo'", (a.hypo,)).fetchone()
    if not row:
        sys.exit(f"FAIL: unknown hypo {a.hypo}")
    mrow = db.execute("SELECT data FROM nodes WHERE id=? AND type='model'", (a.model,)).fetchone()
    if not mrow:
        sys.exit(f"FAIL: unknown model {a.model}")
    md = json.loads(mrow[0])
    if md.get("status") == "done":
        sys.exit(f"FAIL: {a.model} already booked (status=done); no double evidence")
    if not db.execute("SELECT 1 FROM edges WHERE src=? AND rel='TESTS' AND dst=?",
                      (a.model, a.hypo)).fetchone():
        sys.exit(f"FAIL: {a.model} does not TEST {a.hypo} (book via new-model first)")
    parent_id = md.get("parent")
    prow = db.execute("SELECT data FROM nodes WHERE id=?", (parent_id,)).fetchone() if parent_id else None
    if not prow:
        sys.exit(f"FAIL: {a.model} has no parent with a bar")
    parent = json.loads(prow[0])
    if parent.get("subset_mae") is None:
        sys.exit(f"FAIL: parent {parent_id} has no subset_mae")
    bar = parent["subset_mae"] - BAR_STEP
    atype = "support" if mae <= bar else "contradict"
    d = json.loads(row[0])
    c = d.get("confidence", 0.5)
    if atype == "support":
        c += ETA * a.weight * (1 - c)
        rel = "SUPPORTS"
    else:
        c -= ETA * a.weight * c
        rel = "CONTRADICTS"
    c = min(1.0, max(0.0, c))
    d["confidence"] = round(c, 4)
    db.execute("UPDATE nodes SET data=? WHERE id=?", (json.dumps(d), a.hypo))
    md.update({"subset_mae": mae, "status": "done", "note": a.note,
               "evidence_type": atype, "evidence_bar": round(bar, 4)})
    if a.full_mae is not None:
        md["full_mae"] = a.full_mae
    db.execute("UPDATE nodes SET data=? WHERE id=?", (json.dumps(md), a.model))
    db.execute("INSERT OR IGNORE INTO edges VALUES(?,?,?)", (a.model, rel, a.hypo))
    db.commit()
    print(f"{a.hypo} confidence -> {d['confidence']} | {a.model} subset_mae={mae} "
          f"type={atype} bar={bar:.4f} weight={a.weight}")


def cmd_fail(a):
    """Terminal status for a crash/timeout run. Does not move hypothesis confidence."""
    if a.reason not in ("failed", "timeout"):
        sys.exit("FAIL: --reason must be failed|timeout")
    db = con()
    mrow = db.execute("SELECT data FROM nodes WHERE id=? AND type='model'", (a.model,)).fetchone()
    if not mrow:
        sys.exit(f"FAIL: unknown model {a.model}")
    if not db.execute("SELECT 1 FROM nodes WHERE id=? AND type='hypo'", (a.hypo,)).fetchone():
        sys.exit(f"FAIL: unknown hypo {a.hypo}")
    md = json.loads(mrow[0])
    if md.get("status") == "done":
        sys.exit(f"FAIL: {a.model} already status=done; fail is only for unfinished runs")
    if not db.execute("SELECT 1 FROM edges WHERE src=? AND rel='TESTS' AND dst=?",
                      (a.model, a.hypo)).fetchone():
        sys.exit(f"FAIL: {a.model} does not TEST {a.hypo}")
    md.update({"status": a.reason, "note": a.note})
    db.execute("UPDATE nodes SET data=? WHERE id=?", (json.dumps(md), a.model))
    db.commit()
    print(f"{a.model} status={a.reason}")


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


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="c", required=True)
    s.add_parser("seed"); s.add_parser("live-parent"); s.add_parser("bans"); s.add_parser("tree")
    u = s.add_parser("untested"); u.add_argument("--parent", default="")
    h = s.add_parser("new-hypo"); h.add_argument("--text", required=True)
    m = s.add_parser("new-model"); m.add_argument("--parent", required=True); m.add_argument("--hypo", required=True)
    m.add_argument("--switch", required=True); m.add_argument("--params", type=int, default=0)
    e = s.add_parser("evidence"); e.add_argument("--hypo", required=True); e.add_argument("--model", required=True)
    e.add_argument("--log", required=True, help="scp'd run log; MAE/n/type parsed from it")
    e.add_argument("--weight", type=float, default=0.85)
    e.add_argument("--note", default=""); e.add_argument("--full-mae", type=float, default=None)
    f = s.add_parser("fail"); f.add_argument("--hypo", required=True); f.add_argument("--model", required=True)
    f.add_argument("--reason", required=True, choices=["failed", "timeout"])
    f.add_argument("--note", default="")
    b = s.add_parser("ban"); b.add_argument("--rule", required=True); b.add_argument("--ref", default="")
    a = p.parse_args()
    {"seed": cmd_seed, "live-parent": cmd_live, "untested": cmd_untested, "bans": cmd_bans,
     "new-hypo": cmd_new_hypo, "new-model": cmd_new_model, "evidence": cmd_evidence,
     "fail": cmd_fail, "ban": cmd_ban, "tree": cmd_tree}[a.c](a)


if __name__ == "__main__":
    main()
