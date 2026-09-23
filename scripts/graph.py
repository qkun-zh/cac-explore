"""NeuG: Cypher graph via the real NeuG package. Node types: model|hypo|ban. Edges: CHILD_OF|TESTS|SUPPORTS|CONTRADICTS."""
import argparse, ast, json, sys, re
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
DDL = (
    "CREATE NODE TABLE IF NOT EXISTS Node(id STRING, type STRING, data VARCHAR(8192), PRIMARY KEY(id));",
    "CREATE NODE TABLE IF NOT EXISTS Meta(k STRING, v VARCHAR(64), PRIMARY KEY(k));",
    "CREATE REL TABLE IF NOT EXISTS Edge(FROM Node TO Node, rel STRING);",
)


def load_cf():
    if not CF_PATH.exists():
        sys.exit(f"FAIL: missing {CF_PATH}")
    return json.loads(CF_PATH.read_text())


class Graph:
    def __init__(self, read_only=False):
        if DB.exists() and not DB.is_dir():
            sys.exit("FAIL: neug.db is a legacy SQLite file; NeuG expects a directory (move the file aside)")
        import neug
        self._db = neug.Database(str(DB), mode="read-only" if read_only else "read-write")
        self.conn = self._db.connect()
        if not read_only:
            for stmt in DDL:
                self.conn.execute(stmt)
            for k, v in (("model_ctr", "29"), ("hypo_ctr", "0"), ("ban_ctr", "0")):
                self.conn.execute(
                    "MERGE (m:Meta {k: $k}) ON CREATE SET m.v = $v;",
                    parameters={"k": k, "v": v})
            self._ensure_bans()

    def q(self, cypher, **params):
        return list(self.conn.execute(cypher, parameters=params or None))

    def one(self, cypher, **params):
        rows = self.q(cypher, **params)
        return rows[0] if rows else None

    def run(self, cypher, **params):
        self.conn.execute(cypher, parameters=params or None)

    def has_edge(self, src, rel, dst):
        return bool(self.one(
            "MATCH (a:Node {id: $s})-[e:Edge]->(b:Node {id: $d}) "
            "WHERE e.rel = $r RETURN 1;",
            s=src, r=rel, d=dst))

    def add_edge(self, src, rel, dst):
        if not self.has_edge(src, rel, dst):
            self.run(
                "MATCH (a:Node {id: $s}), (b:Node {id: $d}) "
                "CREATE (a)-[:Edge {rel: $r}]->(b);",
                s=src, d=dst, r=rel)

    def get_node(self, nid):
        row = self.one("MATCH (n:Node {id: $id}) RETURN n.type, n.data;", id=nid)
        if not row:
            return None, None
        return row[0], json.loads(row[1])

    def nodes_of(self, ntype):
        return [(r[0], json.loads(r[1])) for r in
                self.q("MATCH (n:Node) WHERE n.type = $t RETURN n.id, n.data;", t=ntype)]

    def all_edges(self):
        return [(r[0], r[1], r[2]) for r in self.q(
            "MATCH (a:Node)-[e:Edge]->(b:Node) RETURN a.id, e.rel, b.id;")]

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
        try:
            self._db.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _ensure_bans(self):
        cf = load_cf()
        have = {d["rule"] for _i, d in self.nodes_of("ban") if "rule" in d}
        if have:
            need = [rule for rule in cf["bans"] if rule not in have]
        else:
            need = list(cf["bans"])
        if not need:
            return
        row = self.one("MATCH (m:Meta {k: $k}) RETURN m.v;", k="ban_ctr")
        n = int(row[0]) if row else 0
        for rule in need:
            n += 1
            self.run("CREATE (:Node {id: $id, type: $type, data: $data});",
                     id=f"B{n:04d}", type="ban", data=json.dumps({"rule": rule}))
        self.run("MATCH (m:Meta {k: $k}) SET m.v = $v;", k="ban_ctr", v=str(n))


def con():
    return Graph()


def nxt(db, k, prefix, width=4):
    row = db.one("MATCH (m:Meta {k: $k}) RETURN m.v;", k=k)
    if not row:
        sys.exit(f"FAIL: missing meta counter {k}")
    n = int(row[0]) + 1
    db.run("MATCH (m:Meta {k: $k}) SET m.v = $v;", k=k, v=str(n))
    return f"{prefix}{n:0{width}d}"


def tok(s):
    return re.findall(r"[a-z]{3,}", s.lower())


def novelty(db, text):
    words = [tok(d["text"]) for _, d in db.nodes_of("hypo")]
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
    try:
        n_models = db.one("MATCH (n:Node) WHERE n.type = $t RETURN count(n) AS c;", t="model")
        if n_models and n_models[0]:
            print("DB already seeded")
            return
        m = {"id": "N0029", "parent": None, "switch": None, "params": 28000000,
             "subset_mae": 25.818, "full_mae": 26.485, "status": "done",
             "note": "TF champion: CountingDINO paper-best + fs=0.5 + MWEx + ADEI t1.5d1m8, ConvNeXt-T required"}
        db.run("CREATE (:Node {id: $id, type: $type, data: $data});",
               id="N0029", type="model", data=json.dumps(m))
        nb = db.one("MATCH (n:Node) WHERE n.type = $t RETURN count(n) AS c;", t="ban")
        print(f"seeded N0029 + {nb[0] if nb else 0} bans")
    finally:
        db.close()


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
    try:
        models = dict(db.nodes_of("model"))
        pid, p = _live(models)
        if pid is None:
            print("no live parent")
            sys.exit(1)
        bans = [d["rule"] for _, d in db.nodes_of("ban")]
        print(json.dumps({"live_parent": pid, "subset_mae": p["subset_mae"],
                          "full_mae": p.get("full_mae"),
                          "next_bar_subset": round(p["subset_mae"] - BAR_STEP, 4),
                          "bans": bans}, indent=1))
    finally:
        db.close()


def _descendants(db, root):
    kids = {}
    for src, _rel, dst in db.all_edges():
        if _rel == "CHILD_OF":
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
    try:
        edges = db.all_edges()
        if a.parent:
            t, _d = db.get_node(a.parent)
            if t != "model":
                sys.exit(f"FAIL: unknown parent {a.parent}")
            scope = _descendants(db, a.parent)
            evidenced = {dst for _s, rel, dst in edges
                         if rel in ("SUPPORTS", "CONTRADICTS") and _s in scope}
        else:
            evidenced = {dst for _s, rel, dst in edges
                         if rel in ("SUPPORTS", "CONTRADICTS")}
        out = []
        for hid, d in db.nodes_of("hypo"):
            c = d.get("confidence", 0.5)
            if hid not in evidenced and REFUTE < c < CONFIRM:
                out.append({"id": hid, "text": d["text"], "confidence": c})
        print(json.dumps(out, indent=1))
    finally:
        db.close()


def cmd_bans(_):
    db = con()
    try:
        print(json.dumps([{"id": i, **d} for i, d in db.nodes_of("ban")], indent=1))
    finally:
        db.close()


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
    try:
        if novelty(db, t) >= 0.82:
            sys.exit("FAIL: novelty gate (jaccard >= 0.82 duplicate)")
        hid = nxt(db, "hypo_ctr", "H")
        db.run("CREATE (:Node {id: $id, type: $type, data: $data});",
               id=hid, type="hypo",
               data=json.dumps({"text": t, "confidence": 0.5}))
        print(hid)
    finally:
        db.close()


def cmd_new_model(a):
    db = con()
    try:
        t, _d = db.get_node(a.parent)
        if t != "model":
            sys.exit(f"FAIL: unknown parent {a.parent}")
        t, _d = db.get_node(a.hypo)
        if t != "hypo":
            sys.exit(f"FAIL: unknown hypo {a.hypo}")
        if not SWITCH_RE.fullmatch(a.switch):
            sys.exit("FAIL: switch must match [a-z][a-z0-9_]+ (it IS the CLI flag name, passed as --<switch>)")
        if a.switch in load_cf()["switches"]:
            sys.exit(f"FAIL: switch --{a.switch} is a closed family (see graph.py bans)")
        args_py = DB.parent / "models" / "cdino" / "tf_pipeline" / "args.py"
        if args_py.exists() and f"--{a.switch}" not in args_py.read_text():
            sys.exit(f"FAIL: flag --{a.switch} not declared in models/cdino/tf_pipeline/args.py")
        mid = nxt(db, "model_ctr", "N")
        db.run("CREATE (:Node {id: $id, type: $type, data: $data});",
               id=mid, type="model",
               data=json.dumps({"id": mid, "parent": a.parent, "switch": a.switch,
                                "params": a.params, "status": "open"}))
        db.add_edge(mid, "CHILD_OF", a.parent)
        db.add_edge(mid, "TESTS", a.hypo)
        print(mid)
    finally:
        db.close()


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
    try:
        t, d = db.get_node(a.hypo)
        if t != "hypo":
            sys.exit(f"FAIL: unknown hypo {a.hypo}")
        t, md = db.get_node(a.model)
        if t != "model":
            sys.exit(f"FAIL: unknown model {a.model}")
        if md.get("status") == "done":
            sys.exit(f"FAIL: {a.model} already booked (status=done); no double evidence")
        if not db.has_edge(a.model, "TESTS", a.hypo):
            sys.exit(f"FAIL: {a.model} does not TEST {a.hypo} (book via new-model first)")
        parent_id = md.get("parent")
        _pt, parent = db.get_node(parent_id) if parent_id else (None, None)
        if not parent:
            sys.exit(f"FAIL: {a.model} has no parent with a bar")
        if parent.get("subset_mae") is None:
            sys.exit(f"FAIL: {parent_id} has no subset_mae")
        bar = parent["subset_mae"] - BAR_STEP
        atype = "support" if mae <= bar else "contradict"
        c = d.get("confidence", 0.5)
        if atype == "support":
            c += ETA * a.weight * (1 - c)
            rel = "SUPPORTS"
        else:
            c -= ETA * a.weight * c
            rel = "CONTRADICTS"
        c = min(1.0, max(0.0, c))
        d["confidence"] = round(c, 4)
        db.run("MATCH (n:Node {id: $id}) SET n.data = $d;",
               id=a.hypo, d=json.dumps(d))
        md.update({"subset_mae": mae, "status": "done", "note": a.note,
                   "evidence_type": atype, "evidence_bar": round(bar, 4)})
        if a.full_mae is not None:
            md["full_mae"] = a.full_mae
        db.run("MATCH (n:Node {id: $id}) SET n.data = $d;",
               id=a.model, d=json.dumps(md))
        db.add_edge(a.model, rel, a.hypo)
        print(f"{a.hypo} confidence -> {d['confidence']} | {a.model} subset_mae={mae} "
              f"type={atype} bar={bar:.4f} weight={a.weight}")
    finally:
        db.close()


def cmd_fail(a):
    """Terminal status for a crash/timeout run. Does not move hypothesis confidence."""
    if a.reason not in ("failed", "timeout"):
        sys.exit("FAIL: --reason must be failed|timeout")
    db = con()
    try:
        t, md = db.get_node(a.model)
        if t != "model":
            sys.exit(f"FAIL: unknown model {a.model}")
        t, _h = db.get_node(a.hypo)
        if t != "hypo":
            sys.exit(f"FAIL: unknown hypo {a.hypo}")
        if md.get("status") == "done":
            sys.exit(f"FAIL: {a.model} already status=done; fail is only for unfinished runs")
        if not db.has_edge(a.model, "TESTS", a.hypo):
            sys.exit(f"FAIL: {a.model} does not TEST {a.hypo}")
        md.update({"status": a.reason, "note": a.note})
        db.run("MATCH (n:Node {id: $id}) SET n.data = $d;",
               id=a.model, d=json.dumps(md))
        print(f"{a.model} status={a.reason}")
    finally:
        db.close()


def cmd_ban(a):
    db = con()
    try:
        bid = nxt(db, "ban_ctr", "B")
        db.run("CREATE (:Node {id: $id, type: $type, data: $data});",
               id=bid, type="ban", data=json.dumps({"rule": a.rule, "ref": a.ref}))
        print(bid)
    finally:
        db.close()


def cmd_tree(_):
    db = con()
    try:
        models = dict(db.nodes_of("model"))
        kids = {}
        for m, d in models.items():
            kids.setdefault(d.get("parent") or ".", []).append(m)

        def show(p, dep):
            for m in sorted(kids.get(p, [])):
                d = models[m]
                print("  " * dep + f"{m} switch={d.get('switch')} mae={d.get('subset_mae')} {d.get('status')}")
                show(m, dep + 1)

        show(".", 0)
    finally:
        db.close()


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
