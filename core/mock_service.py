"""Local HTTP mock of the owner's evaluation service.

Tests protocol conformance of baseline.py / search.py against client.py.
Restarting the mock invalidates any checkpoint that references an old run_id.
Scores are sha256-based noise — cannot assess candidate quality.

Usage:
    python mock_service.py --library synthetic-library --port 8765
    export EVAL_URL=http://127.0.0.1:8765 EVAL_TOKEN=mock
"""

import argparse
import hashlib
import itertools
import json
import threading
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

_RUNS, _JOBS, _LOCK = {}, {}, threading.Lock()
_MANIFEST, _VALID = None, set()


def _enumerate(lib_dir):
    from chemistry import Library, passes_filters
    from rdkit import Chem
    valid = set()
    with Library(lib_dir) as lib:
        _, a, b, c = lib.reaction()
        pools = [lib.ids_for_role(r) for r in (a, b, c) if r]
        for combo in itertools.product(*pools):
            for idx, smi in enumerate(lib.products(combo)):
                mol = Chem.MolFromSmiles(smi) if smi else None
                if mol and passes_filters(mol, lib.manifest["filters"]):
                    pid = "rxn:" + ":".join(map(str, (lib.reaction_id, *combo)))
                    if idx:
                        pid += f":k{idx}"
                    valid.add(pid)
    return valid


def _score(pid):
    h = hashlib.sha256(f"{_MANIFEST['case_digest']}|{pid}".encode()).digest()
    return 4.0 + (int.from_bytes(h[:4], "big") % 500) / 100.0


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _json(self, code, body):
        d = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(d)))
        self.end_headers(); self.wfile.write(d)

    def _ok(self):
        return self.headers.get("Authorization", "").startswith("Bearer ")

    def do_GET(self):
        if not self._ok():
            return self._json(401, {"error": "unauthorized"})
        if self.path == "/v1/cases":
            return self._json(200, [{"id": "synthetic-demo", "backend": "demo",
                "required_count": 3, "target_count": 1, "max_pairs_per_run": 200,
                "affinity_configured": False,
                "case_digest": _MANIFEST["case_digest"]}])
        if self.path.startswith("/v1/runs/"):
            rid = self.path.rsplit("/", 1)[1]
            with _LOCK:
                if rid not in _RUNS:
                    return self._json(404, {"error": "not found"})
                return self._json(200, _RUNS[rid])
        if self.path.startswith("/v1/jobs/"):
            jid = self.path.rsplit("/", 1)[1]
            with _LOCK:
                if jid not in _JOBS:
                    return self._json(404, {"error": "not found"})
                return self._json(200, _JOBS[jid])
        return self._json(404, {"error": "unknown path"})

    def do_POST(self):
        if not self._ok():
            return self._json(401, {"error": "unauthorized"})
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n)) if n else {}

        if self.path == "/v1/runs":
            if body.get("case_id") != "synthetic-demo":
                return self._json(404, {"error": "case unknown"})
            rid = "run-" + uuid.uuid4().hex[:12]
            run = {"id": rid, "case_id": "synthetic-demo",
                   "label": body.get("label"), "code_revision": body.get("code_revision"),
                   "case_digest": _MANIFEST["case_digest"], "created": time.time(),
                   "jobs": [], "pairs_charged": 0}
            with _LOCK:
                _RUNS[rid] = run
            return self._json(200, run)

        if self.path.startswith("/v1/runs/") and self.path.endswith("/jobs"):
            rid = self.path.split("/")[3]
            with _LOCK:
                if rid not in _RUNS:
                    return self._json(404, {"error": "run not found"})
                run = _RUNS[rid]
                req_id = body.get("request_id")
                if req_id:
                    for j in _JOBS.values():
                        if j.get("request_id") == req_id and j["run_id"] == rid:
                            return self._json(200, j)
                products = body.get("products", [])
                op = body.get("operation", "final")
                if op not in ("validate", "oracle", "final"):
                    return self._json(400, {"error": "invalid operation"})
                unknown = [p for p in products if p not in _VALID]
                if unknown:
                    return self._json(400, {"error": f"unknown products: {unknown[:3]}"})
                run["pairs_charged"] += 0 if op == "validate" else len(products)
                items = [{"product": pid, "demo_value": _score(pid)} for pid in products]
                valid = not (op in ("validate", "final") and len(products) < 3)
                jid = "job-" + uuid.uuid4().hex[:12]
                job = {"id": jid, "run_id": rid, "operation": op,
                       "products": products, "request_id": req_id,
                       "state": "succeeded", "charge": 0 if op == "validate" else len(products),
                       "result": {"backend": "demo", "valid": valid,
                                  "items": items, "operation": op}}
                _JOBS[jid] = job
                run["jobs"].append(jid)
            return self._json(200, job)

        return self._json(404, {"error": "unknown path"})


def main():
    global _MANIFEST, _VALID
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    lib = Path(a.library)
    _MANIFEST = json.loads((lib / "library.json").read_text())
    _VALID = _enumerate(lib)
    print(f"Mock on :{a.port}  {len(_VALID)} valid product IDs")
    HTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()