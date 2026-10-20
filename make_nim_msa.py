"""Generate a cached MSA for HIV-1 protease once (CPU-only compatible).

Requires: pip install boltz-community   (no [cuda] extra needed)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TARGET_SEQUENCE = (
    "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYD"
    "QILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
)


def _require_boltz():
    if shutil.which("boltz") is None:
        raise SystemExit(
            "Error: 'boltz' not on PATH.\n"
            "Install with: pip install boltz-community"
        )


def generate(out="msa.a3m", cache=None):
    _require_boltz()
    work = Path(tempfile.mkdtemp(prefix="msa_"))
    yaml = work / "apo.yaml"
    yaml.write_text(
        f"version: 1\nsequences:\n  - protein:\n      id: A\n"
        f'      sequence: "{TARGET_SEQUENCE}"\n'
    )

    cmd = [
        "boltz", "predict", str(yaml),
        "--out_dir", str(work / "out"),
        "--use_msa_server",
        "--accelerator", "cpu",      # force CPU
        "--no_kernels",              # disable Triton kernels on CPU
    ]
    if cache:
        cmd += ["--cache", str(Path(cache).expanduser().resolve())]

    print(f"Generating MSA for {len(TARGET_SEQUENCE)}-residue protein...")
    if cache:
        print(f"Using cache: {cache}")
    print("Running on CPU — structure prediction will be slow; MSA generation is fast.")
    subprocess.run(cmd, check=True, timeout=3600)

    for root, _, files in os.walk(work / "out"):
        for f in files:
            if f.endswith(".a3m"):
                shutil.copy(os.path.join(root, f), out)
                print(f"Wrote {out}")
                return
    raise RuntimeError("boltz ran but produced no .a3m file")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", nargs="?", default="msa.a3m")
    p.add_argument("--cache", help="Boltz cache dir (e.g. ./.boltz)")
    args = p.parse_args()
    generate(args.output, args.cache)