"""Build a synthetic amide-coupling library bundle for offline pipeline testing.

Emits library.json + molecules.sqlite compatible with chemistry.Library.
- Reaction: rxn:1 (amide coupling via join_synthons)
- Acids (role A, mask=1) carry [1*] on the acyl carbon
- Amines (role B, mask=2) carry [2*] on the nitrogen
- Product IDs resolve demo_pool.csv verbatim (rxn:1:1001:2001 ... rxn:1:1012:2012)

Requires RDKit 2024.09.4 (chemistry.RDKIT_VERSION).
"""
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ACIDS = [
    ("CC(=O)[1*]", 1001), ("c1ccccc1C(=O)[1*]", 1002),
    ("CC(C)C(=O)[1*]", 1003), ("O=C([1*])c1ccncc1", 1004),
    ("COc1ccc(C(=O)[1*])cc1", 1005), ("FC(F)(F)c1ccc(C(=O)[1*])cc1", 1006),
    ("CCCC(=O)[1*]", 1007), ("N#Cc1ccc(C(=O)[1*])cc1", 1008),
    ("O=C([1*])C1CCCCC1", 1009), ("Cc1ccc(C(=O)[1*])cc1", 1010),
    ("O=C([1*])c1ccc2ccccc2c1", 1011), ("O=C([1*])c1ccc(Cl)cc1", 1012),
]
AMINES = [
    ("[2*]NCC", 2001), ("[2*]Nc1ccccc1", 2002), ("[2*]NC(C)C", 2003),
    ("[2*]NCC1CCCCC1", 2004), ("[2*]Nc1ccc(OC)cc1", 2005),
    ("[2*]Nc1ccc(C)cc1", 2006), ("[2*]NCCN", 2007),
    ("[2*]N1CCCCC1", 2008), ("[2*]NCCO", 2009),
    ("[2*]Nc1ccncc1", 2010), ("[2*]NCc1ccccc1", 2011),
    ("[2*]Nc1ccc(Cl)cc1", 2012),
]

ROLE_A, ROLE_B, REACTION_ID = 1, 2, 1
SMARTS = "[C:1](=[O:2])[OH:3].[N:4]>>[C:1](=[O:2])[N:4]"

FILTERS = {
    "max_kappa1": 20, "max_molecular_weight": 600, "max_heavy_atoms": 45,
    "min_heavy_atoms": 5, "min_molecular_weight": 150,
    "max_aromatic_rings": 4, "max_aliphatic_rings": 4, "min_rings": 0,
    "max_fsp3": 1.0, "max_spiro_atoms": 2, "max_ring_size": 8,
    "max_aromatic_nitrogen": 4, "max_aromatic_fraction": 1.0, "max_aromatic_carbon": 20,
    "max_halogens": 5, "max_nitrogen": 6, "max_oxygen": 8,
    "max_sulfur": 2, "max_phosphorus": 1, "max_iodine": 1,
    "max_bromine": 2, "max_fluorine": 5,
    "min_heteroatom_fraction": 0.0, "max_heteroatom_fraction": 1.0,
    "max_hbd": 5, "max_amide_bonds": 3, "max_nitrile": 2,
    "max_hydroxyl": 3, "max_triple_bonds": 2, "min_rotatable_bonds": 0,
    "max_rotatable_bonds": 12, "max_bertz_ct": 1000,
    "max_hba": 10, "max_kappa2": 20, "max_kappa3": 20,
    "max_stereocenters": 4, "max_logp": 6.0,
}


def build(out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "molecules.sqlite"; db_path.unlink(missing_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE molecules (mol_id INTEGER PRIMARY KEY, smiles TEXT, role_mask INTEGER);
        CREATE TABLE reactions (rxn_id TEXT PRIMARY KEY, smarts TEXT,
                                roleA INTEGER, roleB INTEGER, roleC INTEGER);
    """)
    for smi, mid in ACIDS:
        conn.execute("INSERT INTO molecules VALUES (?, ?, ?)", (mid, smi, ROLE_A))
    for smi, mid in AMINES:
        conn.execute("INSERT INTO molecules VALUES (?, ?, ?)", (mid, smi, ROLE_B))
    conn.execute("INSERT INTO reactions VALUES (?, ?, ?, ?, ?)",
                 (str(REACTION_ID), SMARTS, ROLE_A, ROLE_B, 0))
    conn.commit(); conn.close()
    manifest = {
        "format_version": 1,
        "rdkit_version": "2024.09.4",
        "allowed_reaction": f"rxn:{REACTION_ID}",
        "case_digest": hashlib.sha256(b"synthetic-demo-bundle-v1").hexdigest(),
        "database_sha256": hashlib.sha256(db_path.read_bytes()).hexdigest(),
        "filters": FILTERS,
    }
    (out_dir / "library.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out_dir}/library.json and molecules.sqlite")
    print(f"  acids: {len(ACIDS)} (1001-1012)  amines: {len(AMINES)} (2001-2012)")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "synthetic-library")