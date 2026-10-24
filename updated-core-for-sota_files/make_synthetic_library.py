"""Dynamically build a synthetic amide-coupling library of any size."""
import hashlib
import json
import math
import sqlite3
import sys
from pathlib import Path

ROLE_A, ROLE_B, REACTION_ID = 1, 2, 1
SMARTS = "[C:1](=[O:2])[OH:3].[N:4]>>[C:1](=[O:2])[N:4]"

FILTERS = {
    "max_kappa1": 20, "max_molecular_weight": 600, "max_heavy_atoms": 45,
    "min_heavy_atoms": 5, "min_molecular_weight": 100,  # <-- Relaxed to 100
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

def _generate_smiles(idx, is_acid):
    cores = ["c1ccccc1", "c1ccncc1", "c1cncnc1", "C1CCCCC1", "C1CCNCC1", 
             "C1CCOCC1", "c1ccsc1", "c1ccoc1", "c1ccc2ccccc2c1", "c1ccc2[nH]ccc2c1"]
    subs = ["", "F", "Cl", "Br", "C", "CC", "OC", "N", "C#N", 
            "C(F)(F)F", "O", "S", "C(=O)O", "N(C)C", "S(=O)(=O)C"]
    
    core = cores[idx % len(cores)]
    sub = subs[(idx // len(cores)) % len(subs)]
    chain = "C" * ((idx // (len(cores) * len(subs))) % 10)
    
    if is_acid:
        if sub:
            if core == "c1ccccc1":
                smi = f"O=C([1*]){chain}c1ccc({sub})cc1" if chain else f"O=C([1*])c1ccc({sub})cc1"
            else:
                smi = f"{sub}{core}{chain}C(=O)[1*]"
        else:
            smi = f"{core}{chain}C(=O)[1*]"
    else:
        if sub:
            if core == "c1ccccc1":
                smi = f"[2*]N{chain}c1ccc({sub})cc1" if chain else f"[2*]Nc1ccc({sub})cc1"
            else:
                smi = f"[2*]N{sub}{core}{chain}"
        else:
            smi = f"[2*]N{core}{chain}" if chain else f"[2*]N{core}"
    
    return smi

def build(out_dir, target_count):
    # Use a 10x buffer to guarantee we have enough valid products
    block_count = math.ceil(math.sqrt(target_count * 10))
    
    ACIDS = [(_generate_smiles(i, True), 1000 + i) for i in range(block_count)]
    AMINES = [(_generate_smiles(i, False), 2000 + i) for i in range(block_count)]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "molecules.sqlite"
    db_path.unlink(missing_ok=True)
    
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
    conn.commit()
    conn.close()
    
    manifest = {
        "format_version": 1,
        "rdkit_version": "2024.09.4",
        "allowed_reaction": f"rxn:{REACTION_ID}",
        "case_digest": hashlib.sha256(f"dynamic-{target_count}".encode()).hexdigest(),
        "database_sha256": hashlib.sha256(db_path.read_bytes()).hexdigest(),
        "filters": FILTERS,
    }
    (out_dir / "library.json").write_text(json.dumps(manifest, indent=2))
    print(f"Built {out_dir}: {len(ACIDS)} acids x {len(AMINES)} amines = {len(ACIDS)*len(AMINES):,} possible products")

if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    out = sys.argv[2] if len(sys.argv) > 2 else f"synthetic-library-{target}"
    build(out, target)
