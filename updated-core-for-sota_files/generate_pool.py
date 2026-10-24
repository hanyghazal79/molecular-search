"""Generate a candidate pool from a frozen chemistry library (dynamic size)."""

import csv
import itertools
import random

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')     # <-- add these 2 lines

from rdkit import Chem
from chemistry import Library, passes_filters

def generate(library, count, attempts, seed):
    _, role_a, role_b, role_c = library.reaction()
    pools = [library.ids_for_role(role) for role in (role_a, role_b, role_c) if role]
    if len(pools) not in (2, 3) or not all(pools):
        raise ValueError("The reaction has an empty or invalid building-block role")
    
    capacity = 1
    for p in pools:
        capacity *= len(p)
    
    rng = random.Random(seed)
    chemical_keys, accepted = set(), []
    
    # Force enumeration for anything up to 2M combos
    use_enumeration = capacity < 2_000_000
    
    if use_enumeration:
        print(f"  [Strategy: ENUMERATION] Pool capacity = {capacity:,}, count = {count:,}")
        all_combos = list(itertools.product(*pools))
        rng.shuffle(all_combos)
        combos = all_combos
    else:
        print(f"  [Strategy: RANDOM SAMPLING] {attempts:,} attempts, capacity = {capacity:,}")
        sampled = set()
        combos = []
        for _ in range(attempts):
            identifiers = tuple(rng.choice(pool) for pool in pools)
            if identifiers in sampled:
                continue
            sampled.add(identifiers)
            combos.append(identifiers)
    
    for identifiers in combos:
        variants = list(enumerate(library.products(identifiers)))
        rng.shuffle(variants)
        for index, smiles in variants:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or not passes_filters(mol, library.manifest["filters"]):
                continue
            key = Chem.MolToInchiKey(mol)
            if not key or key in chemical_keys:
                continue
            chemical_keys.add(key)
            identifier = "rxn:" + ":".join(map(str, (library.reaction_id, *identifiers)))
            if index:
                identifier += f":k{index}"
            accepted.append({"product_id": identifier, "smiles": smiles})
            if len(accepted) >= count:
                return accepted, len(combos)
    
    return accepted, len(combos)

def main(library_dir="synthetic-library", reaction="rxn:1", count=100000, attempts=500, seed=42, output="pool_100k.csv"):
    if count < 1 or attempts < 1:
        raise ValueError("count and attempts must be positive")
    try:
        with Library(library_dir) as library:
            if reaction != library.manifest["allowed_reaction"]:
                raise ValueError(f"reaction '{reaction}' must match '{library.manifest['allowed_reaction']}'")
            accepted, processed = generate(library, count, attempts, seed)
            print(f"  Library case digest: {library.manifest['case_digest']}")
    except (ValueError, OSError) as exc:
        raise RuntimeError(f"Error generating pool: {exc}")

    with open(output, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["product_id", "smiles"])
        writer.writeheader()
        writer.writerows(accepted)
    
    status = "✅" if len(accepted) >= count else "⚠️"
    print(f"  {status} Saved {len(accepted):,} / {count:,} requested; {processed:,} combos processed")
    if len(accepted) < count:
        print(f"  ⚠️  SHORTFALL: requested {count:,}, got {len(accepted):,}")
    
    return accepted

if __name__ == "__main__":
    main()
