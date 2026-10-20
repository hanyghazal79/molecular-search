"""Generate a random candidate pool from the owner's frozen chemistry library."""
import argparse
import csv
import random

from rdkit import Chem

from chemistry import Library, passes_filters


def generate(library, count, attempts, seed):
    _, role_a, role_b, role_c = library.reaction()
    pools = [library.ids_for_role(role) for role in (role_a, role_b, role_c) if role]
    if len(pools) not in (2, 3) or not all(pools):
        raise ValueError("The reaction has an empty or invalid building-block role")
    rng = random.Random(seed)
    sampled, chemical_keys, accepted = set(), set(), []
    for _ in range(attempts):
        identifiers = tuple(rng.choice(pool) for pool in pools)
        if identifiers in sampled:
            continue
        sampled.add(identifiers)
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
                return accepted, len(sampled)
    return accepted, len(sampled)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, help="Directory containing library.json and molecules.sqlite")
    parser.add_argument("--reaction", required=True, help="Allowed reaction, such as rxn:2")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--attempts", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="candidate-pool.csv")
    args = parser.parse_args()
    if args.count < 1 or args.attempts < 1:
        parser.error("--count and --attempts must be positive")
    try:
        with Library(args.library) as library:
            if args.reaction != library.manifest["allowed_reaction"]:
                parser.error("--reaction must match the library's allowed reaction")
            accepted, attempted = generate(library, args.count, args.attempts, args.seed)
            print(f"Library case digest: {library.manifest['case_digest']}")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    with open(args.output, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["product_id", "smiles"])
        writer.writeheader()
        writer.writerows(accepted)
    print(f"Saved {len(accepted)} candidates to {args.output}; {attempted} distinct combinations attempted.")
    if len(accepted) < args.count:
        print("The requested pool size was not reached; increase attempts or inspect the allowed library.")
    print("Local filtering does not establish historical novelty or whole-set diversity; use the evaluation service.")


if __name__ == "__main__":
    main()
