"""Local reaction generation and property checks for a frozen molecular library.

The library contains only molecules.sqlite and library.json. Product indices
refer to distinct canonical products in reaction-enumeration order, not sorted
SMILES. These checks do not establish historical novelty or set diversity.
"""

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from functools import cache
from pathlib import Path

import rdkit
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    GraphDescriptors,
    Lipinski,
    rdMolDescriptors,
)

RDKIT_VERSION = "2024.09.4"
PRODUCT_ID = re.compile(
    r"rxn:([0-9]+):([0-9]+):([0-9]+)(?::([0-9]+))?(?::k([0-9]+))?\Z"
)
FILTER_LIMITS = {
    "max_kappa1": "kappa1",
    "max_molecular_weight": "molecular_weight",
    "max_heavy_atoms": "heavy_atoms",
    "min_heavy_atoms": "heavy_atoms",
    "min_molecular_weight": "molecular_weight",
    "max_aromatic_rings": "aromatic_rings",
    "max_aliphatic_rings": "aliphatic_rings",
    "min_rings": "rings",
    "max_fsp3": "fsp3",
    "max_spiro_atoms": "spiro_atoms",
    "max_ring_size": "largest_ring",
    "max_aromatic_nitrogen": "aromatic_nitrogen",
    "max_aromatic_fraction": "aromatic_fraction",
    "max_aromatic_carbon": "aromatic_carbon",
    "max_halogens": "halogens",
    "max_nitrogen": "N",
    "max_oxygen": "O",
    "max_sulfur": "S",
    "max_phosphorus": "P",
    "max_iodine": "I",
    "max_bromine": "Br",
    "max_fluorine": "F",
    "min_heteroatom_fraction": "heteroatom_fraction",
    "max_heteroatom_fraction": "heteroatom_fraction",
    "max_hbd": "hbd",
    "max_amide_bonds": "amide_bonds",
    "max_nitrile": "nitrile",
    "max_hydroxyl": "hydroxyl",
    "max_triple_bonds": "triple_bonds",
    "min_rotatable_bonds": "rotatable_bonds",
    "max_rotatable_bonds": "rotatable_bonds",
    "max_bertz_ct": "bertz_ct",
    "max_hba": "hba",
    "max_kappa2": "kappa2",
    "max_kappa3": "kappa3",
    "max_stereocenters": "stereocenters",
    "max_logp": "logp",
}
EXCLUDED_PROPERTIES = ("metals", "B", "Si", "Se", "azide", "isocyanide", "allene")
GROUP_PATTERNS = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in {
        "nitrile": "[CX2]#[NX1]",
        "azide": "[NX1]~[NX2]~[NX2,NX3]",
        "isocyanide": "[CX1-]#[NX2+]",
        "allene": "[CX3]=[CX2]=[CX3]",
        "hydroxyl": "[OX2H]",
    }.items()
}
DESCRIPTORS = {
    "molecular_weight": Descriptors.MolWt,
    "rings": Descriptors.RingCount,
    "aromatic_rings": Descriptors.NumAromaticRings,
    "aliphatic_rings": Descriptors.NumAliphaticRings,
    "fsp3": rdMolDescriptors.CalcFractionCSP3,
    "spiro_atoms": rdMolDescriptors.CalcNumSpiroAtoms,
    "amide_bonds": rdMolDescriptors.CalcNumAmideBonds,
    "hbd": Lipinski.NumHDonors,
    "hba": Lipinski.NumHAcceptors,
    "rotatable_bonds": Descriptors.NumRotatableBonds,
    "kappa1": GraphDescriptors.Kappa1,
    "kappa2": GraphDescriptors.Kappa2,
    "kappa3": GraphDescriptors.Kappa3,
    "bertz_ct": GraphDescriptors.BertzCT,
    "logp": Crippen.MolLogP,
}


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_filters(limits):
    if not isinstance(limits, dict) or set(FILTER_LIMITS) - limits.keys():
        raise ValueError("Library is missing required property limits")
    if any(
        type(limits[k]) not in (int, float) or not math.isfinite(limits[k])
        for k in FILTER_LIMITS
    ):
        raise ValueError("Property limits must be finite numbers")


def property_reader(mol):
    """Return a lazy, cached reader for this molecule's published properties."""
    counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    aromatic = Counter(
        atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetIsAromatic()
    )
    heavy = mol.GetNumHeavyAtoms()
    basic = dict(counts)
    basic.update(
        heavy_atoms=heavy,
        aromatic_nitrogen=aromatic["N"],
        aromatic_carbon=aromatic["C"],
        aromatic_fraction=sum(aromatic.values()) / max(heavy, 1),
        heteroatom_fraction=(heavy - counts["C"]) / max(heavy, 1),
        halogens=sum(counts[element] for element in ("F", "Cl", "Br", "I")),
        metals=sum(
            counts[element]
            for element in "Fe Zn Cu Mg Ca Mn Mo Ni Co Pt Au Hg Pd Ru W V Cr Cd Pb As Sb Bi Sn Al Li Na K".split()
        ),
    )

    @cache
    def read(name):
        if name in DESCRIPTORS:
            return DESCRIPTORS[name](mol)
        if name in GROUP_PATTERNS:
            return len(mol.GetSubstructMatches(GROUP_PATTERNS[name]))
        if name == "largest_ring":
            return max(map(len, mol.GetRingInfo().AtomRings()), default=0)
        if name == "triple_bonds":
            return sum(
                bond.GetBondType() == Chem.BondType.TRIPLE for bond in mol.GetBonds()
            )
        if name == "stereocenters":
            return len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        return basic.get(name, 0)

    return read


def passes_filters(mol, limits):
    validate_filters(limits)
    value = property_reader(mol)
    for key, name in FILTER_LIMITS.items():
        actual = value(name)
        if not math.isfinite(actual):
            return False
        if (key.startswith("min_") and actual < limits[key]) or (
            key.startswith("max_") and actual > limits[key]
        ):
            return False
    return all(value(name) == 0 for name in EXCLUDED_PROPERTIES)


def role_mask(value):
    return (
        int.from_bytes(value, "big")
        if isinstance(value, (bytes, bytearray, memoryview))
        else int(value or 0)
    )


def smarts_products(left, right, smarts):
    """Keep the first product of each outcome; canonicalize and retain order."""
    try:
        reaction = AllChem.ReactionFromSmarts(smarts)
        reactants = (Chem.MolFromSmiles(left), Chem.MolFromSmiles(right))
        if reaction is None or any(mol is None for mol in reactants):
            return []
        products = {}
        for outcome in reaction.RunReactants(reactants):
            if outcome:
                try:
                    Chem.SanitizeMol(outcome[0])
                    products.setdefault(Chem.MolToSmiles(outcome[0]), None)
                except (ValueError, RuntimeError):
                    continue
        return list(products)
    except (ValueError, RuntimeError):
        return []


def join_synthons(left, right):
    """Join the neighbors of [1*] and [2*], removing the labelled dummy atoms."""
    try:
        molecules = (Chem.MolFromSmiles(left), Chem.MolFromSmiles(right))
        if any(mol is None for mol in molecules):
            return None
        sites = [
            next(
                (
                    atom
                    for atom in mol.GetAtoms()
                    if atom.GetAtomicNum() == 0 and atom.GetIsotope() == label
                ),
                None,
            )
            for label, mol in enumerate(molecules, 1)
        ]
        if any(atom is None or not atom.GetNeighbors() for atom in sites):
            return None
        first = Chem.RWMol(molecules[0])
        first_site = sites[0].GetIdx()
        first_neighbor = sites[0].GetNeighbors()[0].GetIdx()
        first.RemoveAtom(first_site)
        first_neighbor -= first_neighbor > first_site

        # Build the second fragment from atoms and bond types. This precisely
        # defines the synthon format's bond semantics, including product order.
        second = Chem.RWMol()
        mapping = {
            atom.GetIdx(): second.AddAtom(atom)
            for atom in molecules[1].GetAtoms()
            if atom.GetIdx() != sites[1].GetIdx()
        }
        for bond in molecules[1].GetBonds():
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if begin in mapping and end in mapping:
                second.AddBond(mapping[begin], mapping[end], bond.GetBondType())
        neighbor = mapping[sites[1].GetNeighbors()[0].GetIdx()] + first.GetNumAtoms()
        product = Chem.RWMol(Chem.CombineMols(first, second))
        product.AddBond(first_neighbor, neighbor, Chem.BondType.SINGLE)
        Chem.SanitizeMol(product)
        return Chem.MolToSmiles(product)
    except (ValueError, RuntimeError):
        return None


class Library:
    def __init__(self, directory):
        root = Path(directory).resolve()
        self.manifest = json.loads((root / "library.json").read_text())
        manifest = self.manifest
        if (
            not isinstance(manifest, dict)
            or type(manifest.get("format_version")) is not int
            or manifest["format_version"] != 1
        ):
            raise ValueError("Unsupported library format")
        if (
            manifest.get("rdkit_version") != RDKIT_VERSION
            or rdkit.__version__ != RDKIT_VERSION
        ):
            raise ValueError(f"Use RDKit {RDKIT_VERSION} with this library")
        if not isinstance(manifest.get("allowed_reaction"), str) or not re.fullmatch(
            r"rxn:[0-9]+", manifest["allowed_reaction"]
        ):
            raise ValueError("Invalid library reaction")
        for field in ("case_digest", "database_sha256"):
            if not isinstance(manifest.get(field), str) or not re.fullmatch(
                r"[0-9a-f]{64}", manifest[field]
            ):
                raise ValueError(f"Invalid library {field}")
        validate_filters(manifest.get("filters"))
        self.database = root / "molecules.sqlite"
        if file_sha256(self.database) != manifest["database_sha256"]:
            raise ValueError("Library database hash does not match its manifest")
        self.reaction_id = int(manifest["allowed_reaction"].split(":")[1])
        # The snapshot hash covers this file only. Ignore adjacent WAL/journal
        # files so they cannot silently replace the hashed library contents.
        self.connection = sqlite3.connect(
            self.database.as_uri() + "?mode=ro&immutable=1", uri=True
        )
        try:
            self.connection.execute("PRAGMA query_only=ON")
            self.reaction()
            self.connection.execute(
                "SELECT mol_id,smiles,role_mask FROM molecules LIMIT 0"
            )
        except BaseException:
            self.connection.close()
            raise

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def reaction(self):
        row = self.connection.execute(
            "SELECT smarts,roleA,roleB,roleC FROM reactions WHERE rxn_id=?",
            (str(self.reaction_id),),
        ).fetchone()
        if row is None:
            raise ValueError("Allowed reaction is absent from the library")
        return row[0], *(role_mask(value) for value in row[1:])

    def ids_for_role(self, required):
        if required <= 0:
            return []
        indexed = self.connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='molecule_roles'"
        ).fetchone()
        if indexed:
            bits = [bit for bit in range(required.bit_length()) if required >> bit & 1]
            query = " INTERSECT ".join(
                "SELECT mol_id FROM molecule_roles WHERE bit=?" for _ in bits
            )
            return [
                row[0]
                for row in self.connection.execute(
                    f"SELECT m.mol_id FROM molecules m JOIN ({query}) r ON r.mol_id=m.mol_id",
                    bits,
                )
            ]
        return [
            identifier
            for identifier, mask in self.connection.execute(
                "SELECT mol_id,role_mask FROM molecules"
            )
            if role_mask(mask) & required == required
        ]

    def products(self, molecule_ids):
        if len(molecule_ids) not in (2, 3):
            return []
        rows = [
            self.connection.execute(
                "SELECT smiles,role_mask FROM molecules WHERE mol_id=?", (identifier,)
            ).fetchone()
            for identifier in molecule_ids
        ]
        if any(row is None for row in rows):
            return []
        smarts, role_a, role_b, role_c = self.reaction()
        masks = [role_mask(row[1]) for row in rows]
        has_role = lambda mask, required: required != 0 and mask & required == required
        if has_role(masks[0], role_a) and has_role(masks[1], role_b):
            first, second = rows[0][0], rows[1][0]
        elif has_role(masks[1], role_a) and has_role(masks[0], role_b):
            first, second = rows[1][0], rows[0][0]
        else:
            return []
        if len(rows) == 2:
            if self.reaction_id == 1:
                joined = join_synthons(first, second)
                return [joined] if joined else []
            return smarts_products(first, second, smarts)
        if not has_role(masks[2], role_c):
            return []
        third = rows[2][0]
        if self.reaction_id == 3:
            intermediate = join_synthons(first, second)
            return (
                smarts_products(
                    intermediate, third, "[C:1](=O)[OH].[N:2]>>[C:1](=O)[N:2]"
                )
                if intermediate
                else []
            )
        if self.reaction_id == 5:
            intermediates = smarts_products(
                first, second, "[#6:1][Br].[#6:2][B]([OH])[OH]>>[#6:1][#6:2]"
            )
            return (
                smarts_products(
                    intermediates[0],
                    third,
                    "[#6:1][Cl].[#6:2][B]([OH])[OH]>>[#6:1][#6:2]",
                )
                if intermediates
                else []
            )
        return []

    def resolve(self, product_id):
        match = PRODUCT_ID.fullmatch(product_id)
        if match is None or int(match[1]) != self.reaction_id:
            return None
        molecules = [int(value) for value in match.group(2, 3, 4) if value is not None]
        variants = self.products(molecules)
        index = int(match[5] or 0)
        return variants[index] if index < len(variants) else None
