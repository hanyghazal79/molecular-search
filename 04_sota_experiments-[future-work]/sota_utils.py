"""Shared utilities for SOTA experiments.

Provides:
- SMILES -> PyTorch Geometric graph conversion
- Synthetic Accessibility (SA) score computation
- Pareto dominance & non-dominated sorting
- Amortized MLP model
- Path resolution for import from parent directory
"""

import sys
from pathlib import Path

# Ensure parent directory is on path so `import chemistry`, `import nim_oracle` work
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

try:
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    Data = None


# ---------------------------------------------------------------------------
# SMILES -> Graph conversion
# ---------------------------------------------------------------------------

ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si', 'B', 'Na', 'K', 'Ca', 'Li', 'Mg']
ATOM_TO_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}
ATOM_TO_IDX['*'] = len(ATOM_TYPES)  # catch-all for others
N_ATOM_FEATURES = len(ATOM_TYPES) + 1


def smiles_to_graph(smiles: str) -> 'Data':
    """Convert SMILES to a PyTorch Geometric Data object.

    Node features: one-hot atom type + degree + formal charge + aromatic flag.
    Edge features: bond type one-hot (single, double, triple, aromatic).
    """
    if not HAS_PYG:
        raise RuntimeError("torch_geometric is required for GNN operations")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    x = []
    for atom in mol.GetAtoms():
        feat = [0.0] * (N_ATOM_FEATURES + 3)
        idx = ATOM_TO_IDX.get(atom.GetSymbol(), ATOM_TO_IDX['*'])
        feat[idx] = 1.0
        feat[N_ATOM_FEATURES] = atom.GetDegree() / 4.0
        feat[N_ATOM_FEATURES + 1] = atom.GetFormalCharge()
        feat[N_ATOM_FEATURES + 2] = float(atom.GetIsAromatic())
        x.append(feat)
    x = torch.tensor(x, dtype=torch.float)

    # Edge features (undirected, both directions)
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        bond_feat = [
            float(bt == Chem.BondType.SINGLE),
            float(bt == Chem.BondType.DOUBLE),
            float(bt == Chem.BondType.TRIPLE),
            float(bt == Chem.BondType.AROMATIC),
        ]
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([bond_feat, bond_feat])

    if not edge_index:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)


# ---------------------------------------------------------------------------
# Synthetic Accessibility (SA) score
# ---------------------------------------------------------------------------

def sa_score(smiles: str) -> float:
    """Ertl & Schuffenhauer synthetic accessibility (1=easy, 10=hard).

    Uses RDKit's contrib SA_Score module if available; otherwise falls back
    to a simple heuristic based on ring count and stereocenters.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 10.0
    try:
        from rdkit.Chem import RDConfig
        import os
        sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
        if sa_path not in sys.path:
            sys.path.append(sa_path)
        import sascorer
        return float(sascorer.calculateScore(mol))
    except Exception:
        # Fallback heuristic
        n_rings = mol.GetRingInfo().NumRings()
        n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        n_heavy = mol.GetNumHeavyAtoms()
        return min(10.0, 1.0 + 0.3 * n_rings + 0.2 * n_stereo + 0.01 * n_heavy)


# ---------------------------------------------------------------------------
# Pareto dominance / non-dominated sorting
# ---------------------------------------------------------------------------

def pareto_dominates(a, b) -> bool:
    """True if a dominates b (a is >= b in all objectives, > in at least one).
    Assumes maximization for all objectives.
    """
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a >= b) and np.any(a > b))


def pareto_fronts(points):
    """Return list of fronts, each front a list of indices into `points`.
    points: list of objective vectors.
    """
    points = [np.asarray(p) for p in points]
    remaining = set(range(len(points)))
    fronts = []
    while remaining:
        front = []
        for i in list(remaining):
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                if pareto_dominates(points[j], points[i]):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        fronts.append(front)
        remaining -= set(front)
    return fronts


def pareto_rank(points):
    """Assign each point its Pareto front index (0 = best)."""
    fronts = pareto_fronts(points)
    ranks = [0] * len(points)
    for rank, front in enumerate(fronts):
        for i in front:
            ranks[i] = rank
    return ranks


# ---------------------------------------------------------------------------
# Amortized MLP model
# ---------------------------------------------------------------------------

class AmortizedMLP(nn.Module):
    """Small MLP mapping Morgan fingerprint -> predicted score."""

    def __init__(self, n_bits: int = 2048, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bits, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Common fingerprint helper
# ---------------------------------------------------------------------------

_MORGAN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)


def morgan_np(smiles: str) -> np.ndarray:
    """Return 2048-bit Morgan fingerprint as numpy array or None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = _MORGAN.GetFingerprint(mol)
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)