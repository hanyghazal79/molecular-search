"""Synthetic accessibility complement in (0, 1], higher is better."""

import os
import sys
from dataclasses import dataclass
from typing import List

import numpy as np
from rdkit import Chem, RDConfig

_SA = os.path.join(RDConfig.RDContribDir, "SA_Score")
if _SA not in sys.path:
    sys.path.append(_SA)
import sascorer  # noqa: E402

from reinvent_plugins.components.add_tag import add_tag
from reinvent_plugins.components.component_results import ComponentResults


@add_tag("__parameters")
@dataclass
class Parameters:
    unused: List[List[str]]


@add_tag("__component")
class SAComplement:
    def __init__(self, params: Parameters):
        pass

    def __call__(self, smiles: List[str]) -> ComponentResults:
        out = np.full(len(smiles), np.nan, dtype=np.float32)
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            out[i] = float(np.clip((10.0 - sascorer.calculateScore(mol)) / 9.0, 0.01, 1.0))
        return ComponentResults([out])
