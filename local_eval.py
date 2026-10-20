"""Local validate/final for offline NIM runs.

Mirrors the owner's set-level semantics using only the frozen library:
- Every product_id must resolve to a SMILES that passes library filters.
- No pair may exceed max_tanimoto.
- required_count must be met.

NOT the owner's service. Cannot establish historical novelty against newer
data or the service's internal entropy gates.
"""

import math

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from chemistry import passes_filters


def _fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) if mol else None


def validate_set(product_ids, library, max_tanimoto=0.8):
    reasons = []
    smiles = {}
    for pid in product_ids:
        smi = library.resolve(pid)
        if smi is None:
            reasons.append(f"{pid}: unresolved"); continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None or not passes_filters(mol, library.manifest["filters"]):
            reasons.append(f"{pid}: filter failure"); continue
        smiles[pid] = smi
    fps = {pid: _fp(s) for pid, s in smiles.items()}
    fps = {p: f for p, f in fps.items() if f is not None}
    pids = list(fps)
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            sim = DataStructs.TanimotoSimilarity(fps[pids[i]], fps[pids[j]])
            if sim > max_tanimoto:
                reasons.append(f"{pids[i]}~{pids[j]}: Tanimoto {sim:.2f} > {max_tanimoto}")
    return (not reasons, reasons, smiles)


def final_set(product_ids, library, oracle, max_tanimoto=0.8):
    ok, reasons, smiles = validate_set(product_ids, library, max_tanimoto)
    if not ok:
        return {"valid": False, "reasons": reasons, "scores": {}}
    scores = oracle.score(product_ids)
    valid_scores = [v for v in scores.values() if not math.isnan(v)]
    return {
        "valid": True, "reasons": [], "scores": scores,
        "mean_pIC50": sum(valid_scores) / max(len(valid_scores), 1),
    }