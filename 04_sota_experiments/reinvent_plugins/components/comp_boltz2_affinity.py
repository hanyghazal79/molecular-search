"""Boltz-2 affinity component for REINVENT 4.

Modes:
  - "surrogate": RandomForest on prior Boltz-2 labels (production default)
  - "direct":    live NVIDIA NIM Boltz-2 call (final validation only)

Returns pIC50 (4–9) mapped to (0, 1] for geometric-mean composition.
HTTP 422 (Boltz cannot fold) → np.nan (molecule treated invalid).
HTTP 429 after 3 exponential backoffs / 5xx / network → RuntimeError (run stops).
"""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from reinvent_plugins.components.add_tag import add_tag
from reinvent_plugins.components.component_results import ComponentResults

ENDPOINT = "https://health.api.nvidia.com/v1/biology/mit/boltz2/predict"
_MORGAN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)


def _fp(mol):
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(_MORGAN.GetFingerprint(mol), arr)
    return arr


def _normalise(pic50):
    return float(np.clip((pic50 - 4.0) / 5.0, 0.01, 1.0))


@add_tag("__parameters")
@dataclass
class Parameters:
    # REINVENT combines one entry per endpoint: outer list = endpoints,
    # inner list = TOML array for that endpoint's parameter.
    mode: List[List[str]]
    target_sequence: List[List[str]]
    api_key_env: List[List[str]]
    cache_path: List[List[str]]
    surrogate_path: List[List[str]]
    timeout: List[List[int]]


@add_tag("__component")
class Boltz2Affinity:
    def __init__(self, params: Parameters):
        self.mode = params.mode[0][0]
        self.target = params.target_sequence[0][0]
        self.api_key_env = params.api_key_env[0][0]
        self.cache_path = Path(params.cache_path[0][0])
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.timeout = int(params.timeout[0][0])

        if self.mode == "surrogate":
            path = Path(params.surrogate_path[0][0])
            if not path.exists():
                raise RuntimeError(f"surrogate model not found: {path}")
            bundle = joblib.load(path)
            self.model, self.y_mean, self.y_std = (
                bundle["model"], bundle["y_mean"], bundle["y_std"]
            )
        elif self.mode == "direct":
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"env var {self.api_key_env} not set")
            self.api_key = key
        else:
            raise ValueError(f"unknown mode: {self.mode}")

    def _cache_key(self, smi):
        payload = f"{self.target}|{smi}|200|5"
        return hashlib.sha256(payload.encode()).hexdigest()[:20]

    def _surrogate(self, smiles):
        out = np.full(len(smiles), np.nan, dtype=np.float32)
        rows, idx = [], []
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            rows.append(_fp(mol))
            idx.append(i)
        if rows:
            preds = self.model.predict(np.stack(rows)) * self.y_std + self.y_mean
            for i, p in zip(idx, preds):
                out[i] = _normalise(p)
        return out

    def _direct(self, smiles):
        out = np.full(len(smiles), np.nan, dtype=np.float32)
        for i, smi in enumerate(smiles):
            if Chem.MolFromSmiles(smi) is None:
                continue
            key = self._cache_key(smi)
            if key in self.cache:
                out[i] = self.cache[key]
                continue
            time.sleep(5.0)  # 12 RPM ceiling; matches NimOracle in nim_oracle.py
            try:
                pic50 = self._call_boltz(smi)
            except RuntimeError as exc:
                # Single-candidate failure should not kill the 200-molecule loop.
                # Persist what we have; re-running Cell H resumes from the cache.
                print(f"  [Boltz2 skip] {smi[:40]}: {exc}", flush=True)
                continue
            if pic50 is not None:
                val = _normalise(pic50)   # Python float
                out[i] = val              # coerced to np.float32 in the array
                self.cache[key] = val     # Python float → JSON-serializable
                self.cache_path.write_text(json.dumps(self.cache))
        return out

    def _call_boltz(self, smi):
        payload = {
            "polymers": [{"id": "A", "molecule_type": "protein", "sequence": self.target}],
            "ligands": [{"id": "LIG", "smiles": smi, "predict_affinity": True}],
            "sampling_steps_affinity": 200,
            "diffusion_samples_affinity": 5,
        }
        req = urllib.request.Request(
            ENDPOINT, data=json.dumps(payload).encode(), method="POST",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    body = json.load(r)
                val = body.get("affinities", {}).get("LIG", {}).get("affinity_pic50")
                return None if val is None else float(val[0])
            except urllib.error.HTTPError as exc:
                if exc.code == 422:           # Boltz cannot fold this ligand
                    return None
                if exc.code == 429:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        wait = int(retry_after) if retry_after else 30 * (attempt + 1)
                    except (TypeError, ValueError):
                        wait = 30 * (attempt + 1)
                    print(f"  [Boltz2 429] attempt {attempt+1}/6, sleeping {wait}s",
                          flush=True)
                    time.sleep(wait)
                    continue
                if 500 <= exc.code < 600:      # transient server errors
                    wait = 10 * (attempt + 1)
                    print(f"  [Boltz2 {exc.code}] attempt {attempt+1}/6, sleeping {wait}s",
                          flush=True)
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"Boltz NIM HTTP {exc.code}: "
                    f"{exc.read(512).decode(errors='replace')}"
                ) from None
            except (urllib.error.URLError, OSError) as exc:
                if attempt == 5:
                    raise RuntimeError(f"Boltz NIM request failed: {exc}") from None
                time.sleep(10 * (attempt + 1))
        raise RuntimeError("Boltz NIM rate-limit retries exhausted")
       


    
    def __call__(self, smiles: List[str]) -> ComponentResults:
        vals = self._surrogate(smiles) if self.mode == "surrogate" else self._direct(smiles)
        return ComponentResults([vals])
