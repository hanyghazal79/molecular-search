"""NVIDIA NIM Boltz-2 oracle with disk cache and rate limiting."""
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINT = "https://health.api.nvidia.com/v1/biology/mit/boltz2/predict"

class NimError(RuntimeError):
    pass

class NimOracle:
    def __init__(self, api_key, library, target_sequence,
                 cache_path="nim_cache.json", msa_path=None,
                 sampling_steps=200, diffusion_samples=5,
                 requests_per_minute=12, timeout=600): # <-- Changed to 25 RPM
        self.api_key = api_key
        self.library = library
        self.target_sequence = target_sequence
        self.cache_path = Path(cache_path)
        self.msa_path = msa_path
        self.sampling_steps = sampling_steps
        self.diffusion_samples = diffusion_samples
        self.min_interval = 60.0 / requests_per_minute
        self.timeout = timeout
        self._last_call = 0.0
        self._cache = self._load_cache()

    def _load_cache(self):
        return json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}

    def _save_cache(self):
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache, indent=0))
        tmp.replace(self.cache_path)

    def _key(self, smiles):
        payload = f"{self.target_sequence}|{smiles}|{self.sampling_steps}|{self.diffusion_samples}"
        return hashlib.sha256(payload.encode()).hexdigest()[:20]

    def _wait(self):
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()

    def score(self, product_ids):
        results = {}
        for pid in product_ids:
            smiles = self.library.resolve(pid)
            if smiles is None:
                results[pid] = float("nan")
                continue
            key = self._key(smiles)
            if key in self._cache:
                results[pid] = self._cache[key]
                continue
            try:
                pIC50 = self._call_nim(smiles)
            except NimError as exc:
                print(f"  NIM error on {pid}: {exc}")
                pIC50 = float("nan")
            results[pid] = pIC50
            if not math.isnan(pIC50):
                self._cache[key] = pIC50
                self._save_cache()
        return results


    def score_smiles(self, smiles_list):
        """Score raw SMILES directly, bypassing the library lookup.

        Used by RL and other de novo generators that produce molecules not
        present in the library. Returns {smiles: pIC50 or NaN}.
        """
        results = {}
        for smi in smiles_list:
            if not smi or Chem.MolFromSmiles(smi) is None:
                results[smi] = float("nan")
                continue
            key = self._key(smi)
            if key in self._cache:
                results[smi] = self._cache[key]
                continue
            try:
                pIC50 = self._call_nim(smi)
            except NimError as exc:
                print(f"  NIM error on {smi[:40]}: {exc}")
                pIC50 = float("nan")
            results[smi] = pIC50
            if not math.isnan(pIC50):
                self._cache[key] = pIC50
                self._save_cache()
        return results

    
    
    def _call_nim(self, smiles):
        self._wait()
        payload = {
            "polymers": [{"id": "A", "molecule_type": "protein",
                          "sequence": self.target_sequence}],
            "ligands": [{"id": "LIG", "smiles": smiles, "predict_affinity": True}],
            "sampling_steps_affinity": self.sampling_steps,
            "diffusion_samples_affinity": self.diffusion_samples,
        }
        if self.msa_path:
            payload["msas"] = [{"id": "A", "path": str(self.msa_path)}]

        req = urllib.request.Request(
            ENDPOINT, data=json.dumps(payload).encode(), method="POST",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})
        
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.load(resp)
                    return float(body["affinities"]["LIG"]["affinity_pic50"][0])
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    backoff = 10 * (attempt + 1)         # 10s → 20s → 30s
                    print(f"  Rate limited (429). Sleeping {backoff}s before retry {attempt+1}/3...")
                    time.sleep(backoff)
                    self._last_call = time.monotonic()   # reset spacing clock on retry
                    continue

                
                raise NimError(f"HTTP {exc.code}: {exc.read(512).decode(errors='replace')}") from None
            except (urllib.error.URLError, OSError) as exc:
                raise NimError(f"request failed: {exc}") from None
            except (KeyError, IndexError, TypeError, ValueError):
                return float("nan")
                
        print("  Failed after 3 retries due to rate limiting.")
        return float("nan")
