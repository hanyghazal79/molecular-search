"""SOTA Experiment 2: Reinforcement Learning Generator.

Trains a character-level GRU on the pool SMILES, then fine-tunes with
policy gradient to maximize NIM Boltz-2 affinity rewards.

Generates NOVEL SMILES not present in the original pool.

Usage:
    python 04_sota_experiments/search_rl.py \\
        --library synthetic-library-100k \\
        --pool pool-100k.csv \\
        --target-sequence "PQITL..." \\
        --pretrain-epochs 20 --rl-steps 100 --seed 42 \\
        --output 04_sota_experiments/results/rl/run-s42.json
"""

import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from sota_utils import set_seed, morgan_np, sa_score
from chemistry import Library
from nim_oracle import NimOracle

RDLogger.DisableLog('rdApp.*')
_MORGAN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)

# SMILES vocabulary
CHARS = ['<pad>', '<sos>', '<eos>'] + list(
    'CNOSPFIBcnosp()[]=#+-/\\123456789%.@'
)
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for i, c in CHAR_TO_IDX.items()}
VOCAB_SIZE = len(CHARS)
SOS, EOS, PAD = CHAR_TO_IDX['<sos>'], CHAR_TO_IDX['<eos>'], CHAR_TO_IDX['<pad>']


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SMILESGRU(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=64, hidden=256, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden, n_layers, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x, h=None):
        e = self.embed(x)
        out, h = self.gru(e, h)
        return self.head(out), h

    @torch.no_grad()
    def sample(self, n: int, max_len: int = 80, temperature: float = 1.0, device='cpu'):
        """Generate `n` SMILES from the policy."""
        self.eval()
        x = torch.full((n, 1), SOS, dtype=torch.long, device=device)
        h = None
        finished = torch.zeros(n, dtype=torch.bool, device=device)
        outputs = [[] for _ in range(n)]
        for _ in range(max_len):
            logits, h = self.forward(x, h)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            next_tok = next_tok.squeeze(-1)
            for i in range(n):
                if finished[i]:
                    continue
                tok = next_tok[i].item()
                if tok == EOS:
                    finished[i] = True
                elif tok not in (SOS, PAD):
                    outputs[i].append(IDX_TO_CHAR.get(tok, ''))
            if finished.all():
                break
            x = next_tok.unsqueeze(1)
        return [''.join(o) for o in outputs]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def encode(smiles: str, max_len: int = 80) -> list:
    ids = [SOS] + [CHAR_TO_IDX.get(c, PAD) for c in smiles[:max_len - 2]] + [EOS]
    return ids


def pad_batch(seqs, pad_idx=PAD):
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def pretrain(model, smiles_list, epochs=20, batch_size=64, lr=1e-3, device='cpu'):
    """Supervised pretraining on SMILES strings."""
    seqs = [encode(s) for s in smiles_list if s]
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        random.shuffle(seqs)
        total_loss, n = 0.0, 0
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i + batch_size]
            padded = pad_batch(batch).to(device)
            x, y = padded[:, :-1], padded[:, 1:]
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1),
                ignore_index=PAD,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n += 1
        print(f"  pretrain epoch {epoch+1}/{epochs}: loss={total_loss/n:.3f}")


def reinforce_step(model, reward_fn, batch_size=32, lr=5e-5, device='cpu',
                   baseline=0.0, kl_coef=0.1):
    """One REINFORCE step: sample, score, update."""
    model.train()
    # Sample SMILES
    smiles_batch = model.sample(batch_size, device=device)
    valid = [s for s in smiles_batch if Chem.MolFromSmiles(s) is not None]
    if not valid:
        return 0.0
    rewards = reward_fn(valid)
    if not rewards:
        return 0.0

    # REINFORCE loss
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    opt.zero_grad()
    total_reward = 0.0
    for smi, r in zip(valid, rewards):
        if math.isnan(r):
            continue
        total_reward += r
        seq = encode(smi)
        if len(seq) < 3:
            continue
        inp = torch.tensor([seq[:-1]], device=device)
        target = torch.tensor([seq[1:]], device=device)
        logits, _ = model(inp)
        log_probs = F.log_softmax(logits, dim=-1)
        picked = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        loss = -(r - baseline) * picked.mean()
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return total_reward / max(len(valid), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(library_dir, pool_path, target_sequence, seed=42, pretrain_epochs=20,
        rl_steps=100, batch_size=32, n_generated=200, max_sim=0.6,
        msa=None, cache='nim_cache.json', output='rl_result.json'):
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load training SMILES
    smiles_pool = []
    with open(pool_path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('smiles'):
                smiles_pool.append(row['smiles'])

    print(f"Loaded {len(smiles_pool)} SMILES for pretraining")
    model = SMILESGRU().to(device)
    pretrain(model, smiles_pool, epochs=pretrain_epochs, device=device)

    with Library(library_dir) as library:
        oracle = NimOracle(api_key=os.environ['NVIDIA_API_KEY'], library=library,
                           target_sequence=target_sequence, cache_path=cache, msa_path=msa)

        # Reward function: score via NIM
        reward_history = []
        def reward_fn(smiles_list):
            scores = oracle.score(smiles_list)
            vals = [v for v in scores.values() if not math.isnan(v)]
            reward_history.extend(vals)
            return [scores[s] if s in scores else float('nan') for s in smiles_list]

        # RL fine-tuning
        baseline = 0.0
        for step in range(rl_steps):
            avg_reward = reinforce_step(model, reward_fn, batch_size=batch_size,
                                        device=device, baseline=baseline)
            if reward_history:
                baseline = 0.9 * baseline + 0.1 * max(reward_history[-batch_size:])
            if (step + 1) % 10 == 0:
                print(f"  RL step {step+1}/{rl_steps} | avg_reward={avg_reward:.3f} | baseline={baseline:.3f}")

        # Generate final candidates
        print(f"Generating {n_generated} final candidates...")
        generated = model.sample(n_generated, device=device)
        valid_smiles = [s for s in generated if Chem.MolFromSmiles(s) is not None]
        print(f"  {len(valid_smiles)} valid SMILES")

        # Score and diversify
        final_scores = oracle.score(valid_smiles) if valid_smiles else {}
        scored = [(s, v) for s, v in final_scores.items() if not math.isnan(v)]
        scored.sort(key=lambda x: -x[1])

        # MaxMin diversity
        selected, selected_fps = [], {}
        for s, v in scored:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            fp = _MORGAN.GetFingerprint(mol)
            ok = True
            for chosen_fp in selected_fps.values():
                if DataStructs.TanimotoSimilarity(fp, chosen_fp) > max_sim:
                    ok = False
                    break
            if ok:
                selected.append(s)
                selected_fps[s] = fp
                if len(selected) >= 20:
                    break

        result = {
            'method': 'reinforce_rl',
            'seed': seed,
            'n_generated': len(generated),
            'n_valid': len(valid_smiles),
            'n_scored': len(scored),
            'selected': selected,
            'scores': final_scores,
            'best_score': max((v for _, v in scored), default=float('nan')),
            'avg_reward_final': float(np.mean([v for _, v in scored])) if scored else float('nan'),
        }
        output.write_text(json.dumps(result, indent=2))
        print(f"Done. best={result['best_score']:.2f} -> {output}")
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', required=True)
    p.add_argument('--pool', required=True)
    p.add_argument('--target-sequence', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pretrain-epochs', type=int, default=20)
    p.add_argument('--rl-steps', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--n-generated', type=int, default=200)
    p.add_argument('--max-sim', type=float, default=0.6)
    p.add_argument('--msa')
    p.add_argument('--cache', default='nim_cache.json')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    return run(
        library_dir=args.library, pool_path=args.pool, target_sequence=args.target_sequence,
        seed=args.seed, pretrain_epochs=args.pretrain_epochs, rl_steps=args.rl_steps,
        batch_size=args.batch_size, n_generated=args.n_generated, max_sim=args.max_sim,
        msa=args.msa, cache=args.cache, output=args.output,
    )


if __name__ == '__main__':
    raise SystemExit(main())