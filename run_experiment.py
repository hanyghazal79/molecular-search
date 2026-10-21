"""Dynamically run the FULL Phase 1-3 benchmark for ANY molecule count."""
import os
import shutil
import argparse
import subprocess
from pathlib import Path
import pandas as pd

# Import local modules
from generate_pool import main as gen_main
from search_nim import search_main
from metrics import main as metrics_main

HIV_SEQ = ("PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYD"
           "QILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF")

def main():
    parser = argparse.ArgumentParser(description="Run dynamic molecular search benchmark.")
    parser.add_argument("--count", type=int, required=True, help="Target pool size")
    parser.add_argument("--probes", type=int, default=200, help="NIM probes per strategy")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    parser.add_argument("--required", type=int, default=5, help="Final set size required")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds (e.g., 42,43,44)")
    parser.add_argument("--ablation", action="store_true", help="If set, run greedy and ucb ablation")
    args = parser.parse_args()

    count = args.count
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    print(f"\n{'='*60}")
    print(f"🚀 Benchmarking {count:,} molecules | Seeds: {seeds} | Ablation: {args.ablation}")
    print(f"{'='*60}")

    # 1. Setup paths
    lib_dir = f"synthetic-library-{count}"
    out_dir = Path(f"/kaggle/working/results-{count}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_csv = str(out_dir / f"pool-{count}.csv")
    cache_file = "nim_cache.json"

    # 2. Build Library dynamically
    print(f"\n[1/5] Building library {lib_dir}...")
    subprocess.run(["python", "make_synthetic_library.py", str(count), lib_dir], check=True)

    # 3. Generate Pool
    print(f"\n[2/5] Generating {count:,} candidate pool...")
    gen_main(
        library_dir=lib_dir, reaction="rxn:1", count=count,
        attempts=int(count * 3), seed=42, output=pool_csv
    )

    # 4. Run Strategies
    result_files = []
    
    # Phase 2 Ablation (Seed 42 only) - Greedy & UCB
    if args.ablation:
        for strategy in ["greedy", "ucb"]:
            print(f"\n[3/5] Ablation: {strategy.upper()} (Seed 42)...")
            out_file = str(out_dir / f"{strategy}-s42.json")
            search_main(
                library_dir=lib_dir, pool_path=pool_csv, target_sequence=HIV_SEQ,
                strategy=strategy, seed=42, probes=args.probes, batch_size=args.batch_size,
                required=args.required, max_sim=0.6, cache=cache_file, output=out_file
            )
            result_files.append(out_file)

    # Phase 1 & 2: Random vs EI across all seeds
    for seed in seeds:
        for strategy in ["random", "ei"]:
            print(f"\n[3/5] Running {strategy.upper()} (Seed {seed})...")
            out_file = str(out_dir / f"{strategy}-s{seed}.json")
            search_main(
                library_dir=lib_dir, pool_path=pool_csv, target_sequence=HIV_SEQ,
                strategy=strategy, seed=seed, probes=args.probes, batch_size=args.batch_size,
                required=args.required, max_sim=0.6, cache=cache_file, output=out_file
            )
            result_files.append(out_file)

    # 5. Compile Metrics (Phase 3)
    print(f"\n[4/5] Compiling metrics...")
    csv_out = str(out_dir / "metrics.csv")
    metrics_main(result_files, pool=pool_csv, csv_out=csv_out)

    print(f"\n[5/5] Phase 2 Summary")
    df = pd.read_csv(csv_out)
    summary = df.groupby("checkpoint_strategy").agg({
        "best_per_pair": ["mean", "std", "count"],
        "best_probe_score": ["mean"],
        "final_valid": lambda x: (x == True).sum()
    }).round(3)
    print(summary)
    
    summary.to_csv(out_dir / "summary.csv")
    shutil.make_archive(f"/kaggle/working/results-{count}", 'zip', out_dir)
    print(f"\n✅ Done! Zipped to /kaggle/working/results-{count}.zip")

if __name__ == "__main__":
    main()
