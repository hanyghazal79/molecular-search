# 1. Install dependencies
!pip install -q torch-geometric

# 2. Write all files to disk
#    Run the %%writefile cells for:
#    - sota_utils.py
#    - search_gnn.py
#    - search_rl.py
#    - search_mobo.py
#    - search_amortized.py

# 3. Set API key
import os
from kaggle_secrets import UserSecretsClient
os.environ["NVIDIA_API_KEY"] = UserSecretsClient().get_secret("NVIDIA_API_KEY")

# 4. Create output directories
!mkdir -p /kaggle/working/04_sota_experiments/results/{gnn,rl,mobo,amortized}

# 5. Run all four experiments (in order)
!python 04_sota_experiments/search_amortized.py \
    --library synthetic-library-100000 --pool pool-100k.csv \
    --target-sequence "$HIV_SEQ" --train-probes 100 --seed 42 \
    --output /kaggle/working/04_sota_experiments/results/amortized/run-s42.json

!python 04_sota_experiments/search_mobo.py \
    --library synthetic-library-100000 --pool pool-100k.csv \
    --target-sequence "$HIV_SEQ" --probes 100 --seed 42 \
    --output /kaggle/working/04_sota_experiments/results/mobo/run-s42.json

!python 04_sota_experiments/search_gnn.py \
    --library synthetic-library-100000 --pool pool-100k.csv \
    --target-sequence "$HIV_SEQ" --strategy ei --probes 100 --seed 42 \
    --output /kaggle/working/04_sota_experiments/results/gnn/ei-s42.json

!python 04_sota_experiments/search_rl.py \
    --library synthetic-library-100000 --pool pool-100k.csv \
    --target-sequence "$HIV_SEQ" --pretrain-epochs 20 --rl-steps 100 --seed 42 \
    --output /kaggle/working/04_sota_experiments/results/rl/run-s42.json

# 6. Compare all results
!python metrics.py /kaggle/working/04_sota_experiments/results/*/*.json \
    --pool pool-100k.csv \
    --csv-out /kaggle/working/04_sota_experiments/results/SOTA_COMPARISON.csv