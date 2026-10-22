## 🔮 Future Work: State-of-the-Art (SOTA) AI Methods

While the current pipeline (Random Forest + Expected Improvement) has proven highly successful, I have scaffolded a `04_sota_experiments/` folder in the repository containing four **production-grade** implementations of State-of-the-Art AI methods. 

These scripts are fully written, tested locally for syntax and logic, but **have not yet been executed on the 100K benchmark**. They are designed to be run the moment we have access to the real client scoring case, representing the next frontier of optimization.

### The Scaffolded Methods:

1.  **GNN Surrogate (`search_gnn.py`):** Replaces the Random Forest with a Graph Neural Network (GIN) ensemble. It learns directly from the molecular graph rather than flat fingerprints, which typically improves sample efficiency by 15–30% on complex targets.
2.  **RL Generator (`search_rl.py`):** Instead of selecting from a fixed library, this uses a character-level GRU fine-tuned with REINFORCE. It **generates completely novel SMILES** to maximize the NIM affinity score, discovering molecules outside the client's original library.
3.  **Multi-Objective BO (`search_mobo.py`):** Optimizes three objectives simultaneously: Affinity (pIC50), Synthesizability (SA score), and Novelty (Tanimoto distance to history). It returns a Pareto front of trade-offs.
4.  **Amortized Optimization (`search_amortized.py`):** Trains an MLP once on a small batch, then scores the entire 100,000-molecule pool in a single forward pass without any additional oracle calls. This is designed for zero-cost production re-ranking.

**Status:** Designed and implemented. Pending execution on the real client workspace.