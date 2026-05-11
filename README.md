# Causal Inference: From Identification to Do-Calculus

A series of Jupyter notebooks exploring the theory and practice of causal inference, covering Simpson's paradox, probabilistic graphical models, and do-calculus. Each notebook builds on the last, moving from intuitive examples to a formal framework for reasoning about interventions in observational data.

---

## Project Structure

```
project/
├── data/
│   ├── data1.csv          # Synthetic dosage/recovery dataset (Simpson's paradox)
│   └── data2.txt          # Supporting data for graphical model examples
├── figures/
│   ├── DAG1.png – DAG6.png          # DAG diagrams used throughout the notebooks
│   ├── do_calculus_rules.png        # Visual summary of the three do-calculus rules
│   ├── graph_rules.png              # Meek's orientation rules for CPDAGs
│   └── student_model.png           # Student Bayesian network example
├── notebooks/
│   ├── 01_causal_identification.ipynb
│   ├── 02_probabilistic_graphical_models.ipynb
│   ├── 03_do_calculus.ipynb
│   └── utils.py
└── requirements.txt
```

---

## Notebooks

### 01 — Causal Identification

Introduces the core motivation for causal inference through Simpson's paradox: a synthetic dosage/recovery dataset shows how an aggregate correlation can reverse once we condition on a confounder (severity).

Topics covered:
- Simpson's paradox and confounding
- Conditioning vs. intervening; Pearl's *do(·)* operator
- Potential outcomes and the fundamental problem of causal inference
- Average treatment effect (ATE)
- The adjustment formula and backdoor adjustment
- Illustrated using the 1979 Berkeley admissions data

### 02 — Probabilistic Graphical Models

Extends the single-confounder setting to systems of many interacting variables using Directed Acyclic Graphs (DAGs).

Topics covered:
- DAG structure: nodes, directed edges, acyclicity
- The local Markov assumption and joint distribution factorisation
- Conditional independence and d-separation
- The student Bayesian network as a running example
- Markov equivalence classes and CPDAGs
- Structure learning from observational data (PC algorithm)
- Meek's orientation rules for pruning admissible DAGs

### 03 — Do-Calculus

Addresses the limits of covariate adjustment and introduces a complete formal framework for identifying causal effects from observational data.

Topics covered:
- Conditioning vs. intervention revisited
- Randomised controlled trials (RCTs) and their limitations
- d-separation rules: chains, forks, and colliders
- Collider bias and M-bias
- The backdoor and frontdoor criteria
- When no adjustment set exists (unobserved confounding)
- The three rules of do-calculus

---

## Setup

**Requirements:** Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages include `causal-learn`, `pgmpy`, `networkx`, `matplotlib`, `pandas`, `numpy`, `scikit-learn`, and `graphviz`.

> **Note:** Graphviz also requires a system-level installation. On macOS: `brew install graphviz`. On Ubuntu/Debian: `sudo apt install graphviz`.

Launch the notebooks:

```bash
cd notebooks
jupyter notebook
```

Run them in order (01 → 02 → 03), as later notebooks reference concepts introduced in earlier ones.

---

## Utility Functions (`utils.py`)

Four helper functions used across the notebooks:

- `plot_reg(df)` — fits and overlays a simple linear regression line on the current matplotlib figure
- `nx_to_causallearn_graph(nx_dag)` — converts a NetworkX `DiGraph` into a `causal-learn` `GeneralGraph` for compatibility with structure learning algorithms
- `cpdag_to_nx(cpdag)` — converts a `causal-learn` CPDAG back into two NetworkX graphs: one for identified directed edges, one for undirected (unresolved orientation) edges
- `show_assoc(X, Y, Z, title)` — prints marginal and conditional association tables for binary variables, useful for illustrating confounding and collider bias
