# Causal Inference — A Notebook Series

A self-contained series of Jupyter notebooks covering the foundations of causal inference, from Simpson's paradox through to do-calculus. Each notebook builds on the last, combining theory, simulation, and graphical reasoning.

---

## Project Structure

```
project/
├── data/
│   └── obs_data.txt              # Observational data for the student Bayesian network
├── figures/
│   ├── DAG1.png – DAG6.png       # Causal graphs used throughout the notebooks
│   ├── do_calculus_rules.png     # The three rules of do-calculus
│   ├── graph_rules.png           # Meek's orientation rules
│   └── student_model.png         # Ground-truth student model (Koller & Friedman)
├── notebooks/
│   ├── 01_causal_identification.ipynb
│   ├── 02_probabilistic_graphical_models.ipynb
│   └── 03_do_calculus.ipynb
└── src/
    └── utils.py                  # Shared helper functions
```

---

## Notebooks

### 01 — Causal Identification

Motivates causal inference through Simpson's paradox and the Berkeley admissions data, and introduces the core tools for identifying causal effects from observational data.

- Simpson's paradox — a synthetic dosage/recovery example
- Confounding and why aggregated correlations mislead
- Potential outcomes framework and the fundamental problem of causal inference
- Average treatment effect (ATE) and the identification problem
- Adjustment formula and backdoor adjustment

### 02 — Probabilistic Graphical Models

Extends the single-confounder setting to systems of many interacting variables using Directed Acyclic Graphs (DAGs) and Structural Causal Models (SCMs).

- DAG structure: nodes, directed edges, the local Markov property
- Structure learning from observational data (hill-climb search, BIC scoring)
- Fitting conditional probability distributions via maximum likelihood
- Markov equivalence and CPDAGs
- Meek's orientation rules and how domain knowledge propagates through a graph
- Core assumptions: faithfulness, minimality, modularity

### 03 — Do-Calculus

Addresses causal identification in settings where simple covariate adjustment fails, and introduces the general framework of do-calculus.

- Conditioning vs. intervening: $P(Y \mid X)$ vs. $P(Y \mid do(X))$
- Randomised controlled trials as ground truth for causal effects
- Paths, d-separation, and the three path structures (chain, fork, collider)
- Backdoor criterion and the adjustment formula
- Collider bias and M-bias — when conditioning introduces rather than removes bias
- Front-door criterion — identification under unobserved confounding
- Do-calculus: the three rules and their graphical justification

---

## Dependencies

```
numpy
pandas
matplotlib
scikit-learn
pgmpy
networkx
causallearn
```

Install with:

```bash
pip install numpy pandas matplotlib scikit-learn pgmpy networkx causallearn
```

Graphviz is required for DAG layout:

```bash
# macOS
brew install graphviz

# Ubuntu / Debian
sudo apt-get install graphviz
```

---

## Usage

Clone or download the repository, then launch Jupyter from the `notebooks/` directory (or its parent):

```bash
jupyter notebook
```

The notebooks import from `src/utils.py` using a relative path (`sys.path.append("..")`), so they should be run from inside the `notebooks/` directory, or with the project root on the Python path.

---

## Data

`data/obs_data.txt` contains synthetic observational data for the student Bayesian network from Koller & Friedman's *Probabilistic Graphical Models* (2009). The five variables are:

| Column | Variable | Description |
|--------|----------|-------------|
| I | Intelligence | Student intelligence level |
| D | Difficulty | Course difficulty |
| G | Grade | Course grade |
| S | SAT | SAT score |
| L | Letter | Recommendation letter quality |

---

## References

- Judea Pearl, *Causality: Models, Reasoning, and Inference* (2009)
- Daphne Koller & Nir Friedman, *Probabilistic Graphical Models* (2009)
- Peter Bickel et al., "Sex Bias in Graduate Admissions: Data from Berkeley", *Science* (1975)
