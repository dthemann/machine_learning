# Machine Learning with Python — a hands-on course

A complete, self-contained course in **classical machine learning**, taught in twenty-one
Jupyter notebooks. It takes you from "I can write a `for` loop" to framing a business
problem as a learning problem, choosing and tuning an appropriate model, evaluating it
honestly, explaining what it does, and deploying it responsibly.

Every idea is explained in prose and mathematics, then implemented, then tested on data.
Most algorithms are written twice — once **from scratch in NumPy**, so that nothing is
magic, and once with **scikit-learn**, so that you learn the tools you will actually use.

**What this course does not cover:** deep learning. Neural networks, PyTorch,
convolutional networks and transformers are a large subject with their own prerequisites
and their own course. Where they are the natural continuation of a topic, this course says
so in a sentence and moves on.

---

## Quick start

```bash
conda env create -f environment.yml     # or: mamba env create -f environment.yml
conda activate ml-course
jupyter lab
```

Then open `notebooks/00_course_overview_and_setup.ipynb`. It verifies your installation,
shows the course map, and walks through a complete workflow in ten minutes.

No GPU is needed. Every notebook runs on a laptop CPU in a few minutes. If you do not have
conda, [Miniforge](https://github.com/conda-forge/miniforge) is a small free installer for
Linux, macOS and Windows.

---

## What is in the box

| | |
|---|---|
| **21 notebooks** | ~200 000 words of explanation, 820+ code cells, 480 figures |
| **120+ exercises** | four to six per notebook, graded easy → hard, each with a solution sketch |
| **9 datasets** | five real, four simulated with a known ground truth |
| **~400 references** | textbooks, papers and documentation, with the free ones marked |
| **50–70 hours** | of work if you do the exercises |

```
course/
├── README.md              this file
├── environment.yml        the conda environment
├── REFERENCES.md          the complete bibliography, grouped by topic
├── data/                  bundled datasets + the script that generates them
│   ├── customer_churn.csv      simulated: the running tabular example
│   ├── energy_demand.csv       simulated: hourly demand, 2 years
│   ├── loan_applications.csv   simulated: with a documented bias mechanism
│   ├── reviews.csv             simulated: short product reviews
│   └── make_datasets.py        regenerates all four, deterministically
└── notebooks/
    ├── course_utils.py    plotting style, palette, helpers, dataset loaders
    └── 00_… 20_….ipynb    the course itself
```

---

## The curriculum

### Foundations

| # | Notebook | What you get out of it |
|---|---|---|
| 00 | Course overview and setup | the map, the environment, the conventions, a 10-minute end-to-end demo |
| 01 | Python, NumPy and pandas for ML | vectorised thinking, broadcasting, DataFrames, the `X`/`y` convention |
| 02 | Mathematics essentials | linear algebra, gradients, probability, maximum likelihood, entropy — taught by computing them |
| 03 | Exploratory data analysis | a disciplined look at data before modelling; the grammar of honest charts |
| 04 | Preprocessing and feature engineering | cleaning, imputation, scaling, encoding, feature construction and selection, all inside pipelines |
| 05 | **ML fundamentals** | the learning problem, bias–variance, cross-validation, learning curves, data leakage |

Notebook 05 is the hub of the course. Everything after it assumes you can evaluate a model
without fooling yourself.

### Supervised learning

| # | Notebook | Methods |
|---|---|---|
| 06 | Linear regression and regularisation | OLS, gradient descent, ridge, lasso, elastic net, splines, GLMs |
| 07 | Logistic regression and classification metrics | logistic and softmax regression; precision/recall, ROC, PR, calibration, cost-sensitive thresholds |
| 08 | kNN, naive Bayes, curse of dimensionality | instance-based and generative models; why distance stops working in high dimensions |
| 09 | Decision trees | CART from scratch, impurity criteria, cost-complexity pruning |
| 10 | Ensembles | bagging, random forests, AdaBoost, gradient boosting, stacking — the default for tabular data |
| 11 | SVMs and kernel methods | margins, hinge loss, the kernel trick, SVR, a glimpse of Gaussian processes |
| 12 | Model selection and tuning | grid/random/halving/Bayesian search, nested CV, honest reporting |

### Unsupervised learning

| # | Notebook | Methods |
|---|---|---|
| 13 | Clustering and anomaly detection | k-means, hierarchical, DBSCAN/HDBSCAN, Gaussian mixtures, Isolation Forest, LOF |
| 14 | Dimensionality reduction | PCA, LDA, NMF, matrix factorisation, t-SNE, UMAP |

### Applications

| # | Notebook | Methods |
|---|---|---|
| 15 | Text data with classical ML | tokenisation, TF-IDF, text classifiers, topic models, word embeddings |
| 16 | Time series forecasting | decomposition, exponential smoothing, ARIMA, ML with lag features, backtesting |

### Putting models into the world

| # | Notebook | Topics |
|---|---|---|
| 17 | Interpretability and explainability | permutation importance, PDP/ICE/ALE, LIME, Shapley values, counterfactuals |
| 18 | ML engineering and MLOps | notebooks to modules, persistence, testing, serving, monitoring, drift |
| 19 | Ethics, fairness, privacy | fairness metrics and their impossibility, mitigation, differential privacy, adversarial examples |
| 20 | Capstone project | the whole workflow end to end, three project briefs, a grading rubric |

---

## How each method notebook is structured

Notebooks 06 to 17 all follow the same shape, so you always know where to find things:

1. **Intuition** — what the method is doing, in pictures and plain language.
2. **The mathematics** — the objective, the estimator, the algorithm, derived properly.
3. **From scratch** — a NumPy implementation in under sixty lines, verified against scikit-learn.
4. **The library** — the scikit-learn API, its options and its defaults.
5. **Experiments** — what happens as you vary the data, the noise, the dimension.
6. **Strengths, weaknesses and when to use it** — with at least one failure mode *demonstrated*, not merely asserted.
7. **Tuning guide** — every hyper-parameter that matters, what it controls, what to tune first, and a picture for each: validation curves, 2-D heat-maps, grids of fitted models across the parameter range.
8. **Case study on real data** — the full workflow end to end, with an honest conclusion.
9. **Summary, exercises, references.**

---

## The datasets

**Real data** carries every case study, because real data is messy in ways no simulation
reproduces:

| Dataset | Source | Used for |
|---|---|---|
| Breast cancer Wisconsin (569 × 30) | Street, Wolberg & Mangasarian (1993) | binary classification, medical decisions |
| Wine cultivars (178 × 13) | UCI, via scikit-learn | multi-class classification, clustering |
| Handwritten digits (1797 × 64) | UCI optical digits | multi-class, dimensionality reduction |
| Diabetes progression (442 × 10) | Efron et al. (2004) | regression |
| Airline passengers (144 months) | Box & Jenkins | forecasting |

Four more (`load_california_housing`, `load_penguins`, `load_adult`, `load_newsgroups`)
download on first use and fall back to a stand-in if you are offline.

**Simulated data** ships with the course because when you write the data-generating process
yourself you know the ground truth — you can check whether a model recovers the effect you
put in, demonstrate a bias mechanism exactly, or create a failure mode on demand. The
course always says which kind of data a result came from, and never draws a real-world
conclusion from a simulation. `data/make_datasets.py` regenerates all four deterministically
and documents every mechanism.

---

## Suggested paths

- **The fast path to a working model** (~15 h): 01, 04, 05, 07, 10, 12, 20. Enough to build, tune and honestly evaluate a gradient-boosted model on your own tabular data.
- **The full path**: 00 → 20 in order. This is what the cross-references assume.
- **Theory first**: 02, 05, 06, 07, 08, 09, 11, then the rest.

If you read only one notebook, read 05.

---

## Conventions

- Reproducibility: every notebook sets `RANDOM_STATE = 42` and passes it to every estimator that accepts one; randomness comes from `np.random.default_rng`, never the legacy global API.
- Preprocessing always lives inside a `Pipeline`, so that cross-validation stays honest.
- Plots use a colour-blind-safe palette defined in `course_utils.py`; sequential data uses `viridis`, diverging data `RdBu_r` centred at zero. There are no rainbow colour maps and no dual-axis charts.
- Optional packages (statsmodels, XGBoost, LightGBM, SHAP, Optuna, UMAP, NLTK, gensim, fairlearn, Flask) are imported inside `try/except`. Every notebook runs to the end without them and prints one line where a section is skipped.
- Citations are author–year in the text, with full entries at the end of each notebook and everything collected in `REFERENCES.md`.

---

## Requirements

Python 3.11, and the packages in `environment.yml`. The core is NumPy, pandas, SciPy,
scikit-learn ≥ 1.8, matplotlib, seaborn and JupyterLab. Tested on Linux, macOS (Intel and
Apple silicon) and Windows.

## Licence and attribution

The notebooks are teaching material: use them, adapt them, teach from them. The bundled
CSVs are synthetic and carry no restrictions. The real datasets are redistributed by
scikit-learn under their own terms, and each is cited where it is used. The references are
listed so you can go to the primary sources — please do.
