# Dual Transformer MARK 1
### Graph-Based Anomaly Detection System Using Dual Transformer Architecture

A graph-based anomaly detection framework that combines node attribute representation learning and graph structural representation learning through two parallel Transformer branches.

---

## 1. Project Overview

**Dual Transformer MARK 1** is a graph-based anomaly detection system designed to identify anomalous nodes in graph-structured data.

Traditional anomaly detection methods often focus primarily on node attributes or graph connectivity. However, anomalous nodes may exhibit unusual feature patterns, unusual structural relationships, or both.

To address this, this project implements a Dual Transformer architecture consisting of two parallel learning branches:

- **Attribute Transformer:** Learns representations from node attribute features.
- **Structural Transformer:** Learns representations from graph topology and structural information.

The representations from both branches are used to reconstruct node attributes and graph adjacency information. Reconstruction errors can then be used to identify potentially anomalous nodes.

## 2. Project Objectives

- Implement a Dual Transformer architecture for graph-based anomaly detection.
- Learn meaningful node representations from both attributes and graph structure.
- Incorporate structural information through graph-derived features.
- Reconstruct node attributes and graph connectivity.
- Develop an anomaly detection workflow based on reconstruction errors.

## 3. System Architecture

The proposed framework consists of the following major components:

### A. Input Graph

The input graph is represented using:

- **X:** Node feature matrix.
- **A:** Adjacency matrix representing graph connectivity.
- **E:** Edge information representing relationships between nodes.

### B. Graph Preprocessing

The preprocessing stage prepares the graph for Transformer-based learning.

Operations include:

- Loading the graph dataset.
- Extracting node attributes.
- Constructing graph connectivity information.
- Converting graph edges into an undirected edge index.
- Calculating structural features.
- Normalizing node features.

### C. Attribute Transformer

The Attribute Transformer processes the node attribute matrix (X).

Its purpose is to learn feature-level representations and capture relationships between different node attributes.

Output:

- Attribute embeddings: `h_a`

### D. Structural Transformer

The Structural Transformer processes graph structural information.

Structural information includes graph connectivity and derived node-level structural features.

The structural features used in the implementation include:

- Degree information.
- Betweenness centrality.
- Eigenvector centrality.
- Local clustering coefficient.
- Normalized degree.

Output:

- Structural embeddings: `h_s`

### E. Dual Representation Learning

The attribute and structural representations are learned through two separate Transformer branches.

The resulting embeddings are:

- `h_a`: Attribute representation.
- `h_s`: Structural representation.

These representations are used by the reconstruction components.

### F. Reconstruction

The architecture contains two reconstruction objectives:

1. **Attribute Reconstruction:** Reconstructs the original node feature matrix.

   Output: `X_hat`

2. **Structural Reconstruction:** Reconstructs graph adjacency information.

   Output: `A_hat`

### G. Anomaly Detection

The reconstruction outputs are compared against their corresponding original inputs.

The reconstruction errors provide information about how well each node is represented by the learned model.

Nodes with unusually high reconstruction errors can be considered potential anomalies.

---

## 4. Dataset

The initial implementation uses the **Books dataset**.

| Property | Value |
|---|---:|
| Dataset | Books |
| Number of nodes | 1,418 |
| Number of original edges | 3,695 |
| Number of node features | 21 |
| Anomalous nodes | 28 |
| Normal nodes | 1,390 |
| Undirected edge index entries | 7,390 |

**Input feature matrix:**

`X ∈ R^(1418 × 21)`

**Adjacency matrix:**

`A ∈ R^(1418 × 1418)`

The dataset is loaded from a PyTorch `.pt` file.

Note: The undirected edge index contains both directions of each original edge, assuming no self-loops or duplicate edges.

---

## 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.10 | Core programming language |
| PyTorch | Deep learning implementation |
| PyTorch Geometric | Graph data handling |
| NumPy | Numerical computation |
| SciPy | Scientific computation |
| Pandas | Data processing |
| Scikit-learn | Feature preprocessing and evaluation |
| NetworkX | Graph structural analysis |
| Matplotlib | Visualization |
| Cython | Compiled numerical components |
| tqdm | Progress monitoring |

---

## 6. Project Structure

The following is the intended logical organization. Actual filenames and folders may differ according to the implementation.

```text
MARK 1/
│
├── data/
│   └── Books.pt
│
├── preprocessing/
│   └── graph_preprocessing.py
│
├── models/
│   ├── attribute_transformer.py
│   ├── structural_transformer.py
│   └── dual_transformer.py
│
├── training/
│   └── train.py
│
├── evaluation/
│   └── evaluate.py
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 7. Installation

### Prerequisites

- Python 3.10.11
- Git
- pip

### Step 1: Clone the repository

```bash
git clone https://github.com/Swastik0407/dual-transformer-anomaly-detection.git
cd dual-transformer-anomaly-detection
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### Step 3: Install PyTorch

The development environment uses CPU-based PyTorch.

```bash
pip install torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Install PyTorch Geometric

```bash
pip install torch-geometric==2.5.3
```

For the compatible PyTorch Sparse extension:

```bash
pip install torch-sparse==0.6.18+pt22cpu -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
```

### Step 5: Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## 8. Execution Workflow

The overall execution pipeline follows these stages:

```text
Input Graph Dataset
        |
        v
Graph Preprocessing
        |
        v
Node Attribute Extraction
        |
        v
Structural Feature Calculation
        |
        v
Feature Normalization
        |
        v
+---------------------------+
|                           |
v                           v
Attribute Transformer   Structural Transformer
|                           |
v                           v
Attribute Embedding     Structural Embedding
h_a                         h_s
|                           |
+-------------+-------------+
              |
              v
       Reconstruction
              |
       +------+------+
       |             |
       v             v
 Attribute       Structural
 Reconstruction  Reconstruction
       |             |
       v             v
     X_hat          A_hat
       |             |
       +------+------+
              |
              v
       Reconstruction
             Errors
              |
              v
       Anomaly Detection
```

Run the appropriate training or evaluation script from the project root according to the actual implementation.

---

## 9. Model Outputs

The current model implementation produces four primary tensors:

| Output | Description | Shape |
|---|---|---|
| `h_a` | Attribute embeddings | `(1418, 64)` |
| `h_s` | Structural embeddings | `(1418, 64)` |
| `X_hat` | Reconstructed node features | `(1418, 21)` |
| `A_hat` | Reconstructed adjacency matrix | `(1418, 1418)` |

These outputs represent the learned attribute and structural representations and their corresponding reconstructions.

---

## 10. Current Development Status

**MARK 1 – Initial Implementation**

Completed implementation stages:

- [x] Books dataset loading.
- [x] Graph preprocessing.
- [x] Undirected graph edge construction.
- [x] Structural feature extraction.
- [x] Node feature normalization.
- [x] Attribute Transformer implementation.
- [x] Structural Transformer implementation.
- [x] Dual representation generation.
- [x] Attribute reconstruction.
- [x] Structural reconstruction.
- [ ] Complete training and evaluation pipeline validation.
- [ ] Anomaly score calibration.
- [ ] Performance evaluation and visualization.

This project is currently focused on establishing and validating the complete Dual Transformer workflow.

---

## 11. Research Reference and Acknowledgment

This implementation is developed with reference to the ANIMATE research/codebase and its graph Transformer implementation.

The original repository is used as a reference for relevant implementation components. The MARK 1 project adapts the architecture toward graph-based anomaly detection using separate attribute and structural Transformer branches.

Original repository: https://github.com/Graph-COM/ANIMATE

All original work and third-party components should be credited according to their respective licenses.

---

## 12. Future Scope

Potential future improvements include:

- Refinement of attribute and structural representation learning.
- Improved reconstruction-based anomaly scoring.
- Precision, Recall, F1-score and ROC-AUC evaluation.
- Anomaly score visualization.
- Comparison with baseline graph anomaly detection approaches.
- Evaluation on additional graph datasets.

---

## 13. Author

**Swastik Saha**

B.Tech – Computer Science and Information Technology

Academy of Technology, West Bengal, India

GitHub: https://github.com/Swastik0407

---

*Dual Transformer MARK 1 – Graph-Based Anomaly Detection System.*
