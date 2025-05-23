# Programmation parallèle pour le machine learning
Projet dans le cadre du cours de Programmation parallèle pour le Machine Learning (ENSAE, 3A).

Adaptive benchmarking of self-attention in NumPy, Numba & Cython (+OpenMP), with serveral statistical-based search methods for best parallel settings.

## Objective

The goal of this project is to design and evaluate efficient parallel implementations of the **scaled dot-product attention** mechanism, defined as:

`Attention(Q, K, V) = softmax((Q · K<sup>T</sup>) / √d) · V`

where:

- `Q`, `K`, and `V` are the query, key and value matrices  
- `d` is the dimensionality of the key vectors  

We provide three reference backends—**NumPy** (sequential), **Numba** (JIT-compiled), and **Cython + OpenMP** (multithreaded)—and build an **adaptive benchmarking framework** that automatically tunes:

- number of threads  
- block sizes  
- data types  

using several advanced statistical-based strategies (Random Search, Hyperband, Multi-armed Bandits, CMA-ES, Bayesian Optimization and DOE-RSM). By measuring:

- **speed-up** relative to NumPy (our benchmark)
- **parallel efficiency** (speed-up ÷ thread count)  
- **variability** (standard deviation)  

across a wide range of sequence lengths, our objective is to identify the optimal configuration for each scenario, ensuring both numerical correctness and maximum throughput.  

## Installation

Clone the repo 

```bash
git clone https://github.com/cominho/Programmation-parallele-pour-le-Machine-Learning-3A
```

Install the requirements 

```bash
pip install -r requirements.txt
```

And then run the following code on your terminal 

```bash
rm -rf build/ dist/ *.egg-info
python setup.py build_ext --inplace
pip install -e .
```
Finally, use the notebook "main" (in notebooks/main.ipynb) to run the code. 


