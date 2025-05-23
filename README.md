# Programmation parallele pour le machine learning
Projet dans le cadre du cours de Programmation parallèle pour le Machine Learning (ENSAE, 3A).

Adaptive benchmarking of self-attention in NumPy, Numba & Cython (+OpenMP), with serveral statistical-based search methods for best parallel settings.

## Objective 


The goal of this project is to design and evaluate efficient parallel implementations of the scaled dot-product attention mechanism, defined as:

\[
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\bigl(\tfrac{Q K^\top}{\sqrt{d}}\bigr)\,V,
\]

where \(Q\), \(K\), and \(V\) are the query, key, and value matrices, and \(d\) is the dimension of the key vectors. We provide three reference backends—NumPy (sequential), Numba (JIT-compiled), and Cython with OpenMP multithreading—and develop an adaptive benchmarking framework that automatically tunes threading, block sizes, and data types using several advanced statistical methods. By measuring speed-up, parallel efficiency, and variability across a range of sequence lengths, our objective is to identify the best configuration for each scenario, ensuring both correctness and maximum performance.

## Installation

```bash
git clone https://github.com/cominho/Programmation-parallele-pour-le-Machine-Learning-3A
```
And then run the following code on your terminal 

```bash
rm -rf build/ dist/ *.egg-info
python env.py build_ext --inplace
pip install -e .
```
Then, use the notebook "main" (in notebooks/main.ipynb) to run the code. 


