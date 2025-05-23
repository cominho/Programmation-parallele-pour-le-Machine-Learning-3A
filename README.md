# Programmation parallele pour le machine learning
Projet dans le cadre du cours de Programmation parallèle pour le Machine Learning (ENSAE, 3A).

Adaptive benchmarking of self-attention in NumPy, Numba & Cython (+OpenMP), with serveral statistical-based search methods for best parallel settings.

## Objective 

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


