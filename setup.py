import sys
import os
import numpy as np
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

# Detect macOS to adjust OpenMP flags
is_darwin = sys.platform == "darwin"
if is_darwin:
    prefix = os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew/opt/libomp")
    extra_compile_args = [
        "-Xpreprocessor", "-fopenmp",
        f"-I{prefix}/include"
    ]
    extra_link_args = [
        f"-L{prefix}/lib",
        "-lomp"
    ]
else:
    extra_compile_args = ["-fopenmp"]
    extra_link_args    = ["-fopenmp"]

# Define the Cython extension inside att_bench_lib
extensions = [
    Extension(
        name="att_bench_lib.cython_att",
        sources=["att_bench_lib/cython_att.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="prog_parallele_adaptive",
    version="0.1.0",
    packages=find_packages(),  # will find att_bench_lib and src.benchmark
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': '3'}
    ),
    include_dirs=[np.get_include()],
    install_requires=[
        "numpy", "numba", "cython", "scikit-optimize", "pyDOE2",
        "statsmodels", "matplotlib", "pandas", "rich",
        "scikit-learn", "scipy", "psutil"
    ],
    setup_requires=["Cython"],
    zip_safe=False,
)
