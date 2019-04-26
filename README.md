# PolyspectrumEstimator
This is the code that was used for [Efficient parallel algorithm for estimating higher-order polyspectra](https://arxiv.org/abs/1904.11055)

This repository contains two main files 

ParallelPnNotebook.ipynb: which is roughly an IJulia walkthrough of using the code to calculate all of the polyspectra used in the paper. The functions in this notebook are not fully optimized but should be more readable than the code in the other file.

Bk.jl: This files contains very optimized versions of the main functions, and is also written to account for different sizes in each fourier dimension. The idea with this file is for other users to copy the functions for whatever purposes they need. It can technically be directly imported although that may impose problems.
