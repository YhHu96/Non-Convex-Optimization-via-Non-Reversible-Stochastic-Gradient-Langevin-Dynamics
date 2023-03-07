# Non-Convex-Optimization-via-Non-Reversible-Stochastic-Gradient-Langevin-Dynamics
This folder contains the codes for paper "Non-Convex Optimization via Non-Reversible Stochastic Gradient Langevin Dynamics", which can be read through https://arxiv.org/abs/2004.02823.

The code for Figure 1 and 2 in the paper is in the jupyter notebook files called Figure 1 and Figure 2_Iris and Figure 2_Diabetes. The notebook files also contain the figures and can be read when open the code files. The code for Figure 1 need to include the "seaborn" package. The code for Figure 2 need to include "sklearn" package to download the data.

Figure 3 need to run the code in folder neural network. To run these codes, one needs to include the "tensorflow" and "keras" package in Python environment. And need to replace the "optimizers.py" file in the keras package by the modified "optimizers.py" file in this folder. We added the SGLD and NSGLD method in the optimizers file.
