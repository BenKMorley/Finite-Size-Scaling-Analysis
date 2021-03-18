from getdist import loadMCSamples, plots_edit
import matplotlib.pyplot as plt
import matplotlib
import os
import pdb
from inspect import getmembers
import pickle
from model_definitions import *
from bayesian_functions import *

h5_data_file = "h5data/Bindercrossings.h5"
matplotlib.use('TkAgg')

# How many sample points to use in the MULTINEST algorithm
points = 1000

# Use this to label different runs if you edit something
tag = ""

# Prior Name: To differentiate results which use different priors
prior_name = "A"

# For reproducability
seed = 3475642

N = 4

if N == 2:
  model = model1_1a
  N = 2
  N_s_in = [N]
  Bbar_1 = 0.52
  Bbar_2 = 0.53
  Bbar_s_in = [Bbar_1, Bbar_2]
  g_s_in = [0.1, 0.2, 0.3, 0.5, 0.6]
  L_s_in = [8, 16, 32, 48, 64, 96, 128]
  GL_min = 12.8
  GL_max = 76.8
  Bbar_s = f"{Bbar_1:.3f}_{Bbar_2:.3f}_GL_min{GL_min:.1f}"

  param_names = ["alpha", "f0", "f1", "lambduh", "nu"]
  param_names_latex = ["\\alpha", "f_0", "f_1", "\\beta", "\\nu"]

  alpha_range = [-0.1, 0.1]
  f0_range = [0, 1]
  f1_range = [-2, 2]
  lambduh_range = [0, 2]
  nu_range = [0, 2]
  prior_range = [alpha_range, f0_range, f1_range, lambduh_range, nu_range]

if N == 4:
  model = model1_2a
  N = 4
  N_s_in = [N]
  Bbar_1 = 0.42
  Bbar_2 = 0.43
  Bbar_s_in = [Bbar_1, Bbar_2]
  g_s_in = [0.1, 0.2, 0.3, 0.5, 0.6]
  L_s_in = [8, 16, 32, 48, 64, 96, 128]
  GL_min = 12.8
  GL_max = 76.8
  Bbar_s = f"{Bbar_1:.3f}_{Bbar_2:.3f}_GL_min{GL_min:.1f}"

  param_names = ["alpha1", "alpha2", "f0", "f1", "lambduh", "nu"]
  param_names_latex = ["\\alpha_1", "\\alpha_2", "f_0", "f_1", "\\beta", "\\nu"]

  alpha1_range = [-0.1, 0.1]
  alpha2_range = [-0.1, 0.1]
  f0_range = [0, 1]
  f1_range = [-2, 2]
  lambduh_range = [0, 2]
  nu_range = [0, 2]
  prior_range = [alpha1_range, alpha2_range, f0_range, f1_range, lambduh_range, nu_range]


# Have a look at the log(L) + linear + quadratic term
poly = 1
poly2 = 2
points = 800
def model(*args):
    return polynomial_range_1a(poly, poly2, *args)

N = 2
N_s_in = [N]
Bbar_1 = 0.52
Bbar_2 = 0.53
Bbar_s_in = [Bbar_1, Bbar_2]
g_s_in = [0.1, 0.2, 0.3, 0.5, 0.6]
L_s_in = [8, 16, 32, 48, 64, 96, 128]
GL_min = 8
GL_max = 76.8
Bbar_s = f"{Bbar_1:.3f}_{Bbar_2:.3f}_GL_min{GL_min:.1f}"

param_names = ["alpha", "f0", "f1", "lambduh", "nu", "eps1", "eps2"]
param_names_latex = ["\\alpha", "f_0", "f_1", "\\beta", "\\nu", "\\epsilon_1", "\\epsilon_2"]

alpha_range = [-0.4, 0.4]
f0_range = [0, 1]
f1_range = [-20, 20]
beta_range = [-15, 15]
nu_range = [0, 15]

prior_range = [alpha_range, f0_range, f1_range, beta_range, nu_range, beta_range, beta_range]

n_params = len(prior_range)

in_directory = f"MULTINEST_samples/"

samples, g_s, L_s, Bbar_s, m_s = load_h5_data(h5_data_file, N_s_in, g_s_in, L_s_in, Bbar_s_in, GL_min, GL_max)

run_pymultinest(prior_range, model, GL_min, GL_max, n_params, in_directory,
                            N, g_s, Bbar_s, L_s, samples, m_s, param_names,
                            n_live_points=points, sampling_efficiency=0.3, clean_files=False,
                            tag=tag, prior_name=prior_name, keep_GLmax=False,
                            return_analysis_small=True, seed=seed, param_names_latex=param_names_latex)

samples = loadMCSamples(f"{in_directory}{model.__name__}{tag}_prior{prior_name}_N{N}_GLmin{GL_min:.1f}_p{points}")

g = plots_edit.get_subplot_plotter()
g.triangle_plot(samples, filled=True)

x = plt.gcf()

plt.savefig(f"posterior_N{N}.pdf", dpi=1000)
plt.show()
