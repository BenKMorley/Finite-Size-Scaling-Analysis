###############################################################################
# Copyright (C) 2020
#
# Author: Ben Kitching-Morley bkm1n18@soton.ac.uk
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# See the full license in the file "LICENSE" in the top level distribution
# directory

# The code has been used for the analysis presented in
# "Nonperturbative infrared finiteness in super-renormalisable scalar quantum
# field theory" https://arxiv.org/abs/2009.14768
###############################################################################

from frequentist_run import run_frequentist_analysis
from model_definitions import *
# from bayesian_functions import *
import pdb
import sys

h5_data_file = "./h5data/Bindercrossings.h5"


def get_pvalues_central_fit(N, model_name=None, poly=None, poly2=None):
    """
        This function will reproduce the p-value data against the gL_min cut
        for the central fit.

        INPUTS :
        --------
        N: int, rank of the SU(N) valued fields

        OUTPUTS:
        --------
        pvalues: dict of arrays of floats. Each array is of the length of the
            number of GL_min cut values, and the corresponding p-value to each
            cut is recorded.

            > "pvalues1": Values for model 1 (Lambda_IR = g / (4 pi N))
            > "pvalues2": Values for model 1 (Lambda_IR = 1 / L)
    """
    GL_mins = numpy.array([3.2, 4, 4.8, 6.4, 8, 9.6, 12.8, 14.4,
                           16, 19.2, 24, 25.6, 28.8, 32])
    GL_max = 76.8

    if N == 2:
        N_s = [2]
        Bbar_s = [0.52, 0.53]
        g_s = [0.1, 0.2, 0.3, 0.5, 0.6]
        L_s = [8, 16, 32, 48, 64, 96, 128]

        x0 = [0, 0.5431, -0.03586, 1, 2 / 3]  # EFT values
        x2 = x0

        model1 = model1_1a
        model2 = model2_1a
        param_names = ["alpha", "f0", "f1", "beta", "nu"]

        param_names2 = param_names

        if model_name == "poly":
            def polyno(*args):
                return polynomial_1a(poly, *args)

            x2 = [0, 0.5431, -0.03586, 1, 2 / 3] + [0, ] * poly  # EFT values
            param_names2 = ["alpha", "f0", "f1", "beta", "nu"] + [f"eps{i}" for i in range(1, poly + 1)]

            model2 = polyno

        if model_name == "mixed":
            x2 = [0, 0.5431, -0.03586, 1, 2 / 3]
            param_names2 = ["alpha", "f0", "f1", "beta", "nu"]

            model2 = model_mixed_1a

        if model_name == "mixed2":
            x2 = [0, 0, 0.5431, -0.03586, 1, 2 / 3]
            param_names2 = ["alpha", "alpha2", "f0", "f1", "beta", "nu"]

            model2 = model_mixed_2a

        if model_name == "mixed_fit":
            x2 = [0, 0.5431, -0.03586, 1, 2 / 3, 1]
            param_names2 = ["alpha", "f0", "f1", "beta", "nu", "eps"]

            model2 = model_mixed_fit_1a

        if model_name == "mono":
            def monono(*args):
                return monomial_no_log_1a(poly, *args)

            x2 = [0, 0.5431, -0.03586, 0, 2 / 3]  # EFT values
            param_names2 = ["alpha", "f0", "f1", "beta", "nu"]

            model2 = monono

        if model_name == "poly_range":
            r = numpy.arange(poly, poly2 + 1)
            r = r[r != 0]
            param_names2 = ["\\alpha", "f_0", "f_1", "\\beta", "\\nu"] + ["\\epsilon_{" + f"{i}" + "}" for i in r]
            x2 = [0, 0.5431, -0.03586, 1, 2 / 3] + [0, ] * len(r)

            def polyno(*args):
                return polynomial_range_1a(poly, poly2, *args)

            model2 = polyno

        if model_name == "poly_range_no_log":
            r = numpy.arange(poly, poly2 + 1)
            r = r[r != 0]
            param_names2 = ["\\alpha", "f_0", "f_1", "\\nu"] + ["\\epsilon_{" + f"{i}" + "}" for i in r]
            x2 = [0, 0.5431, -0.03586, 2 / 3] + [0, ] * len(r)

            def polyno(*args):
                return polynomial_range_1a_no_log(poly, poly2, *args)

            model2 = polyno

    if N == 4:
        N_s = [4]
        Bbar_s = [0.42, 0.43]
        g_s = [0.1, 0.2, 0.3, 0.5, 0.6]
        L_s = [8, 16, 32, 48, 64, 96, 128]

        x0 = [0, 0, 0.4459, -0.02707, 1, 2 / 3]  # EFT values
        x2 = [0, 0, 0.4459, -0.02707, 1, 2 / 3]  # EFT values

        model1 = model1_2a
        model2 = model2_2a
        param_names = ["alpha1", "alpha2", "f0", "f1", "beta", "nu"]
        param_names2 = param_names

        if model_name == "poly":
            def polyno(*args):
                return polynomial_2a(poly, *args)

            x2 = [0, 0, 0.4459, -0.02707, 1, 2 / 3] + [0, ] * poly  # EFT values
            param_names2 = ["alpha", "f0", "f1", "beta", "nu"] + [f"eps{i}" for i in range(1, poly + 1)]

            model2 = polyno

        if model_name == "mono":
            def monono(*args):
                return monomial_no_log_2a(poly, *args)

            x2 = [0, 0, 0.4459, -0.02707, 0, 2 / 3]  # EFT values
            param_names2 = ["alpha1", "alpha2", "f0", "f1", "beta", "nu"]

            model2 = monono

        if model_name == "poly_range":
            r = numpy.arange(poly, poly2 + 1)
            r = r[r != 0]
            param_names2 = ["alpha1", "alpha2", "f_0", "f_1", "\\beta", "\\nu"] + ["\\epsilon_{" + f"{i}" + "}" for i in r]
            x2 = [0, 0, 0.4459, -0.02707, 1, 2 / 3] + [0, ] * len(r)

            def polyno(*args):
                return polynomial_range_2a(poly, poly2, *args)

            model2 = polyno

        if model_name == "poly_range_no_log":
            r = numpy.arange(poly, poly2 + 1)
            r = r[r != 0]
            param_names2 = ["alpha1", "alpha2", "f_0", "f_1", "\\nu"] + ["\\epsilon_{" + f"{i}" + "}" for i in r]
            x2 = [0, 0, 0.4459, -0.02707, 2 / 3] + [0, ] * len(r)

            def polyno(*args):
                return polynomial_range_2a_no_log(poly, poly2, *args)

            model2 = polyno

    pvalues_1 = numpy.zeros(len(GL_mins))
    pvalues_2 = numpy.zeros(len(GL_mins))
    BICs_1 = numpy.zeros(len(GL_mins))
    BICs_2 = numpy.zeros(len(GL_mins))

    for i, GL_min in enumerate(GL_mins):
        pvalues_1[i], params1, dof, chisq = \
            run_frequentist_analysis(h5_data_file, model1, N_s, g_s, L_s,
                                     Bbar_s, GL_min, GL_max, param_names, x0,
                                     run_bootstrap=False, retrun_chisq=True)

        pvalues_2[i], params2, dof2, chisq2 = \
            run_frequentist_analysis(h5_data_file, model2, N_s, g_s, L_s,
                                     Bbar_s, GL_min, GL_max, param_names2, x2,
                                     run_bootstrap=False, print_info=False,
                                     retrun_chisq=True)

        BICs_1[i] = len(x0) * numpy.log(dof + len(x0)) + chisq
        BICs_2[i] = len(x2) * numpy.log(dof2 + len(x2)) + chisq2

    pvalues = {}
    pvalues["pvalues1"] = pvalues_1
    pvalues["pvalues2"] = pvalues_2
    # pvalues["BICs_1"] = BICs_1
    # pvalues["BICs_2"] = BICs_2

    # print("##################################################################")

    return pvalues


def get_statistical_errors_central_fit(N, model_name="model1", GL_min_in=None,
                                       Bbar_s_in=None, poly=None):
    """
        This function gets the statistical error bands (and central fit values)
        for the model parameters, and the value of the critical mass at g=0.1
        quoted in the publication.

        INPUTS :
        --------
        N: int, rank of the SU(N) valued fields
        model_name: string, either "model1" or "model2". Determines whether to
            look at the central fit for either Lambda_IR = g / 4 pi N (model1)
            or Lambda_IR = 1 / L (model2)
        GL_min: float, the cut on the minimum value of (ag) * (L / a) in the
            fit. If this is left as None, then the central fits for each model
            are used (GL_min = 12.8 for model1 for example)
        Bbar_s: (2, ) list of floats. The two Bbar values to be used in the
            fit. If left as None, then the central fits for each model are used
            (Bbar_s = [0.52, 0.53] for model1 with N=2 for example)

        OUTPUTS :
        ---------
        results: dictionary containing:
            > "params": list of floats, parameter estimates of the central fit
            > "params_std": list of floats, statistical error on these
                estimates
            > "param_names": list of strings, the names of the parameters in
                the same order as in "params" and "params_std"

            if N == 2 and model_name == "model1":
                > "m_c": float, Estimate of the critical mass at g = 0.1
                > "m_c_error": float, Statistical error on this estimate

            if N == 4 and model_name == "model1":
                > "m_c1": float, Estimate of the critical mass using alpha 1 at
                    g = 0.1
                > "m_c_error1": float, Statistical error on this estimate using
                    alpha1
                > "m_c2": float, Estimate of the critical mass using alpha 2 at
                    g = 0.1
                > "m_c_error2": float, Statistical error on this estimate using
                    alpha2
    """
    g_s = [0.1, 0.2, 0.3, 0.5, 0.6]
    L_s = [8, 16, 32, 48, 64, 96, 128]
    GL_max = 76.8

    if N == 2:
        if model_name == "model1":
            model = model1_1a
            GL_min = 12.8
            Bbar_s = [0.52, 0.53]

        elif model_name == "model2":
            model = model2_1a
            GL_min = 32.0
            Bbar_s = [0.51, 0.52]

        elif callable(model_name):
            model = model_name

        N = 2
        N_s = [N]
        param_names = ["alpha", "f0", "f1", "beta", "nu"]
        x0 = [0, 0.5431, -0.03586, 1, 2 / 3]  # EFT values

        if model_name == "poly":
            def polyno(*args):
                return polynomial_1a(poly, *args)

            x0 = [0, 0.5431, -0.03586, 1, 2 / 3] + [0, ] * poly  # EFT values
            param_names = ["alpha", "f0", "f1", "beta", "nu"] + [f"eps{i}" for i in range(1, poly + 1)]
            Bbar_s = [0.52, 0.53]
            GL_min = 12.8

            model = polyno

        if model_name == "mono":
            def monono(*args):
                return monomial_no_log_1a(poly, *args)

            x0 = [0, 0.5431, -0.03586, 0, 2 / 3]  # EFT values
            param_names = ["alpha", "f0", "f1", "beta", "nu"]
            Bbar_s = [0.51, 0.54]
            GL_min = 32.0

            model = monono

    if N == 4:
        if model_name == "model1":
            model = model1_2a
            GL_min = 12.8

        if model_name == "model2":
            model = model2_2a
            GL_min = 24.0

        x0 = [0, 0, 0.4459, -0.02707, 1, 2 / 3]  # EFT values
        N = 4
        N_s = [N]
        Bbar_s = [0.42, 0.43]
        param_names = ["alpha1", "alpha2", "f0", "f1", "beta", "nu"]

        if model_name == "linear2":
            GL_min = 12.8
            model = linear_2a
            x0 = [0, 0, 0.4459, -0.02707, 1, 2 / 3, 0]
            param_names = ["alpha1", "alpha2", "f0", "f1", "beta", "nu"]

        if model_name == "quadratic2":
            GL_min = 12.8
            model = quadratic_2a
            x0 = [0, 0, 0.4459, -0.02707, 1, 2 / 3, 0]
            param_names = ["alpha1", "alpha2", "f0", "f1", "beta", "nu"]

        if model_name == "cubic2":
            GL_min = 12.8
            model = cubic_2a
            x0 = [0, 0, 0.4459, -0.02707, 1, 2 / 3, 0]
            param_names = ["alpha1", "alpha2", "f0", "f1", "beta", "nu"]

    # Set GL_min and Bbar_s if given
    if GL_min_in is not None:
        GL_min = GL_min_in

    if Bbar_s_in is not None:
        Bbar_s = Bbar_s_in

    # Run once with the full dataset (no resampling)
    pvalue, params_central, dof =\
        run_frequentist_analysis(h5_data_file, model, N_s, g_s, L_s, Bbar_s,
                                 GL_min, GL_max, param_names, x0,
                                 run_bootstrap=False)

    # Run with all the bootstrap samples
    pvalue, params, dof =\
        run_frequentist_analysis(h5_data_file, model, N_s, g_s, L_s, Bbar_s,
                                 GL_min, GL_max, param_names, x0,
                                 print_info=False)

    sigmas = numpy.zeros(len(params_central))
    no_samples = params.shape[0]

    for i, param in enumerate(param_names):
        # Find the standard deviation with respect to the central values
        sigmas[i] = numpy.sqrt(numpy.sum((params[:, i] -
                                          params_central[i]) ** 2) /
                               no_samples)

        print(f"{param} = {params_central[i]} +- {sigmas[i]}")

    # Calculate the value of the non-perterbative critical mass for g = 0.1 and
    # it's statistical error
    if model_name == "model1":
        g = 0.1

        m_c = mPT_1loop(g, N) + g ** 2 * (params_central[0] -
                                          params_central[-2] * K1(g, N))

        if N == 4:
            print("Critical Mass using alpha1:")

        print(f"m_c = {m_c}")

        alphas = params[..., 0]
        betas = params[..., -2]

        m_cs = mPT_1loop(g, N) + g ** 2 * (alphas - betas * K1(g, N))

        m_c_error = numpy.sqrt(numpy.sum((m_cs - m_c) ** 2) /
                               no_samples)

        print(f"m_c_error = {m_c_error}")

    if N == 2:
        results = {}
        results["params_central"] = params_central
        results["params_std"] = sigmas

        if model_name == "model1":
            results["m_c"] = m_c
            results["m_c_error"] = m_c_error

    if N == 4:
        results = {}
        results["params_central"] = params_central
        results["params_std"] = sigmas

        alphas2 = params[..., 1]

        if model_name == "model1":
            m_c2 = mPT_1loop(g, N) + g ** 2 * (params_central[1] -
                                               params_central[-2] * K1(g, N))

            print("Critical Mass using alpha2:")
            print(f"m_c2 = {m_c2}")

            m_c2s = mPT_1loop(g, N) + g ** 2 * (alphas2 - betas * K1(g, N))

            m_c2_error = numpy.sqrt(numpy.sum((m_c2s - m_c2) ** 2) /
                                    no_samples)

            print(f"m_c2_error = {m_c2_error}")

            results["m_c1"] = m_c
            results["m_c_error1"] = m_c_error
            results["m_c2"] = m_c2
            results["m_c_error2"] = m_c2_error

    results["param_names"] = param_names

    return results


def get_systematic_errors(N, model_name="model1", dof=None, poly=None, poly2=None, Bbar_lim=0.0001):
    """
        This function calculates the systematic error bands (and central fit
        values) for the model parameters, and the value of the critical mass at
        g = 0.1 quoted in the publication.

        INPUTS :
        --------
        N: int, rank of the SU(N) valued fields
        model_name: string, either "model1" or "model2". Determines whether to
            look at either Lambda_IR = g / 4 pi N (model1) or
            Lambda_IR = 1 / L (model2)

        OUTPUTS :
        ---------
        results: dictionary containing:
            > "params": list of floats, parameter estimates of the central fit
            > "params_std": list of floats, systematic error on these estimates
            > "param_names": list of strings, the names of the parameters in
                the same order as in "params" and "params_std"

            if N == 2 and model_name == "model1":
                > "m_c": float, Estimate of the critical mass
                > "m_c_error": float, Systematic error on this estimate

            if N == 4 and model_name == "model2":
                > "m_c1": float, Estimate of the critical mass using alpha 1
                > "m_c_error1": float, Systematic error on this estimate using
                    alpha1
                > "m_c2": float, Estimate of the critical mass using alpha 2
                > "m_c_error2": float, Systematic error on this estimate using
                    alpha2
                > "m_c": float, Overall estimate of critical mass (same as
                    alpha 1 result)
                > "m_c_error": float, Overall systematic error when accounting
                    for both alpha values
    """
    GL_mins = numpy.array([0.8, 1.6, 2.4, 3.2, 4, 4.8, 6.4, 8, 9.6, 12.8, 14.4,
                           16, 19.2, 24, 25.6, 28.8, 32])

    g_s = [0.1, 0.2, 0.3, 0.5, 0.6]
    L_s = [8, 16, 32, 48, 64, 96, 128]
    GL_max = 76.8

    if N == 2:
        if model_name == "model1":
            model = model1_1a

        if model_name == "model2":
            model = model2_1a

        N_s = [2]
        Bbar_s = [0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59]
        x0 = [0, 0.5431, -0.03586, 1, 2 / 3]  # EFT values
        param_names = ["alpha", "f0", "f1", "beta", "nu"]

        if model_name == "poly":
            def polyno(*args):
                return polynomial_1a(poly, *args)

            x0 = [0, 0.5431, -0.03586, 1, 2 / 3] + [0, ] * abs(poly)  # EFT values
            param_names = ["alpha", "f0", "f1", "beta", "nu"] + [f"eps{i}" for i in range(1, abs(poly) + 1)]

            model = polyno

        if model_name == "mono":
            def monono(*args):
                return monomial_no_log_1a(poly, *args)

            x0 = [0, 0.5431, -0.03586, 0, 2 / 3]  # EFT values
            param_names = ["alpha", "f0", "f1", "beta", "nu"]

            model = monono

        if model_name == "poly_noscale":
            def polyno(*args):
                return polynomial_no_scaling_1a(poly, *args)

            x0 = [0, 1] + [0, ] * abs(poly)  # EFT values
            param_names = ["alpha", "beta"] + [f"eps{i}" for i in range(1, abs(poly) + 1)]

            model = polyno

        if model_name == "poly_nolog":
            def polyno(*args):
                return polynomial_no_log_1a(poly, *args)

            x0 = [0, 0.5431, -0.03586, 2 / 3] + [0, ] * abs(poly)  # EFT values
            param_names = ["alpha", "f0", "f1", "nu"] + [f"eps{i}" for i in range(1, abs(poly) + 1)]

            model = polyno

        if model_name == "one_plus":
            x0 = [0, 0.5431, -0.03586, 1, 2 / 3, 0]  # EFT values
            param_names = ["alpha", "f0", "f1", "beta", "nu", "eps"]

            model = model_1a_plus

        if model_name == "poly_range":
            r = numpy.arange(poly, poly2 + 1)
            r = r[r != 0]
            param_names = ["\\alpha", "f_0", "f_1", "\\beta", "\\nu"] + ["\\epsilon_{" + f"{i}" + "}" for i in r]
            x0 = [0, 0.5431, -0.03586, 1, 2 / 3] + [0, ] * len(r)

            def polyno(*args):
                return polynomial_range_1a(poly, poly2, *args)

            model = polyno

        if model_name == "poly_range_no_scaling":
            r = numpy.arange(poly, poly2 + 1)
            r = r[r != 0]
            param_names = ["\\alpha", "\\beta"] + ["\\epsilon_{" + f"{i}" + "}" for i in r]
            x0 = [0, 1] + [0, ] * len(r)

            def polyno(*args):
                return polynomial_range_1a_no_scaling(poly, poly2, *args)

            model = polyno

        if model_name == "poly_range_no_scaling_no_log":
            r = numpy.arange(poly, poly2 + 1)
            r = r[r != 0]
            param_names = ["\\alpha"] + ["\\epsilon_{" + f"{i}" + "}" for i in r]
            x0 = [0] + [0, ] * len(r)

            def polyno(*args):
                return polynomial_range_1a_no_scaling_no_log(poly, poly2, *args)

            model = polyno

        if model_name == "poly_range_no_log":
            r = numpy.arange(poly, poly2 + 1)
            r = r[r != 0]
            param_names = ["\\alpha", "f_0", "f_1", "\\nu"] + ["\\epsilon_{" + f"{i}" + "}" for i in r]
            x0 = [0, 0.5431, -0.03586, 2 / 3] + [0, ] * len(r)

            def polyno(*args):
                return polynomial_range_1a_no_log(poly, poly2, *args)

            model = polyno

        if model_name == "g4":
            param_names = ["\\alpha", "f_0", "f_1", "\\beta",  "\\nu", "\\epsilon"]
            x0 = [0, 0.5431, -0.03586, 1, 2 / 3, 0]

            model = model1_1a_g4

        if model_name == "mixed":
            param_names = ["\\alpha", "f_0", "f_1", "\\beta", "\\nu"]
            x0 = [0, 0.5431, -0.03586, 1, 2 / 3]

            model = model_mixed_1a

    if N == 4:
        if model_name == "model1":
            model = model1_2a

        if model_name == "model2":
            model = model2_2a

        N_s = [4]
        Bbar_s = [0.42, 0.43, 0.44, 0.45, 0.46, 0.47]
        x0 = [0, 0, 0.4459, -0.02707, 1, 2 / 3]  # EFT values
        param_names = ["alpha1", "alpha2", "f0", "f1", "beta", "nu"]

        if model_name == "poly":
            def model(*args):
                return polynomial_2a(poly, *args)

            x0 = [0, 0, 0.4459, -0.02707, 1, 2 / 3] + [0, ] * abs(poly)  # EFT values
            param_names = ["alpha1", "alpha2", "f0", "f1", "beta", "nu"] + [f"eps{i}" for i in range(1, abs(poly) + 1)]

        if model_name == "mono":
            def monono(*args):
                return monomial_no_log_2a(poly, *args)

            x0 = [0, 0, 0.4459, -0.02707, 0, 2 / 3]  # EFT values
            param_names = ["alpha1", "alpha2", "f0", "f1", "beta", "nu"]

            model = monono

        if model_name == "poly_noscale":
            def polyno(*args):
                return polynomial_no_scaling_2a(poly, *args)

            x0 = [0, 0, 1] + [0, ] * abs(poly)  # EFT values
            param_names = ["alpha1", "alpha2", "beta"] + [f"eps{i}" for i in range(1, abs(poly) + 1)]

            model = polyno

        if model_name == "poly_nolog":
            def polyno(*args):
                return polynomial_no_log_2a(poly, *args)

            x0 = [0, 0, 1] + [0, ] * poly  # EFT values
            param_names = ["alpha1", "alpha2", "f0", "f1", "nu"] + [f"eps{i}" for i in range(1, abs(poly) + 1)]

            model = polyno

        if model_name == "one_plus":
            x0 = [0, 0, 0.4459, -0.02707, 1, 2 / 3, 0]  # EFT values
            param_names = ["alpha1", "alpha2", "f0", "f1", "beta", "nu", "eps"]

            model = model_2a_plus

        if model_name == "poly_range":
            r = numpy.arange(poly, poly2 + 1)
            r = r[r != 0]
            param_names = ["\\alpha_1", "\\alpha_2", "f_0", "f_1", "\\beta", "\\nu"] + ["\\epsilon_{" + f"{i}" + "}" for i in r]
            x0 = [0, 0, 0.4459, -0.02707, 1, 2 / 3] + [0, ] * len(r)

            def polyno(*args):
                return polynomial_range_2a(poly, poly2, *args)

            model = polyno

        if model_name == "poly_range_no_scaling":
            r = numpy.arange(poly, poly2 + 1)
            r = r[r != 0]
            param_names = ["\\alpha_1", "\\alpha_2", "\\beta"] + ["\\epsilon_{" + f"{i}" + "}" for i in r]
            x0 = [0, 0, 1] + [0, ] * len(r)

            def polyno(*args):
                return polynomial_range_2a_no_scaling(poly, poly2, *args)

            model = polyno

        if model_name == "poly_range_no_scaling_no_log":
            r = numpy.arange(poly, poly2 + 1)
            r = r[r != 0]
            param_names = ["\\alpha_1", "\\alpha_2"] + ["\\epsilon_{" + f"{i}" + "}" for i in r]
            x0 = [0, 0] + [0, ] * len(r)

            def polyno(*args):
                return polynomial_range_2a_no_scaling_no_log(poly, poly2, *args)

            model = polyno

        if model_name == "poly_range_no_log":
            r = numpy.arange(poly, poly2 + 1)
            r = r[r != 0]
            param_names = ["\\alpha_1", "\\alpha_2", "f_0", "f_1", "\\nu"] + ["\\epsilon_{" + f"{i}" + "}" for i in r]
            x0 = [0, 0, 0.4459, -0.02707, 2 / 3] + [0, ] * len(r)

            def polyno(*args):
                return polynomial_range_2a_no_log(poly, poly2, *args)

            model = polyno


    # The minimum number of degrees of freedom needed to consider a fit valid
    if dof is None:
        min_dof = 15

    else:
        min_dof = dof

    n_params = len(param_names)

    # Make a list of all Bbar pairs
    Bbar_list = []
    for i in range(len(Bbar_s)):
        for j in range(i + 1, len(Bbar_s)):
            if abs(Bbar_s[j] - Bbar_s[i]) > Bbar_lim:
                Bbar_list.append([Bbar_s[i], Bbar_s[j]])

    pvalues = numpy.zeros((len(Bbar_list), len(GL_mins)))
    params = numpy.zeros((len(Bbar_list), len(GL_mins), n_params))
    dofs = numpy.zeros((len(Bbar_list), len(GL_mins)))

    for i, Bbar_s in enumerate(Bbar_list):
        Bbar_1, Bbar_2 = Bbar_s
        # print(f"Running fits with Bbar_1 = {Bbar_1}, Bbar_2 = {Bbar_2}")

        for j, GL_min in enumerate(GL_mins):
            pvalues[i, j], params[i, j], dofs[i, j] = \
                run_frequentist_analysis(h5_data_file, model, N_s, g_s, L_s,
                                         Bbar_s, GL_min, GL_max, param_names,
                                         x0, run_bootstrap=False)

    # Extract the index of the smallest GL_min fit that has an acceptable
    # p-value
    r = len(GL_mins)
    best = r - 1

    for i, GL_min in enumerate(GL_mins):
        if numpy.max(pvalues[:, r - 1 - i]) > 0.05:
            best = r - 1 - i

    best_Bbar_index = numpy.argmax(pvalues[:, best])
    best_Bbar = Bbar_list[best_Bbar_index]

    # If there are no acceptable fits then look at the fit with the highest
    # pvalue all other things aside
    if numpy.max(pvalues) < 0.05:
        best_Bbar_index, best = numpy.unravel_index(numpy.argmax(pvalues), pvalues.shape)
        best_Bbar = Bbar_list[best_Bbar_index]

    # print("##################################################################")
    # print("BEST RESULT")
    print("\\Bar{" + "B}" + f" &= {best_Bbar} \\\\ \\nonumber")
    if GL_mins[best] < 12.7:
        print("\\color{" + "green}" + "gL_{" + "min}" + f" &= {GL_mins[best]} \\\\ \\nonumber")

    elif GL_mins[best] < 12.9:
        print("\\color{" + "orange}" + "gL_{" + "min}" + f" &= {GL_mins[best]} \\\\ \\nonumber")

    else:
        print("\\color{" + "red}" + "gL_{" + "min}" + f" &= {GL_mins[best]} \\\\ \\nonumber")

    if pvalues[best_Bbar_index, best] < 0.05:
        print("\\color{" + "red}" + f" p &= {pvalues[best_Bbar_index, best]} \\\\ \\nonumber")

    else:
        print(f"p &= {pvalues[best_Bbar_index, best]} \\\\ \\nonumber ")

    print(f"dof &= {dofs[best_Bbar_index, best]} \\\\ \\nonumber")
    # print("##################################################################")

    params_central = params[best_Bbar_index, best]

    # Find the parameter variation over acceptable fits
    acceptable = numpy.logical_and(
                    numpy.logical_and(pvalues > 0.05, pvalues < 0.95),
                    dofs >= min_dof)

    # if numpy.sum(acceptable) == 0:
    #     # print("##############################################################")
    #     # print("No acceptable fits found!")
    #     return None

    # Find the most extreme values of the parameter estimates that are deemed
    # acceptable
    sys_sigmas = numpy.zeros(n_params)

    for i, param in enumerate(param_names):
        param_small = params[..., i]

        if numpy.sum(acceptable) > 0:
            minimum = numpy.min(param_small[acceptable])
            maximum = numpy.max(param_small[acceptable])

            # Define the systematic error bar by the largest deviation from the
            # central fit by an acceptable fit
            sys_sigmas[i] = max(maximum - params[best_Bbar_index, best, i],
                                params[best_Bbar_index, best, i] -
                                minimum)

        else:
            sys_sigmas[i] = 0

        if abs(sys_sigmas[i]) > abs(params_central[i]):
            print("\\color{red" + "}" + f"{param} &= {params_central[i]} " + "\pm " + f"{sys_sigmas[i]} \\\\ \\nonumber")

        else:
            print(f"{param} &= {params_central[i]} " + "\pm " + f"{sys_sigmas[i]} \\\\ \\nonumber")

    # Find the systematic variation in the critical mass
    if model_name == "model1":
        g = 0.1
        m_c = mPT_1loop(g, N) + g ** 2 * (params[best_Bbar_index, best, 0] -
                                          params[best_Bbar_index, best, -2] *
                                          K1(g, N))
        if N == 4:
            print("Critical Mass using alpha1:")

        print(f"m_c = {m_c}")

        alphas = params[..., 0]
        betas = params[..., -2]

        # Only include parameter estimates from those fits that are acceptable
        alphas = alphas[acceptable]
        betas = betas[acceptable]

        m_cs = mPT_1loop(g, N) + g ** 2 * (alphas - betas * K1(g, N))

        minimum_m = numpy.min(m_cs)
        maximum_m = numpy.max(m_cs)

        m_c_error = max(m_c - minimum_m, maximum_m - m_c)
        print(f"m_c_error = {m_c_error}")

    if N == 2:
        results = {}
        results["params_central"] = params_central
        results["params_std"] = sys_sigmas

        if model_name == "model1":
            results["m_c"] = m_c
            results["m_c_error"] = m_c_error

    if N == 4:
        results = {}
        results["params_central"] = params_central
        results["params_std"] = sys_sigmas

        if model_name == "model1":
            alphas2 = params[..., 1]

            # Only include parameter estimates from those fits that are
            # acceptable
            alphas2 = alphas2[acceptable]

            # Calculate using alpha2
            m_c2 = mPT_1loop(g, N) + g ** 2 * \
                (params[best_Bbar_index, best, 1] -
                 params[best_Bbar_index, best, -2] *
                 K1(g, N))

            print("Critical Mass using alpha2:")

            print(f"m_c2 = {m_c2}")

            m_c2s = mPT_1loop(g, N) + g ** 2 * (alphas2 - betas * K1(g, N))

            minimum_m2 = numpy.min(m_c2s)
            maximum_m2 = numpy.max(m_c2s)

            m_c2_error = max(m_c2 - minimum_m2, maximum_m2 - m_c2)
            print(f"m_c2_error = {m_c2_error}")

            # Get the overall systematic error accounting for both alphas
            m_c_error_overall = max(max(m_c - minimum_m, maximum_m - m_c),
                                    max(m_c - minimum_m2, maximum_m2 - m_c))

            print("Overall result when accounting for both alphas:")
            print(f"m_c = {m_c} +- {m_c_error_overall}")

            results["m_c1"] = m_c
            results["m_c_error1"] = m_c_error
            results["m_c2"] = m_c2
            results["m_c_error2"] = m_c2_error
            results["m_c"] = m_c
            results["m_c_error"] = m_c_error_overall

    results["param_names"] = param_names
    results["GL_min"] = GL_mins[best]

    return results, best_Bbar


# def get_Bayes_factors(N, points=5000, Bbar_fixed=None, model_name="", test_poly=False, poly=None, poly2=None, test_mono=False, lognormal=False):
#     """
#         This function produces the Bayes Factors shown in the publication.

#         INPUTS :
#         --------
#         N: int, rank of the SU(N) valued fields
#         points: int, number of points to use in the MULTINEST algorithm. The
#             higher this is the more accurate the algorithm will be, but at the
#             price of computational cost. To produce the plot of the Bayes
#             factor against gL_min 5000 points were used. For the posterior
#             plots 1000 points were used.

#         OUTPUTS :
#         ---------
#         Bayes_factors: The log10 of the Bayes factor of the
#             Lambda_IR = g / (4 pi N) model over the Lambda_IR = 1 / L model.
#             This is an array of lenght equal to the number of GL_min cuts
#             considered, with each element containin the log Bayes factor of the
#             corresponding GL_min cut.
#     """
#     GL_mins = numpy.array([4, 4.8, 6.4, 8, 9.6, 12.8, 14.4,
#                            16, 19.2, 24, 25.6, 28.8, 32])
#     GL_max = 76.8

#     # Where the output samples will be saved
#     directory = "MULTINEST_samples/"

#     # Use this to label different runs if you edit something
#     tag = ""

#     # Prior Name: To differentiate results which use different priors
#     prior_name = "A"

#     # For reproducability
#     seed = 3475642

#     g_s_in = [0.1, 0.2, 0.3, 0.5, 0.6]
#     L_s_in = [8, 16, 32, 48, 64, 96, 128]
#     GL_max = 76.8

#     mu_s = None
#     sigma_s = None
#     param_idxs = None

#     if N == 2:
#         N = 2
#         N_s_in = [N]

#         if Bbar_fixed is not None:
#             Bbar_s_in = Bbar_fixed

#         else:
#             Bbar_s_in = [0.52, 0.53]

#         model1 = model1_1a
#         param_names = ["alpha", "f0", "f1", "beta", "nu"]

#         alpha_range = [-0.4, 0.4]
#         f0_range = [0, 1]
#         f1_range = [-20, 20]
#         beta_range = [-15, 15]
#         nu_range = [0, 15]

#         prior_range = [alpha_range, f0_range, f1_range, beta_range,
#                        nu_range]

#         n_params = len(prior_range)

#         if test_poly:
#             def model2(*args):
#                 return polynomial_1a(poly, *args)

#             param_names2 = param_names + [f"eps{i}" for i in range(1, poly + 1)]
#             prior_range2 = prior_range + [beta_range for i in range(1, poly + 1)]

#             n_params2 = n_params + poly

#         elif test_mono:
#             def model2(*args):
#                 return monomial_no_log_1a(poly, *args)

#             param_names2 = param_names
#             prior_range2 = prior_range

#             n_params2 = n_params

#         else:
#             model2 = model2_1a
#             n_params2 = n_params
#             prior_range2 = prior_range
#             param_names2 = param_names

#         if not lognormal:
#             if model_name == "poly_range":
#                 r = numpy.arange(poly, poly2 + 1)
#                 r = r[r != 0]
#                 param_names2 = param_names + ["\\epsilon_{" + f"{i}" + "}" for i in r]
#                 prior_range2 = prior_range + [beta_range for i in r]

#                 tag = f"poly{poly}_{poly2}"

#                 def model2(*args):
#                     return polynomial_range_1a(poly, poly2, *args)

#                 n_params2 = n_params + len(r)

#         else:
#             beta_range = [10 ** -5, 10 ** 5]
#             prior_range = [alpha_range, f0_range, f1_range, beta_range,
#                         nu_range]

#             if model_name == "poly_range":
#                 r = numpy.arange(poly, poly2 + 1)
#                 r = r[r != 0]
#                 param_names2 = param_names + ["\\epsilon_{" + f"{i}" + "}" for i in r]
#                 prior_range2 = prior_range + [beta_range for i in r]

#                 tag = f"poly{poly}_{poly2}"

#                 def model2(*args):
#                     return polynomial_range_1a(poly, poly2, *args)

#                 n_params2 = n_params + len(r)

#                 param_idxs = [3] + [(5 + i) for i, x in enumerate(r)]
#                 sigma_s = [0.5, ] * len(param_idxs)
#                 mu_s = [0, ] * len(param_idxs)

#                 if poly == 1 and poly2 == 2:
#                     prior_range2 = prior_range + [[-10 ** -5, -10 ** 5],] + [beta_range,]

#     if N == 4:
#         N = 4
#         N_s_in = [N]

#         if Bbar_fixed is not None:
#             Bbar_s_in = Bbar_fixed

#         else:
#             Bbar_s_in = [0.42, 0.43]

#         x0 = [0, 0, 0.4459, -0.02707, 1, 2 / 3]  # EFT values

#         model1 = model1_2a
#         param_names = ["alpha1", "alpha2", "f0", "f1", "beta", "nu"]
#         alpha_range1 = [-0.4, 0.4]
#         alpha_range2 = [-0.4, 0.4]
#         f0_range = [0, 1]
#         f1_range = [-20, 20]
#         beta_range = [-15, 15]
#         nu_range = [0, 15]

#         prior_range = [alpha_range1, alpha_range2, f0_range, f1_range,
#                        beta_range, nu_range]

#         n_params = len(prior_range)

#         if test_poly:
#             def model2(*args):
#                 return polynomial_2a(poly, *args)

#             param_names2 = param_names + [f"eps{i}" for i in range(1, poly + 1)]
#             prior_range2 = prior_range + [beta_range for i in range(1, poly + 1)]
#             n_params2 = n_params + poly

#         elif test_mono:
#             def model2(*args):
#                 return monomial_no_log_2a(poly, *args)

#             param_names2 = param_names
#             prior_range2 = prior_range

#             n_params2 = n_params

#         else:
#             model2 = model2_2a
#             n_params2 = n_params
#             prior_range2 = prior_range
#             param_names2 = param_names

#         if model_name == "poly_range":
#             r = numpy.arange(poly, poly2 + 1)
#             r = r[r != 0]
#             param_names2 = param_names + ["\\epsilon_{" + f"{i}" + "}" for i in r]
#             prior_range2 = prior_range + [beta_range for i in r]

#             tag = f"poly{poly}_{poly2}"

#             def model2(*args):
#                 return polynomial_range_1a(poly, poly2, *args)

#             n_params2 = n_params + len(r)

#     Bayes_factors = numpy.zeros(len(GL_mins))

#     for i, GL_min in enumerate(GL_mins):
#         samples, g_s, L_s, Bbar_s, m_s = \
#             load_h5_data(h5_data_file, N_s_in, g_s_in, L_s_in, Bbar_s_in,
#                          GL_min, GL_max)

#         # pdb.set_trace()
#         analysis1, best_fit1 = \
#             run_pymultinest(prior_range, model1, GL_min, GL_max, n_params,
#                             directory, N, g_s, Bbar_s, L_s, samples, m_s,
#                             param_names, n_live_points=points,
#                             sampling_efficiency=0.3, clean_files=True,
#                             tag=tag, prior_name=prior_name, keep_GLmax=False,
#                             return_analysis_small=True, seed=seed)

#         analysis2, best_fit2 = \
#             run_pymultinest(prior_range2, model2, GL_min, GL_max, n_params2,
#                             directory, N, g_s, Bbar_s, L_s, samples, m_s,
#                             param_names2, n_live_points=points,
#                             sampling_efficiency=0.3, clean_files=True,
#                             tag=tag, prior_name=prior_name, keep_GLmax=False,
#                             return_analysis_small=True, seed=seed, lognormal=lognormal, mu_s=mu_s,
#                             sigma_s=sigma_s, param_idxs=param_idxs)

#         # This is the log of the Bayes factor equal to the difference in the
#         # log-evidence's between the two models
#         Bayes_factors[i] = analysis1[0] - analysis2[0]

#     # Change log bases to log10 to match the plot in the publication
#     Bayes_factors = Bayes_factors / numpy.log(10)

#     return Bayes_factors
