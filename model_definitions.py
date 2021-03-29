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

import pickle
import os
import numpy
import warnings
import pdb

from scipy.special import gammaincc
import h5py


def load_h5_data(filename, N_s_list, g_s_list, L_s_list, Bbar_list, GL_min=0,
                 GL_max=numpy.inf):
    """
        INPUTS:
        -------
        filename: string, file name of input Binder crossing mass values
        model: Fit anzatz function
        N_s_list: List of ints of N (SU(N) rank) values to be studied. N values
            should be ints
        g_s_list: List of ag (coupling constant in lattice units) values to be
            studied, ag values should be floats
        L_s_list: List of L / a (Lattice Size) values to be studied, L values
            should be ints
        Bbar_s_list: List of Bbar crossing values to be studied, Bbar values
            should be floats
        GL_min: Float, minimum value of the product (ag) * (L / a) to be
            included in the fit
        GL_max: Float, maximum value of the product (ag) * (L / a) to be
            included in the fit

        OUTPUTS:
        --------
        This function will find all entries in the input file with parameters
        corresponding to the inputs. If there are M such entries found then the
        data returned is the following:

        samples: (M, no_samples) array of floats, where no_samples is the
            number of bootstrap samples used in producing the data. Each entry
            represents the crossing point mass of a different bootstrap sample.
        g_s: (M, ) array of floats, contatining the values of the coupling
            constants in lattice units
        L_s: (M, ) array of floats, contatining the values of the lattice size
            in lattice units
        Bbar_s: (M, ) array of floats, contatining the values of the Binder
            Cumulant crossing values
        m_s: (M, ) array of floats, contatining the values of the mass at the
            given value of the Binder Cumulant
    """
    f = h5py.File(filename, 'r')

    Bbar_s = []
    N_s = []
    g_s = []
    L_s = []
    m_s = []
    samples = []

    for Bbar in Bbar_list:
        for N in N_s_list:
            for g in g_s_list:
                for L in L_s_list:
                    ens = f[f'N={N}'][f'g={g:.2f}'][f'L={L}']

                    try:
                        m_s.append(ens[f'Bbar={Bbar:.3f}']['central'][()])

                    except Exception:
                        print(f"WARNING: CONFIGURATION MISSING: N={N}" +
                              f"g={g:.2f}, L={L}, Bbar={Bbar:.3f}")
                        continue

                    samples.append(ens[f'Bbar={Bbar:.3f}']['bs_bins'][()])
                    N_s.append(N)
                    g_s.append(g)
                    L_s.append(L)
                    Bbar_s.append(Bbar)

    f.close()

    # Turn data into numpy arrays
    N_s = numpy.array(N_s)
    g_s = numpy.array(g_s)
    L_s = numpy.array(L_s)
    Bbar_s = numpy.array(Bbar_s)
    m_s = numpy.array(m_s)
    samples = numpy.array(samples)

    # Remove nan values
    keep = numpy.logical_not(numpy.isnan(samples))[:, 0]
    samples = samples[keep]
    N_s = N_s[keep]
    g_s = g_s[keep]
    L_s = L_s[keep]
    Bbar_s = Bbar_s[keep]
    m_s = m_s[keep]

    GL_s = g_s * L_s

    keep = numpy.logical_and(GL_s >= GL_min * (1 - 10 ** -10),
                             GL_s <= GL_max * (1 + 10 ** -10))

    return samples[keep], g_s[keep], L_s[keep], Bbar_s[keep], m_s[keep]


# The one-loop expression of the critical mass
def mPT_1loop(g, N):
    """
        INPUTS :
        --------
        g: float, coupling constant
        N: int, rank of the SU(N) field matrices

        OUTPUTS :
        ---------
        float, value of the one-loop estimate of the critical mass
    """
    Z0 = 0.252731

    return - g * Z0 * (2 - 3 / N ** 2)


# Calculate the lambda terms
def K1(g, N):
    """
        INPUTS :
        --------
        g: float, coupling constant
        N: int, rank of the SU(N) field matrices

        OUTPUTS :
        ---------
        float, value of the log term in the fit anzatz for the critical mass
            with Lambda_IR = g / (4 pi N).
    """
    return numpy.log((g / (4 * numpy.pi * N))) *\
                    ((1 - (6 / N ** 2) + (18 / N ** 4)) / (4 * numpy.pi) ** 2)


def K2(L, N):
    """
        INPUTS :
        --------
        g: float, coupling constant
        N: int, rank of the SU(N) field matrices

        OUTPUTS :
        ---------
        float, value of the log term in the fit anzatz for the critical mass
            with Lambda_IR = L.
    """
    return numpy.log(1 / L) *\
                    ((1 - (6 / N ** 2) + (18 / N ** 4)) / (4 * numpy.pi) ** 2)


# No corrections to scaling model
def model1_1a(N, g, L, Bbar, alpha, f0, f1, lambduh, nu):
    """
        Fit anzatz with Lambda_IR = g / (4 pi N) and a single value of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    result = mPT_1loop(g, N) + g ** 2 *\
        (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
            - lambduh * K1(g, N))

    return result


def model2_1a(N, g, L, Bbar, alpha, f0, f1, lambduh, nu):
    """
        Fit anzatz with Lambda_IR = L and a single value of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                        - lambduh * K2(L, N))


def model1_2a(N, g, L, Bbar, alpha1, alpha2, f0, f1, lambduh, nu):
    """
        Fit anzatz with Lambda_IR = g / (4 pi N) and two values of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha1: As written in fit anzatz
        alpha2: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
            crosses the constant Binder value of Bbar
    """
    Bbar_s = numpy.sort(list(set(Bbar)))
    alpha = numpy.where(Bbar == Bbar_s[0], alpha1, alpha2)

    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                        - lambduh * K1(g, N))


def model2_2a(N, g, L, Bbar, alpha1, alpha2, f0, f1, lambduh, nu):
    """
        Fit anzatz with Lambda_IR = g / (4 pi N) and two values of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha1: As written in fit anzatz
        alpha2: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    Bbar_s = numpy.sort(list(set(Bbar)))
    alpha = numpy.where(Bbar == Bbar_s[0], alpha1, alpha2)

    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                        - lambduh * K2(L, N))


def model1_2a_Bbar_list(Bbar_s, N, g, L, Bbar, alpha1, alpha2, f0, f1, lambduh, nu):
    """
        Fit anzatz with Lambda_IR = g / (4 pi N) and two values of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha1: As written in fit anzatz
        alpha2: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
            crosses the constant Binder value of Bbar
    """
    Bbar_s = numpy.sort(list(set(Bbar_s)))
    alpha = numpy.where(Bbar == Bbar_s[0], alpha1, alpha2)

    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                        - lambduh * K1(g, N))


def model2_2a_Bbar_list(Bbar_s, N, g, L, Bbar, alpha1, alpha2, f0, f1, lambduh, nu):
    """
        Fit anzatz with Lambda_IR = g / (4 pi N) and two values of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha1: As written in fit anzatz
        alpha2: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    Bbar_s = numpy.sort(list(set(Bbar_s)))
    alpha = numpy.where(Bbar == Bbar_s[0], alpha1, alpha2)

    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                        - lambduh * K2(L, N))


def polynomial_range_2a_Bbar_list(Bbar_s, n1, n2, N, g, L, Bbar, alpha1, alpha2, f0, f1, lambduh, nu, *args):
    """
        Fit anzatz with Lambda_IR = L and a single value of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    alpha = numpy.where(Bbar == Bbar_s[0], alpha1, alpha2)

    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                        - lambduh * K2(L, N) + polynomial_range(g * L, n1, n2, N, *args))

def model1_1a_g4(N, g, L, Bbar, alpha, f0, f1, lambduh, nu, eps):
    """
        Fit anzatz with Lambda_IR = g / (4 pi N) and a single value of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    result = mPT_1loop(g, N) + g ** 2 *\
        (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
            - lambduh * K1(g, N) + eps * g ** 2)

    return result


def model_mixed_1a(N, g, L, Bbar, alpha, f0, f1, lambduh, nu):
    result = numpy.where(g / (4 * numpy.pi * N) < 1 / L,
    # result = numpy.where(True,
            mPT_1loop(g, N) + g ** 2 *\
            (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                - lambduh * K2(L / N, N)),
                mPT_1loop(g, N) + g ** 2 *\
            (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                - lambduh * K1(g, N)))

    return result


# def model_mixed_fit_1a(N, g, L, Bbar, alpha, f0, f1, lambduh, nu, eps):
#     result = numpy.where(g < eps / L,
#     # result = numpy.where(g / (4 * numpy.pi * N) < 1 / L,
#             mPT_1loop(g, N) + g ** 2 *\
#             (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
#                 - lambduh * K2(L, N)),
#                 mPT_1loop(g, N) + g ** 2 *\
#             (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
#                 - lambduh * K1(g, N)))

#     return result


def polynomial(x, n, N, *coefs):
    """
        coefs should be of length n representing the coefficients of all
        the terms in the polynomial (no constant)
    """
    total = 0

    if n > 0:
        for i in range(0, n):
            total += coefs[i] * x ** (i + 1)

    # Include the 1 / x term
    if n < 0:
        total += coefs[0] * x ** -1
        for i in range(1, -n):
            total += coefs[i] * x ** i

    # pdb.set_trace()

    return total * ((1 - (6 / N ** 2) + (18 / N ** 4)) / (4 * numpy.pi) ** 2)


def polynomial_range(x, n1, n2, N, *coefs):
    """
        All monomial contributions between integers n1 and n2.
        coefs should be of length representing the coefficients of all
        the terms in the polynomial (no constant).
    """
    total = 0


    r1 = numpy.arange(n1, n2 + 1)
    r1 = r1[r1 != 0]

    for idx, i in enumerate(r1):
        total += coefs[idx] * x ** i

    return total * ((1 - (6 / N ** 2) + (18 / N ** 4)) / (4 * numpy.pi) ** 2)


def polynomial_range_1a(n1, n2, N, g, L, Bbar, alpha, f0, f1, lambduh, nu, *args):
    """
        Fit anzatz with Lambda_IR = L and a single value of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                        - lambduh * K2(L, N) + polynomial_range(g * L, n1, n2, N, *args))


def polynomial_range_2a(n1, n2, N, g, L, Bbar, alpha1, alpha2, f0, f1, lambduh, nu, *args):
    """
        Fit anzatz with Lambda_IR = L and a single value of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    Bbar_s = numpy.sort(list(set(Bbar)))
    alpha = numpy.where(Bbar == Bbar_s[0], alpha1, alpha2)

    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                        - lambduh * K2(L, N) + polynomial_range(g * L, n1, n2, N, *args))


def polynomial_range_1a_no_log(n1, n2, N, g, L, Bbar, alpha, f0, f1, nu, *args):
    """
        Fit anzatz with Lambda_IR = L and a single value of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                        + polynomial_range(g * L, n1, n2, N, *args))


def polynomial_range_2a_no_log(n1, n2, N, g, L, Bbar, alpha1, alpha2, f0, f1, nu, *args):
    """
        Fit anzatz with Lambda_IR = L and a single value of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    Bbar_s = numpy.sort(list(set(Bbar)))
    alpha = numpy.where(Bbar == Bbar_s[0], alpha1, alpha2)

    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                        + polynomial_range(g * L, n1, n2, N, *args))


def polynomial_range_1a_no_scaling(n1, n2, N, g, L, Bbar, alpha, lambduh, *args):
    """
        Fit anzatz with Lambda_IR = L and a single value of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha
                        - lambduh * K2(L, N) + polynomial_range(g * L, n1, n2, N, *args))


def polynomial_range_2a_no_scaling(n1, n2, N, g, L, Bbar, alpha1, alpha2, lambduh, *args):
    """
        Fit anzatz with Lambda_IR = L and a single value of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    Bbar_s = numpy.sort(list(set(Bbar)))
    alpha = numpy.where(Bbar == Bbar_s[0], alpha1, alpha2)

    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha
                        - lambduh * K2(L, N) + polynomial_range(g * L, n1, n2, N, *args))


def polynomial_range_1a_no_scaling_no_log(n1, n2, N, g, L, Bbar, alpha, *args):
    """
        Fit anzatz with Lambda_IR = L and a single value of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha
                        + polynomial_range(g * L, n1, n2, N, *args))


def polynomial_range_2a_no_scaling_no_log(n1, n2, N, g, L, Bbar, alpha1, alpha2, *args):
    """
        Fit anzatz with Lambda_IR = L and a single value of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    Bbar_s = numpy.sort(list(set(Bbar)))
    alpha = numpy.where(Bbar == Bbar_s[0], alpha1, alpha2)

    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha
                        + polynomial_range(g * L, n1, n2, N, *args))


def polynomial_range_2a_Bbar_list(Bbar_s, n1, n2, N, g, L, Bbar, alpha1, alpha2, f0, f1, lambduh, nu, *args):
    """
        Fit anzatz with Lambda_IR = L and a single value of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    alpha = numpy.where(Bbar == Bbar_s[0], alpha1, alpha2)

    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                        - lambduh * K2(L, N) + polynomial_range(g * L, n1, n2, N, *args))


def polynomial_range_2a_no_log_Bbar_list(Bbar_s, n1, n2, N, g, L, Bbar, alpha1, alpha2, f0, f1, nu, *args):
    """
        Fit anzatz with Lambda_IR = L and a single value of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    alpha = numpy.where(Bbar == Bbar_s[0], alpha1, alpha2)

    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                        + polynomial_range(g * L, n1, n2, N, *args))


def polynomial_range_2a_no_scaling_Bbar_list(Bbar_s, n1, n2, N, g, L, Bbar, alpha1, alpha2, lambduh, *args):
    """
        Fit anzatz with Lambda_IR = L and a single value of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    alpha = numpy.where(Bbar == Bbar_s[0], alpha1, alpha2)

    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha
                        - lambduh * K2(L, N) + polynomial_range(g * L, n1, n2, N, *args))


def polynomial_range_2a_no_scaling_no_log_Bbar_list(Bbar_s, n1, n2, N, g, L, Bbar, alpha1, alpha2, *args):
    """
        Fit anzatz with Lambda_IR = L and a single value of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    alpha = numpy.where(Bbar == Bbar_s[0], alpha1, alpha2)

    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha
                        + polynomial_range(g * L, n1, n2, N, *args))


def polynomial_1a(n, N, g, L, Bbar, alpha, f0, f1, lambduh, nu, *args):
    """
        Fit anzatz with Lambda_IR = L and a single value of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                        - lambduh * K2(L, N) + polynomial(g * L, n, N, *args))


def polynomial_2a(n, N, g, L, Bbar, alpha1, alpha2, f0, f1, lambduh, nu, *args):
    """
        Fit anzatz with Lambda_IR = g / (4 pi N) and two values of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha1: As written in fit anzatz
        alpha2: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    Bbar_s = numpy.sort(list(set(Bbar)))
    alpha = numpy.where(Bbar == Bbar_s[0], alpha1, alpha2)

    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                        - lambduh * K2(L, N) + polynomial(g * L, n, N, *args))


def monomial(x, n, N, eps):
    """
        eps should be of length n representing the coefficients of all
        the terms in the monomial (no constant)
    """
    return eps * x ** n * ((1 - (6 / N ** 2) + (18 / N ** 4)) / (4 * numpy.pi) ** 2)


def monomial_1a(n, N, g, L, Bbar, alpha, f0, f1, lambduh, nu, eps):
    """
        Fit anzatz with Lambda_IR = L and a single value of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                        - lambduh * K2(L, N)) + monomial(g * L, n, eps)


def monomial_2a(n, N, g, L, Bbar, alpha1, alpha2, f0, f1, lambduh, nu, eps):
    """
        Fit anzatz with Lambda_IR = g / (4 pi N) and two values of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha1: As written in fit anzatz
        alpha2: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    Bbar_s = numpy.sort(list(set(Bbar)))
    alpha = numpy.where(Bbar == Bbar_s[0], alpha1, alpha2)

    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                        - lambduh * K2(L, N)) + monomial(g * L, n, eps)


def monomial_no_log_1a(n, N, g, L, Bbar, alpha, f0, f1, beta, nu):
    """
        Fit anzatz with Lambda_IR = L and a single value of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        beta: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                        + monomial(g * L, n, N, beta))


def monomial_no_log_2a(n, N, g, L, Bbar, alpha1, alpha2, f0, f1, beta, nu):
    """
        Fit anzatz with Lambda_IR = g / (4 pi N) and two values of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha1: As written in fit anzatz
        alpha2: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    Bbar_s = numpy.sort(list(set(Bbar)))
    alpha = numpy.where(Bbar == Bbar_s[0], alpha1, alpha2)

    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                        + monomial(g * L, n, N, beta))


def polynomial_no_log_1a(n, N, g, L, Bbar, alpha, f0, f1, nu, *args):
    """
        Fit anzatz with Lambda_IR = L and a single value of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        beta: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                        + polynomial(g * L, n, N, *args))


def polynomial_no_log_2a(n, N, g, L, Bbar, alpha1, alpha2, f0, f1, nu, *args):
    """
        Fit anzatz with Lambda_IR = g / (4 pi N) and two values of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha1: As written in fit anzatz
        alpha2: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    Bbar_s = numpy.sort(list(set(Bbar)))
    alpha = numpy.where(Bbar == Bbar_s[0], alpha1, alpha2)

    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                        + polynomial(g * L, n, N, *args))


def polynomial_1a(n, N, g, L, Bbar, alpha, f0, f1, lambduh, nu, *args):
    """
        Fit anzatz with Lambda_IR = L and a single value of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                        - lambduh * K2(L, N) + polynomial(g * L, n, N, *args))


def polynomial_no_scaling_1a(n, N, g, L, Bbar, alpha, lambduh, *args):
    """
        Fit anzatz with Lambda_IR = g / (4 pi N) and two values of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha1: As written in fit anzatz
        alpha2: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha - lambduh * K2(L, N) + polynomial(g * L, n, N, *args))


def polynomial_no_scaling_2a(n, N, g, L, Bbar, alpha1, alpha2, lambduh, *args):
    """
        Fit anzatz with Lambda_IR = g / (4 pi N) and two values of alpha

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha1: As written in fit anzatz
        alpha2: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    Bbar_s = numpy.sort(list(set(Bbar)))
    alpha = numpy.where(Bbar == Bbar_s[0], alpha1, alpha2)

    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha - lambduh * K2(L, N) + polynomial(g * L, n, N, *args))


def model_1a_plus(N, g, L, Bbar, alpha, f0, f1, lambduh, nu, eps):
    result = mPT_1loop(g, N) + g ** 2 *\
        (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
            - lambduh * K1(g, N) + eps / (g * L))

    return result


def model_2a_plus(N, g, L, Bbar, alpha1, alpha2, f0, f1, lambduh, nu, eps):
    Bbar_s = numpy.sort(list(set(Bbar)))
    alpha = numpy.where(Bbar == Bbar_s[0], alpha1, alpha2)

    result = mPT_1loop(g, N) + g ** 2 *\
        (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
            - lambduh * K1(g, N) + eps / (g * L))

    return result


# Model 1 with additional polynomial term of (a / L) ^ n
def monomial_1(n, N, g, L, Bbar, alpha, f0, f1, lambduh, nu, epsilon):
    """
        Fit anzatz with Lambda_IR = g / (4 pi N) and a single value of alpha,
        and an additional monomial term of order n

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent
        epsilon: float, monomial coefficient

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    result = mPT_1loop(g, N) + g ** 2 *\
        (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
            - lambduh * K2(L, N)) + epsilon * (g * L) ** n

    return result


def monomial_2(n, N, g, L, Bbar, alpha1, alpha2, f0, f1, lambduh, nu, epsilon):
    """
        Fit anzatz with Lambda_IR = g / (4 pi N) and a two values of alpha,
        and an additional monomial term of order n

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent
        epsilon: float, monomial coefficient

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    Bbar_s = numpy.sort(list(set(Bbar)))
    alpha = numpy.where(Bbar == Bbar_s[0], alpha1, alpha2)

    return mPT_1loop(g, N) + g ** 2 *\
                    (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
                        - lambduh * K2(L, N)) + epsilon * (g * L) ** n


def linear_1a(*args):
    return monomial_1(1, *args)


def linear_2a(*args):
    return monomial_2(1, *args)


def quadratic_1a(*args):
    return monomial_1(2, *args)


def quadratic_2a(*args):
    return monomial_2(2, *args)


def cubic_1a(*args):
    return monomial_1(3, *args)


def cubic_2a(*args):
    return monomial_2(3, *args)


def polynomial_3(N, g, L, Bbar, alpha, f0, f1, lambduh, nu, epsilon_1, epsilon_2, epsilon_3):
    """
        Fit anzatz with Lambda_IR = g / (4 pi N) and a single value of alpha,
        and an additional polynomial term of order n

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent
        epsilon: float, polynomial coefficient

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    result = mPT_1loop(g, N) + g ** 2 *\
        (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
            - lambduh * K2(L, N)) + epsilon_1 * (g * L) ** 1 +\
                epsilon_2 * (g * L) ** 2 + epsilon_3 * (g * L) ** 3

    return result


def polynomial_3_2a(N, g, L, Bbar, alpha1, alpha2, f0, f1, lambduh, nu, epsilon_1, epsilon_2, epsilon_3):
    """
        Fit anzatz with Lambda_IR = g / (4 pi N) and a single value of alpha,
        and an additional polynomial term of order n

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g: float, coupling constant
        L: int, lattice size
        Bbar: float, value of Bbar where masses intercept
        alpha: As written in fit anzatz
        f0: Equal to f(0) in the fit anzatz
        f1: Equal to f'(0) in the fit anzatz
        lambduh: Coefficient multiplying the log term in the fit anzatz
        nu: float, scaling exponent
        epsilon: float, polynomial coefficient

        OUTPUTS :
        ---------
        float, estimate of the critical mass where the binder cumulant graph
        crosses the constant Binder value of Bbar
    """
    Bbar_s = numpy.sort(list(set(Bbar)))
    alpha = numpy.where(Bbar == Bbar_s[0], alpha1, alpha2)

    result = mPT_1loop(g, N) + g ** 2 *\
        (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1)
            - lambduh * K2(L, N)) + epsilon_1 * (g * L) ** 1 +\
                epsilon_2 * (g * L) ** 2 + epsilon_3 * (g * L) ** 3

    return result


def cov_matrix_calc(g_s, L_s, m_s, samples):
    """
        This function calculates the covariance matrix of the Binder crossing
        data. We know that the correlation between different ensembles is
        exactly zero, and it is set to be so here.

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        g_s: 1D array of floats of length s, where s is the number of data
        points used in the fit. Each element is a value of the coupling, g.
        L_s: 1D array of ints of length s, where s is the number of data points
        used in the fit. Each element is a value of the lattice size, L.
        samples: 2D array of floats of size (s, no_samples), where s is the
        number of data points used in the fit, and no_samples is the number of
        bootstrap samples used for finding the Binder crossing points. Each
        element is a value of the Binder crossing mass, m

        OUTPUTS :
        ---------
        (s, s) matrix of floats, equal to the covariance matrix between the
        binder crossing data, where s is the number of data points used in the
        fit
    """
    # In reality the covariance between different ensembles is 0. We can set
    # it as such to get a more accurate calculation of the covariance matrix
    different_g = numpy.zeros((samples.shape[0], samples.shape[0]))
    different_L = numpy.zeros((samples.shape[0], samples.shape[0]))

    # Check if two ensembles have different physical parameters, e.g. they are
    # different ensembles
    for i in range(samples.shape[0]):
        for j in range(samples.shape[0]):
            different_g[i, j] = g_s[i] != g_s[j]
            different_L[i, j] = L_s[i] != L_s[j]

    # This is true if two data points come different simulations
    different_ensemble = numpy.logical_or(different_L, different_g)

    size = samples.shape[0]
    cov_matrix = numpy.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if different_ensemble[i, j] == 0:
                cov_matrix[i, j] = numpy.mean((samples[i] - m_s[i]) *
                                              (samples[j] - m_s[j]))
                # else the value remains zero as there is no covariance between
                # samples from different ensembles

    return cov_matrix, different_ensemble


def chisq_calc(x, cov_inv, model_function, res_function):
    """
        This function calculates the correlated chi-squared value of the model
        fit given the s data samples.

        INPUTS :
        --------
        x: tuple of floats, of length n_params (number of parameters in the
            fit) containing
        fit parameter values
        cov_inv: The square-root of the inverse of the covariance matrix
        model: A function that has the following callable signature
        model(N, g_s, L_s, Bbar_s, *x), where N is the rank of the field
            symmetry, g_s are the coupling constants, L_s are the lattice sizes
            and Bbar_s are the Bbar intercept values. *x are the parameters of
            the fit, and is of length n_params
        res_function: normalized residual function: See make_res_function

        OUTPUTS :
        ---------
        chisq: float, the value of the correlated chi-squared of the fit
    """
    error = False
    # Caculate the residuals between the model and the data

    # Use the warnings module to catch overflow warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')

        normalized_residuals = res_function(x, cov_inv, model_function)
        chisq = numpy.sum(normalized_residuals ** 2)

        if len(w) > 0:
            normalized_residuals, N, g_s, L_s, Bbar_s, x =\
                res_function(x, cov_inv, model_function, return_details=True)

            # Overflow warning occurs when nu is very small so that
            # (gL) ** -1 / nu is very large. Check that this is the case
            print("Warning: Small nu led to very large (gL) ** -1 / nu")

            # This is to check the warning is indeed as expected
            assert min(g_s * L_s) ** (- 1 / x[-1]) > 10 ** 10

    return chisq


def chisq_pvalue(k, x):
    """
        Calculates the p-value of the fit using the chi-squared distribution

        INPUTS :
        --------
        k: int, the number of degrees of freedom in the fit
        x: float, the chi-squared fit value of the fit

        OUTPUTS :
        ---------
        float, the p-value of the fit quality
    """
    return gammaincc(k / 2, x / 2)


# Try using the scipy least-squares method with Nelder-Mead
def make_res_function(N, m_s, g_s, L_s, Bbar_s):
    """
        This function returns a function called res_function, which calculates
        the normalized residuals of a fit using model_function, to the s data
        points defined by (N, m_s, g_s, L_s, Bbar_s)

        INPUTS :
        --------
        N: int, rank of the SU(N) field matrices
        m_s: 1D array of floats of length s, where s is the number of data
            points used in the fit. Each element is a value of the critical
            mass as determined by the crossing point of the binder cumulant
            with a constant value, Bbar
        Bbar_s: 1D array of floats of length s, where s is the number of data
            points used in the fit. Each element is a value of the constant
            value of the Binder cumumlant that is used to obtain the critcal
            mass.
        g_s: 1D array of floats of length s, where s is the number of data
            points used in the fit. Each element is a value of the coupling, g.
        L_s: 1D array of ints of length s, where s is the number of data points
            used in the fit. Each element is a value of the lattice size, L.
        samples: 2D array of floats of size (s, no_samples), where s is the
            number of data points used in the fit, and no_samples is the number
            of bootstrap samples used for finding the Binder crossing points.
            Each element is a value of the Binder crossing mass, m

        OUTPUTS :
        ---------
        res_function: Used in other functions defined in this publication to
        to calculate the normalized residuals between the data and the fit
    """
    def res_function(x, cov_inv, model_function, return_details=False):
        # Caculate the residuals between the model and the data
        predictions = model_function(N, g_s, L_s, Bbar_s, *x)

        residuals = m_s - predictions

        normalized_residuals = numpy.dot(cov_inv, residuals)

        if return_details:
            return normalized_residuals, N, g_s, L_s, Bbar_s, x

        else:
            return normalized_residuals

    return res_function
