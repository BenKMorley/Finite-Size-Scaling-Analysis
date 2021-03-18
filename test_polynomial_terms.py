from publication_results import *
import pdb
import faulthandler

faulthandler.enable()


# get_systematic_errors(2, "poly3")
# get_systematic_errors(4, "poly3")

# x1 = get_systematic_errors(2, "poly", poly=3)
# x2 = get_systematic_errors(2, "poly", poly=2)
# x3 = get_systematic_errors(2, "poly", poly=3)
# x4 = get_systematic_errors(2, "poly", poly=4)
# x5 = get_systematic_errors(2, "poly", poly=5)


# Bayes_factors2 = {}

# # Bayes_factors2[1] = get_Bayes_factors(2, points=800, test_poly=True, poly=1)

# pdb.set_trace()
# Bayes_factors2[2] = get_Bayes_factors(2, points=400, test_poly=True,
#                                             poly=2)


## PLOT P-VALUE EVIDENCE GRAPH
def pvalue():
    N = 4
    poly = 1
    color_blind_palette = [(0, 0, 0), (230, 159, 0), (86, 180, 233), (0, 158, 115),
                        (240, 228, 66), (0, 114, 178), (213, 94, 0), (204, 121, 167)]
    color_blind_palette = [(a / 255, b / 255, c / 255) for a, b, c in color_blind_palette]

    GL_mins = numpy.array([3.2, 4, 4.8, 6.4, 8, 9.6, 12.8, 14.4, 16, 19.2, 24, 25.6, 28.8, 32])
    GL_min_graph = 3.2

    # Bayes_factors = get_Bayes_factors(N, points=800, test_poly=True, poly=poly)
    Bayes_factors = get_Bayes_factors(N, points=200, test_mono=True, poly=poly)

    min_value = numpy.min(Bayes_factors)
    max_value = numpy.max(Bayes_factors)

    fig, ax = plt.subplots()

    ax.scatter(GL_mins, Bayes_factors, label=r"$\cal{E}$", marker='o', color='none', edgecolor='k')

    plt.fill_between([min(GL_mins) / 2, max(GL_mins) * 2], [-1, -1], [1, 1], color=color_blind_palette[7], alpha=0.5, label=r'Insignificant')
    plt.fill_between([min(GL_mins) / 2, max(GL_mins) * 2], [1, 1], [2, 2], color=color_blind_palette[4], alpha=0.2, label=r'Strong')
    plt.fill_between([min(GL_mins) / 2, max(GL_mins) * 2], [2, 2], [max_value * 4, max_value * 4], color=color_blind_palette[2], alpha=0.2, label=r'Decisive')
    plt.fill_between([min(GL_mins) / 2, max(GL_mins) * 2], [-2, -2], [-1, -1], color=color_blind_palette[4], alpha=0.2)
    plt.fill_between([min(GL_mins) / 2, max(GL_mins) * 2], [-10, -10], [-2, -2], color=color_blind_palette[2], alpha=0.2)

    ax = plt.gca()
    ax.legend(loc=(0.02, 0.3), framealpha=1)
    ax.set_ylabel(r"$\cal{E}$")
    ax.set_ylim(min_value + 1, max_value + 1)
    ax.set_xlim(min(GL_mins) - 1, max(GL_mins) + 1)
    ax.tick_params(direction='in')
    ax.set_xlabel(r"$gL_{min}$")
    ax = plt.gca()
    fig = plt.gcf()
    ax.set_xlim(GL_min_graph, 34)
    ax2 = ax.twinx()
    ax2.set_ylabel(r'$p-value$')

    # results = get_pvalues_central_fit(N, "poly", poly)
    results = get_pvalues_central_fit(N, "mono", poly)

    ax2.scatter(GL_mins, results["pvalues1"], marker='s', label=r"$\Lambda_{IR} \propto g$", color='none', edgecolor=color_blind_palette[1])
    ax2.scatter(GL_mins, results["pvalues2"], marker='^', label=r"$\Lambda_{IR} \propto \frac{1}{L}$", color='none', edgecolor=color_blind_palette[5])

    fig.tight_layout()
    ax2.tick_params(direction='in')
    ax2.legend(loc=(0.02, 0.15), framealpha=1)

    ax.set_yscale('symlog', linthreshy=2.5)
    ax2.annotate(r"$p = 0.05$", xy=(25, 0.07), color='grey')
    ax2.annotate(r"$p = 0.95$", xy=(25, 0.91), color='grey')

    pdb.set_trace()


## Plot central fits
def fit():
    g_s_in = [0.1, 0.2, 0.3, 0.5, 0.6]
    L_s_in = [8, 16, 32, 48, 64, 96, 128]
    N_s_in = [2]
    Bbar_s_in = [0.52, 0.53]
    GL_min = 12.8
    GL_max = 76.8

    samples, g_s, L_s, Bbar_s, m_s = load_h5_data(h5_data_file, N_s_in, g_s_in, L_s_in,
                                                    Bbar_s_in, GL_min, GL_max)

    def colors(g):
        if g == 0.1:
            return 'g'
        if g == 0.2:
            return 'r'
        if g == 0.3:
            return 'k'
        if g == 0.5:
            return 'y'
        if g == 0.6:
            return 'b'

    L_space = numpy.linspace(1 / max(L_s), 1 / min(L_s), 1000)
    results = get_statistical_errors_central_fit(2, model_name="poly", poly=1)['params_central']

    for i, Bbar in enumerate(set(Bbar_s)):
        for g in set(g_s):
            sub_ind = numpy.argwhere(numpy.logical_and(g_s == g, Bbar_s == Bbar))

            plt.scatter(1 / L_s[sub_ind], m_s[sub_ind] / g, color=colors(g))

            if i == 0:
                plt.plot(L_space, polynomial_1a(1, 2, g, 1 / L_space, Bbar, *results) / g, label=f'g = {g}', color=colors(g))
                # plt.plot(L_space, model2_1a(2, g, 1 / L_space, Bbar, *results[:-1]) / g, label=f'g = {g}', color=colors(g))

            if i == 1:
                plt.plot(L_space, polynomial_1a(1, 2, g, 1 / L_space, Bbar, *results) / g, color=colors(g))
                # plt.plot(L_space, model2_1a(2, g, 1 / L_space, Bbar, *results[:-1]) / g, color=colors(g))

    # Now for comparison put the log(g) fit onto the same plot
    L_space = numpy.linspace(1 / max(L_s), 1 / min(L_s), 1000)
    results2 = get_statistical_errors_central_fit(2)['params_central']

    for i, Bbar in enumerate(set(Bbar_s)):
        for g in set(g_s):
            sub_ind = numpy.argwhere(numpy.logical_and(g_s == g, Bbar_s == Bbar))

            plt.scatter(1 / L_s[sub_ind], m_s[sub_ind] / g, color=colors(g))

            if i == 0:
                plt.plot(L_space, model1_1a(2, g, 1 / L_space, Bbar, *results2) / g, label=f'g = {g}', color=colors(g), ls='--')
                # plt.plot(L_space, model2_1a(2, g, 1 / L_space, Bbar, *results2[:-1]) / g, label=f'g = {g}', color=colors(g))

            if i == 1:
                plt.plot(L_space, model1_1a(2, g, 1 / L_space, Bbar, *results2) / g, color=colors(g), ls='--')
                # plt.plot(L_space, model2_1a(2, g, 1 / L_space, Bbar, *results2[:-1]) / g, color=colors(g))

    plt.xlabel('a / L')
    plt.ylabel('m / g')
    plt.legend()
    plt.show()

    def remove_log(g, Bbar, N):
        """
        Gives the L value that removes the log for a given g function
        """
        nu_0 = results2[-1]
        del_nu = results[-2] - nu_0
        del_chi = -del_nu / nu_0 ** 2
        M = (results[1] - Bbar) / results[2]
        C = -((1 - (6 / N ** 2) + (18 / N ** 4)) / (4 * numpy.pi) ** 2)
        G = g ** (-1 / nu_0)
        Y = (C / (del_chi * G * M))
        pdb.set_trace()

        return numpy.exp(- numpy.log(Y) * nu_0)


    g_s = numpy.linspace(0.1, 0.6, 100)
    plt.plot(g_s, remove_log(g_s, 0.52, 2))
    plt.xlabel('g')
    plt.ylabel('L')
    plt.show()

fit()
