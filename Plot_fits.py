from multiprocessing import Pool
from publication_results import get_systematic_errors
from model_definitions import *
import matplotlib.pyplot as plt


h5_data_file = "./h5data/Bindercrossings.h5"


def plot(params):
    N, model_name = params
    # for poly in [-2, -1, 1, 2, 3]:
    for poly in [1]:
        for poly2 in [1]:
        # for poly2 in range(poly, 4):
            if poly2 != 0:
                print("\\subsubsection{" + f"Exponents between {poly} and {poly2}" + "}")
                print("")
                print("\\begin{" + "align}")

                results, Bbar_s_in = get_systematic_errors(N, model_name, poly=poly, poly2=poly2)
                central_values = results["params_central"]
                GL_min = results["GL_min"]
                GL_max = 76.8

                if model_name == "poly_range":
                    if N == 2:
                        def polyno(*args):
                            return polynomial_range_1a(poly, poly2, *args)

                    if N == 4:
                        def polyno(*args):
                            return polynomial_range_2a(poly, poly2, *args)

                        def polyno2(Bbar_s, *args):
                            return polynomial_range_2a_Bbar_list(Bbar_s, poly, poly2, *args)

                        model_Bbar_list = polyno2

                if model_name == "poly_range_no_log":
                    if N == 2:
                        def polyno(*args):
                            return polynomial_range_1a_no_log(poly, poly2, *args)

                    if N == 4:
                        def polyno(*args):
                            return polynomial_range_2a_no_log(poly, poly2, *args)

                        def polyno2(Bbar_s, *args):
                            return polynomial_range_2a_no_log_Bbar_list(Bbar_s, poly, poly2, *args)

                        model_Bbar_list = polyno2

                if model_name == "poly_range_no_scaling":
                    if N == 2:
                        def polyno(*args):
                            return polynomial_range_1a_no_scaling(poly, poly2, *args)

                    if N == 4:
                        def polyno(*args):
                            return polynomial_range_2a_no_scaling(poly, poly2, *args)

                        def polyno2(Bbar_s, *args):
                            return polynomial_range_2a_no_scaling_Bbar_list(Bbar_s, poly, poly2, *args)

                        model_Bbar_list = polyno2

                if model_name == "poly_range_no_scaling_no_log":
                    if N == 2:
                        def polyno(*args):
                            return polynomial_range_1a_no_scaling_no_log(poly, poly2, *args)

                    if N == 4:
                        def polyno(*args):
                            return polynomial_range_2a_no_scaling_no_log(poly, poly2, *args)

                        def polyno2(Bbar_s, *args):
                            return polynomial_range_2a_no_scaling_no_log_Bbar_list(Bbar_s, poly, poly2, *args)

                        model_Bbar_list = polyno2

                if model_name == "model1":
                    if N == 2:
                        model = model1_1a
                    if N == 4:
                        model = model1_2a
                        model_Bbar_list = model1_2a_Bbar_list
                
                elif model_name == "model2":
                    if N == 2:
                        model = model2_1a
                    if N == 4:
                        model = model2_2a
                        model_Bbar_list = model2_2a_Bbar_list

                else:
                    model = polyno

                g_s_in = [0.1, 0.2, 0.3, 0.5, 0.6]
                L_s_in = [8, 16, 32, 48, 64, 96, 128]
                N_s_in = [N]

                samples, g_s, L_s, Bbar_s, m_s = load_h5_data(h5_data_file, N_s_in, g_s_in, L_s_in, Bbar_s_in, GL_min, GL_max)

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

                gL_space = numpy.linspace(min(g_s * L_s), max(g_s * L_s), 1000)

                for g in set(g_s):
                    for i, Bbar in enumerate(set(Bbar_s)):
                        sub_ind = numpy.argwhere(numpy.logical_and(g_s == g, Bbar_s == Bbar))

                        if N == 2:
                            if i == 0:
                                plt.plot(gL_space, model(N, g, gL_space / g, Bbar, *central_values) / g, label=f'g = {g}', color=colors(g))

                            if i == 1:
                                plt.plot(gL_space, model(N, g, gL_space / g, Bbar, *central_values) / g, color=colors(g))

                        if N == 4:
                            if i == 0:
                                plt.plot(gL_space, model_Bbar_list(list(set(Bbar_s)), N, g, gL_space / g, Bbar, *central_values) / g, label=f'g = {g}', color=colors(g))

                            if i == 1:
                                plt.plot(gL_space, model_Bbar_list(list(set(Bbar_s)), N, g, gL_space / g, Bbar, *central_values) / g, color=colors(g))

                        plt.scatter(g * L_s[sub_ind], m_s[sub_ind] / g, color=colors(g))

                print("\\end{align" + "}")
                plt.xlabel("x")
                plt.ylabel("m[B = Bbar] / g")
                plt.title(f"polynomial range = [{poly}, {poly2}]")
                plt.legend()
                plt.savefig(f"graphs/{model_name}_N{N}_fit_plot_{poly}_{poly2}.png", dpi=500)
                plt.close('all')

                print("\\begin{figure}[H]")
                print("\\centering")
                print("\\includegraphics[width=100mm]{" + f"{model_name}" + f"_fit_plot_{poly}_{poly2}.png" + "}")
                print("\\end{figure}")


N = 2
nus = []
stds = []
labels = []
for model_name in ["poly_range"]:
    for poly in range(1, 5):
    # for poly in range(3, 4):
        for poly2 in range(poly, 5):
        # for poly2 in range(4, 5):
            results, Bbar = get_systematic_errors(N, model_name, poly=poly, poly2=poly2)
            params = results["params_central"]
            nu_stds = results["params_std"]
            names = results["param_names"]
            gL_min = results["GL_min"]
            labels.append(f"poly_range = [{poly}, {poly2}], gL_min={gL_min}")

            index = numpy.argwhere(numpy.array(names, dtype=str) == "\\nu")[0, 0]
            nus.append(params[index])
            stds.append(nu_stds[index])

# , "poly_range_no_scaling", "poly_range_no_log", "poly_range_no_scaling_no_log"
fig, ax = plt.subplots()
# nus, stds, labels = pickle.load(open("nu_data.pcl", "rb"))

results, Bbar = get_systematic_errors(N)
nu = results["params_central"][-1]
std = results["params_std"][-1]
gL_min = results["GL_min"]
ax.plot([0, len(nus)], [nu, nu], color='k')
ax.fill_between([0, len(nus)], [nu - std, nu - std], [nu + std, nu + std], color='k', label=f"log(g) fit, gL_min={gL_min}", alpha=0.1)

pickle.dump((nus, stds, labels), open("nu_data.pcl", "wb"))
for i in range(len(nus)):
    ax.errorbar(numpy.arange(len(nus))[i], nus[i], stds[i], label=labels[i], marker='_')

plt.legend(loc=(0.1, 0.6))
plt.ylabel("nu (systematic error bars)")
plt.title(f"N = {N}")
ax.get_xaxis().set_ticks([])
# ax.get_xaxis().set_labels([])
plt.show()

# for param in params:
#   plot(param)
