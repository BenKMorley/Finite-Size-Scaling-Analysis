import pickle
from tqdm import tqdm

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from model_definitions import *
from publication_results import *


GL_mins = numpy.array([4, 4.8, 6.4, 8, 9.6, 12.8, 14.4, 16, 19.2, 24, 25.6, 28.8, 32])
GL_min_graph = 4

color_blind_palette = [(0, 0, 0), (230, 159, 0), (86, 180, 233), (0, 158, 115),
                       (240, 228, 66), (0, 114, 178), (213, 94, 0), (204, 121, 167)]

color_blind_palette = [(a / 255, b / 255, c / 255) for a, b, c in color_blind_palette]

# Do N = 2 First
# min_value = numpy.min(Bayes_factors2)
# max_value = numpy.max(Bayes_factors2)

# fig, ax = plt.subplots()

# ax.scatter(GL_mins, Bayes_factors2, label=r"$\log_{10}(K)$", marker='o', color='none', edgecolor='k')

# # Show the Jeffrey's scale
# plt.fill_between([min(GL_mins) / 2, max(GL_mins) * 2], [-1, -1], [1, 1], color=color_blind_palette[7], alpha=0.5, label=r'$0 < |\log_{10}(K)| < 1$')
# plt.fill_between([min(GL_mins) / 2, max(GL_mins) * 2], [1, 1], [2, 2], color=color_blind_palette[4], alpha=0.2, label=r'$0 < |\log_{10}(K)| < 1$')
# plt.fill_between([min(GL_mins) / 2, max(GL_mins) * 2], [2, 2], [max_value, max_value], color=color_blind_palette[2], alpha=0.2, label=r'$|\log_{10}(K)| > 2$')
# plt.fill_between([min(GL_mins) / 2, max(GL_mins) * 2], [-2, -2], [-1, -1], color=color_blind_palette[4], alpha=0.2)
# plt.fill_between([min(GL_mins) / 2, max(GL_mins) * 2], [-10, -10], [-2, -2], color=color_blind_palette[2], alpha=0.2)

# ax = plt.gca()
# ax.legend(loc=(0.02, 0.3), framealpha=1)
# ax.set_ylabel(r"$\log_{10}(K)$")
# ax.set_ylim(min_value + 1, max_value + 1)
# ax.set_xlim(min(GL_mins) - 1, max(GL_mins) + 1)
# ax.tick_params(direction='in')
# ax.set_xlabel(r"$gL_{min}$")
# ax = plt.gca()
# fig = plt.gcf()
# ax.set_xlim(GL_min_graph, 34)
# ax2 = ax.twinx()
# ax2.set_ylabel(r'$p-value$')

# ax2.scatter(GL_mins, results2[0], marker='s', label=r"$\Lambda_{IR} \propto g$", color='none', edgecolor=color_blind_palette[1])
# ax2.scatter(GL_mins, results2[1], marker='^', label=r"$\Lambda_{IR} \propto \frac{1}{L}$", color='none', edgecolor=color_blind_palette[5])

# # Show our acceptance threshold (alpha)
# ax2.plot(numpy.array(GL_mins) * 2 - 10, [0.05, ] * len(GL_mins), color='grey')
# ax2.plot(numpy.array(GL_mins) * 2 - 10, [0.95, ] * len(GL_mins), color='grey')

# fig.tight_layout()
# ax2.tick_params(direction='in')
# ax2.legend(loc=(0.02, 0.15), framealpha=1)

# # Align 0 for both axes
# left1, right1 = ax.get_ylim()
# left2, right2 = ax2.get_ylim()

# ax.set_ylim(0.5, max(Bayes_factors2) * 1.1)
# ax.set_yscale('symlog', linthreshy=2.5)
# ax2.annotate(r"$p = 0.05$", xy=(25, 0.07), color='grey')
# ax2.annotate(r"$p = 0.95$", xy=(25, 0.91), color='grey')

# plt.show()

N = int(sys.argv[1])
poly = int(sys.argv[2])
poly2 = int(sys.argv[3])

Bayes_factors = get_Bayes_factors(N, model_name="poly_range", points=400, poly=poly, poly2=poly2, Bbar_fixed=[0.52, 0.57])

## Poly(1, 2) for N = 2 with gL_min between 1.6 and 32, and uniform [-15, 15] priors
# Bayes_factors = numpy.array([-305.62662622, -197.46664369, -96.20402337, -67.31837833,
#                             -52.79197141, -23.1886479, -17.50378358, -6.7330111,
#                             5.39603474, 6.03798855, 5.97980565, 4.85348963,
#                             7.73097725, 8.21320593, 8.1712307, 7.54649264])


min_value = numpy.min(Bayes_factors)
max_value = numpy.max(Bayes_factors)

fig, ax = plt.subplots()

ax.scatter(GL_mins, Bayes_factors, label=r"$\cal{E}$", marker='o', color='none', edgecolor='k')

# Show the Jeffrey's scale
plt.fill_between([min(GL_mins) / 2, max(GL_mins) * 2], [-1, -1], [1, 1], color=color_blind_palette[7], alpha=0.5, label=r'Insignificant')
plt.fill_between([min(GL_mins) / 2, max(GL_mins) * 2], [1, 1], [2, 2], color=color_blind_palette[4], alpha=0.2, label=r'Strong')
plt.fill_between([min(GL_mins) / 2, max(GL_mins) * 2], [2, 2], [max_value * 4, max_value * 4], color=color_blind_palette[2], alpha=0.2, label=r'Decisive')
plt.fill_between([min(GL_mins) / 2, max(GL_mins) * 2], [-2, -2], [-1, -1], color=color_blind_palette[4], alpha=0.2)
plt.fill_between([min(GL_mins) / 2, max(GL_mins) * 2], [-1000, -1000], [-2, -2], color=color_blind_palette[2], alpha=0.2)

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
# ax2 = ax.twinx()
# ax2.set_ylabel(r'$p-value$')

# ax2.scatter(GL_mins, results["pvalues1"], marker='s', label=r"$\Lambda_{IR} \propto g$", color='none', edgecolor=color_blind_palette[1])
# ax2.scatter(GL_mins, results["pvalues2"], marker='^', label=r"$\Lambda_{IR} \propto \frac{1}{L}$", color='none', edgecolor=color_blind_palette[5])

# Show our acceptance threshold (alpha)
# ax2.plot(numpy.array(GL_mins) * 2 - 10, [0.05, ] * len(GL_mins), color='grey')
# ax2.plot(numpy.array(GL_mins) * 2 - 10, [0.95, ] * len(GL_mins), color='grey')

fig.tight_layout()
# ax2.tick_params(direction='in')
# ax2.legend(loc=(0.02, 0.15), framealpha=1)

# Align 0 for both axes
left1, right1 = ax.get_ylim()
ax.set_xlabel(r"$gL_{min}$")

ax.set_yscale('symlog', linthreshy=2.5)
# ax2.annotate(r"$p = 0.05$", xy=(25, 0.07), color='grey')
# ax2.annotate(r"$p = 0.95$", xy=(25, 0.91), color='grey')
plt.title(f"poly {poly} {poly2}")

plt.show()
