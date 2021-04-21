from publication_results import *
import matplotlib.pyplot as plt


N = 2
pvalues_array = []
GL_mins = numpy.array([3.2, 4, 4.8, 6.4, 8, 9.6, 12.8, 14.4,
                       16, 19.2, 24, 25.6, 28.8, 32])

pvalues = get_pvalues_central_fit(N)
pvalues_array.append(pvalues["pvalues1"])
pvalues_array.append(pvalues["pvalues2"])
max_pvalues = []

for poly in range(1, 5):
    pvalues = get_pvalues_central_fit(N, model_name="poly_range_no_log", poly=poly, poly2=poly)["pvalues2"]
    pvalues_array.append(pvalues)
    max_pvalues.append(max(pvalues))

pvalues = get_pvalues_central_fit(N, model_name="poly_range_no_log", poly=2, poly2=1)["pvalues2"]
pvalues_array.append(pvalues)

pvalues_array = numpy.array(pvalues_array)

fig, ax = plt.subplots(figsize=(20, 10))

ax.plot(GL_mins, pvalues_array[0], label=r"$\log(g)$")
ax.plot(GL_mins, pvalues_array[1], label=r"$\log(L)$")

for i in range(1, 5):
    ax.plot(GL_mins, pvalues_array[i + 1], label=rf"$x^{i}$ term")
    ax.set_yscale('log')

ax.plot(GL_mins, pvalues_array[-1], label=rf"no $\log(L)$ or monomial")
ax.plot([min(GL_mins), max(GL_mins)], [1, 1], ls='--', color='k', label=r'$p=1$')

plt.legend()
plt.xlabel(r"$gL_{min}$")
plt.ylabel("pvalue")
plt.title(rf"$N = {N}$")
plt.show()

print(max(max_pvalues))
