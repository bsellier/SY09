from numpy.random import default_rng
rng = default_rng()

# Les déplacements |$c$| pour la fonction `gen`
cs = rng.uniform(low=0, high=2, size=500)

# Les erreurs de Bayes entre 0 et 0.5 pour la fonction `gen`. On
# utilise une loi bêta pour avoir une distribution parabolique entre 0
# et 0.5 (plus nombreux au centre).
deltas = rng.beta(2, 2, size=500) / 2

# Un tableau individu-variable des erreurs de Bayes |$\delta^\star$|,
# du paramètre |$c$| et de l'estimation de l'erreur du 1-NN.
df = pd.DataFrame(gen(deltas, cs), columns=["delta", "c", "err"])

x = np.linspace(0, 0.5, 100)
ax = sns.scatterplot(data=df, x="delta", y="err", hue="c", label="c")
sns.lineplot(x=x, y=x, label="Bayes", ax=ax)
sns.lineplot(x=x, y=2*x*(1-x), label="1-NN au pire", ax=ax)
plt.show()
