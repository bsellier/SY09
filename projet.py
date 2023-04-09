import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
from sklearn.decomposition import PCA
import numpy as np

def add_labels(x, y, labels, ax=None):
    """Ajoute les étiquettes `labels` aux endroits définis par `x` et `y`."""

    if ax is None:
        ax = plot.gca()
    for x, y, label in zip(x, y, labels):
        ax.annotate(
            label, [x, y], xytext=(10, -5), textcoords="offset points",
        )

    return ax

########## Import des données

starbucks = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-12-21/starbucks.csv')

print(starbucks.info)

########## Noms des colonnes

'''
['product_name', 'size', 'milk', 'whip', 'serv_size_m_l', 'calories',
 'total_fat_g', 'saturated_fat_g', 'trans_fat_g', 'cholesterol_mg',
 'sodium_mg', 'total_carbs_g', 'fiber_g', 'sugar_g', 'caffeine_mg']
'''

########## Typage des données

starbucks["size"] = pd.Categorical(starbucks["size"], categories = starbucks["size"].unique())

starbucks["milk"] = starbucks['milk'].replace(0,"none")
starbucks["milk"] = starbucks['milk'].replace(1,"nonfat")
starbucks["milk"] = starbucks['milk'].replace(2,"2%")
starbucks["milk"] = starbucks['milk'].replace(3,"soy")
starbucks["milk"] = starbucks['milk'].replace(4,"coconut")
starbucks["milk"] = starbucks['milk'].replace(5,"whole")
starbucks["milk"] = pd.Categorical(starbucks["milk"], categories = starbucks["milk"].unique())

starbucks["whip"] = starbucks["whip"].astype("bool")

########## Exploration variables qualitatives

sns.countplot(x="milk",data=starbucks).set(title='Répartition du type de lait')
plot.show()

sns.countplot(x="whip",data=starbucks).set(title='Répartition de boissons fouettées')
plot.show()

sns.countplot(x="size",data=starbucks).set(title='Répartition de la taille de boisson')
plot.show()

########## Test sur les tailles spécifiques

print(starbucks[starbucks["size"] == "triple"].info)
print(starbucks[starbucks["size"] == "triple"]["product_name"].unique())

print(starbucks[starbucks["size"] == "1 shot"].info)
print(starbucks[starbucks["size"] == "1 shot"]["product_name"].unique())

########## Exploration "product_name"

starbucks["product_name"].count() # 1147 produits (boissons)
starbucks["product_name"].nunique() # 93 noms de boissons uniques

########## Exploration quantitative/qualitative

##### Corrélations

corr = starbucks.corr()
sns.heatmap(corr, annot = True)
plot.show()

##### Taille

sns.boxplot(
    x="size",
    y="sugar_g",
    data=starbucks
    ).set(title='Répartition du Sucre (g) en fonction de la taille de la boisson')
plot.show()

sns.boxplot(
    x="size",
    y="serv_size_m_l",
    data=starbucks
    ).set(title='Répartition de la Taille servie (mL) en fonction de la taille de la boisson')
plot.show()

sns.boxplot(
    x="size",
    y="calories",
    data=starbucks
    ).set(title='Répartition des calories (KCal) en fonction de la taille de la boisson')
plot.show()

##### Boissons fouettées

sns.boxplot(
    x="whip",
    y="sugar_g",
    data=starbucks
    ).set(title='Répartition du Sucre (g) si la boisson a été fouettée')
plot.show()

sns.boxplot(
    x="whip",
    y="serv_size_m_l",
    data=starbucks
    ).set(title='Répartition de la Taille servie (mL) si la boisson a été fouettée')
plot.show()

sns.boxplot(
    x="whip",
    y="calories",
    data=starbucks
    ).set(title='Répartition des calories (KCal) si la boisson a été fouettée')
plot.show()

########## Analyse en Composantes Principales

##### Générer le modèle

n_comp = 5

cls = PCA(n_components = n_comp)
pcs = cls.fit_transform(starbucks.drop(columns=["product_name", "milk", "size", "whip"]))

##### Étudier l'inertie des axes

plot.bar(["Axe 1", "Axe 2", "Axe 3", "Axe 4", "Axe 5"],cls.explained_variance_ratio_)
plot.show()

##### Placement des individus sur 2 axes

# On apprend dans le DF les composantes principales
for i in range(0, n_comp):
    starbucks['PC' + str(i + 1)] = pcs[:, i]
print(starbucks.head())

sns.scatterplot(x="PC1", y="PC2", data=starbucks)
#add_labels(pcs[:, 0], pcs[:, 1], starbucks["product_name"])
plot.show()

sns.scatterplot(x="PC1", y="PC2", hue="size", data=starbucks)
#add_labels(pcs[:, 0], pcs[:, 1], starbucks["product_name"])
plot.show()

sns.scatterplot(x="PC1", y="PC2", hue="whip", data=starbucks)
#add_labels(pcs[:, 0], pcs[:, 1], starbucks["product_name"])
plot.show()

sns.scatterplot(x="PC1", y="PC2", hue="milk", data=starbucks)
#add_labels(pcs[:, 0], pcs[:, 1], starbucks["product_name"])
plot.show()

##### Cercle des corrélations

(fig, ax) = plot.subplots(figsize=(8, 8))
for i in range(0, cls.components_.shape[1]):
    ax.arrow(0,
             0,  # Start the arrow at the origin
             cls.components_[0, i],  #0 for PC1
             cls.components_[1, i],  #1 for PC2
             head_width=0.1,
             head_length=0.1)

    plot.text(cls.components_[0, i] + 0.05,
             cls.components_[1, i] + 0.05,
             starbucks.drop(columns=["product_name", "milk", "size", "whip"]).columns.values[i])

an = np.linspace(0, 2 * np.pi, 100)
plot.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
plot.axis('equal')
ax.set_title('Variable factor map')
plot.show()



