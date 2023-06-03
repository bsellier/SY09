import pandas as pd


def confusion_matrix(z_test, z_pred):
    z_comp = pd.DataFrame({"Réalité": z_test, "Prédiction": z_pred})

    # matrice de comparaison
    mat_comp = (
        z_comp.groupby(["Réalité", "Prédiction"]).size().unstack().fillna(0).astype(int)
    )

    # on transforme en pourcentage
    cat_count = z_test.value_counts().to_frame(name="Count").transpose()

    mat_comp_per = mat_comp.apply(lambda row: (row / cat_count.iloc[0]) * 100)
    mat_comp_per.index.name = "Réalité"
    mat_comp_per.columns.name = "Prédiction"

    return mat_comp_per
