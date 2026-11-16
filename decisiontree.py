import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

iris = load_iris()

X_iris, y_iris = iris.data, iris.target

print(X_iris[:5])
print(y_iris[:5])

dt_classifier = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
dt_classifier.fit(X_iris, y_iris)

y_pred_class = dt_classifier.predict(X_iris)

print(y_pred_class[:10])

print(f"DT Depth:\t{dt_classifier.get_depth()}")
print(f"DT Leaves:\t{dt_classifier.get_n_leaves()}")


# Creazione del grafico
fig, ax = plt.subplots(figsize=(16, 10))

plot_tree(
    dt_classifier,
    feature_names=[
        "Lunghezza sepalo (cm)",
        "Larghezza sepalo (cm)",
        "Lunghezza petalo (cm)",
        "Larghezza petalo (cm)"
    ],
    class_names=[
        "Iris setosa",
        "Iris versicolor",
        "Iris virginica"
    ],
    filled=True,
    fontsize=10,
    ax=ax
)

# Traduzione delle informazioni nei riquadri e sui rami
for testo in ax.texts:
    etichetta = testo.get_text()

    etichetta = etichetta.replace("gini =", "Impurità di Gini =")
    etichetta = etichetta.replace("samples =", "Numero osservazioni =")
    etichetta = etichetta.replace("value =", "Distribuzione delle specie =")
    etichetta = etichetta.replace("class =", "Specie prevista =")
    etichetta = etichetta.replace("True", "Vero")
    etichetta = etichetta.replace("False", "Falso")
    etichetta = etichetta.replace(" <= ", " ≤ ")

    testo.set_text(etichetta)

ax.set_title("Albero decisionale per la classificazione")

plt.tight_layout()
plt.savefig("esercizio.png", dpi=300)
plt.show()
