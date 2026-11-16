
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text


# ==========================================================
# 1. DATASET DRAGON BALL
# ==========================================================

personaggi = [
    "Goku",
    "Vegeta",
    "Gohan",
    "Trunks",
    "Goten",
    "Raditz",
    "Nappa",
    "Piccolo",
    "Krillin",
    "Yamcha",
    "Bulma",
    "Tenshinhan",
    "Chiaotzu",
    "Dende"
]

# Caratteristiche:
# 0 = No
# 1 = Si
#
# Colonna 1: Ha avuto la coda Saiyan?
# Colonna 2: Ha raggiunto il Super Saiyan?

X = np.array([
    [1, 1],  # Goku
    [1, 1],  # Vegeta
    [1, 1],  # Gohan
    [0, 1],  # Trunks
    [0, 1],  # Goten
    [1, 0],  # Raditz
    [1, 0],  # Nappa
    [0, 0],  # Piccolo
    [0, 0],  # Krillin
    [0, 0],  # Yamcha
    [0, 0],  # Bulma
    [0, 0],  # Tenshinhan
    [0, 0],  # Chiaotzu
    [0, 0]   # Dende
])

# Classificazione:
# 0 = Non-Saiyan
# 1 = Saiyan

y = np.array([
    1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0
])


print("\n========== DATASET ==========")
print("Forma di X:", X.shape)
print("Forma di y:", y.shape)
print("Numero personaggi:", len(personaggi))


# ==========================================================
# 2. CREAZIONE E ADDESTRAMENTO DEL MODELLO
# ==========================================================

modello = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

modello.fit(X, y)


# ==========================================================
# 3. PREVISIONI SUI PERSONAGGI DI ADDESTRAMENTO
# ==========================================================

previsioni = modello.predict(X)

print("\n========== PREVISIONI TRAINING ==========")

for nome, reale, previsto in zip(personaggi, y, previsioni):

    classe = "Saiyan" if previsto == 1 else "Non-Saiyan"

    risultato = "CORRETTO" if reale == previsto else "ERRORE"

    print(f"{nome:12} | Previsto: {classe:12} | {risultato}")


accuratezza = modello.score(X, y)

print("\n========== VALUTAZIONE TRAINING ==========")
print(f"Accuracy: {accuratezza * 100:.2f}%")
print(f"Tree Depth: {modello.get_depth()}")
print(f"Tree Leaves: {modello.get_n_leaves()}")


# ==========================================================
# 4. PERSONAGGI NUOVI: TEST
# ==========================================================

nuovi_personaggi = [
    "Freezer",
    "Cabba"
]

nuove_caratteristiche = np.array([
    [0, 0],  # Freezer
    [0, 1]   # Cabba
])

nuove_previsioni = modello.predict(nuove_caratteristiche)

# Classificazioni reali per il confronto
y_reale_nuovi = np.array([0, 1])

print("\n========== NUOVI PERSONAGGI ==========")

for nome, reale, previsto in zip(
    nuovi_personaggi,
    y_reale_nuovi,
    nuove_previsioni
):

    classe = "Saiyan" if previsto == 1 else "Non-Saiyan"

    risultato = "CORRETTO" if reale == previsto else "ERRORE"

    print(f"{nome:12} | Previsto: {classe:12} | {risultato}")


accuratezza_test = np.mean(nuove_previsioni == y_reale_nuovi)

print(f"\nTest Accuracy: {accuratezza_test * 100:.2f}%")
print("Attenzione: il test contiene solamente 2 personaggi.")


# ==========================================================
# 5. REGOLE IMPARATE DAL MODELLO
# ==========================================================

print("\n========== REGOLE IMPARATE ==========")

regole = export_text(
    modello,
    feature_names=[
        "Has had a tail?",
        "Has achieved Super Saiyan?"
    ]
)

print(regole)


# ==========================================================
# 6. IDENTIFICAZIONE DELLE FOGLIE
# ==========================================================

# Foglie raggiunte dai personaggi di addestramento
foglie = modello.apply(X)

gruppi = {}

for nome, foglia in zip(personaggi, foglie):
    gruppi.setdefault(foglia, []).append(nome)


# Foglie raggiunte dai personaggi nuovi
foglie_nuove = modello.apply(nuove_caratteristiche)

gruppi_test = {}

for nome, foglia, previsione in zip(
    nuovi_personaggi,
    foglie_nuove,
    nuove_previsioni
):

    classe = "Saiyan" if previsione == 1 else "Non-Saiyan"

    gruppi_test.setdefault(foglia, []).append(
        f"{nome} -> {classe}"
    )


print("\n========== PERSONAGGI PER FOGLIA ==========")

for foglia, nomi in gruppi.items():
    print(f"Foglia {foglia}: {', '.join(nomi)}")


# ==========================================================
# 7. GRAFICO DECISION TREE PERSONALIZZATO
# ==========================================================

fig, ax = plt.subplots(figsize=(24, 17))

nodi = plot_tree(
    modello,

    feature_names=[
        "Has had a tail?",
        "Has achieved Super Saiyan?"
    ],

    class_names=[
        "Non-Saiyan",
        "Saiyan"
    ],

    filled=True,
    rounded=True,
    node_ids=True,
    impurity=True,
    proportion=False,
    precision=2,
    fontsize=10,
    ax=ax
)

# ==========================================================
# 8. AGGIUNTA DEI NOMI NELLE FOGLIE
# ==========================================================

for nodo in nodi:

    testo = nodo.get_text()
    prima_riga = testo.splitlines()[0]

    # Consideriamo soltanto i riquadri dei nodi,
    # ignorando le scritte True e False sui collegamenti
    if not prima_riga.startswith("node #"):
        continue

    # Recuperiamo il numero identificativo del nodo
    id_nodo = int(prima_riga.replace("node #", ""))

    # Verifichiamo se è una foglia
    if modello.tree_.children_left[id_nodo] == -1:

        nomi_training = gruppi.get(id_nodo, [])
        nomi_test = gruppi_test.get(id_nodo, [])

        testo += "\n\nTRAINING CHARACTERS:\n"
        testo += "\n".join(nomi_training)

        if nomi_test:
            testo += "\n\nNEW TEST CHARACTERS:\n"
            testo += "\n".join(nomi_test)

        nodo.set_text(testo)

# ==========================================================
# 9. TITOLI E SPIEGAZIONE DEL GRAFICO
# ==========================================================

fig.suptitle(
    "DRAGON BALL - DECISION TREE CLASSIFICATION",
    fontsize=21,
    fontweight="bold",
    y=0.97
)

ax.set_title(
    "Classification of Saiyan and Non-Saiyan characters",
    fontsize=14,
    pad=20
)

# Nascondiamo gli assi esterni
ax.axis("off")


# ==========================================================
# 10. LEGENDA ESPLICATIVA
# ==========================================================

spiegazione = (
    "FEATURES: 0 = NO | 1 = YES\n"
    "TRUE = the condition in the node is satisfied (<= 0.5)\n"
    "FALSE = the condition in the node is not satisfied (> 0.5)\n"
    "GINI = impurity of the node (0.0 means a perfectly pure group)\n"
    "SAMPLES = number of characters reaching the node\n"
    "VALUE = number of characters in each class [Non-Saiyan, Saiyan]\n"
    "CLASS = classification predicted by the model\n"
    "TRAINING CHARACTERS = examples used to train the model\n"
    "NEW TEST CHARACTERS = unseen examples classified after training"
)

fig.text(
    0.5,
    0.025,
    spiegazione,
    ha="center",
    va="bottom",
    fontsize=10,
    bbox=dict(
        boxstyle="round,pad=0.8",
        facecolor="whitesmoke",
        edgecolor="gray"
    )
)


# Spazio per titolo e legenda
plt.tight_layout(rect=[0, 0.16, 1, 0.92])
plt.savefig("eserciziomiotree.png", dpi=300)
plt.show()
