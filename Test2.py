import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from itertools import combinations

# Beispiel Datensatz laden (ersetze diesen Schritt mit deinem eigenen Titanic-Datensatz)
df_titanic_train = pd.read_csv('titanic_training.csv')

# Extrahieren des Titels aus dem Namen
df_titanic_train['Title'] = df_titanic_train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Mapping der Titel zu numerischen Werten
title_mapping = {
    "Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Dr": 4, "Rev": 5, "Col": 6,
    "Major": 7, "Mlle": 8, "Countess": 9, "Ms": 10, "Lady": 11, "Jonkheer": 12,
    "Don": 13, "Dona": 14, "Mme": 15, "Capt": 16, "Sir": 17
}
df_titanic_train['Title'] = df_titanic_train['Title'].map(title_mapping)

# Features und Zielvariable
X = df_titanic_train.drop(['Survived', 'Cabin', 'Ticket', 'Name'], axis=1)
y = df_titanic_train['Survived']

# Konvertierung von Kategorien in numerische Werte
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
X['Embarked'] = X['Embarked'].map({'Q': 0, 'S': 1, 'C': 2})

# Fehlende Werte entfernen
X.dropna(inplace=True)
y = y[X.index]

# Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ergebnisliste für verschiedene Kombinationen und n_neighbors
results = []

# Über sämtliche Feature-Kombinationen und n_neighbors iterieren
for i in range(1, len(X_train.columns) + 1):
    for combination in combinations(X_train.columns, i):
        X_subset = X_train[list(combination)]
        
        for n in range(1, 25):
            knn = KNeighborsClassifier(n_neighbors=n)
            # 5-fache Kreuzvalidierung auf dem aktuellen Subset der Daten
            scores = cross_val_score(knn, X_subset, y_train, cv=5)
            accuracy = scores.mean()
            # Ergebnis hinzufügen
            results.append((combination, n, accuracy))

            # Zwischenergebnisse direkt ausgeben
            print(f"Kombination: {combination}, n_neighbors: {n}, Genauigkeit: {accuracy:.4f}")

# Ergebnisse nach Genauigkeit sortieren
results.sort(key=lambda x: x[2], reverse=True)

# Beste 5 Kombinationen ausgeben
print("\nDie 5 besten Kombinationen:")
for i in range(5):
    print(f"Kombination: {results[i][0]}, n_neighbors: {results[i][1]}, Genauigkeit: {results[i][2]:.4f}")

# Schlechteste 5 Kombinationen ausgeben
print("\nDie 5 schlechtesten Kombinationen:")
for i in range(1, 6):
    print(f"Kombination: {results[-i][0]}, n_neighbors: {results[-i][1]}, Genauigkeit: {results[-i][2]:.4f}")
