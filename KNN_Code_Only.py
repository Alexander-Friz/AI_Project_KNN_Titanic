df_titanic_train = pd.read_csv('titanic_training.csv')
df_titanic_test = pd.read_csv('titanic_test.csv')


print(df_titanic_train.head())

print(df_titanic_train.describe())

print(df_titanic_train.info())
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Drop unnecessary columns and create X and y
X_train = df_titanic_train.drop(['Survived', 'Cabin', 'Ticket', 'Name'], axis=1)
y_train = df_titanic_train['Survived']

# Map categorical variables
X_train['Sex'] = X_train['Sex'].map({'male': 0, 'female': 1})

# Map Embarked and then drop rows with any remaining missing values
X_train['Embarked'] = X_train['Embarked'].map({'Q': 0, 'S': 1, 'C': 2})
#X_train.dropna(inplace=True)  # Drop rows with any missing values

#Wir extrahieren Titel aus den Namen, z.B. "Mr", "Mrs", "Miss"
#X_train['Title'] = X_train['Name'].str.extract(' ([A-Za-z]+).', expand=False)
#X_train['Title'] = X_train['Title'].map({"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Dr": 4, "Rev": 5, "Col": 6, "Major": 7, "Mlle": 8, "Countess": 9, "Ms": 10, "Lady": 11, "Jonkheer": 12, "Don": 13, "Dona": 14, "Mme": 15, "Capt": 16, "Sir": 17})

# Drop the original Name column as it's no longer needed
#X_train.drop('Name', axis=1, inplace=True)
X_train.dropna(inplace=True)  # Drop rows with any missing values

# Ensure that the corresponding rows are removed from y_train
y_train = y_train[X_train.index]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the k-nearest neighbors model
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred = knn.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
#Hyperparameter Tuning
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Define the kNN model
knn = KNeighborsClassifier()

# Define the hyperparameter grid to search
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],   # Number of neighbors to use
    'weights': ['uniform', 'distance'], # Weight function used in prediction
    'metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metric
}
# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters and best score from the grid search
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Cross-Validation Accuracy:", best_score)
# Use the best estimator from the grid search
best_knn = grid_search.best_estimator_

# Make predictions on the validation set
y_pred = best_knn.predict(X_val)

# Calculate and print the accuracy on the validation set
accuracy = accuracy_score(y_val, y_pred)
print("Validation Set Accuracy:", accuracy)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Validierungssatz erstellen
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# k-nearest Neighbour Modell trainieren
knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train, y_train)

# Vorhersagen und Genauigkeit
y_pred = knn.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
best_n = 1
best_accuracy = 0

for n in range(1, 25):  
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n = n

print(f"Beste Genauigkeit: {best_accuracy:.4f} mit n_neighbors = {best_n}")

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Define the range of neighbors to test
neighbors_range = range(1, 27)
train_scores = []
valid_scores = []

# Train and validate for each number of neighbors
for n_neighbors in neighbors_range:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_train_pred = knn.predict(X_train)
    y_val_pred = knn.predict(X_val)
    train_scores.append(accuracy_score(y_train, y_train_pred))
    valid_scores.append(accuracy_score(y_val, y_val_pred))

# Plot accuracy vs. number of neighbors
plt.figure(figsize=(10, 6))
plt.plot(neighbors_range, train_scores, 'o-', color='r', label='Training Accuracy')
plt.plot(neighbors_range, valid_scores, 'o-', color='g', label='Validation Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Neighbors for KNN')
plt.legend()
plt.grid(True)
plt.show()
