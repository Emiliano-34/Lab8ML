from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np

data = fetch_openml(name="glass", version=1, as_frame=True)
X, y = data.data, data.target

y = y.astype('category').cat.codes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Desempeño antes de SMOTE (Hold-Out):")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(knn, X, y, cv=skf, scoring='accuracy')
print(f"Accuracy promedio (10-Fold CV, antes de SMOTE): {np.mean(cv_scores):.2f}")

print("\nAntes de SMOTE:", Counter(y_train))
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print("Después de SMOTE:", Counter(y_train_sm))

knn.fit(X_train_sm, y_train_sm)
y_pred_sm = knn.predict(X_test)

print("\nDesempeño después de SMOTE (Hold-Out):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_sm):.2f}")
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred_sm))

cv_scores_sm = cross_val_score(knn, X, y, cv=skf, scoring='accuracy')
print(f"Accuracy promedio (10-Fold CV, después de SMOTE): {np.mean(cv_scores_sm):.2f}")
