import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

df = pd.read_csv('./framingham.csv')

# Crear un imputador para llenar los valores faltantes con la media
imputer = SimpleImputer(strategy='mean')

# Dividir los datos en características (X) y etiquetas (y)
X = df.drop('TenYearCHD', axis=1)  # Características
y = df['TenYearCHD']  # Etiquetas

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Eliminar filas con valores faltantes en los conjuntos de entrenamiento y prueba
rows_with_nan_train = np.isnan(X_train).any(axis=1)
X_train = X_train[~rows_with_nan_train]
y_train = y_train[~rows_with_nan_train]

rows_with_nan_test = np.isnan(X_test).any(axis=1)
X_test = X_test[~rows_with_nan_test]
y_test = y_test[~rows_with_nan_test]

# Estandarizar las características para que tengan media 0 y desviación estándar 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear un modelo de regresión logística
modelo = LogisticRegression()

# Entrenar el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')

# Mostrar la matriz de confusión y el informe de clasificación
conf_matrix = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión:')
print(conf_matrix)

report = classification_report(y_test, y_pred)
print('Informe de Clasificación:')
print(report)