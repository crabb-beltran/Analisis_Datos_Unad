import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('./data.csv')

# Dividir los datos en características (X) y etiquetas (y)
X = df[['metro']]  # Característica: área en metros cuadrados
y = df['precio']   # Etiqueta: precio

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear un modelo de regresión lineal
modelo = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Calcular el coeficiente de determinación (R²) para evaluar el rendimiento
r2 = modelo.score(X_test, y_test)

# Imprimir el coeficiente de determinación (R²)
print(f"Coeficiente de Determinación (R²): {r2}")

# Visualizar los resultados
plt.scatter(X_test, y_test, color='blue', label='Datos reales')
plt.plot(X_test, y_pred, color='red', linewidth=3, label='Regresión Lineal')
plt.xlabel('Área en metros cuadrados')
plt.ylabel('Precio')
plt.legend()
plt.show()