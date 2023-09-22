import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.cm as cm

# Cargar los datos
df = pd.read_csv('./framingham.csv')

# Remover campos con datos faltantes
df = df.dropna()

# Establecer una semilla para reproducibilidad
np.random.seed(1234)

# Tomar una muestra aleatoria de 500 observaciones
smp = df.sample(n=500)

# Gráfico de dispersión bivariado y curva de regresión logística
plt.figure(figsize=(12, 6))

# Preparar los datos para la regresión logística
X = smp[['age']]  # Variable independiente
y = smp['TenYearCHD']  # Variable dependiente

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

report = classification_report(y_test, y_pred, zero_division=1)
print('Informe de Clasificación:')
print(report)

# Gráfica de la curva de regresión logística
x = np.linspace(X['age'].min(), X['age'].max(), 100).reshape(-1, 1)
x_df = pd.DataFrame({'age': x[:, 0]})
x_scaled = scaler.transform(x_df)
y_prob = modelo.predict_proba(x_scaled)[:, 1]

# Colormap de estilo arcoíris para colorear los puntos
colors = cm.rainbow(np.linspace(0, 1, len(smp)))

# Gráfico de puntos con regresión logística
plt.scatter(smp['age'], smp['TenYearCHD'], color=colors, label='Puntos de datos')
plt.plot(x, y_prob, color='black', label='Regresión Logística')
plt.xlabel('Age')
plt.ylabel('TenYearCHD')
plt.title('Binned TenYearCHD vs Age with Logistic Regression')
plt.legend()
plt.show()