import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar el archivo CSV (asegúrate de que no tiene encabezados)
column_names = [
    "num_class", "alcohol", "malic_acid", "ash", "alcalinity_of_ash",
    "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols",
    "proanthocyanins", "color_intensity", "hue", "od280_od315_of_diluted_wines",
    "proline"
]
df = pd.read_csv('./wine.data', header=None, names=column_names)

# Dividir los datos en características (X) y etiquetas (y)
X = df.drop('num_class', axis=1)  # Características
y = df['num_class']  # Etiquetas

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de árbol de decisión
modelo_arbol = DecisionTreeClassifier()

# Entrenar el modelo con los datos de entrenamiento
modelo_arbol.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = modelo_arbol.predict(X_test)

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