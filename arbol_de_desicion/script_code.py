# Importar las bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree

# Nombres de las columnas en la misma secuencia que aparecen en el archivo CSV
nombres_columnas = [
    "num_class", "alcohol", "malic_acid", "ash", "alcalinity_of_ash",
    "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols",
    "proanthocyanins", "color_intensity", "hue", "od280_od315_of_diluted_wines",
    "proline"
]

# Cargar el archivo CSV sin encabezados y usando los nombres de columna proporcionados
df = pd.read_csv('./wine.data', header=None, names=nombres_columnas)
df

# Seleccionar las columnas relevantes (variables independientes) para el análisis
# Excluir la columna "num_class" ya que es la variable objetivo
features = [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
    'color_intensity', 'hue', 'od280_od315_of_diluted_wines', 'proline'
]

# Crear conjuntos de características (X) y la variable objetivo (y)
X = df[features]
y = df['num_class']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de árbol de decisión
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
accuracy = (y_pred == y_test).mean()
print(f'Precisión del modelo: {accuracy * 100:.2f}%')

# Crear y mostrar el árbol de decisión
plt.figure(figsize=(15, 10))
tree.plot_tree(model, feature_names=features, class_names=[str(i) for i in range(1, 4)], filled=True, rounded=True)
plt.title("Árbol de Decisión para Clasificación de Vinos")
plt.show()