# DOCUMENTACION PROYECTO 

## 1. Nombre :computer:

Análisis de datos.

## 2. Descripción 	:bookmark_tabs:

Este monorepo integra los diferentes proyectos que desarrollaran a lo largo del semestre en el curso de análisis de datos.

## 3. Lógica del Desarrollo  :speech_balloon:

- */Tarea 2: Regresión Lineal/*

*¿Qué es?:* La regresión lineal es una técnica estadística que se utiliza para modelar la relación entre una variable dependiente (la variable que se está tratando de predecir) y una o más variables independientes (predictores o variables de entrada). Se asume que esta relación es lineal, lo que significa que se puede representar mediante una ecuación de una línea recta.

*Beneficios:*
Simplicidad: La regresión lineal es fácil de entender e interpretar. La relación entre las variables se representa de manera clara y comprensible.
Interpretación de coeficientes: Permite la interpretación de los coeficientes del modelo, lo que significa que se pueden identificar las contribuciones relativas de cada predictor en la variable dependiente.
Predicción: Se puede utilizar para hacer predicciones numéricas en función de los valores de las variables predictoras.

- */Tarea 2: Regresión Lógistica/*
  
*¿Qué es?:* La regresión logística es una técnica utilizada para modelar y predecir variables binarias o categóricas (por ejemplo, sí/no, 0/1). A diferencia de la regresión lineal, la regresión logística se utiliza para problemas de clasificación en lugar de predicción numérica. Estima la probabilidad de que una observación pertenezca a una categoría específica.

*Beneficios:*
Clasificación binaria: Es efectiva para problemas de clasificación binaria, como la detección de spam o diagnóstico médico.
Probabilidades estimadas: Proporciona estimaciones de probabilidades de pertenencia a una clase, lo que permite tomar decisiones basadas en umbrales de probabilidad.
Flexibilidad: Se puede extender a problemas de clasificación multiclase utilizando técnicas como la regresión logística multinomial.

- */Tarea 2: Árbol de Desiciones/*

*¿Qué es?:* Un árbol de decisión es una técnica de aprendizaje automático que crea un modelo en forma de un árbol. Cada nodo del árbol representa una pregunta o una decisión basada en características particulares, y las ramas del árbol representan las posibles respuestas o resultados. Los árboles de decisión se utilizan tanto en problemas de clasificación como en problemas de regresión.

*Beneficios:*
Interpretación visual: Los árboles de decisión son fáciles de visualizar y entender, lo que facilita la interpretación de cómo se toman las decisiones.
No requiere supuestos lineales: A diferencia de la regresión lineal, los árboles de decisión pueden manejar relaciones no lineales entre variables.
Flexibilidad: Pueden utilizarse tanto para problemas de clasificación como de regresión y son adecuados para conjuntos de datos complejos.
En resumen, el análisis de regresión lineal se utiliza para modelar relaciones lineales entre variables, la regresión logística es efectiva para problemas de clasificación binaria o categórica, y los árboles de decisión son flexibles y se utilizan para problemas de clasificación y regresión, proporcionando una interpretación visual de las decisiones del modelo. La elección de la técnica depende del tipo de problema y los objetivos de análisis.

## 4. Roadmap - Ideas :roller_coaster:
* [x] Creación del monorepo.
* [x] Cargues de Dataset.
* [x] Desarrollo de scripts en python.

## 5. Autores 🧑‍💻
- Cristian Beltrán -- Student

## 6. Referencias :books:
> Monorepos. [github.com](https://github.com/Igvir/monorepo-guidelines)

## 9. Estado del Proyecto - Fases Devops :construction:
* [x] Fase de Planeación (Entendimiento del brief. Roadmap)
* [x] Fase de Construcción (Generación de Diseño y Código del desarrollo)
* [ ] Fase de Integración Continua (Testeo, calidad con sonar. Pruebas unitarias)
* [ ] Fase de Implementación o Despliegue continuo (Instalación en los ambientes qa, staging, production con gitlab ci/cd)
* [ ] Fase de Gestionar
* [ ] Fase de Feedback Continuo (Retroalimentación Cliente y Usuario)
