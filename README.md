# 🧠 Assignment 2: Principal Component Analysis & Neural Networks

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-ffffff?logo=matplotlib&logoColor=black)

Este repositorio contiene la implementación y el análisis de dos enfoques fundamentales en Machine Learning y Visión por Computadora: reducción de dimensionalidad para reconocimiento facial mediante **PCA (Eigenfaces)** y clasificación de dígitos manuscritos utilizando arquitecturas de **Redes Neuronales (MLP y CNN)**.

---

## 📌 Información Académica

| Atributo | Detalle |
| :--- | :--- |
| **Curso** | Machine Learning II (2025-G1-910040-4-PEUCD) |
| **Autor** | Sebastian Rios, Wilder Teddy |
| **Enfoque** | Reducción de Dimensionalidad, Visión por Computadora, Deep Learning |

---

## 🗂️ Estructura del Proyecto

El repositorio está organizado en dos módulos principales correspondientes a cada parte de la asignación:

```text
assignment2/
├── data/                      # Datasets de entrenamiento y prueba (.txt y crudos)
├── report/
│   └── figuras/               # Gráficos generados por los notebooks
├── src/
│   ├── ParteI_Eigenfaces.ipynb        # Implementación de PCA y Regresión Logística
│   └── ParteII_NeuralNetworks.ipynb   # Modelos MLP y CNN en PyTorch
├── utils/
│   ├── pca_utils.py           # Funciones auxiliares matemáticas (SVD, centrado)
│   └── nn_utils.py            # Loops de entrenamiento, evaluación y graficado
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Documentación principal
```

---

## 🧑‍💻 Parte I: Eigenfaces para Reconocimiento Facial

**Archivo principal:** `src/ParteI_Eigenfaces.ipynb`

Implementación del clásico método de **Eigenfaces** basado en el Análisis de Componentes Principales (PCA) para comprimir y clasificar rostros humanos.

* **Dataset:** 540 imágenes de entrenamiento y 100 imágenes de prueba (resolución de 50x50 píxeles, en escala de grises).
* **Pipeline Matemático:**
  1. Centrado de los datos restando el **rostro promedio**.
  2. Extracción de los *eigenfaces* mediante la Descomposición en Valores Singulares (**SVD**) optimizada.
  3. Proyección de las imágenes originales sobre un subespacio de dimensión **r** (componentes principales).
  4. Entrenamiento de un clasificador lineal (Regresión Logística sin intercepto) en el espacio latente.
  5. Reconstrucción de imágenes y cálculo del **Error de Frobenius**.

### Resultados del Rendimiento vs. Componentes (r)

| Componentes (r) | Accuracy en Test | Observación del Error de Reconstrucción |
| :---: | :---: | :--- |
| **10** | ~ 59.0% | Error alto; rostros reconstruidos borrosos pero con silueta general. |
| **100** | ~ 95.0% | El error decrece drásticamente y se estabiliza. Detalles finos recuperados. |
| **130** | **~ 97.0%** | **Máximo rendimiento predictivo.** Equilibrio óptimo entre compresión y señal. |

> **Visualizaciones generadas:** `mean_face.png`, `eigenfaces.png`, `accuracy_vs_r.png`, `error_reconstruccion.png`.

---

## 🤖 Parte II: Redes Neuronales Profundas (PyTorch)

**Archivo principal:** `src/ParteII_NeuralNetworks.ipynb`

Diseño, entrenamiento y evaluación comparativa de tres arquitecturas de redes neuronales sobre el dataset **MNIST** (dígitos manuscritos de 28x28 píxeles).

### Arquitecturas Implementadas

1. **MLP (Multilayer Perceptron):** Red densa tradicional con topología de capas `784 → 256 → 128 → 10`.
2. **CNN Básica:** Red Neuronal Convolucional con 2 capas convolucionales seguidas de operaciones de `MaxPooling` y capas lineales.
3. **CNN Mejorada:** Extensión de la CNN básica incorporando técnicas modernas de regularización: **Batch Normalization** y **Dropout**, diseñadas para acelerar la convergencia y mitigar el sobreajuste.

### Comparativa de Rendimiento

| Modelo | Accuracy en Test | Ventajas / Observaciones |
| :--- | :---: | :--- |
| **MLP** | 98.05% | Rápido de entrenar, pero ignora la topología espacial 2D de la imagen. |
| **CNN Básica** | 98.69% | Extrae características locales (bordes, curvas) mejorando la precisión. |
| **CNN Mejorada** | **99.24%** | Mayor estabilidad durante el entrenamiento y la **mejor generalización**. |

> **Curvas de Entrenamiento:** `curva_mlp.png`, `curva_cnn_basica.png`, `curva_cnn_mejorada.png`.

---

## 🏆 Conclusiones Globales

* **Eficacia de PCA:** El algoritmo PCA demostró ser una técnica de compresión formidable, reduciendo el espacio dimensional de 2500 píxeles a tan solo ~130 variables latentes conservando la varianza necesaria para clasificar rostros con un **97% de precisión**.
* **Supremacía Convolucional:** Las CNNs demostraron una superioridad empírica sobre las redes densas (MLP) al aprovechar de forma nativa la estructura jerárquica espacial de las imágenes.
* **Importancia de la Regularización:** La incorporación de `BatchNorm` y `Dropout` en la CNN mejorada fue clave para cruzar la barrera del **99.2%**, demostrando que el control del sobreajuste es tan importante como la capacidad del modelo.

---

## 🚀 Configuración y Ejecución del Entorno

### Requisitos Previos
Asegúrese de tener instaladas las bibliotecas detalladas en `requirements.txt`:
`numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch`, `torchvision`, `tqdm`, `imageio`, `seaborn`.

### Pasos de Instalación

1. **Clonar el repositorio y abrir la terminal en la raíz del proyecto.**
2. **Crear un entorno virtual aislado:**
   ```bash
   python -m venv .venv
   ```
3. **Activar el entorno:**
   * En Windows (PowerShell):
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   * En Linux / macOS:
     ```bash
     source .venv/bin/activate
     ```
4. **Instalar dependencias:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. **Ejecutar los notebooks:** Abra `src/ParteI_Eigenfaces.ipynb` o `src/ParteII_NeuralNetworks.ipynb` en su entorno de Jupyter preferido.
