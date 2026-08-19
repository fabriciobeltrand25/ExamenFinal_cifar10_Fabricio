# Clasificador CIFAR-10 - Examen Nube UTH · Streamlit

Este proyecto es una aplicación web interactiva desarrollada con **Streamlit** y **TensorFlow / Keras**, capaz de clasificar imágenes en tiempo real utilizando un modelo de Red Neuronal Convolucional (CNN) entrenado con una muestra ligera del dataset **CIFAR-10**. 

Además, incorpora la **API de Groq (Llama 3.1)** para generar reportes e interpretaciones técnicas sobre la predicción realizada por el modelo de visión por computadora.

---

## 🚀 Características Principales

* **Entrenamiento Ligero y Rápido:** CNN optimizada para ejecutarse en la nube sin requerir almacenamiento pesado.
* **Procesamiento de 6 Clases Reales:** Identificación de Avión, Carro, Humano, Perro, Vaca y Barco.
* **Doble Fuente de Entrada:** Permite subir archivos locales (`.jpg`, `.jpeg`, `.png`) o capturar imágenes directamente mediante la cámara del dispositivo.
* **Integración con LLM (Groq API):** Genera informes contextuales instantáneos basados en el resultado obtenido.
* **Métricas Integradas:** Código listo para validar matriz de confusión, reporte de clasificación y visualización aleatoria de inferencias en Google Colab.

---

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3.10+
* **Framework Web:** Streamlit
* **Deep Learning:** TensorFlow / Keras
* **Procesamiento de Datos e Imágenes:** NumPy, Pandas, Pillow (PIL)
* **Visualización de Métricas:** Matplotlib, Seaborn, Scikit-Learn
* **LLM / Asistente IA:** OpenAI Client / Groq API (`llama-3.1-8b-instant`)

---

## 📁 Estructura del Repositorio

```text
├── app.py                   # Código principal de la interfaz en Streamlit
├── modelo_examen.keras      # Modelo CNN entrenado guardado
├── requirements.txt         # Dependencias del proyecto para Streamlit Cloud
└── README.md                # Documentación del repositorio
