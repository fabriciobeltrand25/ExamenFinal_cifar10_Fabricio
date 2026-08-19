import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# ----------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ----------------------------------------------------
st.set_page_config(
    page_title="Clasificador CIFAR-10 - EXAMEN--UTH 2026 - Fabricio Beltrand",
    page_icon="🤖",
    layout="wide"
)

# ----------------------------------------------------
# CARGA DEL MODELO Y CONFIGURACIÓN
# ----------------------------------------------------
MODEL_PATH = "modelo_examen.keras"
CLASES = ["Avión", "Carro", "Humano", "Perro", "Vaca", "Barco"]
IMG_SIZE = (32, 32)

@st.cache_resource(show_spinner="Cargando modelo de IA...")
def cargar_modelo():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        st.error(f"No se encontró el archivo del modelo `{MODEL_PATH}` en el directorio principal.")
        return None

modelo = cargar_modelo()

# Metadata descriptiva para los elementos clasificados
INFO_OBJETOS = {
    "Avión": {"categoria": "Transporte Aéreo", "entorno": "Cielo / Aeropuertos"},
    "Carro": {"categoria": "Transporte Terrestre", "entorno": "Carreteras / Vías Urbanas"},
    "Humano": {"categoria": "Ser Vivo / Persona", "entorno": "Entorno Social / Urbano"},
    "Perro": {"categoria": "Ser Vivo / Mamífero Doméstico", "entorno": "Hogares / Áreas Verdes"},
    "Vaca": {"categoria": "Ser Vivo / Mamífero de Granja", "entorno": "Campos / Zonas Rurales"},
    "Barco": {"categoria": "Transporte Marítimo", "entorno": "Océanos / Ríos / Puertos"}
}

# ----------------------------------------------------
# INTERFAZ GRÁFICA DE USUARIO
# ----------------------------------------------------
st.title("🤖 Clasificador de Objetos e Imágenes - CIFAR-10- EXAMEN--UTH 2026 - Fabricio Beltrand")
st.caption("Examen de Computación en la Nube | Desarrollado por: **Fabricio Beltrand**")

col_izq, col_der = st.columns([1, 1])

with col_izq:
    st.subheader("1. Captura o Carga de Imagen")
    metodo = st.radio("Fuente de la imagen:", ["Subir archivo", "Usar cámara"], horizontal=True, key="fuente_img")
    
    imagen_pil = None
    if metodo == "Subir archivo":
        archivo = st.file_uploader("Seleccione una imagen...", type=["jpg", "jpeg", "png"], key="uploader_file")
        if archivo:
            imagen_pil = Image.open(archivo)
    else:
        foto = st.camera_input("Tome una fotografía", key="camera_input")
        if foto:
            imagen_pil = Image.open(foto)

    if imagen_pil:
        st.image(imagen_pil, caption="Imagen seleccionada", use_container_width=True)

with col_der:
    st.subheader("2. Diagnóstico y Análisis del Modelo")
    
    if imagen_pil and modelo:
        # Preprocesamiento de la imagen para el modelo (32x32 RGB)
        img_rgb = imagen_pil.convert("RGB")
        img_resized = img_rgb.resize(IMG_SIZE)
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # Predicción del modelo de red neuronal
        predicciones = modelo.predict(img_batch, verbose=0)[0]
        idx_clase = np.argmax(predicciones)
        confianza = float(predicciones[idx_clase]) * 100
        
        clase_detectada = CLASES[idx_clase]
        info = INFO_OBJETOS.get(clase_detectada, {"categoria": "General", "entorno": "N/A"})

        # Mostrar Resultados Principales
        st.markdown(f"### **Objeto Detectado:** {clase_detectada}")
        st.write(f"**Categoría:** {info['categoria']}")
        st.write(f"**Entorno Frecuente:** {info['entorno']}")
        st.metric(label="Porcentaje de Confianza", value=f"{confianza:.2f}%")

        st.divider()

        # ----------------------------------------------------
        # GRÁFICAS Y DISTRIBUCIÓN DE PROBABILIDADES
        # ----------------------------------------------------
        st.subheader("📊 Distribución de Probabilidades por Clase")
        
        # Crear DataFrame para la gráfica y tabla
        df_probs = pd.DataFrame({
            "Clase": CLASES,
            "Probabilidad (%)": predicciones * 100
        }).sort_values(by="Probabilidad (%)", ascending=False)

        # Gráfico de barras interactivo nativo de Streamlit
        st.bar_chart(
            df_probs.set_index("Clase"), 
            height=250,
            use_container_width=True
        )

        # Tabla detallada con los porcentajes
        st.dataframe(
            df_probs.style.format({"Probabilidad (%)": "{:.2f}%"}),
            hide_index=True,
            use_container_width=True
        )

    elif not imagen_pil:
        st.info("Por favor cargue o tome una fotografía para realizar la predicción.")

st.divider()
st.caption("Servicio Web de Computación en la Nube | UTH 2026 - Fabricio Beltrand")
