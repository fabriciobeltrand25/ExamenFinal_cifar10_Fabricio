import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from openai import OpenAI

# Configuración principal de la página
st.set_page_config(
    page_title="Clasificador CIFAR-10 - Examen Nube UTH",
    page_icon="🤖",
    layout="wide"
)

# ----------------------------------------------------
# CARGA DE MODELO Y CONFIGURACIÓN
# ----------------------------------------------------
MODEL_PATH = "modelo_examen.keras"
CLASES = ["Avión", "Carro", "Humano", "Perro", "Vaca", "Barco"]
IMG_SIZE = (32, 32)

@st.cache_resource(show_spinner="Cargando modelo de IA...")
def cargar_modelo():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        st.error(f"No se encontró el modelo `{MODEL_PATH}` en el directorio principal.")
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
st.title("🤖 Clasificador de Objetos e Imágenes - CIFAR-10")
st.caption("Examen de Computación en la Nube | Desarrollado por: **Fabricio Beltrand**")

col_izq, col_der = st.columns([1, 1])

with col_izq:
    st.subheader("1. Captura o Carga de Imagen")
    metodo = st.radio("Fuente de la imagen:", ["Subir archivo", "Usar cámara"], horizontal=True)
    
    imagen_pil = None
    if metodo == "Subir archivo":
        archivo = st.file_uploader("Seleccione una imagen...", type=["jpg", "jpeg", "png"])
        if archivo:
            imagen_pil = Image.open(archivo)
    else:
        foto = st.camera_input("Tome una fotografía")
        if foto:
            imagen_pil = Image.open(foto)

    if imagen_pil:
        st.image(imagen_pil, caption="Imagen seleccionada", use_column_width=True)

with col_der:
    st.subheader("2. Diagnóstico y Análisis del Modelo")
    
    if imagen_pil and modelo:
        # Preprocesamiento de la imagen para el modelo (32x32 RGB)
        img_rgb = imagen_pil.convert("RGB")
        img_resized = img_rgb.resize(IMG_SIZE)
        img_array = np.array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # Predicción del modelo de red neuronal
        predicciones = modelo.predict(img_batch, verbose=0)[0]
        idx_clase = np.argmax(predicciones)
        confianza = float(predicciones[idx_clase]) * 100
        
        clase_detectada = CLASES[idx_clase]
        info = INFO_OBJETOS.get(clase_detectada, {"categoria": "General", "entorno": "N/A"})

        # Mostrar Resultados
        st.markdown(f"### **Objeto Detectado:** {clase_detectada}")
        st.write(f"**Categoría:** {info['categoria']}")
        st.write(f"**Entorno Frecuente:** {info['entorno']}")
        st.metric(label="Porcentaje de Confianza", value=f"{confianza:.2f}%")

        st.divider()

        # ----------------------------------------------------
        # INTEGRACIÓN CON LA API DE GROQ
        # ----------------------------------------------------
        st.subheader("💡 Análisis Descriptivo e Interpretación (IA Groq)")
        
        api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

        if api_key:
            with st.spinner("Generando análisis detallado con Groq..."):
                prompt = f"""
                Actúa como un sistema de asistencia de visión por computadora experto.
                El usuario ha subido una imagen que fue clasificada por un modelo CNN de CIFAR-10 con los siguientes datos:

                - Objeto Predicho: {clase_detectada}
                - Categoría: {info['categoria']}
                - Confianza del Modelo: {confianza:.2f}%

                Por favor proporciona un breve informe estructurado que incluya:
                1. Una descripción técnica breve de la categoría detectada ({clase_detectada}).
                2. Características visuales clave que un modelo CNN suele buscar para identificar esta categoría.
                3. Recomendaciones o contexto sobre la precisión del modelo en entornos de nube.
                """

                try:
                    client = OpenAI(
                        api_key=api_key,
                        base_url="https://api.groq.com/openai/v1"
                    )

                    respuesta = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": "Eres un asistente de inteligencia artificial y visión por computadora."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3
                    )

                    st.info(respuesta.choices[0].message.content)

                except Exception as e:
                    st.warning(f"Error al conectar con la API de Groq: {e}")
        else:
            st.warning("⚠️ Agregue su `GROQ_API_KEY` en los Secretos de Streamlit Cloud o en `.streamlit/secrets.toml` para activar el informe asistido por Groq.")

    elif not imagen_pil:
        st.info("Por favor cargue o tome una fotografía para realizar la predicción.")

st.divider()
st.caption("Servicio Web de Computación en la Nube | UTH 2026 - Fabricio Beltrand")