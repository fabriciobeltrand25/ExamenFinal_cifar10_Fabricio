import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Configuración principal de la página
st.set_page_config(
    page_title="Clasificador CIFAR-10 - Examen Nube UTH",
    page_icon="📊",
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

# Metadata para las categorías
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
st.title("📊 Clasificador CIFAR-10 - Examen Nube UTH")
st.caption("Desarrollado por: **Fabricio Beltrand**")

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
    st.subheader("2. Diagnóstico y Resultados")
    
    if imagen_pil and modelo:
        # Preprocesamiento de la imagen para el modelo (32x32 RGB)
        img_rgb = imagen_pil.convert("RGB")
        img_resized = img_rgb.resize(IMG_SIZE)
        img_array = np.array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # Predicción del modelo
        predicciones = modelo.predict(img_batch, verbose=0)[0]
        idx_clase = np.argmax(predicciones)
        confianza = float(predicciones[idx_clase]) * 100
        clase_detectada = CLASES[idx_clase]
        info = INFO_OBJETOS.get(clase_detectada, {"categoria": "General", "entorno": "N/A"})

        # Mostrar Métricas principales
        st.markdown(f"### **Predicción:** {clase_detectada}")
        
        m1, m2 = st.columns(2)
        m1.metric(label="Confianza Principal", value=f"{confianza:.2f}%")
        m2.metric(label="Categoría", value=info['categoria'])

        st.divider()

        # ----------------------------------------------------
        # VISUALIZACIÓN DE GRÁFICAS
        # ----------------------------------------------------
        st.subheader("📈 Distribución de Probabilidades por Clase")
        
        # Crear DataFrame para la gráfica
        df_probs = pd.DataFrame({
            "Clase": CLASES,
            "Probabilidad (%)": predicciones * 100
        }).sort_values(by="Probabilidad (%)", ascending=True)

        # Gráfica de barras horizontales con Plotly
        fig_bar = px.bar(
            df_probs,
            x="Probabilidad (%)",
            y="Clase",
            orientation="h",
            text="Probabilidad (%)",
            color="Probabilidad (%)",
            color_continuous_scale="Blues"
        )
        fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_bar.update_layout(
            xaxis_range=[0, 105],
            height=300,
            margin=dict(l=0, r=20, t=10, b=10),
            showlegend=False
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)

        # Medidor de Confianza (Gauge Chart)
        st.subheader("🎯 Indicador de Precisión")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confianza,
            number={'suffix': "%"},
            title={'text': f"Certeza de {clase_detectada}"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, 50], 'color': "#ffebee"},
                    {'range': [50, 80], 'color': "#e8f5e9"},
                    {'range': [80, 100], 'color': "#c8e6c9"}
                ]
            }
        ))
        fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    elif not imagen_pil:
        st.info("Por favor cargue o tome una fotografía para realizar la predicción y visualizar las gráficas.")

st.divider()
st.caption("Servicio Web de Computación en la Nube | UTH 2026 - Fabricio Beltrand")
