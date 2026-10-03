import os
import tempfile

import streamlit as st
from PIL import Image
from teachable_machine import TeachableMachine


# ==========================================
# CONFIGURACIÓN
# ==========================================

st.set_page_config(
    page_title="Belen AI",
    page_icon="🤖",
    layout="centered"
)

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"


# ==========================================
# CARGAR MODELO
# ==========================================

@st.cache_resource
def cargar_modelo():

    if not os.path.exists(MODEL_PATH):
        st.error("❌ No se encontró el archivo keras_model.h5")
        st.stop()

    if not os.path.exists(LABELS_PATH):
        st.error("❌ No se encontró el archivo labels.txt")
        st.stop()

    modelo = TeachableMachine(
        model_path=MODEL_PATH,
        labels_file_path=LABELS_PATH,
        model_type="h5"
    )

    return modelo


model = cargar_modelo()


# ==========================================
# INTERFAZ
# ==========================================

st.title("🤖 Belen AI")

st.write(
    "Toma una foto y el modelo analizará la imagen."
)


# ==========================================
# CÁMARA
# ==========================================

foto = st.camera_input(
    "📷 Toma una foto"
)


# ==========================================
# ANALIZAR FOTO
# ==========================================

if foto is not None:

    # Abrir imagen
    imagen = Image.open(foto).convert("RGB")

    st.image(
        imagen,
        caption="Imagen capturada",
        use_container_width=True
    )

    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as archivo:

        ruta_temporal = archivo.name

        imagen.save(
            ruta_temporal,
            format="JPEG"
        )

    try:

        # ======================================
        # PREDICCIÓN
        # ======================================

        resultado = model.classify_image(
            ruta_temporal
        )

        # Nombre que entrega el modelo
        nombre = resultado["class_name"]

        # ======================================
        # LIMPIAR NOMBRE
        # "0 Belen" → "Belen"
        # ======================================

        partes = nombre.split(" ", 1)

        if len(partes) == 2 and partes[0].isdigit():
            nombre = partes[1]

        # ======================================
        # PROBABILIDAD
        # ======================================

        confianza = float(
            resultado["class_confidence"]
        )

        # ======================================
        # MOSTRAR RESULTADO
        # ======================================

        st.subheader("✨ Resultado")

        st.success(
            f"**{nombre}**"
        )

        st.write(
            f"Probabilidad: **{confianza * 100:.2f}%**"
        )

    except Exception as error:

        st.error(
            "❌ Ocurrió un error al analizar la imagen."
        )

        st.write(
            str(error)
        )

    finally:

        # Eliminar archivo temporal
        if os.path.exists(ruta_temporal):

            os.remove(
                ruta_temporal
            )
