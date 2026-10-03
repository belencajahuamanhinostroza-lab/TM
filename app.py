import os
import tempfile

import streamlit as st
from PIL import Image
from teachable_machine import TeachableMachine


# ---------------------------------------
# CONFIGURACIÓN
# ---------------------------------------

st.set_page_config(
    page_title="Belen AI",
    page_icon="🤖",
    layout="centered"
)

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"


# ---------------------------------------
# ESTILO SIMPLE
# ---------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #f8f6ff;
}

h1 {
    color: #4f3b78;
}

.resultado {
    background: white;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    margin-top: 20px;
    border: 1px solid #ddd4f5;
}

.nombre {
    font-size: 36px;
    font-weight: bold;
    color: #6d4cc2;
}

.probabilidad {
    font-size: 22px;
    color: #555;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------
# TÍTULO
# ---------------------------------------

st.title("🤖 Belen AI")

st.write(
    "Toma una foto y el modelo analizará si corresponde a Belen."
)


# ---------------------------------------
# CARGAR MODELO
# ---------------------------------------

@st.cache_resource
def cargar_modelo():

    if not os.path.exists(MODEL_PATH):
        st.error("No se encontró keras_model.h5")
        st.stop()

    if not os.path.exists(LABELS_PATH):
        st.error("No se encontró labels.txt")
        st.stop()

    return TeachableMachine(
        model_path=MODEL_PATH,
        labels_file_path=LABELS_PATH,
        model_type="h5"
    )


model = cargar_modelo()


# ---------------------------------------
# CÁMARA
# ---------------------------------------

foto = st.camera_input("📷 Toma una foto")


# ---------------------------------------
# PREDICCIÓN
# ---------------------------------------

if foto is not None:

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

        # ---------------------------------------
        # CLASIFICAR IMAGEN
        # ---------------------------------------

        resultado = model.classify_image(
            ruta_temporal
        )

        nombre = resultado["class_name"]

        # Quitar "0 " de "0 Belen"
        partes = nombre.split(" ", 1)

        if len(partes) == 2 and partes[0].isdigit():
            nombre = partes[1]

        confianza = float(
            resultado["class_confidence"]
        )

        # ---------------------------------------
        # RESULTADO
        # ---------------------------------------

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
            os.remove(ruta_temporal)
