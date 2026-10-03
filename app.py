import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from teachable_machine import TeachableMachine


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Belen AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"


# ============================================================
# ESTILOS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(220, 207, 255, 0.55),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 15%,
            rgba(236, 226, 255, 0.65),
            transparent 35%
        ),
        #f8f7fc;
}

/* =========================
   TÍTULO
   ========================= */

.main-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 42px;
    font-weight: 800;
    color: #2c2440;
    margin-bottom: 4px;
}

.subtitle {
    font-size: 17px;
    color: #777087;
    margin-bottom: 30px;
}


/* =========================
   TARJETAS
   ========================= */

.card {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid rgba(120, 100, 170, 0.10);
    border-radius: 24px;
    padding: 26px;
    box-shadow: 0 12px 35px rgba(76, 61, 112, 0.08);
    margin-bottom: 20px;
}


/* =========================
   RESULTADO
   ========================= */

.result-card {
    background: linear-gradient(
        135deg,
        #eee7ff,
        #ffffff
    );

    border: 1px solid #ddd1ff;
    border-radius: 28px;

    padding: 35px 25px;

    text-align: center;

    box-shadow:
        0 15px 40px rgba(110, 82, 180, 0.12);
}

.result-label {
    color: #81769b;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 1px;
}

.result-name {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #6244b2;
    font-size: 40px;
    font-weight: 800;
    margin-top: 8px;
}

.confidence {
    color: #3e354f;
    font-size: 20px;
    font-weight: 600;
    margin-top: 8px;
}


/* =========================
   SIDEBAR
   ========================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #f0eaff 0%,
        #f8f6ff 100%
    );

    border-right: 1px solid #e2daf4;
}

.sidebar-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 26px;
    font-weight: 800;
    color: #4f3b78;
}

.sidebar-text {
    color: #746b87;
    font-size: 15px;
    line-height: 1.6;
}


/* =========================
   MÉTRICAS
   ========================= */

.metric-card {
    background: rgba(255, 255, 255, 0.90);
    border-radius: 20px;
    padding: 20px;
    text-align: center;

    border: 1px solid #eee9f8;

    box-shadow:
        0 8px 25px rgba(76, 61, 112, 0.05);
}

.metric-number {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: #6545b7;
}

.metric-text {
    color: #81788f;
    font-size: 14px;
}


/* =========================
   CÁMARA
   ========================= */

[data-testid="stCameraInput"] {
    border-radius: 22px;
    overflow: hidden;
}


/* =========================
   FOOTER
   ========================= */

.footer {
    text-align: center;
    color: #9991a9;
    font-size: 13px;
    padding: 35px 0 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNCIONES
# ============================================================

@st.cache_resource
def load_model():
    """
    Carga el modelo usando teachable-machine.

    Esta librería maneja específicamente:
    - DepthwiseConv2D con groups=1
    - modelos Sequential anidados
    - exports H5 antiguos de Teachable Machine
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontró '{MODEL_PATH}'"
        )

    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            f"No se encontró '{LABELS_PATH}'"
        )

    model = TeachableMachine(
        model_path=MODEL_PATH,
        labels_file_path=LABELS_PATH,
        model_type="h5"
    )

    return model


def get_label():
    """
    Lee labels.txt.

    Convierte:
        0 Belen

    en:
        Belen
    """

    if not os.path.exists(LABELS_PATH):
        return "Belen"

    with open(
        LABELS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        line = file.readline().strip()

    if not line:
        return "Belen"

    parts = line.split(maxsplit=1)

    if len(parts) == 2 and parts[0].isdigit():
        return parts[1]

    return line


def classify_uploaded_image(model, uploaded_file):
    """
    Guarda temporalmente la imagen capturada por Streamlit
    y utiliza classify_image() de TeachableMachine.
    """

    image = Image.open(uploaded_file).convert("RGB")

    # Archivo temporal para que teachable-machine pueda
    # trabajar con la imagen mediante su API oficial.
    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as temp_file:

        temp_path = temp_file.name

        image.save(
            temp_path,
            format="JPEG",
            quality=95
        )

    try:

        result = model.classify_image(temp_path)

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)

    return image, result


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🤖 Belen AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p class="sidebar-text">
        Sistema de reconocimiento de imágenes
        entrenado con Teachable Machine.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 🧠 Modelo")

    st.markdown(
        """
        <div class="card">
            <b>Clasificador de imágenes</b>
            <br><br>
            Captura una fotografía y el modelo
            analizará automáticamente la imagen.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 📁 Archivos")

    if os.path.exists(MODEL_PATH):
        st.success("✓ keras_model.h5")
    else:
        st.error("✗ Falta keras_model.h5")

    if os.path.exists(LABELS_PATH):
        st.success("✓ labels.txt")
    else:
        st.error("✗ Falta labels.txt")


# ============================================================
# CABECERA
# ============================================================

st.markdown(
    '<div class="main-title">Reconocimiento de Imágenes</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Captura una imagen y deja que Belen AI la analice.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CARGAR MODELO
# ============================================================

try:

    model = load_model()

except Exception as error:

    st.error("❌ No se pudo cargar el modelo.")

    st.markdown(
        """
        <div class="card">
            <b>Información del error</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.code(
        str(error),
        language="text"
    )

    st.stop()


# ============================================================
# INFORMACIÓN
# ============================================================

label = get_label()

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
        <div class="metric-card">

            <div class="metric-number">
                224×224
            </div>

            <div class="metric-text">
                Resolución de entrada
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-number">
                1
            </div>

            <div class="metric-text">
                Clase: {label}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="metric-card">

            <div class="metric-number">
                AI
            </div>

            <div class="metric-text">
                Clasificación automática
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


# ============================================================
# CÁMARA
# ============================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.markdown("### 📷 Captura una imagen")

st.markdown(
    """
    <p style="
        color:#81788f;
        margin-bottom:15px;
    ">
        Toma una fotografía utilizando la cámara
        de tu dispositivo.
    </p>
    """,
    unsafe_allow_html=True
)

image_file = st.camera_input(
    "Toma una fotografía"
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PREDICCIÓN
# ============================================================

if image_file is not None:

    try:

        image, result = classify_uploaded_image(
            model,
            image_file
        )

        # --------------------------------------------
        # DATOS DEL MODELO
        # --------------------------------------------

        class_index = int(
            result["class_index"]
        )

        confidence = float(
            result["class_confidence"]
        )

        predictions = np.asarray(
            result["predictions"],
            dtype=float
        ).flatten()

        # La etiqueta del archivo labels.txt es
        # la que mostramos al usuario.
        predicted_label = label

        # --------------------------------------------
        # COLUMNAS
        # --------------------------------------------

        left, right = st.columns(
            [1, 1],
            gap="large"
        )


        # ============================================
        # IMAGEN
        # ============================================

        with left:

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.markdown(
                "### 🖼️ Imagen capturada"
            )

            st.image(
                image,
                use_container_width=True
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        # ============================================
        # RESULTADO
        # ============================================

        with right:

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.markdown(
                "### ✨ Resultado"
            )

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-label">
                        RESULTADO DETECTADO
                    </div>

                    <div class="result-name">
                        {predicted_label}
                    </div>

                    <div class="confidence">
                        {confidence * 100:.2f}% de confianza
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                "<br>",
                unsafe_allow_html=True
            )

            # ========================================
            # ESTADO
            # ========================================

            if confidence >= 0.80:

                st.success(
                    "✓ El modelo tiene alta confianza "
                    "en esta clasificación."
                )

            elif confidence >= 0.50:

                st.warning(
                    "⚠️ El modelo tiene confianza "
                    "moderada en esta clasificación."
                )

            else:

                st.info(
                    "ℹ️ La confianza del modelo es baja. "
                    "Prueba con otra fotografía."
                )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        # ====================================================
        # PROBABILIDADES
        # ====================================================

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "### 📊 Probabilidad del modelo"
        )

        # Tu modelo tiene una sola salida.
        # La mostramos como la probabilidad de Belen.

        probability_df = pd.DataFrame({
            "Clase": [predicted_label],
            "Probabilidad": [
                confidence * 100
            ]
        })

        probability_df["Probabilidad"] = (
            probability_df["Probabilidad"]
            .round(2)
        )

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )

        st.progress(
            min(max(confidence, 0.0), 1.0)
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # INFORMACIÓN TÉCNICA
        # ====================================================

        with st.expander(
            "🔎 Ver información técnica"
        ):

            st.write(
                "**Clase detectada:**",
                predicted_label
            )

            st.write(
                "**Índice de clase:**",
                class_index
            )

            st.write(
                "**Confianza:**",
                f"{confidence:.6f}"
            )

            st.write(
                "**Entrada del modelo:**",
                "224 × 224 × 3"
            )

            st.write(
                "**Número de salidas:**",
                len(predictions)
            )


    except Exception as error:

        st.error(
            "❌ Ocurrió un error durante la predicción."
        )

        st.code(
            str(error),
            language="text"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Belen AI · Clasificación de imágenes con Inteligencia Artificial
    </div>
    """,
    unsafe_allow_html=True
)
