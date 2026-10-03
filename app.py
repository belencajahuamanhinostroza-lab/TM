import os
import platform

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Belen AI",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
IMAGE_PATH = "OIG5.jpg"

IMAGE_SIZE = (224, 224)


# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(226, 214, 255, 0.65),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(242, 232, 255, 0.75),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #faf8ff 0%,
                #f4efff 50%,
                #ffffff 100%
            );
    }

    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* HEADER */

    .top-header {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 25px;
    }

    .ai-icon {
        width: 54px;
        height: 54px;
        border-radius: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(
            135deg,
            #8b5cf6,
            #a78bfa
        );
        box-shadow:
            0 10px 30px rgba(124, 92, 246, 0.25);
        font-size: 27px;
    }

    .title-area h1 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 30px;
        font-weight: 800;
        color: #29233d;
        margin: 0;
        letter-spacing: -1px;
    }

    .title-area p {
        color: #827a94;
        margin: 3px 0 0 0;
        font-size: 14px;
    }

    /* CARDS */

    .card {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(139, 92, 246, 0.10);
        border-radius: 25px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow:
            0 12px 40px rgba(57, 38, 95, 0.07);
        backdrop-filter: blur(10px);
    }

    .card-title {
        color: #312847;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .card-subtitle {
        color: #8b8499;
        font-size: 13px;
        margin-bottom: 18px;
    }

    /* WELCOME */

    .welcome-card {
        background: linear-gradient(
            135deg,
            rgba(255,255,255,0.96),
            rgba(245,240,255,0.96)
        );
        border-radius: 28px;
        padding: 30px;
        margin-bottom: 22px;
        border: 1px solid rgba(139, 92, 246, 0.12);
        box-shadow: 0 15px 45px rgba(70, 45, 110, 0.08);
    }

    .welcome-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 25px;
        font-weight: 800;
        color: #302640;
        margin-bottom: 8px;
    }

    .welcome-text {
        color: #777087;
        font-size: 14px;
        line-height: 1.7;
        max-width: 700px;
    }

    /* RESULT */

    .result-card {
        text-align: center;
        background: linear-gradient(
            145deg,
            #ffffff,
            #f6f0ff
        );
        border-radius: 28px;
        padding: 30px 20px;
        border: 1px solid rgba(139, 92, 246, 0.12);
        box-shadow: 0 15px 45px rgba(70, 45, 110, 0.09);
    }

    .result-label {
        color: #9189a0;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 700;
    }

    .result-name {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 35px;
        font-weight: 800;
        color: #7047d7;
        margin-top: 7px;
    }

    .result-confidence {
        color: #5e586a;
        font-size: 16px;
        margin-top: 5px;
    }

    /* CAMERA */

    [data-testid="stCameraInput"] {
        border-radius: 20px;
        overflow: hidden;
    }

    [data-testid="stFileUploader"] {
        border-radius: 20px;
    }

    /* BUTTON */

    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 15px;
        padding: 12px 20px;
        background: linear-gradient(
            135deg,
            #7c4dff,
            #9b76ff
        );
        color: white;
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        font-size: 14px;
        box-shadow:
            0 8px 22px rgba(124, 77, 255, 0.22);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow:
            0 12px 28px rgba(124, 77, 255, 0.30);
    }

    /* SIDEBAR */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #f8f5ff 0%,
                #f1ebff 100%
            );
        border-right: 1px solid rgba(124, 77, 255, 0.08);
    }

    section[data-testid="stSidebar"] h2 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #302640;
    }

    .sidebar-card {
        background: rgba(255,255,255,0.75);
        border: 1px solid rgba(124,77,255,0.10);
        border-radius: 20px;
        padding: 18px;
        margin-bottom: 15px;
    }

    .sidebar-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        color: #403653;
        margin-bottom: 8px;
    }

    .sidebar-text {
        color: #777087;
        font-size: 13px;
        line-height: 1.6;
    }

    /* STATUS */

    .status {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: #f1edff;
        color: #7352c7;
        border-radius: 50px;
        padding: 7px 13px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #8b5cf6;
        display: inline-block;
    }

    /* FOOTER */

    .footer {
        text-align: center;
        color: #aaa3b3;
        font-size: 12px;
        padding-top: 15px;
        padding-bottom: 10px;
    }

    /* METRICS */

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.8);
        border-radius: 18px;
        padding: 15px;
        border: 1px solid rgba(124,77,255,0.08);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FUNCIONES
# ============================================================

@st.cache_resource
def load_ai_model():
    """
    Carga el modelo de Teachable Machine.
    compile=False evita problemas innecesarios al cargar
    modelos entrenados para clasificación.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontró el archivo '{MODEL_PATH}'."
        )

    return load_model(
        MODEL_PATH,
        compile=False
    )


@st.cache_data
def load_labels():
    """
    Lee las etiquetas desde labels.txt.

    Acepta formatos como:
        0 Belen
        1 Perro

    o simplemente:
        Belen
        Perro
    """

    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            f"No se encontró el archivo '{LABELS_PATH}'."
        )

    labels = []

    with open(
        LABELS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1])
            else:
                labels.append(line)

    return labels


def prepare_image(image):
    """
    Prepara la imagen exactamente con el formato
    utilizado habitualmente por Teachable Machine:
    224x224 y normalización entre -1 y 1.
    """

    image = image.convert("RGB")

    image = ImageOps.fit(
        image,
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(image)

    normalized_image = (
        image_array.astype(np.float32) / 127.0
    ) - 1.0

    data = np.ndarray(
        shape=(1, 224, 224, 3),
        dtype=np.float32
    )

    data[0] = normalized_image

    return data


def get_prediction(image, model, labels):
    """
    Realiza la predicción y devuelve:
    nombre de clase, confianza y todas las probabilidades.
    """

    data = prepare_image(image)

    prediction = model.predict(
        data,
        verbose=0
    )

    probabilities = prediction[0]

    # En caso de que haya más clases en el modelo
    # que etiquetas en labels.txt.
    number_of_classes = len(probabilities)

    if len(labels) < number_of_classes:
        labels = labels + [
            f"Clase {i}"
            for i in range(
                len(labels),
                number_of_classes
            )
        ]

    # En caso de que haya más etiquetas que clases.
    labels = labels[:number_of_classes]

    best_index = int(
        np.argmax(probabilities)
    )

    best_label = labels[best_index]

    confidence = float(
        probabilities[best_index]
    )

    results = pd.DataFrame(
        {
            "Clase": labels,
            "Probabilidad": probabilities
        }
    )

    results["Porcentaje"] = (
        results["Probabilidad"] * 100
    ).round(2)

    results = results.sort_values(
        by="Probabilidad",
        ascending=False
    ).reset_index(drop=True)

    return (
        best_label,
        confidence,
        results
    )


# ============================================================
# CARGAR MODELO Y ETIQUETAS
# ============================================================

try:
    model = load_ai_model()
    labels = load_labels()

except Exception as error:
    st.error(
        "No fue posible cargar el modelo."
    )

    st.code(
        str(error)
    )

    st.info(
        "Verifica que keras_model.h5 y labels.txt "
        "estén en la misma carpeta que app.py."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="top-header">

        <div class="ai-icon">
            🤖
        </div>

        <div class="title-area">
            <h1>Belen AI</h1>
            <p>Reconocimiento inteligente de imágenes</p>
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-card">

            <div class="sidebar-title">
                🤖 Asistente IA
            </div>

            <div class="sidebar-text">
                Esta aplicación utiliza un modelo de
                inteligencia artificial entrenado con
                Teachable Machine para reconocer imágenes.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-card">

            <div class="sidebar-title">
                📷 ¿Cómo funciona?
            </div>

            <div class="sidebar-text">
                1. Permite el acceso a la cámara.<br><br>
                2. Toma una fotografía.<br><br>
                3. La IA analiza la imagen.<br><br>
                4. Se muestra la clase detectada y
                su nivel de confianza.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="sidebar-card">

            <div class="sidebar-title">
                🧠 Modelo
            </div>

            <div class="sidebar-text">
                Entrada: 224 × 224 px<br>
                Clases disponibles: {len(labels)}<br>
                Modelo: Teachable Machine
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        f"Python {platform.python_version()}"
    )


# ============================================================
# PRESENTACIÓN
# ============================================================

st.markdown(
    """
    <div class="welcome-card">

        <div class="status">
            <span class="status-dot"></span>
            IA lista para analizar
        </div>

        <div class="welcome-title">
            Hola 👋
        </div>

        <div class="welcome-text">
            Toma una fotografía y deja que el modelo
            de inteligencia artificial analice su contenido.
            El resultado mostrará la categoría detectada
            y la probabilidad estimada.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CÁMARA
# ============================================================

st.markdown(
    """
    <div class="card">

        <div class="card-title">
            📸 Captura una imagen
        </div>

        <div class="card-subtitle">
            Coloca el objeto frente a la cámara y toma una foto.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)

img_file_buffer = st.camera_input(
    "Toma una fotografía"
)


# ============================================================
# PROCESAMIENTO
# ============================================================

if img_file_buffer is not None:

    image = Image.open(
        img_file_buffer
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(
        [1, 1],
        gap="large"
    )

    # --------------------------------------------------------
    # IMAGEN
    # --------------------------------------------------------

    with col1:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    🖼️ Imagen capturada
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )

    # --------------------------------------------------------
    # PREDICCIÓN
    # --------------------------------------------------------

    try:

        (
            best_label,
            confidence,
            results
        ) = get_prediction(
            image,
            model,
            labels
        )

        confidence_percentage = (
            confidence * 100
        )

        with col2:

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-label">
                        Resultado de la IA
                    </div>

                    <div class="result-name">
                        {best_label}
                    </div>

                    <div class="result-confidence">
                        Confianza:
                        <strong>
                            {confidence_percentage:.2f}%
                        </strong>
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                "<br>",
                unsafe_allow_html=True
            )

            # Mensaje según confianza
            if confidence >= 0.80:

                st.success(
                    "La IA tiene una alta confianza "
                    "en esta clasificación."
                )

            elif confidence >= 0.50:

                st.warning(
                    "La IA encontró una coincidencia "
                    "moderada. Puedes intentar otra foto."
                )

            else:

                st.info(
                    "La confianza es baja. "
                    "Prueba con una imagen más clara."
                )

    except Exception as error:

        st.error(
            "Ocurrió un error durante la predicción."
        )

        st.code(
            str(error)
        )

        st.stop()


    # ========================================================
    # PROBABILIDADES
    # ========================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                📊 Probabilidades
            </div>

            <div class="card-subtitle">
                Distribución de la predicción del modelo.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # Tabla
    table = results[
        ["Clase", "Porcentaje"]
    ].copy()

    table["Porcentaje"] = (
        table["Porcentaje"].astype(str)
        + "%"
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )

    # Gráfico
    chart_data = results[
        ["Clase", "Probabilidad"]
    ].copy()

    chart_data = chart_data.set_index(
        "Clase"
    )

    st.bar_chart(
        chart_data,
        use_container_width=True
    )


else:

    # ========================================================
    # ESTADO INICIAL
    # ========================================================

    st.markdown(
        """
        <div class="card" style="text-align:center;">

            <div style="
                font-size:50px;
                margin-bottom:10px;
            ">
                📷
            </div>

            <div class="card-title">
                Esperando una fotografía
            </div>

            <div class="card-subtitle">
                Utiliza la cámara de arriba para comenzar
                el reconocimiento.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Belen AI · Clasificación de imágenes con
        Inteligencia Artificial
    </div>
    """,
    unsafe_allow_html=True
)
