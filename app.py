import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Belen AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# ESTILO — AI ASSISTANT / LAVANDA
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700&display=swap');

:root {
    --bg: #f5f0fb;
    --bg2: #eee8f8;
    --white: #ffffff;
    --purple: #9b72df;
    --purple-dark: #7952bd;
    --purple-soft: #eee5ff;
    --text: #282331;
    --muted: #8c8497;
    --line: #e8e0f1;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 5%, rgba(221,194,255,.55), transparent 28%),
        radial-gradient(circle at 90% 20%, rgba(255,213,236,.45), transparent 30%),
        linear-gradient(145deg, #f9f5fc 0%, #f0eaf8 48%, #f8edf5 100%);
    color: var(--text);
}

#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

.block-container {
    max-width: 1180px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
}


/* HEADER */

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
}

.profile {
    display: flex;
    align-items: center;
    gap: 11px;
}

.avatar {
    width: 43px;
    height: 43px;
    border-radius: 50%;
    background: linear-gradient(145deg, #c5a2ef, #8f66d2);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    font-weight: 700;
    box-shadow: 0 7px 20px rgba(139,100,205,.2);
}

.hello {
    color: #948c9d;
    font-size: .7rem;
}

.name {
    color: #302a39;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1rem;
    font-weight: 700;
}

.ai-pill {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 8px 13px;
    border-radius: 999px;
    background: rgba(255,255,255,.72);
    border: 1px solid rgba(255,255,255,.9);
    color: var(--purple-dark);
    font-size: .68rem;
    font-weight: 700;
    box-shadow: 0 7px 22px rgba(103,75,137,.08);
}

.ai-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--purple);
}


/* HERO */

.hero {
    position: relative;
    overflow: hidden;
    border-radius: 28px;
    padding: 31px 34px;
    min-height: 205px;
    margin-bottom: 22px;
    background:
        radial-gradient(circle at 82% 25%, rgba(177,128,240,.25), transparent 25%),
        linear-gradient(135deg, rgba(255,255,255,.94), rgba(239,230,250,.92));
    border: 1px solid rgba(255,255,255,.9);
    box-shadow: 0 18px 45px rgba(104,78,128,.10);
}

.hero-orb {
    position: absolute;
    width: 155px;
    height: 155px;
    right: 55px;
    top: 25px;
    border-radius: 50%;
    background:
        radial-gradient(circle at 30% 25%, #e5c9ff, #a678df 45%, #7d58bb);
    opacity: .85;
    filter: blur(.2px);
    box-shadow:
        0 20px 45px rgba(135,93,202,.20),
        inset -18px -18px 30px rgba(79,48,122,.16);
}

.hero-small {
    position: relative;
    z-index: 2;
    color: #9279ad;
    font-size: .68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.hero-title {
    position: relative;
    z-index: 2;
    max-width: 600px;
    margin-top: 8px;
    color: #302a39;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.25rem;
    font-weight: 700;
    line-height: 1.06;
    letter-spacing: -1.1px;
}

.hero-text {
    position: relative;
    z-index: 2;
    max-width: 570px;
    color: #81788c;
    font-size: .84rem;
    line-height: 1.5;
    margin-top: 11px;
}

.hero-tag {
    position: relative;
    z-index: 2;
    display: inline-flex;
    margin-top: 15px;
    padding: 8px 13px;
    border-radius: 999px;
    background: #9b72df;
    color: white;
    font-size: .66rem;
    font-weight: 700;
}


/* CARDS */

.card {
    background: rgba(255,255,255,.86);
    border: 1px solid rgba(255,255,255,.95);
    border-radius: 23px;
    padding: 20px;
    box-shadow: 0 15px 35px rgba(106,78,131,.09);
    margin-bottom: 18px;
}

.card-title {
    color: #322c3a;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: .98rem;
    font-weight: 700;
}

.card-subtitle {
    color: #958c9e;
    font-size: .72rem;
    margin-top: 3px;
    margin-bottom: 14px;
}


/* CAMERA */

.camera-shell {
    background: rgba(255,255,255,.76);
    border: 1px solid rgba(255,255,255,.95);
    border-radius: 23px;
    padding: 15px;
    box-shadow: 0 15px 35px rgba(106,78,131,.09);
}

.camera-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 2px 5px 12px;
}

.camera-title {
    color: #342d3d;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: .82rem;
    font-weight: 700;
}

.camera-status {
    display: flex;
    align-items: center;
    gap: 6px;
    color: #8c68bd;
    font-size: .65rem;
    font-weight: 700;
}

.camera-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #9b72df;
}


/* BUTTONS */

.stButton > button {
    background: linear-gradient(135deg, #a47ade, #8c61d1) !important;
    color: white !important;
    border: 0 !important;
    border-radius: 999px !important;
    min-height: 45px !important;
    padding: .65rem 1.4rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: .75rem !important;
    font-weight: 700 !important;
    box-shadow: 0 9px 20px rgba(137,96,202,.20) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #9164d2, #794ebc) !important;
    transform: translateY(-1px);
}


/* CAMERA INPUT */

[data-testid="stCameraInput"] section {
    background: #faf8fc !important;
    border: 1px dashed #d8cde5 !important;
    border-radius: 17px !important;
}

[data-testid="stCameraInput"] button {
    background: #9b72df !important;
    color: white !important;
    border: 0 !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
}


/* PREDICTION */

.prediction {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #a77cdf, #8960c9);
    color: white;
    border-radius: 23px;
    padding: 23px;
    min-height: 205px;
    box-shadow: 0 17px 35px rgba(116,76,173,.20);
}

.prediction:after {
    content: "";
    position: absolute;
    width: 175px;
    height: 175px;
    right: -70px;
    top: -80px;
    border-radius: 50%;
    border: 1px solid rgba(255,255,255,.17);
    box-shadow:
        0 0 0 25px rgba(255,255,255,.05),
        0 0 0 50px rgba(255,255,255,.035);
}

.prediction-label {
    position: relative;
    z-index: 2;
    color: #e9dcfa;
    font-size: .65rem;
    text-transform: uppercase;
    letter-spacing: .8px;
    font-weight: 700;
}

.prediction-title {
    position: relative;
    z-index: 2;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    margin-top: 8px;
}

.prediction-score {
    position: relative;
    z-index: 2;
    color: #eee4fa;
    font-size: .82rem;
    margin-top: 5px;
}

.prediction-pill {
    position: relative;
    z-index: 2;
    display: inline-block;
    margin-top: 19px;
    padding: 7px 11px;
    border-radius: 999px;
    background: rgba(255,255,255,.16);
    border: 1px solid rgba(255,255,255,.18);
    font-size: .64rem;
    font-weight: 700;
}


/* STATS */

.stat {
    background: rgba(255,255,255,.72);
    border: 1px solid rgba(255,255,255,.9);
    border-radius: 17px;
    padding: 15px;
}

.stat-label {
    color: #968d9f;
    font-size: .63rem;
    text-transform: uppercase;
    letter-spacing: .7px;
    font-weight: 700;
}

.stat-value {
    color: #362f3e;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    margin-top: 5px;
}


/* SIDEBAR */

[data-testid="stSidebar"] {
    background: #f5f0fb !important;
    border-right: 1px solid #e6ddef !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #342d3d !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

[data-testid="stSidebar"] p {
    color: #8c8395 !important;
}


/* ALERTS */

[data-testid="stAlert"] {
    border-radius: 15px !important;
}


/* DATAFRAME */

[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    overflow: hidden !important;
}


/* FOOTER */

.footer {
    margin-top: 30px;
    padding-top: 18px;
    border-top: 1px solid #e1d8e9;
    display: flex;
    justify-content: space-between;
    color: #a198a9;
    font-size: .64rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# ARCHIVOS DEL MODELO
# ============================================================

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"


@st.cache_resource
def load_ai_model():
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        return None


@st.cache_data
def load_labels():
    labels = []

    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                # Admite:
                # "0 Belen"
                # "Belen"
                parts = line.split(maxsplit=1)

                if len(parts) == 2 and parts[0].isdigit():
                    labels.append(parts[1])
                else:
                    labels.append(line)

    except FileNotFoundError:
        return []

    return labels


model = load_ai_model()
labels = load_labels()


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="topbar">

    <div class="profile">

        <div class="avatar">
            ✦
        </div>

        <div>
            <div class="hello">
                Good morning
            </div>

            <div class="name">
                Belen AI
            </div>
        </div>

    </div>

    <div class="ai-pill">
        <span class="ai-dot"></span>
        AI READY
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-orb"></div>

    <div class="hero-small">
        Smart vision · AI assistant
    </div>

    <div class="hero-title">
        Hi! How can I help you today?
    </div>

    <div class="hero-text">
        Take a picture and let your trained AI model analyze
        what it sees. Your result will appear instantly below.
    </div>

    <div class="hero-tag">
        ✦ &nbsp; START AI ANALYSIS
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div style="
        color:#8c68bd;
        font-size:.65rem;
        font-weight:700;
        text-transform:uppercase;
        letter-spacing:1px;
    ">
        AI Assistant
    </div>

    <div style="
        color:#342d3d;
        font-family:'Plus Jakarta Sans';
        font-size:1.1rem;
        font-weight:700;
        margin-top:4px;
    ">
        Model settings
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.write("Modelo:", MODEL_PATH)
    st.write("Labels:", LABELS_PATH)

    if labels:
        st.write("Clases detectadas:")

        for label in labels:
            st.markdown(
                f"**✦ {label}**"
            )


# ============================================================
# ERROR MODELO
# ============================================================

if model is None:

    st.error(
        "No se encontró o no se pudo cargar `keras_model.h5`. "
        "Coloca el archivo junto a este `app.py`."
    )

    st.stop()


# ============================================================
# CÁMARA + INFORMACIÓN
# ============================================================

left, right = st.columns(
    [1.05, .95],
    gap="large"
)


with left:

    st.markdown("""
    <div class="camera-shell">

        <div class="camera-head">

            <div class="camera-title">
                Take a photo
            </div>

            <div class="camera-status">
                <span class="camera-dot"></span>
                READY
            </div>

        </div>
    """, unsafe_allow_html=True)

    img_file_buffer = st.camera_input(
        "Camera",
        label_visibility="collapsed"
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


with right:

    st.markdown(f"""
    <div class="card">

        <div class="card-title">
            Your AI model
        </div>

        <div class="card-subtitle">
            Trained image classifier
        </div>

        <div class="stat">
            <div class="stat-label">
                Available classes
            </div>

            <div class="stat-value">
                {len(labels)}
            </div>
        </div>

    </div>
    """, unsafe_allow_html=True)

    if labels:
        for label in labels:
            st.markdown(f"""
            <div class="stat" style="margin-bottom:9px;">
                <div class="stat-label">
                    AI class
                </div>

                <div class="stat-value">
                    ✦ {label}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# PREDICCIÓN
# ============================================================

if img_file_buffer is not None:

    try:

        image = Image.open(
            img_file_buffer
        ).convert("RGB")

        image_resized = image.resize(
            (224, 224)
        )

        image_array = np.asarray(
            image_resized,
            dtype=np.float32
        )

        normalized = (
            image_array / 127.0
        ) - 1.0

        data = np.ndarray(
            shape=(1, 224, 224, 3),
            dtype=np.float32
        )

        data[0] = normalized

        prediction = model.predict(
            data,
            verbose=0
        )[0]

        best_index = int(
            np.argmax(prediction)
        )

        best_probability = float(
            prediction[best_index]
        )

        if best_index < len(labels):
            best_label = labels[best_index]
        else:
            best_label = f"Class {best_index}"

        # ====================================================
        # RESULTADO
        # ====================================================

        st.markdown(
            "<div style='height:22px'></div>",
            unsafe_allow_html=True
        )

        result_col, image_col = st.columns(
            [1, 1],
            gap="large"
        )

        with result_col:

            st.markdown(f"""
            <div class="prediction">

                <div class="prediction-label">
                    AI detected
                </div>

                <div class="prediction-title">
                    {best_label}
                </div>

                <div class="prediction-score">
                    Confidence:
                    <strong>
                        {best_probability:.1%}
                    </strong>
                </div>

                <div class="prediction-pill">
                    ✦ ANALYSIS COMPLETE
                </div>

            </div>
            """, unsafe_allow_html=True)


        with image_col:

            st.markdown("""
            <div class="card">

                <div class="card-title">
                    Your photo
                </div>

                <div class="card-subtitle">
                    Image analyzed by the AI model
                </div>
            """, unsafe_allow_html=True)

            st.image(
                image,
                use_container_width=True
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


        # ====================================================
        # PROBABILIDADES
        # ====================================================

        st.markdown(
            "<div style='height:4px'></div>",
            unsafe_allow_html=True
        )

        st.markdown("""
        <div class="card">

            <div class="card-title">
                What the AI sees
            </div>

            <div class="card-subtitle">
                Confidence for each trained category
            </div>
        """, unsafe_allow_html=True)

        names = []

        for i in range(len(prediction)):

            if i < len(labels):
                names.append(labels[i])
            else:
                names.append(f"Class {i}")

        probability_df = {
            "Class": names,
            "Confidence": prediction
        }

        import pandas as pd

        df = pd.DataFrame(
            probability_df
        )

        df["Confidence"] = df["Confidence"].round(4)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            df.set_index("Class")["Confidence"],
            use_container_width=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    except Exception as e:

        st.error(
            f"Error procesando la imagen: {str(e)}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    <div>
        BELEN AI · IMAGE CLASSIFICATION
    </div>

    <div>
        KERAS · TEACHABLE MACHINE · STREAMLIT
    </div>

</div>
""", unsafe_allow_html=True)
