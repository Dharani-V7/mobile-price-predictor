import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Mobile Price Predictor",
    page_icon="📱",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("model.pkl")

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 48px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    margin-bottom: 30px;
}

.section-title {
    font-size: 32px;
    font-weight: 650;
    margin-top: 35px;
    margin-bottom: 15px;
}

.card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.3);
    margin-bottom: 15px;
}

.metric-card {
    text-align: center;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.3);
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 2px solid rgba(128,128,128,0.4);
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    '<div class="main-title">📱 Mobile Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict the price category of a mobile phone using its technical specifications.</div>',
    unsafe_allow_html=True
)

st.divider()

# ============================================================
# MACHINE LEARNING OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">🤖 Machine Learning Overview</div>',
    unsafe_allow_html=True
)

st.write(
    """
    Our system uses a **Random Forest Classification algorithm** to predict
    the price category of a mobile phone based on its technical specifications.
    """
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Algorithm", "Random Forest")

with col2:
    st.metric("Dataset", "2,000")

with col3:
    st.metric("Features", "20")

with col4:
    st.metric("Accuracy", "88%")

# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ How It Works</div>',
    unsafe_allow_html=True
)

step1, step2, step3, step4 = st.columns(4)

with step1:
    st.subheader("01")
    st.write("**Enter Specifications**")
    st.write("Provide the technical specifications of the mobile phone.")

with step2:
    st.subheader("02")
    st.write("**Prepare Input**")
    st.write("The input is arranged in the same feature format used during training.")

with step3:
    st.subheader("03")
    st.write("**Random Forest Prediction**")
    st.write("Multiple decision trees analyze the mobile specifications.")

with step4:
    st.subheader("04")
    st.write("**View Result**")
    st.write("The model predicts one of four mobile price categories.")

# ============================================================
# PROJECT OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">📚 Project Overview</div>',
    unsafe_allow_html=True
)

st.write("### Problem")

st.write(
    "Predict the price category of a mobile phone based on its technical specifications."
)

st.write("### Dataset")

st.write(
    "The dataset contains 2,000 mobile phone records with 20 input features."
)

st.write("### Target Variable")

st.code("price_range")

st.write("### Price Categories")

category_col1, category_col2, category_col3, category_col4 = st.columns(4)

with category_col1:
    st.info("0\n\n**Budget Phone**")

with category_col2:
    st.info("1\n\n**Affordable Phone**")

with category_col3:
    st.info("2\n\n**Premium Phone**")

with category_col4:
    st.info("3\n\n**Flagship Phone**")

# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">📊 Model Performance</div>',
    unsafe_allow_html=True
)

st.write(
    "The Random Forest classifier was evaluated using accuracy, precision, recall and F1-score."
)

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric("Accuracy", "88%")

with metric2:
    st.metric("Precision", "88%")

with metric3:
    st.metric("Recall", "88%")

with metric4:
    st.metric("F1-Score", "88%")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.markdown(
    '<div class="section-title">📋 Classification Performance by Class</div>',
    unsafe_allow_html=True
)

report_data = {
    "Price Category": [
        "Budget (0)",
        "Affordable (1)",
        "Premium (2)",
        "Flagship (3)"
    ],
    "Precision": [0.95, 0.82, 0.81, 0.93],
    "Recall": [0.96, 0.84, 0.79, 0.93],
    "F1-Score": [0.96, 0.83, 0.80, 0.93]
}

report_df = pd.DataFrame(report_data)

st.dataframe(
    report_df,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

st.markdown(
    '<div class="section-title">🔲 Confusion Matrix</div>',
    unsafe_allow_html=True
)

st.write(
    "The confusion matrix shows how many mobile phones were correctly "
    "and incorrectly classified into each price category."
)

cm = [
    [96, 4, 0, 0],
    [5, 84, 11, 0],
    [0, 14, 79, 7],
    [0, 0, 7, 93]
]

fig, ax = plt.subplots(figsize=(7, 5))

im = ax.imshow(cm)

ax.set_xlabel("Predicted Price Category")
ax.set_ylabel("Actual Price Category")
ax.set_title("Random Forest Confusion Matrix")

ax.set_xticks([0, 1, 2, 3])
ax.set_yticks([0, 1, 2, 3])

ax.set_xticklabels([
    "Budget",
    "Affordable",
    "Premium",
    "Flagship"
])

ax.set_yticklabels([
    "Budget",
    "Affordable",
    "Premium",
    "Flagship"
])

for i in range(4):
    for j in range(4):
        ax.text(
            j,
            i,
            cm[i][j],
            ha="center",
            va="center"
        )

fig.colorbar(im, ax=ax)

st.pyplot(fig)

# ============================================================
# INFERENCE
# ============================================================

st.markdown(
    '<div class="section-title">🧠 What Do These Results Tell Us?</div>',
    unsafe_allow_html=True
)

st.write(
    """
    The Random Forest classifier achieved an overall accuracy of **88%**
    on the test dataset.

    The model performed particularly well for the **Budget** and
    **Flagship** categories, with F1-scores of **0.96** and **0.93**
    respectively.

    The **Affordable** and **Premium** categories were somewhat more
    difficult for the model to distinguish, with F1-scores of **0.83**
    and **0.80**.

    The confusion matrix shows that most predictions are concentrated
    along the diagonal, indicating that the majority of mobile phones
    were correctly classified.
    """
)

# ============================================================
# PREDICTOR
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">🔮 Mobile Price Predictor</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the specifications of a mobile phone to predict its price category."
)

col1, col2, col3 = st.columns(3)

with col1:

    battery_power = st.number_input(
        "Battery Power (mAh)",
        min_value=500,
        max_value=2000,
        value=1000
    )

    blue = st.selectbox(
        "Bluetooth",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    clock_speed = st.number_input(
        "Clock Speed (GHz)",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.1
    )

    dual_sim = st.selectbox(
        "Dual SIM",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    fc = st.number_input(
        "Front Camera (MP)",
        min_value=0,
        max_value=20,
        value=5
    )

    four_g = st.selectbox(
        "4G",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    int_memory = st.number_input(
        "Internal Memory (GB)",
        min_value=2,
        max_value=64,
        value=16
    )

with col2:

    m_dep = st.number_input(
        "Mobile Depth (cm)",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1
    )

    mobile_wt = st.number_input(
        "Mobile Weight (g)",
        min_value=80,
        max_value=200,
        value=140
    )

    n_cores = st.number_input(
        "Number of Cores",
        min_value=1,
        max_value=8,
        value=4
    )

    pc = st.number_input(
        "Primary Camera (MP)",
        min_value=0,
        max_value=20,
        value=10
    )

    px_height = st.number_input(
        "Pixel Resolution Height",
        min_value=0,
        max_value=2000,
        value=500
    )

    px_width = st.number_input(
        "Pixel Resolution Width",
        min_value=500,
        max_value=2000,
        value=1000
    )

    ram = st.number_input(
        "RAM (MB)",
        min_value=256,
        max_value=4000,
        value=2000
    )

with col3:

    sc_h = st.number_input(
        "Screen Height (cm)",
        min_value=5,
        max_value=20,
        value=12
    )

    sc_w = st.number_input(
        "Screen Width (cm)",
        min_value=0,
        max_value=20,
        value=7
    )

    talk_time = st.number_input(
        "Talk Time (hours)",
        min_value=2,
        max_value=20,
        value=10
    )

    three_g = st.selectbox(
        "3G",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    touch_screen = st.selectbox(
        "Touch Screen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    wifi = st.selectbox(
        "Wi-Fi",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

# ============================================================
# PREDICTION
# ============================================================

st.write("")

if st.button(
    "🔮 Predict Mobile Category",
    use_container_width=True
):

    input_data = pd.DataFrame([[
        battery_power,
        blue,
        clock_speed,
        dual_sim,
        fc,
        four_g,
        int_memory,
        m_dep,
        mobile_wt,
        n_cores,
        pc,
        px_height,
        px_width,
        ram,
        sc_h,
        sc_w,
        talk_time,
        three_g,
        touch_screen,
        wifi
    ]], columns=[
        "battery_power",
        "blue",
        "clock_speed",
        "dual_sim",
        "fc",
        "four_g",
        "int_memory",
        "m_dep",
        "mobile_wt",
        "n_cores",
        "pc",
        "px_height",
        "px_width",
        "ram",
        "sc_h",
        "sc_w",
        "talk_time",
        "three_g",
        "touch_screen",
        "wifi"
    ])

    prediction = model.predict(input_data)[0]

    price_categories = {
        0: "Budget Phone",
        1: "Affordable Phone",
        2: "Premium Phone",
        3: "Flagship Phone"
    }

    result = price_categories[prediction]

    st.markdown(
        f"""
        <div class="result-box">
            <h2>🎯 Recommended Category</h2>
            <h1>{result}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )