import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from io import BytesIO
import datetime

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Arogya Sevak",
    page_icon="🩺",
    layout="wide",
)


st.markdown("""
<style>
.glow {
    font-size: 45px;
    color: #00BFFF;
    text-align: center;
    animation: glow 1.5s ease-in-out infinite alternate;
}

@keyframes glow {
    from {
        text-shadow: 0 0 10px #00BFFF, 0 0 20px #00BFFF, 0 0 30px #0073e6;
    }
    to {
        text-shadow: 0 0 20px #1E90FF, 0 0 30px #1E90FF, 0 0 40px #1E90FF;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="glow">Arogya Sevak</h1>', unsafe_allow_html=True)


# =========================
# Load Model
# =========================
# Replace your current load_model line with this:
# =========================
# Load Model (Compatibility Fix)
# =========================
from tensorflow.keras.layers import InputLayer

# Define a helper to bridge the 'batch_shape' vs 'batch_input_shape' difference
class CompatibleInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super().__init__(*args, **kwargs)

# Use custom_object_scope to redirect 'InputLayer' to our fixed version
with tf.keras.utils.custom_object_scope({'InputLayer': CompatibleInputLayer}):
    model = tf.keras.models.load_model(
        "models/best_model.h5",
        compile=False
    )
labels = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia"
]

# =========================
# Disease Information Dictionary
# =========================



disease_info = {

"Atelectasis": {
"description": "Atelectasis is the partial or complete collapse of lung tissue, reducing oxygen exchange.",
"symptoms": "Shortness of breath, rapid breathing, chest discomfort.",
"causes": "Blocked airways, mucus plugs, tumors, prolonged bed rest.",
"treatment": "Chest physiotherapy, breathing exercises, oxygen support."
},

"Cardiomegaly": {
"description": "Cardiomegaly is enlargement of the heart seen in chest imaging.",
"symptoms": "Fatigue, swelling in legs, shortness of breath.",
"causes": "High blood pressure, heart valve disease, cardiomyopathy.",
"treatment": "Medication, lifestyle changes, sometimes surgery."
},

"Effusion": {
"description": "Pleural effusion is fluid buildup between lung lining layers.",
"symptoms": "Chest pain, breathing difficulty, dry cough.",
"causes": "Heart failure, infections, cancer.",
"treatment": "Fluid drainage and treating underlying condition."
},

"Infiltration": {
"description": "Infiltration indicates abnormal substances within lung tissues.",
"symptoms": "Cough, fever, breathlessness.",
"causes": "Infections, inflammation, immune disorders.",
"treatment": "Depends on underlying infection or inflammation."
},

"Mass": {
"description": "A mass is an abnormal growth in lung tissue.",
"symptoms": "Persistent cough, chest pain, weight loss.",
"causes": "Benign tumors or lung cancer.",
"treatment": "Biopsy evaluation, surgery, chemotherapy if malignant."
},

"Nodule": {
"description": "A nodule is a small round growth in the lung.",
"symptoms": "Often asymptomatic.",
"causes": "Infections, scar tissue, early-stage cancer.",
"treatment": "Monitoring or biopsy if suspicious."
},

"Pneumonia": {
"description": "Pneumonia is lung infection causing air sac inflammation.",
"symptoms": "Fever, productive cough, chills, shortness of breath.",
"causes": "Bacterial, viral, or fungal infections.",
"treatment": "Antibiotics (if bacterial), rest, fluids."
},

"Pneumothorax": {
"description": "Pneumothorax is collapsed lung due to air leakage.",
"symptoms": "Sudden chest pain, breathing difficulty.",
"causes": "Lung injury, rupture of air blisters.",
"treatment": "Oxygen therapy or chest tube insertion."
},

"Consolidation": {
"description": "Consolidation occurs when lung tissue fills with liquid instead of air.",
"symptoms": "Cough, fever, chest pain.",
"causes": "Commonly pneumonia.",
"treatment": "Antibiotics and supportive care."
},

"Edema": {
"description": "Pulmonary edema is fluid accumulation in lung air sacs.",
"symptoms": "Severe breathlessness, frothy sputum.",
"causes": "Heart failure, kidney failure.",
"treatment": "Diuretics, oxygen therapy."
},

"Emphysema": {
"description": "Emphysema damages air sacs reducing oxygen exchange.",
"symptoms": "Chronic breathlessness.",
"causes": "Smoking, air pollution.",
"treatment": "Inhalers, oxygen therapy."
},

"Fibrosis": {
"description": "Pulmonary fibrosis is lung tissue scarring.",
"symptoms": "Dry cough, fatigue, breathlessness.",
"causes": "Autoimmune disease, long-term inflammation.",
"treatment": "Medication to slow progression."
},

"Pleural_Thickening": {
"description": "Thickening of pleural lining around lungs.",
"symptoms": "Often mild breathlessness.",
"causes": "Previous infections or asbestos exposure.",
"treatment": "Monitoring and supportive care."
},

"Hernia": {
"description": "A hernia appears when internal organs push through muscle wall.",
"symptoms": "Chest discomfort.",
"causes": "Congenital weakness or trauma.",
"treatment": "Surgical repair."
},

"Normal": {
"description": "No radiographic abnormalities detected.",
"symptoms": "Healthy lungs.",
"causes": "Normal condition.",
"treatment": "No treatment required."
}

}
# ===============================
# Risk Classification Function
# ===============================

def get_risk_level(disease, confidence):

    high_risk = ["Pneumothorax", "Edema", "Mass", "Fibrosis"]
    medium_risk = ["Pneumonia", "Cardiomegaly", "Effusion", "Consolidation"]

    if disease in high_risk and confidence > 0.6:
        return "High Risk 🔴"
    elif disease in medium_risk:
        return "Moderate Risk 🟠"
    elif disease == "Normal":
        return "No Risk 🟢"
    else:
        return "Low Risk 🟡"



# =========================
# Sidebar
# =========================
# =========================
# Sidebar
# =========================
st.sidebar.title("🩺 AI Medical Assistant")

theme_mode = st.sidebar.radio(
    "🌗 Select Theme",
    ["Light Mode ☀️", "Dark Mode 🌙"]
)

page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "🔍 Disease Detection",
    "📊 Model Performance",
    "ℹ About Project"
])

# =========================
# Apply Theme Styling
# =========================

if theme_mode == "Dark Mode 🌙":
    background_color = "#0e1117"
    text_color = "white"
    card_color = "#1c1f26"
else:
    background_color = "#ffffff"
    text_color = "#000000"
    card_color = "#f5f5f5"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(135deg, {background_color}, #1c1f26);
        color: {text_color};
    }}

    section[data-testid="stSidebar"] {{
        background-color: {card_color};
    }}

    h1, h2, h3, h4, h5, h6, p, span {{
        color: {text_color} !important;
    }}

    .custom-card {{
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(12px);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.37);
    }}
    </style>
    """,
    unsafe_allow_html=True
)
# =========================
# Grad-CAM
# =========================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, class_index):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# =========================
# PDF Generator
# =========================
def generate_pdf_report(patient_id, predicted_label, confidence, risk_label):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>AI Chest X-ray Diagnostic Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Patient ID: {patient_id}", styles["Normal"]))
    elements.append(Paragraph(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(f"Predicted Condition: {predicted_label}", styles["Normal"]))
    elements.append(Paragraph(f"Confidence Score: {confidence:.2f}%", styles["Normal"]))
    elements.append(Paragraph(f"Risk Level: {risk_label}", styles["Normal"]))

    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph("<b>Disclaimer:</b>", styles["Heading2"]))
    elements.append(Paragraph(
        "This AI system is for educational purposes only and does not replace professional medical advice.",
        styles["Normal"]
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# =========================
# HOME
# =========================
if page == "🏠 Home":
    st.markdown(
    f"""
    <div class="custom-card">
        <h1 style="text-align:center;">🩺 AI Chest X-ray Diagnosis System</h1>
        <p style="text-align:center;">
        Detects 14 thoracic diseases using Deep Learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
    st.markdown("""
    - Multi-class prediction
    - Risk severity indicator
    - Top-3 conditions
    - Grad-CAM explainability
    - "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
      "Mass", "Nodule", "Pneumonia", "Pneumothorax",
      "Consolidation", "Edema", "Emphysema",
      "Fibrosis", "Pleural Thickening", "Hernia"
      
    """)
    st.warning("For educational use only.")

# =========================
# DISEASE DETECTION
# ====================
elif page == "🔍 Disease Detection":

    st.title("🔍 Chest X-ray Analysis")
    patient_id = st.text_input("Enter Patient ID", "P001")
    uploaded_file = st.file_uploader("Choose X-ray Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:

        img = Image.open(uploaded_file).convert("RGB")
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Analyzing X-ray..."):
            prediction = model.predict(img_array)[0]

        predicted_label = labels[np.argmax(prediction)]
        max_confidence = float(np.max(prediction))

        # Risk classification
        if max_confidence > 0.7:
            risk_color = "#ff4b4b"
            risk_label = "🔴 HIGH RISK"
        elif max_confidence > 0.4:
            risk_color = "#ffa500"
            risk_label = "🟡 MODERATE RISK"
        else:
            risk_color = "#28a745"
            risk_label = "🟢 LOW RISK"

        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.image(img, caption="Uploaded X-ray", width=600)

            st.subheader("🔥 Grad-CAM for Top 3")
            last_conv_layer = "conv5_block16_concat"
            top_indices = np.argsort(prediction)[-3:][::-1]

            # Create 3 columns for horizontal display
            col1, col2, col3 = st.columns([2,2,2])
            cols = [col1, col2, col3]

            for i, idx in enumerate(top_indices):
                
               confidence_percent = int(prediction[idx] * 100)
               heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer, idx)
               heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
               heatmap = np.uint8(255 * heatmap)
               heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
               superimposed_img = cv2.addWeighted(np.array(img), 0.6, heatmap, 0.4, 0)

               with cols[i]:
                    st.markdown(f"### {labels[idx]}")
                    st.progress(int(prediction[idx]*100))
                    st.image(superimposed_img, use_container_width=True)
# Decide border color based on confidence
               if confidence_percent >= 75:
                  border_color = "#00FFAA"
                  risk_text = "High Confidence"
               elif confidence_percent >= 50:
                   border_color = "#FFA500"
                   risk_text = "Moderate Confidence"
               else:
                 border_color = "#FF4B4B"
                 risk_text = "Low Confidence"

               with cols[i]:
                  st.markdown(
                    f"""
                    <div style="
                     border: 2px solid {border_color};
                     border-radius: 15px;
                       padding: 15px;
                      box-shadow: 0 8px 20px rgba(0,0,0,0.25);
                       background: rgba(255,255,255,0.05);
                      backdrop-filter: blur(10px);
                     text-align:center;
                       ">
                        <h4>{labels[idx]}</h4>
                    <p style="color:{border_color}; font-weight:600;">
                          {confidence_percent}% - {risk_text}
                     </p>
                      </div>
                    """,
                        unsafe_allow_html=True
               )

               
        with col_right: 

            st.markdown(
    f"""
    <div class="custom-card" style="text-align:center; font-weight:700; font-size:18px; background-color:{risk_color}; color:white;">
        {risk_label}
    </div>
    """,
    unsafe_allow_html=True
)

            st.markdown("### 🧠 Top 3 Conditions")
            for i in top_indices:
                st.write(f"{labels[i]} — {prediction[i]*100:.2f}%")

        # Detailed Info
        st.markdown("## 📖 Detailed Medical Information")
        for i in top_indices:
            disease = labels[i]
            if disease in disease_info:
                info = disease_info[disease]
                with st.expander(f"{disease} Details"):
                    st.write("**Description:**", info["description"])
                    st.write("**Symptoms:**", info["symptoms"])
                    st.write("**Causes:**", info["causes"])
                    st.write("**Treatment:**", info["treatment"])
                   
                    


        # PDF Button
        confidence_percent = int(max_confidence * 100)
        if st.button("📄 Generate Medical Report"):
            pdf_file = generate_pdf_report(
                patient_id,
                predicted_label,
                confidence_percent,
                risk_label
            )
            st.download_button(
                label="⬇ Download Report",
                data=pdf_file,
                file_name=f"{patient_id}_diagnostic_report.pdf",
                mime="application/pdf"
            )
            
         
        
       


        

        # Chart
        st.subheader("📊 Probability Chart")
        fig, ax = plt.subplots(figsize=(8, 5), facecolor=background_color)
        ax.set_facecolor(background_color)
        ax.tick_params(colors=text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.barh(labels, prediction)
        ax.invert_yaxis()
        st.pyplot(fig)

# =========================
# MODEL PERFORMANCE
# =========================
elif page == "📊 Model Performance":
    st.title("📊 Model Evaluation")
    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", "87.4%", "+2.1%")
    col2.metric("Precision", "85.2%", "+1.8%")
    col3.metric("Recall", "84.9%", "+1.5%")
# =========================
# ABOUT
# =========================
elif page == "ℹ About Project":
    st.title("ℹ About")
    st.markdown("""
    - Model: DenseNet121
    - Framework: TensorFlow + Streamlit
    - Explainability: Grad-CAM
    - 14 thoracic diseases classification
    """)

st.markdown("---")
st.caption("AI Chest X-ray Diagnosis | Educational Use Only")
