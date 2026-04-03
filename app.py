import streamlit as st
import shap
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# 页面配置
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Liver Fibrosis Risk Prediction",
    page_icon="🏥",
    layout="wide",
)

# ─────────────────────────────────────────────
# 风险分层阈值（X-tile 计算结果）
# ─────────────────────────────────────────────
CUTOFF_LOW  = 0.15    # < 15.00%  → Low Risk
CUTOFF_HIGH = 0.5108  # ≥ 51.08% → High Risk

def get_risk_group(prob):
    if prob < CUTOFF_LOW:
        return "🟢 Low Risk",  "#28a745", "low"
    elif prob < CUTOFF_HIGH:
        return "🟠 Medium Risk", "#fd7e14", "medium"
    else:
        return "🔴 High Risk",  "#dc3545", "high"

RECOMMENDATIONS = {
    "low": (
        "The predicted probability of significant liver fibrosis is **low**. "
        "Routine follow-up is recommended. Continue healthy lifestyle habits, "
        "control body weight, and monitor liver function annually."
    ),
    "medium": (
        "The predicted probability of significant liver fibrosis is **moderate**. "
        "Enhanced clinical surveillance is advised. Consider liver elastography "
        "or biopsy for further evaluation, and optimise management of metabolic "
        "risk factors (obesity, dyslipidaemia, insulin resistance)."
    ),
    "high": (
        "The predicted probability of significant liver fibrosis is **high**. "
        "Prompt consultation with a hepatologist is strongly recommended. "
        "Comprehensive assessment including liver biopsy should be considered, "
        "along with aggressive management of underlying MAFLD and comorbidities."
    ),
}

# ─────────────────────────────────────────────
# 加载模型（缓存，只加载一次）
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "1.pkl")
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["features"]

# ─────────────────────────────────────────────
# SHAP log-odds 转换
# ─────────────────────────────────────────────
def compute_shap_logodds(model, X_input):
    logit = lambda p: np.log(np.clip(p, 1e-10, 1-1e-10) /
                             (1 - np.clip(p, 1e-10, 1-1e-10)))
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_input)

    if isinstance(shap_vals, list):
        sv_prob = shap_vals[1][0]
        ev_prob = explainer.expected_value[1]
    else:
        sv_prob = shap_vals[0, :, 1]
        ev_prob = explainer.expected_value[1]

    ev_lo      = logit(ev_prob)
    fx_prob    = ev_prob + sv_prob.sum()
    fx_lo      = logit(fx_prob)
    delta_lo   = fx_lo - ev_lo
    total_prob = sv_prob.sum()
    scale      = delta_lo / total_prob if abs(total_prob) > 1e-10 else 0.0
    shap_lo    = sv_prob * scale
    prob       = 1 / (1 + np.exp(-fx_lo))
    return ev_lo, shap_lo, fx_lo, prob

# ─────────────────────────────────────────────
# Force plot 绘制
# ─────────────────────────────────────────────
def draw_force_plot(ev_lo, shap_lo, X_input, fx, prob, risk_label, risk_color):
    shap.force_plot(
        ev_lo, shap_lo, X_input.iloc[0],
        matplotlib=True, show=False, text_rotation=0,
    )
    fig = plt.gcf()
    fig.set_size_inches(18, 3.5)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.72, bottom=0.28)
    fig.text(
        0.5, 0.97,
        f"f(x) = {fx:.4f}  │  Predicted Probability = {prob*100:.1f}%  │  {risk_label}",
        ha="center", va="top", fontsize=11, fontweight="bold",
        color=risk_color, transform=fig.transFigure,
    )
    return fig

# ═════════════════════════════════════════════
# 特征输入配置（根据你的数据集定义）
# ═════════════════════════════════════════════
FEATURE_CONFIG = {
    "Age":                 dict(label="Age (years)",                        type="number", min=18,    max=90,    default=50,    step=1,    fmt="%.0f"),
    "Gender":              dict(label="Gender (1=Male / 0=Female)",         type="select", options=[1,0],       default=1),
    "BMI":                 dict(label="BMI (kg/m²)",                        type="number", min=15.0,  max=60.0,  default=26.0,  step=0.1,  fmt="%.1f"),
    "Hypertension":        dict(label="Hypertension (1=Yes / 0=No)",        type="select", options=[0,1],       default=0),
    "Diabetes":            dict(label="Diabetes (1=Yes / 0=No)",            type="select", options=[0,1],       default=0),
    "Smoking":             dict(label="Smoking (1=Yes / 0=No)",             type="select", options=[0,1],       default=0),
    "Alcohol":             dict(label="Alcohol use (1=Yes / 0=No)",         type="select", options=[0,1],       default=0),
    "FBG":                 dict(label="Fasting Blood Glucose (mmol/L)",     type="number", min=2.0,   max=30.0,  default=5.5,   step=0.1,  fmt="%.1f"),
    "FastingInsulin":      dict(label="Fasting Insulin (pmol/L)",           type="number", min=1.0,   max=500.0, default=60.0,  step=1.0,  fmt="%.0f"),
    "TotalCholesterol":    dict(label="Total Cholesterol (mmol/L)",         type="number", min=1.0,   max=15.0,  default=4.5,   step=0.1,  fmt="%.1f"),
    "AST_ALT_Ratio":       dict(label="AST/ALT Ratio",                      type="number", min=0.1,   max=10.0,  default=0.9,   step=0.01, fmt="%.2f"),
    "GGT":                 dict(label="GGT (U/L)",                          type="number", min=1.0,   max=500.0, default=35.0,  step=1.0,  fmt="%.0f"),
    "ALP":                 dict(label="ALP (U/L)",                          type="number", min=1.0,   max=500.0, default=80.0,  step=1.0,  fmt="%.0f"),
    "Albumin":             dict(label="Albumin (g/L)",                      type="number", min=10.0,  max=60.0,  default=42.0,  step=0.1,  fmt="%.1f"),
    "Platelet":            dict(label="Platelet (×10⁹/L)",                 type="number", min=10.0,  max=700.0, default=200.0, step=1.0,  fmt="%.0f"),
    "Ferritin":            dict(label="Ferritin (ng/mL)",                   type="number", min=1.0,   max=2000.0,default=150.0, step=1.0,  fmt="%.0f"),
    "CRP":                 dict(label="CRP (mg/L)",                         type="number", min=0.1,   max=200.0, default=2.0,   step=0.1,  fmt="%.1f"),
    "LiverSteatosisGrade": dict(label="Liver Steatosis Grade (0–3)",        type="select", options=[0,1,2,3],   default=1),
    "LiverParenchyma":     dict(label="Liver Parenchyma (0=Normal / 1=Coarse)", type="select", options=[0,1],  default=0),
    "LiverRightLobe":      dict(label="Liver Right Lobe Length (cm)",       type="number", min=8.0,   max=25.0,  default=13.0,  step=0.1,  fmt="%.1f"),
    "LiverLeftLobe":       dict(label="Liver Left Lobe Length (cm)",        type="number", min=3.0,   max=15.0,  default=6.0,   step=0.1,  fmt="%.1f"),
    "SpleenLength":        dict(label="Spleen Length (mm)",                 type="number", min=50.0,  max=200.0, default=100.0, step=0.1,  fmt="%.1f"),
    "SpleenThickness":     dict(label="Spleen Thickness (mm)",              type="number", min=20.0,  max=80.0,  default=35.0,  step=0.1,  fmt="%.1f"),
    "PortalVeinDiameter":  dict(label="Portal Vein Diameter (mm)",          type="number", min=5.0,   max=25.0,  default=11.0,  step=0.1,  fmt="%.1f"),
    "PortalVeinVelocity":  dict(label="Portal Vein Velocity (cm/s)",        type="number", min=5.0,   max=50.0,  default=18.0,  step=0.1,  fmt="%.1f"),
    "GallbladderWall":     dict(label="Gallbladder Wall Thickness (mm)",    type="number", min=1.0,   max=10.0,  default=2.5,   step=0.1,  fmt="%.1f"),
    "Gallstones":          dict(label="Gallstones (1=Yes / 0=No)",          type="select", options=[0,1],       default=0),
    "CommonBileDuct":      dict(label="Common Bile Duct Diameter (mm)",     type="number", min=1.0,   max=15.0,  default=5.0,   step=0.1,  fmt="%.1f"),
    "Ascites":             dict(label="Ascites (1=Yes / 0=No)",             type="select", options=[0,1],       default=0),
    "FIB4":                dict(label="FIB-4 Index",                        type="number", min=0.1,   max=20.0,  default=1.2,   step=0.01, fmt="%.2f"),
}

# ═════════════════════════════════════════════
# 主界面
# ═════════════════════════════════════════════
st.title("🏥 Liver Fibrosis Risk Prediction in MAFLD")
st.markdown(
    "Based on hepatic ultrasound features and clinical indicators. "
    "Enter the patient's values below and click **Submit Prediction**."
)
st.divider()

# 加载模型
try:
    model, features = load_model()
except FileNotFoundError:
    st.error(
        "❌ Model file `1.pkl` not found.  \n"
        "Please place `1.pkl` in the same directory as `app.py` and push to GitHub."
    )
    st.stop()

# ─────────────────────────────────────────────
# 输入表单（每行 3 列）
# ─────────────────────────────────────────────
st.subheader("📋 Please enter the following clinical and ultrasound features:")

active  = [f for f in features if f in FEATURE_CONFIG]
missing = [f for f in features if f not in FEATURE_CONFIG]
if missing:
    st.warning(f"⚠️ Features not configured (defaults to 0): {missing}")

input_values = {}
rows = [active[i:i+3] for i in range(0, len(active), 3)]

for row in rows:
    cols = st.columns(3)
    for col, feat in zip(cols, row):
        cfg = FEATURE_CONFIG[feat]
        with col:
            if cfg["type"] == "number":
                input_values[feat] = st.number_input(
                    cfg["label"],
                    min_value=float(cfg["min"]),
                    max_value=float(cfg["max"]),
                    value=float(cfg["default"]),
                    step=float(cfg["step"]),
                    format=cfg["fmt"],
                    key=feat,
                )
            else:
                input_values[feat] = st.selectbox(
                    cfg["label"],
                    options=cfg["options"],
                    index=cfg["options"].index(cfg["default"]),
                    key=feat,
                )

for feat in features:
    if feat not in input_values:
        input_values[feat] = 0

st.divider()

# ─────────────────────────────────────────────
# 预测按钮
# ─────────────────────────────────────────────
if st.button("🔍  Submit Prediction", type="primary", use_container_width=True):

    X_input = pd.DataFrame([{f: input_values[f] for f in features}])

    # 输入特征表
    st.subheader("📊 Model Input Features")
    st.dataframe(X_input.style.format("{:.3g}"), use_container_width=True)

    st.subheader("🎯 Prediction Result & Explanation")

    with st.spinner("Computing SHAP values..."):
        ev_lo, shap_lo, fx, prob = compute_shap_logodds(model, X_input)

    risk_label, risk_color, risk_key = get_risk_group(prob)

    # 指标卡片
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.metric("Estimated Probability", f"{prob*100:.1f}%")
    with col2:
        st.metric("f(x)  [log-odds]", f"{fx:.4f}")
    with col3:
        st.markdown(
            f"<div style='background:{risk_color};color:white;padding:14px 20px;"
            f"border-radius:10px;font-size:1.25rem;font-weight:bold;text-align:center;"
            f"margin-top:4px'>{risk_label}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # 临床建议
    with st.expander("📌 Clinical Recommendation", expanded=True):
        st.info(RECOMMENDATIONS[risk_key])

    # SHAP Force Plot
    st.markdown("#### SHAP Force Plot")
    st.caption("Red bars push prediction higher (increases fibrosis risk); blue bars push it lower.")
    with st.spinner("Rendering force plot..."):
        fig = draw_force_plot(ev_lo, shap_lo, X_input, fx, prob, risk_label, risk_color)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

   
