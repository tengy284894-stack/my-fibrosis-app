import streamlit as st
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── 加载模型 ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("1.pkl", "rb") as f:
        obj = pickle.load(f)
    return obj

obj = load_model()
model    = obj["model"]
features = obj["features"]
shap_rel = obj["shap_relative_contribution"]
cv_mean  = obj["cv_auc_mean"]
cv_std   = obj["cv_auc_std"]

# ── 页面设置 ──────────────────────────────────────────────
st.set_page_config(page_title="肝纤维化风险预测", page_icon="🏥", layout="centered")
st.title("🏥 肝纤维化风险预测模型")
st.caption(f"基于随机森林分类器 | 交叉验证 AUC: {cv_mean:.3f} ± {cv_std:.3f}")

# ── 输入表单 ──────────────────────────────────────────────
st.subheader("请输入以下临床特征：")

col1, col2 = st.columns(2)
with col1:
    age          = st.number_input("Age（年龄）",                    min_value=0.0,   max_value=120.0,  value=50.0,  step=1.0)
    ast_alt      = st.number_input("AST_ALT_Ratio（AST/ALT比值）",   min_value=0.0,   max_value=20.0,   value=1.0,   step=0.01)
    spleen_thick = st.number_input("SpleenThickness（脾脏厚度 mm）", min_value=0.0,   max_value=200.0,  value=35.0,  step=0.1)
    spleen_len   = st.number_input("SpleenLength（脾脏长度 mm）",    min_value=0.0,   max_value=300.0,  value=100.0, step=0.1)

with col2:
    platelet  = st.number_input("Platelet（血小板 ×10⁹/L）", min_value=0.0, max_value=1000.0, value=200.0, step=1.0)
    ferritin  = st.number_input("Ferritin（铁蛋白 ng/mL）",  min_value=0.0, max_value=5000.0, value=100.0, step=1.0)
    liver_par = st.number_input("LiverParenchyma（肝实质）", min_value=0.0, max_value=10.0,   value=1.0,   step=0.1)

# ── 预测按钮 ──────────────────────────────────────────────
if st.button("🔍 开始预测", use_container_width=True):
    input_values = {
        "AST_ALT_Ratio":   ast_alt,
        "SpleenThickness": spleen_thick,
        "SpleenLength":    spleen_len,
        "Age":             age,
        "Platelet":        platelet,
        "Ferritin":        ferritin,
        "LiverParenchyma": liver_par,
    }
    X = np.array([[input_values[f] for f in features]])
    prob = model.predict_proba(X)[0][1]

    # ── 结果展示 ──
    st.subheader("📊 预测结果与解释")
    st.metric("肝纤维化估计概率", f"{prob*100:.1f}%")

    if prob >= 0.3:
        st.error("⚠️ 您当前处于**高风险**肝纤维化状态，建议尽快就医进行进一步检查。")
    else:
        st.success("✅ 您当前处于**低风险**肝纤维化状态，请继续保持健康生活方式并定期复查。")

   # ── SHAP 力图（Force Plot）──
    st.subheader("🔬 各特征对预测的贡献（SHAP）")

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    shap_arr  = np.array(shap_vals)

    # 取 class=1 的 SHAP 值和期望值
    if shap_arr.ndim == 3:
        if shap_arr.shape[0] == X.shape[0]:   # (1,7,2)
            sv = shap_arr[0, :, 1]
            ev = float(np.array(explainer.expected_value)[1])
        else:                                   # (2,1,7)
            sv = shap_arr[1, 0, :]
            ev = float(np.array(explainer.expected_value)[1])
    elif shap_arr.ndim == 2:
        sv = shap_arr[0, :]
        ev = float(np.array(explainer.expected_value).flat[1])
    else:
        sv = shap_arr
        ev = float(np.array(explainer.expected_value).flat[0])

    sv = sv.flatten().astype(float)

    # 画力图
    import pandas as pd
    sample_display = pd.Series(
        {features[i]: float(X[0][i]) for i in range(len(features))}
    )

    shap.force_plot(
        ev, sv, sample_display,
        matplotlib=True, show=False
    )
    fig = plt.gcf()
    fig.set_size_inches(20, 4)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.25)
    st.pyplot(fig, use_container_width=True)
    plt.close()

 
