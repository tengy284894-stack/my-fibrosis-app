import shap
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

# ── 风险分层阈值（X-tile计算结果）──────────────────────────────
CUTOFF_LOW  = 0.15    # <15.00%  → 低风险
CUTOFF_HIGH = 0.5108  # ≥51.08% → 高风险
# 15.00% ~ 51.08% → 中风险

def get_risk_group(prob):
    """根据预测概率返回风险分组标签"""
    if prob < CUTOFF_LOW:
        return "低风险 (Low Risk)",  "#4CAF50"   # 绿色
    elif prob < CUTOFF_HIGH:
        return "中风险 (Medium Risk)", "#FF9800"  # 橙色
    else:
        return "高风险 (High Risk)",  "#F44336"   # 红色

# ── 1. 加载模型 ──────────────────────────────────────────────
with open("C:\\Users\\admin\\Desktop\\1.pkl", "rb") as f:
    obj = pickle.load(f)

model    = obj["model"]
features = obj["features"]

# ── 2. 加载数据 ──────────────────────────────────────────────
df = pd.read_excel("C:\\Users\\admin\\Desktop\\验证集完整.xlsx")
X  = df[features]
y  = df["Fibrosis"]

# ── 3. 计算 SHAP（log-odds 空间）────────────────────────────
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)   # shape: (n_samples, n_features, 2)

ev_prob   = explainer.expected_value[1]
shap_prob = shap_values[:, :, 1]

logit = lambda p: np.log(p / (1 - p + 1e-10))

ev_logodds      = logit(ev_prob)
fx_prob_all     = ev_prob + shap_prob.sum(axis=1)
fx_logodds_all  = logit(fx_prob_all)
delta_logodds   = fx_logodds_all - ev_logodds
total_shap_prob = shap_prob.sum(axis=1)

scale = np.where(
    np.abs(total_shap_prob) > 1e-10,
    delta_logodds / total_shap_prob,
    0
)
shap_logodds = shap_prob * scale[:, np.newaxis]

print(f"Base value (log-odds): {ev_logodds:.4f}")
print(f"风险分层阈值: 低风险 < {CUTOFF_LOW*100:.2f}% ≤ 中风险 < {CUTOFF_HIGH*100:.2f}% ≤ 高风险")
print("-" * 60)

# ── 4. 选择样本并绘图 ─────────────────────────────────────────
sample_index = 33   # ← 改这个数字查看不同样本（0 ~ len(df)-1）

sv   = shap_logodds[sample_index]
fx   = ev_logodds + sv.sum()
prob = 1 / (1 + np.exp(-fx))

risk_label, risk_color = get_risk_group(prob)

print(f"样本 {sample_index}")
print(f"  f(x)       = {fx:.4f}")
print(f"  预测概率    = {prob:.3f} ({prob*100:.1f}%)")
print(f"  真实标签    = {y.iloc[sample_index]}")
print(f"  风险分层    = {risk_label}")

# ── 5. 绘制 force plot ────────────────────────────────────────
shap.force_plot(
    ev_logodds,
    sv,
    X.iloc[sample_index],
    matplotlib=True,
    show=False,
    text_rotation=0,
)

fig = plt.gcf()
fig.set_size_inches(20, 4)
plt.subplots_adjust(left=0.02, right=0.98, top=0.75, bottom=0.25)

# 在图顶部添加风险分层标注
fig.text(
    0.5, 0.97,
    f"f(x) = {fx:.4f}  |  预测概率 = {prob*100:.1f}%  |  风险分层：{risk_label}",
    ha="center", va="top",
    fontsize=13, fontweight="bold",
    color=risk_color,
    transform=fig.transFigure
)

plt.savefig("C:\\Users\\admin\\Desktop\\force_plot.svg",
            format="svg", bbox_inches="tight", dpi=150)
plt.show()
print(f"\n图片已保存至桌面 force_plot.svg")


