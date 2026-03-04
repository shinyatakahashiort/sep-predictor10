
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from predictor import SEPredictor

# Page config
st.set_page_config(
    page_title="小児屈折予測 AI",
    page_icon="👁️",
    layout="wide"
)

# Initialize predictor (cached to reload only when needed)
@st.cache_resource
def get_predictor():
    return SEPredictor(model_dir='.')

try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました: {e}")
    st.stop()

# Title and Description
st.title("👁️ 小児の術後屈折予測 AI")
st.markdown("""
このアプリケーションは、小児白内障手術後の屈折値（SE: Spherical Equivalent）を予測するAIツールです。
**MLP**, **ExtraTrees**, **CatBoost** の3つの機械学習モデルのアンサンブルにより、高精度な予測を提供します。
""")

# Sidebar for Inputs
st.sidebar.header("患者データの入力")

def user_input_features():
    age = st.sidebar.number_input("年齢 (歳)", min_value=0, max_value=20, value=7, step=1)

    gender_label = st.sidebar.radio("性別", ("男性", "女性"))
    gender = 0 if gender_label == "女性" else 1 # Assuming 0=F, 1=M based on training

    # Biometric parameters
    st.sidebar.markdown("### 生体計測値")
    # Based on training stats, K is around 7.72, likely radius in mm
    k_mm = st.sidebar.number_input("角膜曲率半径 K (mm)", min_value=6.0, max_value=10.0, value=7.72, step=0.01)

    al = st.sidebar.number_input("眼軸長 AL (mm)", min_value=15.0, max_value=35.0, value=24.0, step=0.1)
    lt = st.sidebar.number_input("水晶体厚 LT (mm)", min_value=2.0, max_value=6.0, value=3.5, step=0.01)
    acd = st.sidebar.number_input("前房深度 ACD (mm)", min_value=1.5, max_value=6.0, value=3.75, step=0.01)

    data = {
        '年齢': age,
        '性別': gender,
        'K': k_mm,
        'AL': al,
        'LT': lt,
        'ACD': acd
    }
    return data

input_data = user_input_features()

# Display Input Data
st.header("1. 入力確認")
input_df = pd.DataFrame([input_data])
# Show clearer labels for display
display_df = input_df.copy()
display_df['性別'] = display_df['性別'].map({0: '女性', 1: '男性'})
st.dataframe(display_df)

# Prediction Button
if st.button("予測を実行"):
    st.header("2. 予測結果")

    with st.spinner('予測中...'):
        ensemble_pred, individual_preds = predictor.predict(input_data)

    final_pred = ensemble_pred[0]

    # Main Result
    st.success(f"### 予測 術後等価球面度数 (SE): {final_pred:.2f} D")

    # Detailed Breakdown
    st.subheader("モデル別予測内訳")
    cols = st.columns(len(individual_preds))
    for i, (name, pred) in enumerate(individual_preds.items()):
        with cols[i]:
            st.metric(label=name, value=f"{pred[0]:.2f} D")

    # Visualization
    st.subheader("アンサンブル予測の内訳")

    display_names = {'mlp': 'MLP', 'extratrees': 'ExtraTrees', 'catboost': 'CatBoost'}
    preds_values = [p[0] for p in individual_preds.values()]
    model_keys = list(individual_preds.keys())
    model_labels = [display_names.get(k, k) for k in model_keys]
    weights = predictor.weights
    weight_values = [weights.get(k, 1 / len(model_keys)) for k in model_keys]
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4),
                                    gridspec_kw={'width_ratios': [3, 1]})
    fig.subplots_adjust(wspace=0.35)

    # --- Left panel: individual predictions ---
    ax1.barh(model_labels, preds_values, color=colors, alpha=0.75, height=0.5)

    x_min, x_max = ax1.get_xlim()
    span = x_max - x_min if x_max != x_min else 1
    for i, (label, pred, weight) in enumerate(zip(model_labels, preds_values, weight_values)):
        offset = span * 0.02
        x_pos = pred + offset
        ax1.text(x_pos, i, f'{pred:.2f} D  (weight {weight:.0%})',
                 va='center', ha='left', fontsize=10, fontweight='bold')

    ax1.axvline(x=final_pred, color='crimson', linestyle='--', linewidth=2.5,
                label=f'Ensemble: {final_pred:.2f} D', zorder=5)
    ax1.set_xlabel('Predicted SE (D)', fontsize=11)
    ax1.set_title('Individual Model Predictions vs Ensemble', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.tick_params(axis='y', labelsize=11)

    # --- Right panel: model weights ---
    left = 0
    for label, weight, color in zip(model_labels, weight_values, colors):
        ax2.barh(['Model Weights'], [weight], left=left, color=color,
                 alpha=0.85, label=label, height=0.4)
        if weight > 0.05:
            ax2.text(left + weight / 2, 0, f'{label}\n{weight:.0%}',
                     va='center', ha='center', fontsize=9,
                     fontweight='bold', color='white')
        left += weight

    ax2.set_xlim(0, 1)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=9)
    ax2.set_title('Model Weights', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelleft=False)

    st.pyplot(fig)

    # Interpretation Note
    st.info("""
    **注釈:**
    * 予測値は術後の屈折誤差の目安です。
    * 機械学習モデルは過去のデータセットに基づいて予測を行っています。
    * **CatBoost, MLP, ExtraTrees** の加重平均を使用しています。
    """)

# Footer
st.markdown("---")
st.markdown("Developed with Streamlit and Python.")
