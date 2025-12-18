
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
    st.subheader("予測の信頼性分布")
    
    # Create a simple distribution plot of the individual predictions
    preds = [p[0] for p in individual_preds.values()]
    model_names = list(individual_preds.keys())
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=preds, y=model_names, palette="viridis", ax=ax)
    ax.axvline(x=final_pred, color='r', linestyle='--', label=f'Ensemble: {final_pred:.2f}')
    ax.set_xlabel("Predicted SE (D)")
    ax.set_title("Individual Model Predictions vs Ensemble")
    ax.legend()
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
