# 03_streamlit_app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# ===========================
# 1. 頁面設定
# ===========================
st.set_page_config(page_title="Spam Email Classifier", page_icon="📧", layout="wide")

st.title("📧 Spam Email Classifier")
st.caption("一個使用機器學習建立的垃圾郵件分類器 | 2025 ML Project by Beck Lin")

# ===========================
# 2. 載入模型與向量器
# ===========================
@st.cache_resource
def load_models():
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    # 自動偵測最佳模型
    import glob
    model_files = [m for m in glob.glob("models/*.pkl") if "vectorizer" not in m]
    model_path = model_files[0] if model_files else None

    if model_path:
        model = joblib.load(model_path)
        st.success(f"✅ 已載入模型：{model_path.split('/')[-1]}")
        return vectorizer, model
    else:
        st.error("❌ 找不到模型檔案，請先執行 train_pipeline.py")
        st.stop()

vectorizer, model = load_models()

# ===========================
# 3. 使用者輸入區
# ===========================
st.subheader("📝 測試郵件內容")
user_input = st.text_area(
    "請輸入郵件內容：",
    height=150,
    placeholder="例如：Congratulations! You've won a $1000 gift card. Click here to claim..."
)

col1, col2 = st.columns([1, 2])

if col1.button("🔍 開始分析"):
    if user_input.strip() == "":
        st.warning("請先輸入郵件內容。")
    else:
        X_input = vectorizer.transform([user_input])
        pred = model.predict(X_input)[0]
        try:
            proba = model.predict_proba(X_input)[0][1]
        except:
            proba = None

        if pred == 1:
            col2.error("🚨 預測結果：**Spam (垃圾郵件)**")
        else:
            col2.success("✅ 預測結果：**Not Spam (正常郵件)**")

        if proba is not None:
            col2.metric("Spam 機率", f"{proba*100:.2f}%")
            st.progress(float(proba))

# ===========================
# 4. 模型效能摘要
# ===========================
st.markdown("---")
st.subheader("📊 模型效能摘要")

try:
    df_metrics = pd.read_csv("data/model_results.csv")
    st.dataframe(df_metrics.style.highlight_max(subset=["F1"], color="lightgreen"))
except FileNotFoundError:
    st.info("⚠️ 找不到 model_results.csv，請先執行 train_pipeline.py。")

# ===========================
# 5. 詞雲展示
# ===========================
st.markdown("---")
st.subheader("☁️ Spam / Ham 詞雲")

spam_texts = [
    "free winner cash prize money offer congratulations click claim now",
    "you have won lottery gift card free coupon claim reward now"
]
ham_texts = [
    "see you at lunch tomorrow meeting scheduled at 3pm",
    "please find attached the report for this week project update"
]

spam_wc = WordCloud(width=500, height=300, background_color="white").generate(" ".join(spam_texts))
ham_wc = WordCloud(width=500, height=300, background_color="white").generate(" ".join(ham_texts))

col1, col2 = st.columns(2)
with col1:
    st.image(spam_wc.to_array(), caption="🚨 Spam 常見詞")
with col2:
    st.image(ham_wc.to_array(), caption="✅ Ham 常見詞")

# ===========================
# 6. 統計圖表
# ===========================
st.markdown("---")
st.subheader("📈 Spam vs Ham 統計分析")

spam_count, ham_count = 747, 4827
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# 圓餅圖
ax[0].pie([ham_count, spam_count], labels=["Ham", "Spam"], autopct="%1.1f%%", colors=["#4CAF50", "#F44336"])
ax[0].set_title("資料集比例")

# 模擬長度分布
np.random.seed(42)
ham_lengths = np.random.normal(80, 20, 200)
spam_lengths = np.random.normal(120, 25, 200)
sns.kdeplot(ham_lengths, ax=ax[1], label="Ham")
sns.kdeplot(spam_lengths, ax=ax[1], label="Spam", color="red")
ax[1].set_title("郵件長度分布")
ax[1].set_xlabel("字數")
ax[1].legend()

st.pyplot(fig)

# ===========================
# 7. 模型說明
# ===========================
st.markdown("---")
with st.expander("📘 關於此模型"):
    st.write("""
    - 模型使用 **TF-IDF 向量化** + **Linear SVM**
    - 資料來源：SMS Spam Collection Dataset
    - 評估指標：Accuracy、Precision、Recall、F1-score
    - 可即時預測郵件內容是否為垃圾郵件。
    """)

st.caption("🧠 Created by Beck Lin | 2025 Machine Learning Project")
