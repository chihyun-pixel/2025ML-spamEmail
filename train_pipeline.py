"""
train_pipeline.py
整合版訓練腳本：
自動完成資料下載、預處理、模型訓練、成果視覺化與結果輸出
"""

import os
import re
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

# ===============================
# 0. NLTK 初始化
# ===============================
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ===============================
# 1. 下載資料
# ===============================
print("📥 載入資料中...")
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
print(f"✅ 成功載入資料，共 {df.shape[0]} 筆樣本")

# ===============================
# 2. 文字清理
# ===============================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

print("🧹 清理文字中...")
df['clean_text'] = df['message'].apply(clean_text)

# ===============================
# 3. 向量化
# ===============================
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_text']).toarray()
y = np.where(df['label'] == 'spam', 1, 0)

print("🔠 向量化完成，特徵維度：", X.shape)

# ===============================
# 4. 資料分割
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 5. 模型訓練
# ===============================
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results = []

print("🤖 開始模型訓練...\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
    except:
        roc = np.nan

    results.append([name, acc, prec, rec, f1, roc])
    print(f"{name}: F1={f1:.4f}, Accuracy={acc:.4f}")

# ===============================
# 6. 選出最佳模型
# ===============================
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"])
best_row = results_df.loc[results_df['F1'].idxmax()]
best_model_name = best_row['Model']
best_model = models[best_model_name]

print("\n🏆 最佳模型:", best_model_name)
print(results_df)

# ===============================
# 7. 儲存成果
# ===============================
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 儲存處理後資料、模型、向量器
joblib.dump((X_train, X_test, y_train, y_test), "data/processed_spam_data.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(best_model, f"models/{best_model_name.replace(' ', '_')}.pkl")

# 儲存模型結果表格供 Streamlit 使用
results_df.to_csv("data/model_results.csv", index=False)

print("\n💾 模型與資料已儲存完成！")
print("📁 data/processed_spam_data.pkl")
print("📁 data/model_results.csv")
print("📁 models/tfidf_vectorizer.pkl")
print(f"📁 models/{best_model_name.replace(' ', '_')}.pkl")

# ===============================
# 8. 混淆矩陣視覺化
# ===============================
cm = confusion_matrix(y_test, best_model.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"{best_model_name} Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\n✅ 訓練流程完成！可直接用於 Streamlit Demo。")
