# train_model_improved.py
import os
import joblib
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# configurações
os.makedirs("models", exist_ok=True)
nltk.download('stopwords', quiet=True)
stop_pt = stopwords.words("portuguese")

DATA_PATH = os.path.join("data", "emails.csv")  # ajuste se necessário
df = pd.read_csv(DATA_PATH)
df['text'] = df['text'].fillna("")
# mapear rótulos (ajuste se seus rótulos forem diferentes)
df['label_bin'] = df['label'].map({'Produtivo': 1, 'Improdutivo': 0})
df = df.dropna(subset=['label_bin'])

X = df['text'].astype(str)
y = df['label_bin'].astype(int)

# split estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# vetor + classificador
vec = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=10000,
    lowercase=True,
    stop_words=stop_pt,
    token_pattern=r'\b[^\d\W]+\b'  # ignora tokens apenas numéricos
)

base_clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
calibrated = CalibratedClassifierCV(base_estimator=base_clf, cv=5, method='sigmoid')

pipeline = Pipeline([
    ("tfidf", vec),
    ("clf", calibrated)
])

print("Treinando modelo (pode demorar dependendo do tamanho dos dados)...")
pipeline.fit(X_train, y_train)

# previsões e probabilidades calibradas
pred = pipeline.predict(X_test)
proba = pipeline.predict_proba(X_test)[:, 1]  # probabilidade da classe 'Produtivo'

print("\nRelatório de classificação:")
print(classification_report(y_test, pred, target_names=["Improdutivo", "Produtivo"]))
print("Acurácia:", accuracy_score(y_test, pred))
try:
    print("ROC AUC:", roc_auc_score(y_test, proba))
except Exception:
    pass
print("Matriz de confusão:\n", confusion_matrix(y_test, pred))

# salvar
joblib.dump(pipeline, "models/email_classifier.joblib")
print("\nModelo salvo em models/email_classifier.joblib")
