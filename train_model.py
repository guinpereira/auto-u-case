# train_model_improved.py
# -*- coding: utf-8 -*-
"""
Script para Treinamento e Avaliação do Modelo de Classificação de E-mails.

Este script utiliza um pipeline de Machine Learning (TF-IDF + Regressão Logística Calibrada)
para classificar e-mails em 'Produtivo' (requer ação) ou 'Improdutivo' (não requer ação).
O modelo final é salvo em 'models/email_classifier.joblib' para ser usado na aplicação Flask (app.py).
"""

# --- Importações de Bibliotecas ---
import os
import joblib           # Para salvar e carregar o modelo treinado (serialização).
import re               # Para expressões regulares (mantido, mas não essencial neste script).
import pandas as pd     # Para manipulação de dados e leitura do arquivo CSV.
import numpy as np      # Para operações numéricas (boa prática em ML).

# Importações NLTK (Natural Language Toolkit) para pré-processamento de texto.
import nltk
from nltk.corpus import stopwords

# Importações Scikit-learn para construção do pipeline de ML.
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV # Para calibrar as probabilidades do modelo.

# --- Configurações Iniciais e Preparação de Dados ---

# Cria o diretório 'models' se ele não existir, para salvar o modelo treinado.
os.makedirs("models", exist_ok=True)

# Baixa os recursos de 'stopwords' do NLTK. 'quiet=True' evita mensagens de console se já estiver baixado.
# Esta linha é crucial para que o vetorizador funcione corretamente.
nltk.download('stopwords', quiet=True)
# Define a lista de stopwords em português a ser usada no pré-processamento.
stop_pt = stopwords.words("portuguese")

# Define o caminho para o arquivo de dados (e-mails).
DATA_PATH = os.path.join("data", "emails.csv") 
# Carrega o dataset de e-mails usando Pandas.
df = pd.read_csv(DATA_PATH)

# Trata valores ausentes (NaN) na coluna 'text', substituindo-os por string vazia.
df['text'] = df['text'].fillna("")

# Mapeia os rótulos de string para binários (0 e 1): 'Produtivo' -> 1, 'Improdutivo' -> 0.
df['label_bin'] = df['label'].map({'Produtivo': 1, 'Improdutivo': 0})

# Remove quaisquer linhas que não puderam ser mapeadas (garante que X e Y sejam válidos).
df = df.dropna(subset=['label_bin'])

# Define as features (X) como o texto do e-mail e o target (y) como o rótulo binário.
X = df['text'].astype(str)
y = df['label_bin'].astype(int)

# --- Divisão dos Dados (Split) ---

# Divide os dados em conjuntos de treino e teste (85% treino, 15% teste).
# 'stratify=y' garante que a proporção das classes (Produtivo/Improdutivo) seja mantida em ambos os conjuntos.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# --- Construção do Pipeline de Machine Learning ---

# 1. Vetorizador (TF-IDF - Term Frequency-Inverse Document Frequency)
vec = TfidfVectorizer(
    ngram_range=(1,2),      # Inclui unigramas e bigramas (palavras e pares de palavras).
    max_features=10000,     # Limita o vocabulário às 10000 palavras/bigramas mais frequentes.
    lowercase=True,         # Converte todo o texto para minúsculas.
    stop_words=stop_pt,     # Remove as stopwords em português.
    token_pattern=r'\b[^\d\W]+\b'  # Padrão de tokenização: ignora tokens que são apenas números.
)

# 2. Classificador Base (Regressão Logística)
base_clf = LogisticRegression(
    max_iter=2000,           # Aumenta o número máximo de iterações para garantir convergência.
    class_weight="balanced", # Pesa as amostras para lidar com possíveis desequilíbrios de classe.
    solver="liblinear"       # Algoritmo de otimização eficiente.
)

# 3. Calibrador de Probabilidade
# Envolve o classificador base para garantir que as saídas 'predict_proba' sejam probabilidades reais.
calibrated = CalibratedClassifierCV(base_estimator=base_clf, cv=5, method='sigmoid')

# 4. Pipeline Completo
# Junta o vetorizador e o classificador calibrado em uma única sequência de processamento.
pipeline = Pipeline([
    ("tfidf", vec),
    ("clf", calibrated)
])

# --- Treinamento e Avaliação ---

print("Treinando modelo (pode demorar dependendo do tamanho dos dados)...")
# Treina o pipeline usando os dados de treino.
pipeline.fit(X_train, y_train)

# Faz previsões (rótulos binários) e extrai as probabilidades da classe positiva (Produtivo).
pred = pipeline.predict(X_test)
proba = pipeline.predict_proba(X_test)[:, 1]  # Probabilidade da classe 'Produtivo' (índice 1).

# Imprime o relatório detalhado de classificação.
print("\nRelatório de classificação:")
print(classification_report(y_test, pred, target_names=["Improdutivo", "Produtivo"]))
print("Acurácia:", accuracy_score(y_test, pred))

# Tenta calcular o ROC AUC, uma métrica importante para modelos binários.
try:
    print("ROC AUC:", roc_auc_score(y_test, proba))
except Exception:
    # Se o cálculo falhar (e.g., apenas uma classe presente no teste), ignora.
    pass
    
# Imprime a Matriz de Confusão para visualizar erros e acertos.
print("Matriz de confusão:\n", confusion_matrix(y_test, pred))

# --- Salvar o Modelo ---

# Salva o pipeline treinado (incluindo vetorizador e classificador) em um arquivo para uso posterior no Flask.
joblib.dump(pipeline, "models/email_classifier.joblib")
print("\nModelo salvo em models/email_classifier.joblib")
