# -*- coding: utf-8 -*-
"""
Script para Treinamento e Avaliação do Modelo de Classificação de E-mails.

Este script executa um pipeline de Machine Learning completo:
1.  Lê os dados de 'data/emails.csv'.
2.  Pré-processa o texto (limpeza, TF-IDF).
3.  Treina um modelo de Regressão Logística Calibrada.
4.  Avalia a performance do modelo.
5.  Salva o modelo treinado em 'models/email_classifier.joblib' para uso pela aplicação Flask.

Foi projetado para ser executado de forma robusta em ambientes de build automatizado como o Render.
"""

# --- Importações de Bibliotecas ---
import os
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

def train_and_save_model():
    """
    Função principal que encapsula todo o processo de treinamento e salvamento do modelo.

    Returns:
        bool: True se o treinamento e salvamento foram bem-sucedidos, False caso contrário.
    """
    print("--- Iniciando processo de treinamento do modelo ---")

    # --- 1. Configurações Iniciais e Preparação de Dados ---
    try:
        print("Criando diretório 'models' se necessário...")
        os.makedirs("models", exist_ok=True)
        
        print("Baixando recursos do NLTK (stopwords)...")
        nltk.download('stopwords', quiet=True)
        stop_pt = stopwords.words("portuguese")

        DATA_PATH = os.path.join("data", "emails.csv")
        print(f"Lendo o arquivo de dados de '{DATA_PATH}'...")
        df = pd.read_csv(DATA_PATH)
        
    except FileNotFoundError:
        print(f"\nERRO CRÍTICO: O arquivo de dados '{DATA_PATH}' não foi encontrado.")
        print("Verifique se o arquivo está no local correto e foi commitado no Git.")
        return False
    except Exception as e:
        print(f"\nERRO CRÍTICO inesperado durante a configuração: {e}")
        return False

    print("Pré-processando os dados...")
    df['text'] = df['text'].fillna("")
    df['label_bin'] = df['label'].map({'Produtivo': 1, 'Improdutivo': 0})
    df = df.dropna(subset=['label_bin'])
    
    if len(df) < 20:
        print(f"\nAVISO: O dataset possui apenas {len(df)} amostras. "
              "Isso é muito pouco para um treinamento significativo e pode causar erros.")
        return False
        
    X = df['text'].astype(str)
    y = df['label_bin'].astype(int)

    print("Dividindo os dados em conjuntos de treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # --- 2. Construção do Pipeline de Machine Learning ---
    print("Construindo o pipeline do modelo...")
    vec = TfidfVectorizer(
        ngram_range=(1, 2), max_features=10000, lowercase=True,
        stop_words=stop_pt, token_pattern=r'\b[^\d\W]+\b'
    )
    base_clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="liblinear"
    )
    # Usando cv=3 para ser um pouco mais rápido no build do Render
    # Linha corrigida
    calibrated = CalibratedClassifierCV(estimator=base_clf, cv=3, method='sigmoid')
    
    pipeline = Pipeline([
        ("tfidf", vec), 
        ("clf", calibrated)
    ])
    
    # --- 3. Treinamento e Avaliação ---
    print("Treinando o modelo... (Isso pode levar alguns instantes)")
    pipeline.fit(X_train, y_train)
    
    print("\n--- Avaliação do Modelo (usando amostra de teste) ---")
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    print("\nRelatório de Classificação:")
    print(classification_report(y_test, pred, target_names=["Improdutivo", "Produtivo"]))
    
    print(f"Acurácia: {accuracy_score(y_test, pred):.4f}")
    
    # O cálculo de ROC AUC pode falhar se o conjunto de teste for muito pequeno ou tiver apenas uma classe
    try:
        print(f"ROC AUC: {roc_auc_score(y_test, proba):.4f}")
    except ValueError:
        print("ROC AUC: Não pôde ser calculado (provavelmente apenas uma classe na amostra de teste).")
        
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, pred))
    
    # --- 4. Salvar o Modelo ---
    model_path = "models/email_classifier.joblib"
    print(f"\nSalvando o modelo treinado em '{model_path}'...")
    joblib.dump(pipeline, model_path)
    
    print(f"✅ Modelo salvo com sucesso!")
    return True

# --- Bloco de Execução Principal ---
# Este bloco só será executado quando o script for chamado diretamente
# (ex: `python train_model.py`)
if __name__ == "__main__":
    success = train_and_save_model()
    
    if success:
        print("\nProcesso de treinamento concluído com sucesso.")
    else:
        print("\nO processo de treinamento falhou. Verifique os erros acima.")
        # Sai com um código de erro, o que pode ajudar a falhar o build do Render explicitamente
        exit(1)