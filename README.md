# 📧 Classificador de E-mails – Case Prático AutoU

Aplicação web em **Python + Flask** que:
1. **Classifica e-mails** em **Produtivo** ou **Improdutivo** usando um modelo local de Machine Learning.
2. **Sugere uma resposta automática** com base na classificação.


## 🖥️ Demonstração
- **Aplicação hospedada:** [https://auto-u-case.onrender.com](https://auto-u-case.onrender.com)  
- **Vídeo de apresentação:** `https://youtu.be/BjhwxLOMIz8`



## 🚀 Tecnologias
- **Python 3.10+**
- Flask
- scikit-learn (Logistic Regression + TF-IDF)
- TailwindCSS (interface)
- joblib / pandas / nltk


## ✨ Funcionalidades
- Upload de arquivo `.txt` ou `.pdf` ou inserção de texto direto.
- Classificação usando **modelo local** (TF-IDF + Regressão Logística calibrada).
- Heurística que identifica **número de pedido** e melhora a precisão.
- Resposta automática personalizada com base no conteúdo.


## 🛠️ Como Rodar Localmente

# 1. Clonar o repositório
git clone https://github.com/guinpereira/auto-u-case
cd auto-u-case

# 2. Criar ambiente virtual
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Treinar o modelo local (necessário antes de rodar o app)
python train_model_improved.py

# 5. Iniciar o servidor/rodar aplicação
python app.py


## ☁️ Como Fazer o Deploy (Render)

A aplicação está hospedada no Render e configurada para deploy contínuo a partir da branch `main`. As configurações utilizadas no serviço do Render são:

-   **Build Command**: `pip install -r requirements.txt && python train_model_improved.py`
-   **Start Command**: `gunicorn app:app`
-   **Variáveis de Ambiente**: Para uso das APIs, as chaves `OPENAI_API_KEY` e `GEMINI_API_KEY` devem ser configuradas diretamente no ambiente do Render para maior segurança.

**Observação Importante:** O comando de build garante que, a cada novo deploy, as dependências sejam instaladas e o modelo de machine learning seja treinado novamente, garantindo que o arquivo `email_classifier.joblib` esteja sempre presente e atualizado.


## 🔒 Segurança (uso de APIs)

Atualmente a aplicação permite informar a chave da API (OpenAI ou Gemini) diretamente pela interface.

👉 Recomendação para produção:

Remover o campo de chave da interface.

Configurar as variáveis de ambiente do sistema:

OPENAI_API_KEY

GEMINI_API_KEY

Dessa forma, as chaves ficam protegidas e não precisam ser informadas manualmente.


## 🔮 Melhorias Futuras

Fine-tuning com modelos de linguagem (transformers) para melhor compreensão semântica.

Dataset maior e balanceado: incluir mais e-mails reais ou públicos para aumentar acurácia.

Melhorar UI/UX.

Usar um conjunto maior e mais balanceado de e-mails reais ou públicos.

Criar um pipeline para coletar e-mails já classificados (Produtivo/Improdutivo).

Remover a opção de inserir chave API na interface.

Usar somente variáveis de ambiente para armazenar credenciais.

Multi-classe: permitir novas categorias (ex.: Financeiro, Suporte, Comercial).

Analisar outros modelos como BERT, DistilBERT ou similares em relação à compreensão semântica.

Feedback Loop

usuários poderiam corrigir classificações e alimentar o modelo para treino contínuo.

Fila assíncrona

Processamento em background para alto volume de e-mails (ex.: Celery + Redis).