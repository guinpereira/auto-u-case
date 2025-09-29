# 📧 Classificador de E-mails – Case Prático AutoU

Aplicação web em **Python + Flask** que:
1. **Classifica e-mails** em **Produtivo** ou **Improdutivo** usando um modelo local de Machine Learning.
2. **Sugere uma resposta automática** com base na classificação.


## 🖥️ Demonstração
- **Aplicação hospedada:** [https://auto-u-case.onrender.com](https://auto-u-case.onrender.com)  
- **Vídeo de apresentação:** `https://youtu.be/<SEU-VIDEO>`



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


🔮 Melhorias Futuras

Dataset maior e balanceado: incluir mais e-mails reais ou públicos para aumentar acurácia.

Fine-tuning com modelos de linguagem (transformers) para melhor compreensão semântica.

Multi-classe: permitir novas categorias (ex.: Financeiro, Suporte, Comercial).

Feedback loop: permitir que o usuário corrija a classificação e alimente o treino contínuo.

Fila assíncrona: para processar alto volume de e-mails em background.

Melhorar UI/UX.

Implementar modelo de feedback do usuário, conforme o modelo erra ou acerta o usuário aponta e ele já se auto corrige.