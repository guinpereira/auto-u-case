# 📧 Classificador de E-mails – Case Prático AutoU

Aplicação web em **Python + Flask** que:
1. **Classifica e-mails** em **Produtivo** ou **Improdutivo** usando um modelo local de Machine Learning.
2. **Sugere uma resposta automática** com base na classificação.

> **Demo online:** [adicione_aqui_o_link_do_deploy]  
> **Vídeo de apresentação:** [adicione_aqui_o_link_do_video]



## 🚀 Tecnologias
- **Python 3.10+**
- Flask
- scikit-learn (Logistic Regression + TF-IDF)
- TailwindCSS (interface)
- joblib / pandas / nltk


## 💡 Funcionalidades
- Upload de arquivo `.txt` ou `.pdf` **ou** texto colado manualmente.
- Escolha entre **modelo local** ou APIs (OpenAI/Gemini) para classificação.
- Exibição da **confiança (%)** da predição.
- **Resposta automática** pronta para copiar.


## 🛠️ Como Rodar Localmente

# 1. Clonar o repositório
git clone https://github.com/<seu_usuario>/<nome_repo>.git
cd <nome_repo>

# 2. Criar ambiente virtual
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Treinar o modelo local (necessário antes de rodar o app)
python train_model_improved.py

# 5. Iniciar o servidor
python app.py


🔮 Melhorias Futuras

Dataset maior e balanceado: incluir mais e-mails reais ou públicos para aumentar acurácia.

Fine-tuning com modelos de linguagem (transformers) para melhor compreensão semântica.

Multi-classe: permitir novas categorias (ex.: Financeiro, Suporte, Comercial).

Feedback loop: permitir que o usuário corrija a classificação e alimente o treino contínuo.

Fila assíncrona: para processar alto volume de e-mails em background.