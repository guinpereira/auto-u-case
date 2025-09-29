# AutoU Email Classifier

Aplicação web que **classifica e sugere respostas automáticas para e-mails** em Português, separando-os em **Produtivo** ou **Improdutivo**.

## ✨ Funcionalidades
- Upload de arquivo `.txt` ou `.pdf` ou inserção de texto direto.
- Classificação usando **modelo local** (TF-IDF + Regressão Logística calibrada).
- Heurística que identifica **número de pedido** e melhora a precisão.
- Resposta automática personalizada com base no conteúdo.

## 🖥️ Demonstração
- **Aplicação hospedada:** `https://<SEU-LINK-DEPLOY>`  
- **Vídeo de apresentação:** `https://youtu.be/<SEU-VIDEO>`

## 🚀 Como rodar localmente

### 1. Clonar repositório
```bash
git clone https://github.com/<SEU_USUARIO>/email-classifier.git
cd email-classifier
