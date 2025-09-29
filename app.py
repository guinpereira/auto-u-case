# -*- coding: utf-8 -*-
"""
Aplicação Flask para classificar e sugerir respostas a e-mails.

Funcionalidades:
- Suporta múltiplos modos de classificação: 'local', 'openai', 'gemini'.
- Recebe a chave de API diretamente da interface para testes práticos.
- Extrai texto de arquivos .txt e .pdf enviados.
- Apresenta os resultados de forma clara na interface.
"""

# --- Importações de Bibliotecas ---
import os
import io
import re
import json
import traceback
from flask import Flask, request, render_template
import joblib  # Para carregar o modelo de machine learning local
import PyPDF2  # Para ler arquivos PDF

# --- Importações Opcionais para as APIs de IA ---
try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# --- Configuração Inicial do Flask e do Modelo Local ---
app = Flask(__name__)

MODEL_PATH = "models/email_classifier.joblib"
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"Modelo local carregado com sucesso de '{MODEL_PATH}'")
else:
    print(f"Aviso: Modelo local não foi encontrado em '{MODEL_PATH}'. O modo local não funcionará.")


# --- Funções Auxiliares ---

def extract_text_from_pdf(file_stream):
    """Extrai o texto de um objeto de arquivo PDF."""
    try:
        reader = PyPDF2.PdfReader(file_stream)
        text_pages = [page.extract_text() for page in reader.pages if page.extract_text()]
        return "\n".join(text_pages)
    except Exception as e:
        print(f"Erro ao extrair texto do PDF: {e}")
        return ""
    
def detect_order_info(text):
    """
    Detecta número de pedido ou identificadores longos.
    Retorna o número se encontrar, senão None.
    """
    text_low = text.lower()
    m = re.search(r'pedido[:\s-]*([0-9]{4,})', text_low)
    if m:
        return m.group(1)
    m2 = re.search(r'\b([0-9]{6,})\b', text_low)
    if m2:
        return m2.group(1)
    return None


def template_reply(label, text=""):
    """
    Gera resposta automática com heurística para pedidos/nota fiscal.
    """
    text_lower = text.lower()
    order_id = detect_order_info(text)

    if label == "Produtivo":
        if order_id:
            return (f"Olá! Obrigado pelo contato. "
                    f"Identificamos o pedido {order_id} em sua mensagem. "
                    "Estamos verificando e retornaremos com a nota fiscal ou atualização do status em breve.")
        if any(k in text_lower for k in ["nota fiscal", "nota-fiscal", "nf-e", "nota_fiscal"]):
            return ("Olá! Obrigado pelo contato. "
                    "Vamos verificar a nota fiscal solicitada e retornaremos assim que possível.")
        if any(k in text_lower for k in ["cancelamento", "devolução", "reembolso", "assinatura"]):
            return ("Olá! Recebemos sua solicitação. "
                    "Nossa equipe está analisando e retornará em breve.")
        return ("Olá! Recebemos sua solicitação e vamos analisar. "
                "Por favor, confirme o número do seu pedido ou envie mais detalhes.")
    else:
        return ("Olá! Agradecemos a sua mensagem. "
                "Entraremos em contato se for necessária alguma ação. "
                "Tenha um ótimo dia!")



# --- Funções de Classificação com Modelos de IA ---

def classify_with_gemini(text, api_key):
    """Classifica e gera resposta usando a API do Gemini."""
    if genai is None:
        return "Erro", 0.0, "A biblioteca do Google Gemini não está instalada. Rode: pip install google-generativeai"

    try:
        genai.configure(api_key=api_key)

        model_gemini = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={"temperature": 0}
        )

        prompt = (
            "Você é um assistente de e-mail. Classifique o e-mail a seguir em 'Produtivo' ou 'Improdutivo' "
            "e gere uma resposta automática apropriada. Retorne SOMENTE um JSON válido com os campos: "
            "'label' (string), 'confidence' (float entre 0.0 e 1.0), e 'reply' (string). "
            f"E-mail para análise: '''{text}'''"
        )

        response = model_gemini.generate_content(prompt)

        # Garante que o texto da resposta seja capturado corretamente
        if hasattr(response, "text") and response.text:
            content = response.text
        elif hasattr(response, "candidates") and response.candidates:
            content = response.candidates[0].content.parts[0].text
        else:
            raise ValueError("Não foi possível extrair o texto da resposta do Gemini.")

        # Limpa e converte para JSON
        content = content.strip().lstrip("```json").rstrip("```")
        parsed = json.loads(content)

        return parsed.get("label"), float(parsed.get("confidence", 0)), parsed.get("reply", "")

    except Exception as e:
        print(f"Erro na API do Gemini: {e}")
        return "Erro", 0.0, f"Ocorreu um erro ao comunicar com a API do Gemini: {e}"


def classify_with_openai(text, api_key):
    """Classifica e gera resposta usando a API da OpenAI."""
    if openai is None:
        return "Erro", 0.0, "A biblioteca da OpenAI não está instalada. Rode: pip install openai"

    try:
        openai.api_key = api_key

        prompt = (
            "Você é um assistente de e-mail. Classifique o e-mail a seguir em 'Produtivo' ou 'Improdutivo' "
            "e gere uma resposta automática apropriada. Retorne SOMENTE um JSON válido com os campos: "
            "'label' (string), 'confidence' (float entre 0.0 e 1.0), e 'reply' (string). "
            f"E-mail para análise: '''{text}'''"
        )

        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        content = resp['choices'][0]['message']['content']
        parsed = json.loads(content)
        return parsed.get("label"), float(parsed.get("confidence", 0)), parsed.get("reply", "")

    except Exception as e:
        print(f"Erro na API da OpenAI: {e}")
        return "Erro", 0.0, f"Ocorreu um erro ao comunicar com a API da OpenAI: {e}"



def classify_local(text):
    """
    Usa o modelo local + heurística + threshold de confiança.
    """
    if model is None:
        return "Erro", 0.0, "O modelo local não está carregado. Treine o modelo primeiro."

    try:
        proba_prod = float(model.predict_proba([text])[0][1])  # probabilidade da classe 'Produtivo'

        # Heurística: se detectar pedido/nota, aumenta a confiança
        if detect_order_info(text):
            proba_prod = max(proba_prod, 0.85)

        # Thresholds
        if proba_prod >= 0.65:
            label = "Produtivo"
        elif proba_prod >= 0.5:
            # zona de incerteza: trata como improdutivo mas avisa
            label = "Improdutivo"
            print("Aviso: classificação incerta, revisão humana sugerida.")
        else:
            label = "Improdutivo"

        reply = template_reply(label, text)
        return label, proba_prod, reply
    except Exception as e:
        print(f"Erro no modelo local: {e}")
        return "Erro", 0.0, "Ocorreu um erro ao usar o modelo local."



# --- Rotas ---

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    try:
        text_input = request.form.get("text_input", "").strip()
        mode = request.form.get("mode", "local")
        api_key = request.form.get("api_key", "").strip()
        file = request.files.get("file")

        text = text_input
        if not text and file and file.filename:
            filename = file.filename.lower()
            if filename.endswith(".pdf"):
                text = extract_text_from_pdf(io.BytesIO(file.read()))
            elif filename.endswith(".txt"):
                text = file.read().decode("utf-8", errors="ignore")

        if not text:
            return render_template("index.html", error="Nenhum texto ou arquivo válido foi enviado.")

        label, confidence, reply = "", 0.0, ""

        if mode == "local":
            label, confidence, reply = classify_local(text)
        elif mode == "gemini":
            key_to_use = api_key or os.getenv("GEMINI_API_KEY")
            if not key_to_use:
                return render_template("index.html", error="Chave de API do Gemini não fornecida na interface nem encontrada no ambiente.")
            label, confidence, reply = classify_with_gemini(text, key_to_use)
        elif mode == "openai":
            key_to_use = api_key or os.getenv("OPENAI_API_KEY")
            if not key_to_use:
                return render_template("index.html", error="Chave de API da OpenAI não fornecida na interface nem encontrada no ambiente.")
            label, confidence, reply = classify_with_openai(text, key_to_use)
        else:
            return render_template("index.html", error="Modo de operação inválido selecionado.")

        if label == "Erro":
            return render_template("index.html", error=reply, original_text=text)

        return render_template("index.html",
                               original_text=text,
                               result_label=label,
                               confidence=confidence,
                               suggested_reply=reply)

    except Exception as e:
        print(f"Erro geral na rota /process: {e}")
        traceback.print_exc()
        return render_template("index.html", error="Ocorreu um erro inesperado no servidor. Verifique o console para mais detalhes.")


# --- Execução ---
if __name__ == "__main__":
    app.run(debug=True, port=5000)
