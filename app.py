# -*- coding: utf-8 -*-
"""
Módulo Principal da Aplicação Flask para Classificação de E-mails.

Esta aplicação web permite que o usuário insira o conteúdo de um e-mail 
ou carregue um arquivo (.txt ou .pdf) para classificação. 
O processo de classificação pode ser executado em três modos:
1. 'local': Usando um modelo de Machine Learning treinado previamente (e-mail_classifier.joblib).
2. 'openai': Usando a API GPT (necessita de chave).
3. 'gemini': Usando a API Gemini (necessita de chave).

O resultado inclui o rótulo da classificação ('Produtivo' ou 'Improdutivo'),
o nível de confiança e uma sugestão de resposta automática.
"""

# --- Importações de Bibliotecas Essenciais ---
import os           # Para interagir com o sistema operacional (ex: variáveis de ambiente, caminhos de arquivo)
import io           # Para trabalhar com streams de I/O em memória (necessário para ler arquivos enviados)
import re           # Para expressões regulares (usado na detecção de números de pedido/nota)
import json         # Para manipulação de dados JSON (usado para comunicar com as APIs de IA)
import traceback    # Para imprimir o stack trace de erros (útil para debug)
from flask import Flask, request, render_template # Framework web principal e suas funções

# Biblioteca para carregar o modelo de machine learning local (serialização/desserialização)
import joblib 
# Para ler arquivos PDF e extrair seu texto
import PyPDF2

# --- Importações Opcionais para as APIs de IA ---
# As importações são envolvidas em try/except para que a aplicação funcione
# mesmo que as bibliotecas de IA não estejam instaladas (o modo 'local' ainda será funcional).
try:
    import openai
except ImportError:
    openai = None # Se falhar, a variável é setada para None, desabilitando o modo 'openai'.

try:
    import google.generativeai as genai
except ImportError:
    genai = None # Se falhar, a variável é setada para None, desabilitando o modo 'gemini'.

# --- Configuração Inicial do Flask e do Modelo Local ---
# Inicializa a aplicação Flask.
app = Flask(__name__)

# Define o caminho para o arquivo do modelo de ML local.
MODEL_PATH = "models/email_classifier.joblib"
# Variável para armazenar o modelo carregado.
model = None

# Verifica se o arquivo do modelo existe no caminho especificado.
if os.path.exists(MODEL_PATH):
    try:
        # Carrega o modelo de ML usando joblib.
        model = joblib.load(MODEL_PATH)
        print(f"Modelo local carregado com sucesso de '{MODEL_PATH}'")
    except Exception as e:
        print(f"Erro ao carregar o modelo local: {e}")
        model = None
else:
    # Aviso caso o modelo não seja encontrado.
    print(f"Aviso: Modelo local não foi encontrado em '{MODEL_PATH}'. O modo local não funcionará.")


# --- Funções Auxiliares de Processamento e Heurística ---

def extract_text_from_pdf(file_stream):
    """
    Extrai o texto contido em um objeto de arquivo PDF (stream de bytes).

    Args:
        file_stream (io.BytesIO): Stream de bytes do arquivo PDF.

    Returns:
        str: Todo o texto extraído do PDF, ou uma string vazia em caso de erro.
    """
    try:
        # Cria um objeto leitor de PDF a partir do stream.
        reader = PyPDF2.PdfReader(file_stream)
        # Itera sobre todas as páginas e extrai o texto, juntando-o em uma única string.
        text_pages = [page.extract_text() for page in reader.pages if page.extract_text()]
        return "\n".join(text_pages)
    except Exception as e:
        # Loga o erro de extração e retorna uma string vazia.
        print(f"Erro ao extrair texto do PDF: {e}")
        return ""
    
def detect_order_info(text):
    """
    Detecta números de pedido, nota fiscal ou identificadores longos no texto.
    
    A presença de tais números é uma forte heurística para um e-mail 'Produtivo' (que requer ação).

    Args:
        text (str): O corpo do e-mail.

    Returns:
        str or None: O número de pedido encontrado ou None se nenhum for detectado.
    """
    text_low = text.lower()
    
    # Padrão 1: Busca por 'pedido' seguido opcionalmente por : ou - e, em seguida, pelo menos 4 dígitos.
    m = re.search(r'pedido[:\s-]*([0-9]{4,})', text_low)
    if m:
        return m.group(1) # Retorna o grupo de captura (os dígitos do pedido).
        
    # Padrão 2: Busca por uma sequência isolada de 6 ou mais dígitos (pode ser um ID de rastreio ou pedido).
    m2 = re.search(r'\b([0-9]{6,})\b', text_low)
    if m2:
        return m2.group(1)
        
    return None # Retorna None se nenhum padrão for encontrado.


def template_reply(label, text=""):
    """
    Gera uma resposta automática padronizada com base no rótulo de classificação
    e ajustada por heurística para e-mails 'Produtivos'.

    Args:
        label (str): Rótulo de classificação ('Produtivo' ou 'Improdutivo').
        text (str): O corpo do e-mail original para análise de palavras-chave.

    Returns:
        str: A sugestão de resposta.
    """
    text_lower = text.lower()
    order_id = detect_order_info(text)

    # Lógica para e-mails que requerem ação (Produtivo)
    if label == "Produtivo":
        # Caso 1: Se um número de pedido foi detectado.
        if order_id:
            return (f"Olá! Obrigado pelo contato. "
                    f"Identificamos o pedido {order_id} em sua mensagem. "
                    "Estamos verificando e retornaremos com a nota fiscal ou atualização do status em breve.")
        
        # Caso 2: Se palavras-chave de nota fiscal forem encontradas.
        if any(k in text_lower for k in ["nota fiscal", "nota-fiscal", "nf-e", "nota_fiscal"]):
            return ("Olá! Obrigado pelo contato. "
                    "Vamos verificar a nota fiscal solicitada e retornaremos assim que possível.")
        
        # Caso 3: Se palavras-chave de cancelamento/devolução forem encontradas.
        if any(k in text_lower for k in ["cancelamento", "devolução", "reembolso", "assinatura"]):
            return ("Olá! Recebemos sua solicitação. "
                    "Nossa equipe está analisando e retornará em breve.")
        
        # Caso 4: Resposta Produtiva genérica (se nenhuma heurística específica se aplicar).
        return ("Olá! Recebemos sua solicitação e vamos analisar. "
                "Por favor, confirme o número do seu pedido ou envie mais detalhes.")
    
    # Lógica para e-mails que não requerem ação imediata (Improdutivo)
    else:
        return ("Olá! Agradecemos a sua mensagem. "
                "Entraremos em contato se for necessária alguma ação. "
                "Tenha um ótimo dia!")



# --- Funções de Classificação com Modelos de IA ---

def classify_with_gemini(text, api_key):
    """
    Classifica o e-mail e gera resposta usando a API do Gemini.
    
    O modelo é instruído a retornar um JSON estruturado para facilitar o parsing.

    Args:
        text (str): O corpo do e-mail.
        api_key (str): Chave de API do Google Gemini.

    Returns:
        tuple: (label, confidence, reply) ou ("Erro", 0.0, mensagem de erro).
    """
    if genai is None:
        return "Erro", 0.0, "A biblioteca do Google Gemini não está instalada. Rode: pip install google-generativeai"

    try:
        # Configura a chave de API para a sessão.
        genai.configure(api_key=api_key)

        # Inicializa o modelo Gemini (gemini-2.5-flash) com baixa temperatura para respostas mais determinísticas.
        model_gemini = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={"temperature": 0}
        )

        # Prompt de engenharia: Define o papel do assistente e a estrutura de saída JSON.
        prompt = (
            "Você é um assistente de e-mail. Classifique o e-mail a seguir em 'Produtivo' ou 'Improdutivo' "
            "e gere uma resposta automática apropriada. Retorne SOMENTE um JSON válido com os campos: "
            "'label' (string), 'confidence' (float entre 0.0 e 1.0), e 'reply' (string). "
            f"E-mail para análise: '''{text}'''"
        )

        # Faz a chamada à API.
        response = model_gemini.generate_content(prompt)

        # Trata as diferentes formas de extrair o texto da resposta (por segurança).
        if hasattr(response, "text") and response.text:
            content = response.text
        elif hasattr(response, "candidates") and response.candidates:
            content = response.candidates[0].content.parts[0].text
        else:
            raise ValueError("Não foi possível extrair o texto da resposta do Gemini.")

        # Limpa o texto da resposta (remove '```json' e '```' que a IA pode incluir) e converte para JSON.
        content = content.strip().lstrip("```json").rstrip("```")
        parsed = json.loads(content)

        # Extrai os valores do JSON e garante que a confiança seja um float.
        return parsed.get("label"), float(parsed.get("confidence", 0)), parsed.get("reply", "")

    except Exception as e:
        print(f"Erro na API do Gemini: {e}")
        # Retorna erro amigável em caso de falha na API.
        return "Erro", 0.0, f"Ocorreu um erro ao comunicar com a API do Gemini: {e}"


def classify_with_openai(text, api_key):
    """
    Classifica o e-mail e gera resposta usando a API da OpenAI (GPT).

    Args:
        text (str): O corpo do e-mail.
        api_key (str): Chave de API da OpenAI.

    Returns:
        tuple: (label, confidence, reply) ou ("Erro", 0.0, mensagem de erro).
    """
    if openai is None:
        return "Erro", 0.0, "A biblioteca da OpenAI não está instalada. Rode: pip install openai"

    try:
        # Configura a chave de API.
        openai.api_key = api_key

        # Prompt de engenharia: Semelhante ao Gemini, define o papel e a estrutura JSON.
        prompt = (
            "Você é um assistente de e-mail. Classifique o e-mail a seguir em 'Produtivo' ou 'Improdutivo' "
            "e gere uma resposta automática apropriada. Retorne SOMENTE um JSON válido com os campos: "
            "'label' (string), 'confidence' (float entre 0.0 e 1.0), e 'reply' (string). "
            f"E-mail para análise: '''{text}'''"
        )

        # Faz a chamada à API de Chat Completion.
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0 # Baixa temperatura para resultados mais consistentes.
        )

        # Extrai o conteúdo da resposta e converte para JSON.
        content = resp['choices'][0]['message']['content']
        parsed = json.loads(content)
        
        # Extrai e retorna os resultados.
        return parsed.get("label"), float(parsed.get("confidence", 0)), parsed.get("reply", "")

    except Exception as e:
        print(f"Erro na API da OpenAI: {e}")
        # Retorna erro amigável em caso de falha na API.
        return "Erro", 0.0, f"Ocorreu um erro ao comunicar com a API da OpenAI: {e}"


def classify_local(text):
    """
    Usa o modelo de Machine Learning local para classificação,
    aplicando heurísticas e thresholds de confiança.

    Args:
        text (str): O corpo do e-mail.

    Returns:
        tuple: (label, confidence, reply) ou ("Erro", 0.0, mensagem de erro).
    """
    if model is None:
        # Verifica se o modelo foi carregado com sucesso na inicialização.
        return "Erro", 0.0, "O modelo local não está carregado. Treine o modelo primeiro."

    try:
        # Calcula a probabilidade de pertencer à classe 'Produtivo'.
        # Assume que 'Produtivo' é a segunda classe (índice 1).
        proba_prod = float(model.predict_proba([text])[0][1])

        # Heurística de Reforço: Se detectar um número de pedido/nota, 
        # a confiança na classe 'Produtivo' é elevada, se já for alta.
        if detect_order_info(text):
            # Aumenta a probabilidade para, no mínimo, 85%.
            proba_prod = max(proba_prod, 0.85)

        # Aplica os Thresholds (Limiares) de Decisão.
        if proba_prod >= 0.65:
            # Alta confiança em Produtivo.
            label = "Produtivo"
            confidence = proba_prod
        elif proba_prod >= 0.5:
            # Zona de Incerteza (entre 50% e 65%): assume Improdutivo por precaução, mas avisa.
            label = "Improdutivo"
            confidence = proba_prod
            print("Aviso: classificação incerta, revisão humana sugerida.")
        else:
            # Baixa confiança em Produtivo (assume Improdutivo).
            label = "Improdutivo"
            confidence = proba_prod
            
        # Gera a resposta automática usando a função auxiliar.
        reply = template_reply(label, text)
        return label, confidence, reply
        
    except Exception as e:
        print(f"Erro no modelo local: {e}")
        return "Erro", 0.0, "Ocorreu um erro ao usar o modelo local."



# --- Rotas da Aplicação Flask ---

@app.route("/", methods=["GET"])
def index():
    """
    Rota principal (GET). 
    Apenas renderiza o template 'index.html' (a interface do usuário).
    """
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    """
    Rota de processamento (POST). 
    Recebe os dados do formulário (texto, modo, chave de API, arquivo)
    e executa a classificação de acordo com o modo escolhido.
    """
    try:
        # 1. Coleta e Sanitiza os Dados do Formulário
        text_input = request.form.get("text_input", "").strip()
        mode = request.form.get("mode", "local")
        api_key = request.form.get("api_key", "").strip()
        file = request.files.get("file")

        # Inicializa o texto a ser analisado.
        text = text_input
        
        # 2. Processamento de Arquivos
        if not text and file and file.filename:
            filename = file.filename.lower()
            
            # Se for PDF, extrai o texto usando a função auxiliar.
            if filename.endswith(".pdf"):
                # io.BytesIO(file.read()) cria um stream em memória para PyPDF2 ler o arquivo.
                text = extract_text_from_pdf(io.BytesIO(file.read()))
            
            # Se for TXT, lê o conteúdo como texto simples.
            elif filename.endswith(".txt"):
                # Decodifica o conteúdo do arquivo com tratamento de erro.
                text = file.read().decode("utf-8", errors="ignore")

        # 3. Validação do Texto
        if not text:
            return render_template("index.html", error="Nenhum texto ou arquivo válido foi enviado.")

        # 4. Execução da Classificação Baseada no Modo
        label, confidence, reply = "", 0.0, ""

        if mode == "local":
            label, confidence, reply = classify_local(text)
        
        elif mode == "gemini":
            # Usa a chave da interface ou busca na variável de ambiente.
            key_to_use = api_key or os.getenv("GEMINI_API_KEY")
            if not key_to_use:
                return render_template("index.html", error="Chave de API do Gemini não fornecida na interface nem encontrada no ambiente.")
            label, confidence, reply = classify_with_gemini(text, key_to_use)
            
        elif mode == "openai":
            # Usa a chave da interface ou busca na variável de ambiente.
            key_to_use = api_key or os.getenv("OPENAI_API_KEY")
            if not key_to_use:
                return render_template("index.html", error="Chave de API da OpenAI não fornecida na interface nem encontrada no ambiente.")
            label, confidence, reply = classify_with_openai(text, key_to_use)
            
        else:
            # Tratamento para modo inválido (segurança).
            return render_template("index.html", error="Modo de operação inválido selecionado.")

        # 5. Tratamento de Erros de Classificação
        if label == "Erro":
            # Retorna o erro específico da função de classificação.
            return render_template("index.html", error=reply, original_text=text)

        # 6. Renderização de Resultados
        # Retorna a página inicial com os resultados da classificação.
        return render_template("index.html",
                               original_text=text,
                               result_label=label,
                               confidence=confidence,
                               suggested_reply=reply)

    except Exception as e:
        # Tratamento de erro geral para qualquer falha não esperada na rota.
        print(f"Erro geral na rota /process: {e}")
        traceback.print_exc() # Imprime o stack trace para o console para debug.
        return render_template("index.html", error="Ocorreu um erro inesperado no servidor. Verifique o console para mais detalhes.")


# --- Execução da Aplicação ---
if __name__ == "__main__":
    # Garante que o servidor Flask seja executado apenas quando o script for chamado diretamente.
    # debug=True: Permite recarregamento automático e exibe o console de debug do Flask.
    # port=5000: Define a porta de execução padrão.
    app.run(debug=True, port=5000)
