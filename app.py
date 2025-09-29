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

🔧 Melhorias adicionadas (do código do GPT):
- Heurísticas extras (acadêmicas, anexos, comunicação formal).
- Ajuste do threshold (>=0.6 Produtivo, 0.5–0.6 zona incerta).
- Respostas contextuais aprimoradas no template_reply.

✅ Melhorias de UX (Experiência do Usuário) implementadas:
- Bug 1: A seleção do modelo de IA agora persiste entre as análises.
- Bug 2: O envio de um arquivo agora tem prioridade sobre o texto digitado na caixa.
- Bug 3: A lógica de backend agora suporta o preview do arquivo no frontend.
- Feature: O campo de API Key é escondido se a chave já existir no ambiente do servidor.
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
    # ### CORREÇÃO DE ERRO DE DIGITAÇÃO ###
    # Corrigido de 'google.generai' para 'google.generativeai'
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

# Bloco de inicialização: tenta carregar o modelo de ML na memória quando a aplicação inicia.
# Isso evita a necessidade de recarregar o modelo a cada requisição, melhorando a performance.
print("--- INICIALIZANDO APLICAÇÃO ---")
# Verifica se o arquivo do modelo existe no caminho especificado.
if os.path.exists(MODEL_PATH):
    try:
        # Carrega o modelo de ML usando joblib.
        model = joblib.load(MODEL_PATH)
        print(f"✅ Modelo local carregado com sucesso de '{MODEL_PATH}'")
    except Exception as e:
        # Se ocorrer um erro durante o carregamento (ex: arquivo corrompido),
        # o erro é logado e a variável 'model' permanece como None.
        print(f"❌ Erro ao carregar o modelo local: {e}")
        model = None
else:
    # Aviso caso o modelo não seja encontrado no deploy. Isso pode acontecer se
    # o script de treinamento não foi executado com sucesso durante o build.
    print(f"⚠️  Aviso: Modelo local não foi encontrado em '{MODEL_PATH}'. O modo 'local' não funcionará.")
print("--- APLICAÇÃO PRONTA ---")


# --- Funções Auxiliares de Processamento e Heurística ---

def extract_text_from_pdf(file_stream):
    """
    Extrai o texto contido em um objeto de arquivo PDF (stream de bytes).

    Args:
        file_stream (io.BytesIO): Stream de bytes do arquivo PDF, vindo do request do Flask.

    Returns:
        str: Todo o texto extraído do PDF, concatenado, ou uma string vazia em caso de erro.
    """
    try:
        # Cria um objeto leitor de PDF a partir do stream de bytes em memória.
        reader = PyPDF2.PdfReader(file_stream)
        # Usa uma list comprehension para iterar sobre todas as páginas,
        # extrair o texto de cada uma e juntar tudo em uma única string com quebras de linha.
        text_pages = [page.extract_text() for page in reader.pages if page.extract_text()]
        return "\n".join(text_pages)
    except Exception as e:
        # Loga o erro de extração no console do servidor para debug.
        print(f"Erro ao extrair texto do PDF: {e}")
        return ""
    
def detect_order_info(text):
    """
    Detecta números de pedido, nota fiscal ou identificadores longos no texto usando expressões regulares.
    
    A presença de tais números é uma forte heurística para um e-mail 'Produtivo' (que requer ação).

    Args:
        text (str): O corpo do e-mail.

    Returns:
        str or None: O número de pedido encontrado ou None se nenhum for detectado.
    """
    text_low = text.lower()
    
    # Padrão 1: Busca por 'pedido' seguido opcionalmente por ':', espaço ou '-' e, em seguida, pelo menos 4 dígitos.
    # Ex: "pedido 1234", "pedido:5678", "pedido-9101"
    m = re.search(r'pedido[:\s-]*([0-9]{4,})', text_low)
    if m:
        return m.group(1) # Retorna o grupo de captura (apenas os dígitos do pedido).
        
    # Padrão 2: Busca por uma sequência isolada de 6 ou mais dígitos.
    # O `\b` significa "word boundary" (fronteira de palavra), garantindo que não pegamos
    # parte de um número maior, como um CEP ou telefone.
    # Ex: "ID 123456", "rastreio 987654321"
    m2 = re.search(r'\b([0-9]{6,})\b', text_low)
    if m2:
        return m2.group(1)
        
    return None # Retorna None se nenhum padrão for encontrado.


# <<< INÍCIO DA MELHORIA DO GPT >>>
# ### INCREMENTO: ASSINATURA DINÂMICA ###
# A assinatura da função foi modificada para aceitar 'user_name'
def template_reply(label, text="", user_name=""):
    """
    Gera uma resposta automática padronizada com base no rótulo de classificação
    e ajustada por heurística para e-mails 'Produtivos'.

    🔧 Melhorias: respostas contextuais para anexos, comunicação formal e termos acadêmicos.
    """
    text_lower = text.lower()
    order_id = detect_order_info(text)
    
    # Variável para construir o corpo da resposta antes de adicionar a assinatura.
    reply_body = ""

    # Lógica para e-mails que requerem ação (Produtivo)
    if label == "Produtivo":
        if order_id:
            reply_body = (f"Prezado(a),\n\nObrigado pelo contato. "
                          f"Identificamos o pedido #{order_id} em sua mensagem. "
                          "Estamos verificando e retornaremos com a nota fiscal ou atualização do status em breve.")
        elif any(k in text_lower for k in ["nota fiscal", "nota-fiscal", "nf-e", "nota_fiscal"]):
            reply_body = ("Prezado(a),\n\nObrigado pelo contato. "
                          "Vamos verificar a nota fiscal solicitada e retornaremos assim que possível.")
        elif any(k in text_lower for k in ["cancelamento", "devolução", "reembolso", "assinatura"]):
            reply_body = ("Prezado(a),\n\nRecebemos sua solicitação. "
                          "Nossa equipe está analisando e retornará em breve.")
        elif any(k in text_lower for k in ["anexo", "slides", "currículo", "documento"]):
            reply_body = ("Prezado(a),\n\nObrigado pelo envio do material. "
                          "Ele será muito útil e já estamos organizando para utilização.")
        elif any(k in text_lower for k in ["professor", "aluno", "disciplina", "tarefa", "projeto", "atividade"]):
            reply_body = ("Prezado(a) Professor(a),\n\nAgradecemos a mensagem e o envio. "
                          "Estamos acompanhando com atenção.")
        elif any(k in text_lower for k in ["prezado", "atenciosamente"]):
            reply_body = ("Prezado(a),\n\nAgradecemos o contato e confirmamos o recebimento da sua mensagem. "
                          "Nossa equipe está à disposição.")
        else:
            reply_body = ("Prezado(a),\n\nRecebemos sua solicitação e vamos analisar. "
                          "Por favor, confirme o número do seu pedido ou envie mais detalhes, se aplicável.")
    
    # Lógica para e-mails que não requerem ação imediata (Improdutivo)
    else:
        reply_body = ("Prezado(a),\n\nAgradecemos a sua mensagem. Entraremos em contato se for necessária alguma ação.")

    # ### INCREMENTO: ASSINATURA DINÂMICA ###
    # Lógica para criar a assinatura. Se o nome for fornecido no formulário, usa-o.
    # Caso contrário, usa um placeholder genérico que incentiva o preenchimento.
    signature = f"Atenciosamente,\n{user_name}" if user_name else "Atenciosamente,\n[Seu Nome]"
    
    # Concatena o corpo da resposta com a assinatura para formar a resposta final.
    return f"{reply_body}\n\n{signature}"
# <<< FIM DA MELHORIA DO GPT >>>


# --- Funções de Classificação com Modelos de IA ---

def classify_with_gemini(text, api_key):
    """
    Classifica o e-mail e gera resposta usando a API do Gemini.
    """
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
            "e gere o CORPO de uma resposta automática apropriada. Retorne SOMENTE um JSON válido com os campos: "
            "'label' (string), 'confidence' (float entre 0.0 e 1.0), e 'reply' (string, sem assinatura). "
            f"E-mail para análise: '''{text}'''"
        )
        response = model_gemini.generate_content(prompt)
        if hasattr(response, "text") and response.text:
            content = response.text
        elif hasattr(response, "candidates") and response.candidates:
            content = response.candidates[0].content.parts[0].text
        else:
            raise ValueError("Não foi possível extrair o texto da resposta do Gemini.")
        content = content.strip().lstrip("```json").rstrip("```")
        parsed = json.loads(content)
        return parsed.get("label"), float(parsed.get("confidence", 0)), parsed.get("reply", "")

    except Exception as e:
        print(f"Erro na API do Gemini: {e}")
        return "Erro", 0.0, f"Ocorreu um erro ao comunicar com a API do Gemini: {e}"

def classify_with_openai(text, api_key):
    """
    Classifica o e-mail e gera resposta usando a API da OpenAI (GPT).
    """
    if openai is None:
        return "Erro", 0.0, "A biblioteca da OpenAI não está instalada. Rode: pip install openai"

    try:
        openai.api_key = api_key
        prompt = (
            "Você é um assistente de e-mail. Classifique o e-mail a seguir em 'Produtivo' ou 'Improdutivo' "
            "e gere o CORPO de uma resposta automática apropriada. Retorne SOMENTE um JSON válido com os campos: "
            "'label' (string), 'confidence' (float entre 0.0 e 1.0), e 'reply' (string, sem assinatura). "
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


def classify_local(text, user_name=""):
    """
    Usa o modelo de Machine Learning local para classificação.
    """
    if model is None:
        return "Erro", 0.0, "O modelo local não está carregado. Treine o modelo primeiro."
    try:
        proba_prod = float(model.predict_proba([text])[0][1])
        if detect_order_info(text):
            proba_prod = max(proba_prod, 0.85)
        if any(k in text.lower() for k in ["anexo", "slides", "professor", "disciplina", "currículo", "documento"]):
            proba_prod = max(proba_prod, 0.8)

        if proba_prod >= 0.6:
            label = "Produtivo"
            confidence = proba_prod
        else:
            label = "Improdutivo"
            confidence = 1 - proba_prod
            if proba_prod >= 0.5:
                print("Aviso: classificação incerta (zona 0.5–0.6), revisão humana sugerida.")
        
        reply = template_reply(label, text, user_name)
        return label, confidence, reply
        
    except Exception as e:
        print(f"Erro no modelo local: {e}")
        return "Erro", 0.0, "Ocorreu um erro ao usar o modelo local."


# --- Rotas da Aplicação Flask ---

# ### INCREMENTO: VERIFICAÇÃO DE CHAVES DE AMBIENTE ###
# A rota principal agora verifica se as chaves de API existem no ambiente do servidor
# e passa essa informação para o template.
@app.route("/", methods=["GET"])
def index():
    """
    Rota principal (GET). 
    Renderiza o template 'index.html', passando o status das chaves de API do ambiente.
    """
    # os.getenv() busca uma variável de ambiente. Retorna None se não encontrar.
    # A conversão para bool (!!os.getenv(...)) resulta em True se a chave existir e False se não.
    gemini_key_exists = bool(os.getenv("GEMINI_API_KEY"))
    openai_key_exists = bool(os.getenv("OPENAI_API_KEY"))
    
    print(f"Status da chave Gemini no ambiente: {gemini_key_exists}")
    print(f"Status da chave OpenAI no ambiente: {openai_key_exists}")
    
    # Passa as variáveis booleanas para o template.
    return render_template("index.html", 
                           gemini_key_exists=gemini_key_exists, 
                           openai_key_exists=openai_key_exists)


@app.route("/process", methods=["POST"])
def process():
    """
    Rota de processamento (POST). 
    Recebe os dados do formulário, executa a classificação e renderiza os resultados.
    """
    try:
        # 1. Coleta e Sanitiza os Dados do Formulário
        text_input = request.form.get("text_input", "").strip()
        mode = request.form.get("mode", "local")
        api_key = request.form.get("api_key", "").strip()
        file = request.files.get("file")
        user_name = request.form.get("user_name", "").strip()
        
        # ### INCREMENTO: VERIFICAÇÃO DE CHAVES DE AMBIENTE (RE-RENDERIZAÇÃO) ###
        # É importante verificar o status das chaves aqui também, para que a informação
        # seja passada de volta ao template em caso de erro ou sucesso.
        gemini_key_exists = bool(os.getenv("GEMINI_API_KEY"))
        openai_key_exists = bool(os.getenv("OPENAI_API_KEY"))
        # Monta um dicionário com os dados a serem passados de volta para o template.
        # Isso evita repetição de código nos retornos.
        render_data = {
            "selected_mode": mode,
            "user_name": user_name,
            "gemini_key_exists": gemini_key_exists,
            "openai_key_exists": openai_key_exists
        }

        text = None
        
        # Correção Bug 2: Prioridade do Arquivo
        if file and file.filename:
            filename = file.filename.lower()
            if filename.endswith(".pdf"):
                text = extract_text_from_pdf(io.BytesIO(file.read()))
            elif filename.endswith(".txt"):
                text = file.read().decode("utf-8", errors="ignore")
        
        if not text:
            text = text_input

        # 3. Validação do Texto Final
        if not text:
            render_data["error"] = "Nenhum texto ou arquivo válido foi enviado."
            return render_template("index.html", **render_data)

        # 4. Execução da Classificação Baseada no Modo
        label, confidence, reply = "", 0.0, ""

        if mode == "local":
            label, confidence, reply = classify_local(text, user_name)
        
        elif mode == "gemini" or mode == "openai":
            reply_body = ""
            key_to_use = api_key # Por padrão, usa a chave do formulário.
            
            if mode == "gemini":
                # Se a chave de ambiente do Gemini existir, ela tem prioridade.
                if gemini_key_exists:
                    key_to_use = os.getenv("GEMINI_API_KEY")
                # Se não existir nem no ambiente nem no formulário, retorna erro.
                if not key_to_use:
                    render_data["error"] = "Chave de API do Gemini não fornecida."
                    return render_template("index.html", **render_data)
                label, confidence, reply_body = classify_with_gemini(text, key_to_use)

            else: # openai
                # Se a chave de ambiente do OpenAI existir, ela tem prioridade.
                if openai_key_exists:
                    key_to_use = os.getenv("OPENAI_API_KEY")
                # Se não existir nem no ambiente nem no formulário, retorna erro.
                if not key_to_use:
                    render_data["error"] = "Chave de API da OpenAI não fornecida."
                    return render_template("index.html", **render_data)
                label, confidence, reply_body = classify_with_openai(text, key_to_use)
            
            # Adiciona a assinatura dinâmica à resposta vinda da API
            if label != "Erro":
                signature = f"\n\nAtenciosamente,\n{user_name}" if user_name else "\n\nAtenciosamente,\n[Seu Nome]"
                reply = reply_body + signature
            else:
                reply = reply_body # Em caso de erro, a resposta já é a mensagem de erro.
        else:
            render_data["error"] = "Modo de operação inválido."
            return render_template("index.html", **render_data)

        # 5. Tratamento de Erros de Classificação
        if label == "Erro":
            render_data["error"] = reply
            render_data["original_text"] = text
            return render_template("index.html", **render_data)

        # 6. Renderização de Resultados
        # Adiciona os resultados ao dicionário de dados e renderiza o template.
        render_data.update({
            "original_text": text,
            "result_label": label,
            "confidence": confidence,
            "suggested_reply": reply,
        })
        return render_template("index.html", **render_data)

    except Exception as e:
        print(f"Erro geral na rota /process: {e}")
        traceback.print_exc()
        return render_template("index.html", error="Ocorreu um erro inesperado no servidor.")


# --- Execução da Aplicação ---
if __name__ == "__main__":
    app.run(debug=True, port=5000)