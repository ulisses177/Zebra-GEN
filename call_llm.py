import requests
import json
import traceback

OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEBUG = True

def call_llm(prompt, show_tokens=True, temperature=0.2):
    """
    Chama a API do Ollama para gerar uma resposta à partir do prompt fornecido.
    A resposta é processada de forma a juntar os tokens recebidos via stream.
    Retorna o texto gerado.
    
    Args:
        prompt (str): O prompt para a LLM
        show_tokens (bool): Se deve mostrar os tokens sendo gerados
        temperature (float): Temperatura para geração (0.0 a 1.0)
    """
    data = {
        "model": "mistral-small:24b-instruct-2501-q4_K_M",
        "prompt": prompt,
        "stream": True,
        "temperature": temperature
    }
    
    if DEBUG:
        print(f"DEBUG: Enviando prompt para a API. Tamanho do prompt: {len(prompt)} caracteres.")
    
    try:
        response = requests.post(OLLAMA_API_URL, json=data, stream=True, timeout=30)
        if response.status_code == 200:
            full_response = ""
            print("\nGerando resposta da LLM:", end=" ", flush=True)
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line)
                    except Exception as e:
                        if DEBUG:
                            print(f"\nDEBUG: Erro ao decodificar JSON: {e}")
                        continue
                    if 'response' in json_response:
                        token = json_response['response']
                        full_response += token
                        if show_tokens:
                            print(token, end="", flush=True)
            print()  # Nova linha ao final
            return full_response.strip()
        else:
            print(f"Erro na API: Código {response.status_code}")
            if DEBUG:
                print(f"DEBUG: Resposta da API: {response.text}")
            return prompt
    except Exception as e:
        print(f"Erro na chamada à LLM: {str(e)}")
        if DEBUG:
            traceback.print_exc()
        return prompt