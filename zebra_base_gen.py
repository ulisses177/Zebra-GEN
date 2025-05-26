import requests
import json
import traceback

OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEBUG = True

def call_llm(prompt, show_tokens=True):
    """
    Chama a API do Ollama para gerar uma resposta a partir do prompt fornecido.
    A resposta é processada em stream, concatenando os tokens recebidos.
    Retorna o texto completo gerado.
    """
    data = {
        "model": "mistral-small-pt",  # ou outro modelo que você prefira
        "prompt": prompt,
        "stream": True
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

def generate_name_and_dimension():
    prompt = """
Por favor, gere apenas as chaves "name" e "dimension" para um quebra-cabeça de lógica com animais em uma fazenda.
- "name": deve ser uma string que identifique o puzzle (ex.: "Quebra-cabeça: Animais na Fazenda").
- "dimension": deve ser um número inteiro (ex.: 5).

Gere a saída no seguinte formato (sem as chaves externas):
"name": "Quebra-cabeça: Animais na Fazenda",
"dimension": 5
"""
    return call_llm(prompt)

def generate_domain():
    prompt = """
Por favor, gere apenas a seção "domain" para um quebra-cabeça de lógica com animais em uma fazenda.
Os atributos obrigatórios devem incluir: "Animal", "Cor", "Localização", "Comida" e "Som". 
Para cada atributo, liste exatamente 5 valores coerentes com o tema.
Exemplo (a saída deve ser semelhante a isto, mas com valores diferentes):
{
  "Animal": ["Cachorro", "Gato", "Vaca", "Cavalo", "Ovelha"],
  "Cor": ["Preto", "Branco", "Marrom", "Cinza", "Amarelo"],
  "Localização": ["Estábulo", "Galpão", "Casa", "Campo", "Sítio"],
  "Comida": ["Feno", "Ração", "Grãos", "Carne", "Vegetais"],
  "Som": ["Latido", "Miado", "Mugido", "Relincho", "Balido"]
}
Gere apenas o conteúdo do domain (sem a chave "domain").
"""
    return call_llm(prompt)

def generate_constraints():
    prompt = """
Por favor, gere apenas a seção "constraints" para um quebra-cabeça de lógica com animais em uma fazenda.
A lista deve conter pelo menos 6 restrições, usando os seguintes tipos:
- "position": ex.: { "type": "position", "position": 0, "attribute": "Animal", "value": "Cachorro" }
- "direct": ex.: { "type": "direct", "if": {"attribute": "Cor", "value": "Cinza"}, "then": {"attribute": "Animal", "value": "Cabra"} }
- "ordered": ex.: { "type": "ordered", "left": {"attribute": "Localização", "value": "Estábulo"}, "right": {"attribute": "Localização", "value": "Galpão"}, "immediate": true }
- "neighbor": ex.: { "type": "neighbor", "if": {"attribute": "Som", "value": "Latido"}, "neighbor": {"attribute": "Animal", "value": "Cachorro"} }
Gere a lista completa (sem a chave "constraints").
"""
    return call_llm(prompt)

def generate_fixed():
    prompt = """
Por favor, gere apenas a seção "fixed" para um quebra-cabeça de lógica com animais em uma fazenda.
Esta seção deve ser um objeto com pelo menos 2 entradas, onde as chaves são índices (como "0", "2") e os valores são objetos com um atributo e seu valor.
Exemplo de saída:
{
  "0": {"Animal": "Cachorro"},
  "2": {"Comida": "Feno"}
}
Gere apenas o conteúdo do fixed (sem a chave "fixed").
"""
    return call_llm(prompt)

def main():
    # Gera cada parte separadamente
    print("\nGerando name e dimension...")
    name_dimension = generate_name_and_dimension()
    
    print("\nGerando domain...")
    domain = generate_domain()
    
    print("\nGerando constraints...")
    constraints = generate_constraints()
    
    print("\nGerando fixed...")
    fixed = generate_fixed()

    # Monta o JSON final
    final_json_str = "{\n"
    final_json_str += name_dimension.strip().rstrip(',') + ",\n"
    final_json_str += "\"domain\": " + domain.strip().rstrip(',') + ",\n"
    final_json_str += "\"constraints\": " + constraints.strip().rstrip(',') + ",\n"
    final_json_str += "\"fixed\": " + fixed.strip().rstrip(',') + "\n"
    final_json_str += "}\n"

    print("\nJSON final gerado (texto bruto):")
    print(final_json_str)

    try:
        final_json = json.loads(final_json_str)
        print("\nJSON válido gerado com sucesso:")
        print(json.dumps(final_json, indent=2, ensure_ascii=False))
    except json.JSONDecodeError as e:
        print("\nErro ao decodificar o JSON final:")
        print(final_json_str)
        print("Detalhes do erro:", e)

if __name__ == "__main__":
    main()
