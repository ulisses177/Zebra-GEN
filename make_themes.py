import random
import requests
import json
import traceback
from call_llm import call_llm
# Lista de temas
themes = [
    "Pokemon! O nome dos pokemons deve ser um dos atributos.",
    "Cidades! Os nomes das cidades devem ser os atributos.",
    "Carros! Os modelos de carros são os atributos.",
    "Animais! Os nomes dos animais devem ser os atributos.",
    "Profissões! Os nomes das profissões devem ser os atributos."
]

# Seleciona 3 temas aleatórios para exemplo
random_themes = random.sample(themes, min(3, len(themes)))
random_themes_str = "\n".join(random_themes)

# Função para chamar a LLM
prompt = f"""
Você é um redator de puzzles e deve compor um texto final que apresente de forma clara e organizada. Você deve gerar um tema para um puzzle, seja criativo NÃO GERE O PUZZLE, APENAS O TEMA, em um breve parágrafo apenas.

Aqui estão alguns temas já gerados:
{random_themes_str}
"""
print("Gerando temas...")
generated_themes = []
qtd_themes = 50

for i in range(qtd_themes):
    theme = call_llm(prompt, temperature=1.4)
    generated_themes.append(theme)
    themes.append(theme)  # Adiciona o tema gerado à lista original
    print(f'    "{theme}",')  # Formatação para fácil cópia/cola

# Salvar todos os temas em um arquivo JSON
all_themes = {
    "themes": themes
}
with open("puzzle_themes.json", "w", encoding="utf-8") as f:
    json.dump(all_themes, f, indent=2, ensure_ascii=False)

print("\nTemas salvos em 'puzzle_themes.json'")

