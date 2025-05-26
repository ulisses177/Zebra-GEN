#!/usr/bin/env python3
import sys
from call_llm import call_llm

def main():
    # Se o usuário não informar tema, usa um padrão.
    theme = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Culinária"
    
    prompt = (
        f"Crie um puzzle lógico com o tema '{theme}'.\n"
        "Para isso, liste os atributos que compõem cada item do puzzle, "
        "os valores possíveis para cada atributo e as restrições que relacionem esses atributos. "
        "O puzzle deve ter 5 itens e seguir o formato dos tradicionais puzzles de lógica. "
        "Forneça a resposta em português, de forma clara, estruturando a resposta em seções: "
        "1) Atributos e seus valores, "
        "2) Lista de restrições (dicas) que conectam os atributos."
    )
    
    resposta = call_llm(prompt)
    print("Resposta da LLM para geração do puzzle:")
    print(resposta)

if __name__ == '__main__':
    main() 