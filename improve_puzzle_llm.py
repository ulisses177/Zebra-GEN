#!/usr/bin/env python3
from call_llm import call_llm
from puzzle_examples import ZEBRA_PUZZLE

def get_statement(puzzle):
    """
    Monta o enunciado do puzzle a partir dos seus dados.
    """
    statement = f"{puzzle.name}\n"
    statement += f"Você tem {puzzle.dimension} itens dispostos em linha, numerados de 1 a {puzzle.dimension}.\n"
    statement += "Cada item possui os seguintes atributos, com os respectivos valores possíveis:\n"
    for cat, values in puzzle.domain.items():
        statement += f"  - {cat}: " + ", ".join(values) + ".\n"
    statement += "\nDicas:\n"
    for i, constraint in enumerate(puzzle.constraints, start=1):
        if constraint["type"] == "direct":
            text = (f"Se um item tem {constraint['if']['attribute']} igual a {constraint['if']['value']}, "
                    f"então ele deve ter {constraint['then']['attribute']} igual a {constraint['then']['value']}.")
        elif constraint["type"] == "neighbor":
            text = (f"Se um item tem {constraint['if']['attribute']} igual a {constraint['if']['value']}, "
                    f"pelo menos um dos vizinhos deve ter {constraint['neighbor']['attribute']} igual a {constraint['neighbor']['value']}.")
        elif constraint["type"] == "ordered":
            text = (f"O item com {constraint['left']['attribute']} igual a {constraint['left']['value']} "
                    f"está antes do item com {constraint['right']['attribute']} igual a {constraint['right']['value']}.")
        else:
            text = "Restrição desconhecida."
        statement += f"{i}. {text}\n"
    return statement

def main():
    # Usaremos o puzzle Zebra definido em puzzle_examples.py
    puzzle = ZEBRA_PUZZLE
    
    # Repetir o enunciado do puzzle
    statement = get_statement(puzzle)
    
    # Cria duas perguntas a partir do valor fixo do puzzle (o 'fixed')
    # Exemplo: Casa 1 com "Nacionalidade: Norueguês" e Casa 3 com "Bebida: Leite"
    question1 = ("Dada a fixação de que a Casa 1 tem Nacionalidade 'Norueguês', "
                 "quais conclusões dedutivas podem ser extraídas para o posicionamento dos demais atributos?")
    question2 = ("Sabendo que a Casa 3 tem Bebida 'Leite', que inferências lógicas podem orientar a distribuição dos demais valores?")
    
    think_part = "<think> Analise todas as restrições do puzzle e, a partir das fixações, determine quais relações obrigatórias entre atributos podem ser inferidas para completar a solução. Sejam claros quanto aos passos lógicos e dedutivos. </think>"
    
    prompt = (
        f"{statement}\n\n"
        "Agora, responda as seguintes duas perguntas retextualizadas:\n"
        f"1. {question1}\n"
        f"2. {question2}\n\n"
        f"{think_part}\n\n"
        "Por fim, forneça as respostas para ambas as perguntas, explicando detalhadamente os passos dedutivos e a lógica utilizada para chegar a uma solução completa do puzzle."
    )
    
    resposta = call_llm(prompt)
    print("Resposta da LLM para a versão melhorada do puzzle:")
    print(resposta)

if __name__ == '__main__':
    main() 