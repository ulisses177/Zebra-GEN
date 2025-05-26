#!/usr/bin/env python3
import sys
import json
import argparse
from call_llm import call_llm
from puzzle_examples import generate_puzzle
from zebra_gen import solve_puzzle, generate_enunciado
from tqdm import tqdm

def get_attributes_and_values(theme, dimension):
    """
    Retorna uma lista de atributos e um dicionário de valores possíveis para cada atributo,
    de acordo com um tema específico. Se não houver mapeamento para o tema, retorna atributos genéricos.
    """
    if theme.lower() == "culinária":
        attributes = ["Prato", "Bebida", "Sobremesa", "Ingrediente", "Cozinheiro"]
        values = {
            "Prato": [f"Prato_{i}" for i in range(1, dimension+1)],
            "Bebida": [f"Bebida_{i}" for i in range(1, dimension+1)],
            "Sobremesa": [f"Sobremesa_{i}" for i in range(1, dimension+1)],
            "Ingrediente": [f"Ingrediente_{i}" for i in range(1, dimension+1)],
            "Cozinheiro": [f"Cozinheiro_{i}" for i in range(1, dimension+1)]
        }
        return attributes, values
    else:
        # Atributos genéricos
        attributes = [f"Atributo_{i}" for i in range(1, 6)]
        values = {att: [f"Valor_{att}_{j}" for j in range(1, dimension+1)] for att in attributes}
        return attributes, values

def extract_two_fixed_questions(fixed, dimension):
    """
    A partir do dicionário fixed do puzzle (fixações) extrai duas questões para servir de base ao prompt,
    retextualizando as pistas fixas. Se houver menos de 2 pistas, utiliza as disponíveis ou cria mensagens genéricas.
    """
    fixed_items = sorted(list(fixed.items()), key=lambda x: x[0])
    questions = []
    for pos, assign in fixed_items[:2]:
        # pos é o índice (0-based). Convertemos para 1-based para o enunciado.
        for attr, val in assign.items():
            quest = (f"Dada a fixação de que o item {pos+1} tem {attr} '{val}', "
                     f"quais inferências dedutivas podem ser extraídas para distribuir os demais valores?")
            questions.append(quest)
    # Caso haja menos de 2 fixações, adiciona uma pergunta genérica.
    while len(questions) < 2:
        questions.append("Quais inferências lógicas podem guiar a atribuição dos demais valores?")
    return questions[0], questions[1]

def compute_constraints_distribution(constraints):
    """
    Computa a distribuição (contagem) de cada tipo de restrição.
    """
    dist = {}
    for c in constraints:
        ctype = c.get("type", "desconhecido")
        dist[ctype] = dist.get(ctype, 0) + 1
    return dist

def main():
    parser = argparse.ArgumentParser(
        description="Gera puzzle, resolve e cria dataset com chain-of-thought."
    )
    parser.add_argument("--theme", type=str, default="Culinária", help="Tema do puzzle (ex: Culinária)")
    parser.add_argument("--dimension", type=int, default=5, help="Dimensão do puzzle (número de itens)")
    args = parser.parse_args()

    theme = args.theme
    dimension = args.dimension

    # Barra de progresso geral: 4 passos principais
    pbar = tqdm(total=4, desc="Progresso do Dataset", ncols=80)
    
    # Passo 1: Obter atributos e valores
    attributes, values = get_attributes_and_values(theme, dimension)
    pbar.set_description("Atributos e valores obtidos")
    pbar.update(1)

    # Passo 2: Gerar o puzzle e resolver (com otimizações na geração de restrições)
    puzzle = generate_puzzle(dimension, attributes, values)
    # Inclui o tema no nome do puzzle para deixar explícito.
    puzzle.name = f"Puzzle '{theme}' {dimension}x{len(attributes)}"
    
    try:
        solution, solve_log = solve_puzzle(puzzle)
    except Exception as e:
        print(f"Erro ao resolver o puzzle: {e}")
        pbar.close()
        return
    pbar.set_description("Puzzle gerado e resolvido")
    pbar.update(1)

    # Passo 3: Gerar enunciado e montar prompt para a LLM
    puzzle_text = generate_enunciado(puzzle.name, puzzle.dimension, puzzle.domain, puzzle.constraints)
    question1, question2 = extract_two_fixed_questions(puzzle.fixed, puzzle.dimension)
    think_part = "<think> Analise as restrições listadas e, a partir das pistas fixas, deduza os relacionamentos obrigatórios entre os atributos necessários para completar a solução. Explique seu raciocínio passo a passo. </think>"
    
    prompt = (
        f"{puzzle_text}\n\n"
        "Agora, responda as seguintes duas perguntas:\n"
        f"1. {question1}\n"
        f"2. {question2}\n\n"
        f"{think_part}\n\n"
        "Por favor, forneça as respostas para ambas as perguntas, detalhando os passos lógicos e dedutivos para chegar à solução completa."
    )
    
    chain_of_thought = call_llm(prompt)
    pbar.set_description("Chain-of-thought obtida")
    pbar.update(1)
    
    # Passo 4: Calcular distribuição das restrições e salvar o dataset
    constraints_distribution = compute_constraints_distribution(puzzle.constraints)
    dataset = {
        "theme": theme,
        "dimension": dimension,
        "attributes": attributes,
        "puzzle_name": puzzle.name,
        "puzzle_enunciado": puzzle_text,
        "puzzle_fixed": puzzle.fixed,
        "puzzle_solution": solution,
        "constraints_distribution": constraints_distribution,
        "chain_of_thought": chain_of_thought,
        "solve_log": solve_log
    }
    
    safe_theme = theme.replace(" ", "_")
    filename = f"puzzle_dataset_{safe_theme}_{dimension}.json"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        pbar.set_description("Dataset salvo")
        pbar.update(1)
        pbar.close()
        print(f"\nDataset salvo em: {filename}")
    except Exception as e:
        print(f"Erro ao salvar dataset: {e}")
        pbar.close()

if __name__ == '__main__':
    main() 