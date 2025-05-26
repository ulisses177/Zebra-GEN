#!/usr/bin/env python3
import itertools
import copy
import random
import json
import re
from call_llm import call_llm
import os
import time  # Adicionar no topo do arquivo

# ======================================================
# FUNÇÕES DE PUZZLE (adaptadas do zebra-gen.py)
# ======================================================

def constraint_to_text(constraint):
    ctype = constraint["type"]
    if ctype == "position":
        pos = constraint["position"] + 1  # itens numerados a partir de 1
        return f"O item {pos} tem {constraint['attribute']} igual a {constraint['value']}."
    elif ctype == "direct":
        return (f"Se um item tem {constraint['if']['attribute']} igual a {constraint['if']['value']}, "
                f"então esse mesmo item tem {constraint['then']['attribute']} igual a {constraint['then']['value']}.")
    elif ctype == "ordered":
        left = constraint["left"]
        right = constraint["right"]
        if constraint.get("immediate", False):
            return (f"O item com {left['attribute']} igual a {left['value']} está imediatamente à esquerda "
                    f"do item com {right['attribute']} igual a {right['value']}.")
        else:
            return (f"Todos os itens com {left['attribute']} igual a {left['value']} devem vir antes "
                    f"de um item com {right['attribute']} igual a {right['value']}.")
    elif ctype == "neighbor":
        return (f"Se um item tem {constraint['if']['attribute']} igual a {constraint['if']['value']}, "
                f"então pelo menos um dos itens vizinhos tem {constraint['neighbor']['attribute']} igual a {constraint['neighbor']['value']}.")
    else:
        return "Restrição desconhecida."

def check_constraints_param(items, constraints):
    for constraint in constraints:
        ctype = constraint["type"]
        if ctype == "position":
            pos = constraint["position"]
            attr = constraint["attribute"]
            expected_val = constraint["value"]
            if items[pos].get(attr) is not None and items[pos][attr] != expected_val:
                return False
        elif ctype == "direct":
            cond = constraint["if"]
            then = constraint["then"]
            for item in items:
                if item.get(cond["attribute"]) == cond["value"]:
                    if item.get(then["attribute"]) is not None and item[then["attribute"]] != then["value"]:
                        return False
        elif ctype == "ordered":
            left = constraint["left"]
            right = constraint["right"]
            immediate = constraint.get("immediate", False)
            if immediate:
                for i in range(len(items)):
                    if items[i].get(left["attribute"]) == left["value"]:
                        if i+1 < len(items):
                            if items[i+1].get(right["attribute"]) is not None and items[i+1][right["attribute"]] != right["value"]:
                                return False
                        else:
                            return False
                    if items[i].get(right["attribute"]) == right["value"]:
                        if i-1 >= 0:
                            if items[i-1].get(left["attribute"]) is not None and items[i-1][left["attribute"]] != left["value"]:
                                return False
                        else:
                            return False
            else:
                left_indices = [i for i, item in enumerate(items) if item.get(left["attribute"]) == left["value"]]
                right_indices = [i for i, item in enumerate(items) if item.get(right["attribute"]) == right["value"]]
                for li in left_indices:
                    if all(ri <= li for ri in right_indices):
                        return False
        elif ctype == "neighbor":
            cond = constraint["if"]
            neighbor_req = constraint["neighbor"]
            for i, item in enumerate(items):
                if item.get(cond["attribute"]) == cond["value"]:
                    neighbors = []
                    if i - 1 >= 0:
                        neighbors.append(items[i-1])
                    if i + 1 < len(items):
                        neighbors.append(items[i+1])
                    all_assigned = True
                    found = False
                    for n in neighbors:
                        if n.get(neighbor_req["attribute"]) is None:
                            all_assigned = False
                        elif n.get(neighbor_req["attribute"]) == neighbor_req["value"]:
                            found = True
                    if all_assigned and not found:
                        return False
        else:
            raise ValueError("Tipo de restrição desconhecido: " + ctype)
    return True

def generate_candidates_for_item(i, used, fixed_assignments, domain):
    candidate_options = {}
    for cat in domain.keys():
        if i in fixed_assignments and cat in fixed_assignments[i]:
            candidate_options[cat] = [fixed_assignments[i][cat]]
        else:
            candidate_options[cat] = [val for val in domain[cat] if val not in used[cat]]
    cats = list(domain.keys())
    for values in itertools.product(*(candidate_options[cat] for cat in cats)):
        candidate = dict(zip(cats, values))
        yield candidate

def backtrack(i, items, used, log, domain, fixed_assignments, constraints, start_time, timeout=120):
    """
    Versão com timeout do algoritmo de backtracking.
    
    Args:
        ...
        start_time (float): Tempo de início do backtracking
        timeout (int): Tempo máximo em segundos para o backtracking
    """
    # Verificar timeout
    if time.time() - start_time > timeout:
        return None
        
    if i == len(items):
        if check_constraints_param(items, constraints):
            log.append("Solução completa encontrada!")
            return items
        else:
            return None
            
    candidates = list(generate_candidates_for_item(i, used, fixed_assignments, domain))
    for candidate in candidates:
        items[i] = candidate
        for cat in candidate:
            if not (i in fixed_assignments and cat in fixed_assignments[i]):
                used[cat].add(candidate[cat])
        log.append(f"Atribuindo item {i+1}: {candidate}")
        if check_constraints_param(items, constraints):
            sol = backtrack(i + 1, items, used, log, domain, fixed_assignments, constraints, start_time, timeout)
            if sol is not None:
                return sol
        log.append(f"Backtrack no item {i+1}: {candidate}")
        for cat in candidate:
            if not (i in fixed_assignments and cat in fixed_assignments[i]):
                used[cat].remove(candidate[cat])
        items[i] = {cat: None for cat in domain.keys()}
    return None

def solve_puzzle(domain, constraints, fixed_assignments, dimension):
    items = [{cat: None for cat in domain.keys()} for _ in range(dimension)]
    for i, fixed in fixed_assignments.items():
        for cat, val in fixed.items():
            items[i][cat] = val
    used = {cat: set() for cat in domain.keys()}
    for i in fixed_assignments:
        for cat, val in fixed_assignments[i].items():
            used[cat].add(val)
    log = []
    start_time = time.time()
    solution = backtrack(0, items, used, log, domain, fixed_assignments, constraints, start_time)
    if solution is None and time.time() - start_time >= 2:
        log.append("Timeout: O backtracking excedeu o tempo limite de 2 segundos")
    return solution, log

def generate_enunciado(puzzle_name, dimension, domain, constraints):
    text = f"{puzzle_name}\n"
    text += f"Você tem {dimension} itens (por exemplo, casas ou carros) dispostos em linha, numerados de 1 a {dimension}.\n"
    text += "Cada item possui os seguintes atributos e seus valores possíveis:\n"
    for cat, values in domain.items():
        text += f"  - {cat}: " + ", ".join(values) + ".\n"
    text += "\nDicas do puzzle:\n"
    for i, constraint in enumerate(constraints, start=1):
        text += f"{i}. {constraint_to_text(constraint)}\n"
    return text

def generate_candidate_clues(solution, domain):
    candidates = []
    M = len(solution)
    attributes = list(domain.keys())
    # Easy clues: posição fixa
    for i in range(M):
        for attr in attributes:
            clue = {
                "type": "position",
                "position": i,
                "attribute": attr,
                "value": solution[i][attr],
                "difficulty": "easy"
            }
            candidates.append(clue)
    # Medium clues: relação direta entre atributos no mesmo item
    for attr1 in attributes:
        for attr2 in attributes:
            if attr1 == attr2:
                continue
            mapping = {}
            for i in range(M):
                mapping[solution[i][attr1]] = solution[i][attr2]
            for val1, val2 in mapping.items():
                clue = {
                    "type": "direct",
                    "if": {"attribute": attr1, "value": val1},
                    "then": {"attribute": attr2, "value": val2},
                    "difficulty": "medium"
                }
                candidates.append(clue)
    # Hard clues: pistas de ordem imediata e de vizinhança (se M > 1)
    if M > 1:
        for i in range(M - 1):
            for attr1 in attributes:
                for attr2 in attributes:
                    if attr1 == attr2:
                        continue
                    clue = {
                        "type": "ordered",
                        "left": {"attribute": attr1, "value": solution[i][attr1]},
                        "right": {"attribute": attr2, "value": solution[i+1][attr2]},
                        "immediate": True,
                        "difficulty": "hard"
                    }
                    candidates.append(clue)
        for i in range(M):
            for attr1 in attributes:
                for attr2 in attributes:
                    if attr1 == attr2:
                        continue
                    neighbor_vals = set()
                    if i - 1 >= 0:
                        neighbor_vals.add(solution[i-1][attr2])
                    if i + 1 < M:
                        neighbor_vals.add(solution[i+1][attr2])
                    for n_val in neighbor_vals:
                        clue = {
                            "type": "neighbor",
                            "if": {"attribute": attr1, "value": solution[i][attr1]},
                            "neighbor": {"attribute": attr2, "value": n_val},
                            "difficulty": "hard"
                        }
                        candidates.append(clue)
    return candidates

def generate_logical_deduction(selected_clues):
    difficulty_order = {"easy": 1, "medium": 2, "hard": 3}
    sorted_clues = sorted(selected_clues, key=lambda clue: difficulty_order.get(clue.get("difficulty", "medium"), 2))
    deduction_steps = []
    for i, clue in enumerate(sorted_clues, start=1):
        deduction_steps.append(f"{i}. {constraint_to_text(clue)}")
    return deduction_steps

def generate_puzzle(solution, domain, clue_counts):
    candidates = generate_candidate_clues(solution, domain)
    random.shuffle(candidates)
    clues_easy = [c for c in candidates if c.get("difficulty") == "easy"]
    clues_medium = [c for c in candidates if c.get("difficulty") == "medium"]
    clues_hard = [c for c in candidates if c.get("difficulty") == "hard"]
    
    selected_clues = []
    selected_clues.extend(clues_easy[:clue_counts.get("easy", 0)])
    selected_clues.extend(clues_medium[:clue_counts.get("medium", 0)])
    selected_clues.extend(clues_hard[:clue_counts.get("hard", 0)])
    
    fixed_assignments = {}
    other_constraints = []
    for clue in selected_clues:
        if clue["type"] == "position":
            pos = clue["position"]
            if pos not in fixed_assignments:
                fixed_assignments[pos] = {}
            fixed_assignments[pos][clue["attribute"]] = clue["value"]
        else:
            other_constraints.append(clue)
    
    dimension = len(solution)
    sol, log = solve_puzzle(domain, other_constraints, fixed_assignments, dimension)
    if sol is None:
        return None
    
    enunciado = generate_enunciado("Puzzle Gerado Automaticamente", dimension, domain, other_constraints)
    deduction = generate_logical_deduction(selected_clues)
    
    return {
        "enunciado": enunciado,
        "constraints": selected_clues,
        "deduction": deduction,
        "solution": sol,
        "log": log
    }

# ======================================================
# FUNÇÕES DE GERAÇÃO VIA LLM
# ======================================================

def extract_json(response_text):
    """
    Tenta extrair um objeto JSON válido da resposta da LLM.
    Procura a primeira ocorrência de "{" e a última de "}" e retorna a substring.
    """
    response_text = response_text.strip()
    start = response_text.find('{')
    end = response_text.rfind('}')
    if start != -1 and end != -1 and start < end:
        json_str = response_text[start:end+1]
        return json_str
    else:
        return response_text

def generate_entries(theme, dimension):
    """
    Solicita à LLM que gere uma estrutura JSON contendo as entradas para um puzzle,
    conforme o tema e a dimensão desejada.
    Retorna um JSON com as chaves "domain" e "solution".
    """
    prompt = f"""
Você é um gerador de puzzles e deve produzir uma estrutura JSON válida, sem nenhum texto adicional, com o seguinte formato:

{{
  "domain": {{
    "Attribute1": ["Value1", "Value2", ..., "Value{dimension}"],
    "Attribute2": ["Value1", "Value2", ..., "Value{dimension}"],
    ...,
    "Attribute{dimension}": ["Value1", "Value2", ..., "Value{dimension}"]
  }},
  "solution": [
    {{"Attribute1": "V1", "Attribute2": "V1", ..., "Attribute{dimension}": "V1"}},
    {{"Attribute1": "V2", "Attribute2": "V2", ..., "Attribute{dimension}": "V2"}},
    ...,
    {{"Attribute1": "V{dimension}", "Attribute2": "V{dimension}", ..., "Attribute{dimension}": "V{dimension}"}}
  ]
}}

Utilize o tema "{theme}" para definir nomes criativos e coerentes para os atributos e para os valores.
Cada atributo deve ter exatamente {dimension} valores e a solução deve conter exatamente {dimension} itens, com cada item usando um valor único por atributo.
Certifique-se de retornar APENAS o JSON, sem nenhum texto adicional.
"""
    response_text = call_llm(prompt, show_tokens=True)
    json_str = extract_json(response_text)
    try:
        entries = json.loads(json_str)
    except Exception as e:
        print("Erro ao converter a resposta da LLM para JSON:")
        print("Texto bruto da resposta:", response_text)
        print("Texto extraído:", json_str)
        raise e
    return entries

def finalize_puzzle_output(context_json, enunciado, deduction, solution):
    """
    Usa os termos gerados anteriormente para produzir um texto final organizado.
    """
    deduction_text = "\n".join(deduction)
    solution_text = "\n".join([f"Item {i+1}: " + ", ".join([f"{k}: {v}" for k, v in item.items()]) for i, item in enumerate(solution)])
    
    prompt = f"""
Você é um redator de puzzles e deve compor um texto final que apresente de forma clara e organizada as seguintes seções:

1. Contexto de Geração:
Mostre o objeto JSON abaixo, que contém os termos gerados anteriormente (domínio e solução):

{context_json}

2. Enunciado do Puzzle:
{enunciado}

3. Passos Dedutivos (Caminho Lógico):
{deduction_text}

4. Solução Final:
{solution_text}

Formate o texto final com títulos para cada seção e utilize uma linguagem clara e precisa.
"""
    final_text = call_llm(prompt, show_tokens=True)
    return final_text.strip()

# ======================================================
# FUNÇÃO PARA EXTRAIR A INFORMAÇÃO PARA A PERGUNTA DE RETROALIMENTAÇÃO
# ======================================================

def extract_feedback_info(deduction_list, solution):
    """
    A partir dos últimos passos dedutivos (deduction_list) e da solução,
    extrai as informações necessárias para gerar a pergunta de feedback e a resposta correta.
    
    Se o último passo dedutivo contém uma pista de vizinhança (palavra "vizinho(s)"),
    a função formata a pergunta específica, por exemplo:
    
      "Com base na dedução, quem é vizinho do item com <atributo_condição> igual a <valor_condição>?"
    
    A resposta completa (correct_answer) mostrará a solução final e, logo abaixo, responderá à questão.
    """
    # Obtém o último passo dedutivo
    last_deduction = deduction_list[-1]

    # Se for uma dedução do tipo 'vizinhança'
    if re.search(r"vizin", last_deduction, re.IGNORECASE):
        pattern = r"Se um item tem\s+(.+?)\s+igual a\s+(.+?),\s+então pelo menos um dos itens vizinhos tem\s+(.+?)\s+igual a\s+([^.]+)\."
        match = re.search(pattern, last_deduction)
        if match:
            cond_attribute = match.group(1).strip()
            cond_value = match.group(2).strip()
            neighbor_attribute = match.group(3).strip()
            neighbor_value = match.group(4).strip()
            feedback_question = f"Com base na dedução, quem é vizinho do item com {cond_attribute} igual a {cond_value}?"
    else:
        # Padrão para deduções diretas do tipo "O item X tem ... igual a ..."
        pattern = r"\d+\.\s+O item (\d+)\s+tem\s+([^ ]+)\s+igual a\s+([^.]+)\."
        match = re.search(pattern, last_deduction)
        if match:
            item = match.group(1).strip()
            attribute = match.group(2).strip()
            value = match.group(3).strip()
            feedback_question = f"Com base na dedução, qual é o {attribute} do item {item}?"

    # Monta a representação completa da solução
    solution_lines = []
    for i, sol in enumerate(solution, start=1):
        sol_line = f"Item {i}: " + ", ".join([f"{k}: {v}" for k, v in sol.items()])
        solution_lines.append(sol_line)
    solution_str = "\n".join(solution_lines)

    # Define a resposta correta conforme o tipo de dedução identificado
    if re.search(r"vizin", last_deduction, re.IGNORECASE) and match:
        correct_answer = (f"Solução Final:\n{solution_str}\n\n"
                          f"Resposta: O item vizinho possui {neighbor_attribute} igual a {neighbor_value}.")
    elif match:
        correct_answer = (f"Solução Final:\n{solution_str}\n\n"
                          f"Resposta: O {attribute} do item {item} é {value}.")
    else:
        feedback_question = "Qual é o resultado final deduzido?"
        correct_answer = ""
    
    return feedback_question, correct_answer

# ======================================================
# FUNÇÃO MAIN – GERANDO O DATASET
# ======================================================

def main():
    # Criar pasta datasets se não existir
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
        print("Pasta 'datasets' criada.")

    # Carregar temas do arquivo JSON
    try:
        with open("puzzle_themes.json", "r", encoding="utf-8") as f:
            themes_data = json.load(f)
            themes = themes_data["themes"]
    except FileNotFoundError:
        themes = [
            "Pokemon! O nome dos pokemons deve ser um dos atributos.",
            "Cidades! Os nomes das cidades devem ser os atributos.",
            "Carros! Os modelos de carros são os atributos.",
            "Animais! Os nomes dos animais devem ser os atributos.",
            "Profissões! Os nomes das profissões devem ser os atributos."
        ]
        print("Arquivo puzzle_themes.json não encontrado. Usando temas padrão.")
    
    # Lista de dimensões possíveis para os puzzles
    dimensions = [3, 4, 5]
    
    dataset = []
    total_entries = 500
    checkpoint_interval = 5  # Salvar a cada 5 entradas
    
    for i in range(total_entries):
        theme = random.choice(themes)
        dim = random.choice(dimensions)
        print(f"\n=== Gerando entrada {i+1} com dimensão {dim} usando o tema '{theme}' ===")
        
        # Tentativas máximas por entrada
        max_attempts = 2
        success = False
        
        for attempt in range(max_attempts):
            try:
                # Gerar os termos (domain e solution) via LLM
                entries = generate_entries(theme, dim)
                
                domain = entries.get("domain")
                solution = entries.get("solution")
                if domain is None or solution is None:
                    print(f"Tentativa {attempt + 1}: Estrutura inválida gerada pela LLM.")
                    continue

                # Validar se todos os atributos do domínio existem na solução
                valid_structure = True
                for item in solution:
                    if not all(attr in item for attr in domain.keys()):
                        print(f"Tentativa {attempt + 1}: Solução não contém todos os atributos do domínio.")
                        valid_structure = False
                        break
                
                if not valid_structure:
                    continue

                context_json = json.dumps(entries, indent=2, ensure_ascii=False)
                clue_counts = {"easy": dim, "medium": dim, "hard": dim}
                puzzle = generate_puzzle(solution, domain, clue_counts)
                
                if puzzle is None:
                    print(f"Tentativa {attempt + 1}: Não foi possível gerar um puzzle com as configurações escolhidas.")
                    continue

                deduction_list = puzzle.get("deduction", [])
                if not deduction_list:
                    print(f"Tentativa {attempt + 1}: Nenhum passo dedutivo gerado.")
                    continue

                # Se chegou até aqui, tudo deu certo
                last_deduction = deduction_list[-1]
                feedback_question, correct_answer = extract_feedback_info(deduction_list, solution)
                enunciado_final = puzzle["enunciado"] + "\n\nPergunta Retroalimentada: " + feedback_question

                dataset_entry = {
                    "domain": domain,
                    "enunciado": enunciado_final,
                    "deduction": deduction_list,
                    "feedback_question": feedback_question,
                    "correct_answer": correct_answer,
                    "solution": puzzle.get("solution"),
                    "log": puzzle.get("log")
                }
                dataset.append(dataset_entry)
                success = True
                break

            except Exception as ex:
                print(f"Tentativa {attempt + 1}: Erro durante a geração: {str(ex)}")
                if attempt + 1 < max_attempts:
                    print("Tentando novamente...")
                continue
        
        if not success:
            print(f"Falha em todas as tentativas para entrada {i+1}. Continuando para a próxima...")
            continue

        # Checkpoint a cada 5 entradas bem sucedidas
        if success and len(dataset) % checkpoint_interval == 0:
            checkpoint_file = f"datasets/dataset_checkpoint_{len(dataset)}.json"
            try:
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(dataset, f, indent=2, ensure_ascii=False)
                print(f"\nCheckpoint salvo em '{checkpoint_file}'")
            except Exception as e:
                print(f"\nErro ao salvar checkpoint: {str(e)}")
    
    # Salvar dataset final
    final_file = f"datasets/dataset_final_{len(dataset)}_entries.json"
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\nDataset final com {len(dataset)} entradas foi gerado e salvo em '{final_file}'.")

if __name__ == '__main__':
    main()
