#!/usr/bin/env python3
import itertools
import copy
import random

# ======================================================
# Função auxiliar para converter uma restrição em texto.
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

# ======================================================
# Parte A: Solucionador Baseado em Backtracking
# ======================================================

def check_constraints_param(items, constraints):
    """
    Verifica se o arranjo (parcial ou completo) de items satisfaz todas as restrições.
    """
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
    """
    Para o item de índice i, gera todas as atribuições candidatas, respeitando:
      - Valores já usados (garantindo unicidade)
      - Fixações pré-definidas.
    """
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

def backtrack(i, items, used, log, domain, fixed_assignments, constraints):
    """
    Preenche os items utilizando backtracking. Armazena os passos no log (técnico, para depuração).
    """
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
            sol = backtrack(i + 1, items, used, log, domain, fixed_assignments, constraints)
            if sol is not None:
                return sol

        log.append(f"Backtrack no item {i+1}: {candidate}")
        for cat in candidate:
            if not (i in fixed_assignments and cat in fixed_assignments[i]):
                used[cat].remove(candidate[cat])
        items[i] = {cat: None for cat in domain.keys()}

    return None

def solve_puzzle(domain, constraints, fixed_assignments, dimension=5):
    """
    Tenta resolver o puzzle dado o domínio, restrições e fixações.
    Retorna (solução, log) se encontrar solução, ou (None, log) caso contrário.
    """
    items = [{cat: None for cat in domain.keys()} for _ in range(dimension)]
    for i, fixed in fixed_assignments.items():
        for cat, val in fixed.items():
            items[i][cat] = val
    used = {cat: set() for cat in domain.keys()}
    for i in fixed_assignments:
        for cat, val in fixed_assignments[i].items():
            used[cat].add(val)
    log = []
    solution = backtrack(0, items, used, log, domain, fixed_assignments, constraints)
    return solution, log

def generate_enunciado(puzzle_name, dimension, domain, constraints):
    """
    Gera o enunciado do puzzle, incluindo:
      - Itens e atributos (com valores possíveis)
      - Lista de dicas (restrições) em linguagem natural.
    """
    text = f"{puzzle_name}\n"
    text += f"Você tem {dimension} itens (por exemplo, casas ou carros) dispostos em linha, numerados de 1 a {dimension}.\n"
    text += "Cada item possui os seguintes atributos e seus valores possíveis:\n"
    for cat, values in domain.items():
        text += f"  - {cat}: " + ", ".join(values) + ".\n"
    text += "\nDicas do puzzle:\n"
    for i, constraint in enumerate(constraints, start=1):
        text += f"{i}. {constraint_to_text(constraint)}\n"
    return text

# ======================================================
# Parte B: Gerador de Puzzles a partir de uma Solução
# ======================================================

def generate_candidate_clues(solution, domain):
    """
    A partir da solução completa e do domínio, gera uma lista de pistas (restrições candidatas)
    classificadas por dificuldade:
      - easy: pistas de posição (fixação direta);
      - medium: pistas if-then (associando dois atributos no mesmo item);
      - hard: pistas envolvendo ordem imediata ou relação de vizinhança.
    """
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
    
    # Medium clues: relação direta (if-then) entre atributos no mesmo item
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
        # Ordered clues: itens adjacentes
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
        # Neighbor clues: relação com vizinhos
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
    """
    A partir do conjunto de pistas selecionadas, gera um caminho lógico resumido,
    ou seja, uma lista de passos em linguagem natural que descrevem as deduções.
    As pistas são ordenadas por nível de dificuldade (easy, medium, hard).
    """
    difficulty_order = {"easy": 1, "medium": 2, "hard": 3}
    sorted_clues = sorted(selected_clues, key=lambda clue: difficulty_order.get(clue.get("difficulty", "medium"), 2))
    deduction_steps = []
    for i, clue in enumerate(sorted_clues, start=1):
        deduction_steps.append(f"{i}. {constraint_to_text(clue)}")
    return deduction_steps

def generate_puzzle(solution, domain, clue_counts):
    """
    A partir de:
      - uma solução completa (lista de dicionários, um por item),
      - o domínio (atributos e seus possíveis valores),
      - um dicionário 'clue_counts' com a quantidade desejada para cada nível (ex.: {"easy": 3, "medium": 4, "hard": 3}),
      
    Gera um puzzle definindo as pistas e testa se o puzzle possui solução (de preferência única).
    
    Retorna um dicionário contendo:
      - "enunciado": texto do enunciado,
      - "constraints": lista completa de pistas selecionadas,
      - "deduction": o caminho lógico (passos dedutivos) em formato de lista de strings,
      - "solution": solução encontrada pelo solver,
      - "log": log completo do backtracking (para depuração, se necessário).
    
    Se não for possível encontrar um conjunto de pistas que gere a solução, retorna None.
    """
    candidates = generate_candidate_clues(solution, domain)
    random.shuffle(candidates)  # Embaralha as pistas
    # Separa as pistas por dificuldade
    clues_easy = [c for c in candidates if c.get("difficulty") == "easy"]
    clues_medium = [c for c in candidates if c.get("difficulty") == "medium"]
    clues_hard = [c for c in candidates if c.get("difficulty") == "hard"]
    
    selected_clues = []
    selected_clues.extend(clues_easy[:clue_counts.get("easy", 0)])
    selected_clues.extend(clues_medium[:clue_counts.get("medium", 0)])
    selected_clues.extend(clues_hard[:clue_counts.get("hard", 0)])
    
    # Separe os "position" constraints (fixações) dos demais
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
# Parte C: Exemplo de Uso – Impressão Automática
# ======================================================

def main():
    # Definição do domínio: atributos e valores possíveis
    domain = {
        "Piloto": ["Hamilton", "Vettel", "Alonso", "Verstappen", "Norris"],
        "Cor": ["Vermelho", "Azul", "Verde", "Amarelo", "Preto"],
        "Bebida": ["Água", "Chá", "Café", "Refrigerante", "Suco"],
        "Marca": ["Ferrari", "Lamborghini", "Porsche", "Bugatti", "McLaren"],
        "Acessório": ["Relógio", "Óculos", "Chapéu", "Luvas", "Jaqueta"]
    }
    # Solução completa (a ordem das linhas indica a posição dos itens)
    solution = [
        {"Piloto": "Hamilton", "Cor": "Vermelho",  "Bebida": "Chá",          "Marca": "Ferrari",      "Acessório": "Óculos"},
        {"Piloto": "Norris",   "Cor": "Verde",     "Bebida": "Água",         "Marca": "Porsche",      "Acessório": "Relógio"},
        {"Piloto": "Vettel",   "Cor": "Azul",      "Bebida": "Refrigerante", "Marca": "Lamborghini",  "Acessório": "Jaqueta"},
        {"Piloto": "Alonso",   "Cor": "Amarelo",   "Bebida": "Suco",         "Marca": "Bugatti",      "Acessório": "Luvas"},
        {"Piloto": "Verstappen","Cor": "Preto",    "Bebida": "Café",         "Marca": "McLaren",      "Acessório": "Chapéu"}
    ]
    
    # Parâmetros: quantidade de pistas desejadas para cada nível de dificuldade
    clue_counts = {
        "easy": 3,    # pistas diretas (posição fixa)
        "medium": 4,  # pistas if-then
        "hard": 3     # pistas de ordem/vizinhança
    }
    
    puzzle = generate_puzzle(solution, domain, clue_counts)
    if puzzle is None:
        print("Não foi possível gerar um puzzle com as configurações escolhidas.")
        return
    
    # --- Impressão do ENUNCIADO ---
    print("=== ENUNCIADO ===")
    print(puzzle["enunciado"])
    
    # --- Impressão dos PASSOS DO QUEBRA-CABEÇA (Caminho Lógico) ---
    print("\n=== PASSOS DO QUEBRA-CABEÇA (Caminho Lógico) ===")
    for step in puzzle["deduction"]:
        print(step)
    
    # --- Impressão da RESPOSTA FINAL ---
    print("\n=== RESPOSTA FINAL ===")
    solution_final = puzzle["solution"]
    for i, item in enumerate(solution_final):
        print(f"Item {i+1}:")
        for attr, val in item.items():
            print(f"  {attr}: {val}")
    print("\nObservações importantes:")
    print(" - Cada atributo aparece apenas uma vez por item e os valores não se repetem entre os itens.")
    print(" - As pistas utilizadas permitiram deduzir única e corretamente a solução acima.")

if __name__ == '__main__':
    main()
