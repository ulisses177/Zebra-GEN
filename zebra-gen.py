#!/usr/bin/env python3
import itertools

# === Parte 1: Motor do Puzzle (Backtracking, Verificação, etc.) ===

def check_constraints_param(items, constraints):
    """
    Verifica, de forma genérica, se o arranjo (parcial ou completo) de itens
    satisfaz todas as restrições definidas na lista 'constraints'.
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
                            return False  # não há item à direita
                    if items[i].get(right["attribute"]) == right["value"]:
                        if i-1 >= 0:
                            if items[i-1].get(left["attribute"]) is not None and items[i-1][left["attribute"]] != left["value"]:
                                return False
                        else:
                            return False  # não há item à esquerda
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
    Para o item (ex.: uma casa ou um carro) de índice i, gera todas as atribuições
    candidatas (dicionários) respeitando:
      - os valores já usados (garantindo unicidade)
      - as fixações pré-definidas (fixed_assignments)
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
    Preenche os itens (de índice 0 a n-1) com atribuições candidatas, utilizando
    backtracking e verificando as restrições.
    
    Nota: valores fixos (definidos em fixed_assignments) não são adicionados/retirados de 'used'.
    """
    if i == len(items):
        if check_constraints_param(items, constraints):
            return items
        else:
            return None

    # "Congela" os candidatos para a posição atual
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
    Inicializa as estruturas e resolve o puzzle via backtracking.
    Retorna a solução e o log dos passos.
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

# === Parte 2: Gerador de Enunciado (traduzindo domínios e restrições para texto) ===

def constraint_to_text(constraint):
    """
    Converte uma restrição (dicionário) para uma frase em português.
    """
    ctype = constraint["type"]
    if ctype == "position":
        pos = constraint["position"] + 1  # casas numeradas de 1 a n
        return f"A casa {pos} deve ter {constraint['attribute']} igual a {constraint['value']}."
    elif ctype == "direct":
        return (f"Se um item tem {constraint['if']['attribute']} igual a {constraint['if']['value']}, "
                f"então esse mesmo item deve ter {constraint['then']['attribute']} igual a {constraint['then']['value']}.")
    elif ctype == "ordered":
        left = constraint["left"]
        right = constraint["right"]
        if constraint.get("immediate", False):
            return (f"O item com {left['attribute']} igual a {left['value']} está imediatamente à esquerda "
                    f"do item com {right['attribute']} igual a {right['value']}.")
        else:
            return (f"Todos os itens com {left['attribute']} igual a {left['value']} devem estar posicionados "
                    f"antes de algum item com {right['attribute']} igual a {right['value']}.")
    elif ctype == "neighbor":
        return (f"Se um item tem {constraint['if']['attribute']} igual a {constraint['if']['value']}, "
                f"então pelo menos um dos itens vizinhos deve ter {constraint['neighbor']['attribute']} igual a {constraint['neighbor']['value']}.")
    else:
        return "Restrição desconhecida."

def generate_enunciado(puzzle_name, dimension, domain, constraints):
    """
    Gera o enunciado do puzzle (em linguagem natural) a partir dos domínios e restrições.
    
    O enunciado descreve:
      - O número de itens e as categorias com seus valores possíveis.
      - A lista de dicas (restrições) traduzidas para linguagem natural.
    """
    text = f"{puzzle_name}\n"
    text += f"Você tem {dimension} itens (por exemplo, casas ou carros) dispostos em linha, numerados de 1 a {dimension}.\n"
    text += "Cada item possui os seguintes atributos, com os respectivos valores possíveis:\n"
    for cat, values in domain.items():
        text += f"  - {cat}: " + ", ".join(values) + ".\n"
    text += "\nDicas:\n"
    for i, constraint in enumerate(constraints, start=1):
        text += f"{i}. {constraint_to_text(constraint)}\n"
    return text

# === Parte 3: Dados de Exemplo para Dois Puzzles ===

def main():
    # ----- Puzzle Zebra Clássico -----
    zebra_domain = {
        "Nacionalidade": ["Norueguês", "Dinamarquês", "Britânico", "Alemão", "Sueco"],
        "Cor": ["Amarelo", "Azul", "Vermelho", "Verde", "Branco"],
        "Bebida": ["Água", "Chá", "Leite", "Café", "Cerveja"],
        "Cigarro": ["Dunhill", "Blend", "Pall Mall", "Prince", "BlueMaster"],
        "Animal": ["Gatos", "Cavalos", "Pássaros", "Zebra", "Cachorros"]
    }
    zebra_constraints = [
        { "type": "position", "position": 0, "attribute": "Nacionalidade", "value": "Norueguês" },
        { "type": "position", "position": 2, "attribute": "Bebida", "value": "Leite" },
        { "type": "direct", "if": {"attribute": "Nacionalidade", "value": "Britânico"},
                         "then": {"attribute": "Cor", "value": "Vermelho"} },
        { "type": "direct", "if": {"attribute": "Nacionalidade", "value": "Dinamarquês"},
                         "then": {"attribute": "Bebida", "value": "Chá"} },
        { "type": "direct", "if": {"attribute": "Nacionalidade", "value": "Alemão"},
                         "then": {"attribute": "Cigarro", "value": "Prince"} },
        { "type": "direct", "if": {"attribute": "Nacionalidade", "value": "Sueco"},
                         "then": {"attribute": "Animal", "value": "Cachorros"} },
        { "type": "ordered", "left": {"attribute": "Cor", "value": "Verde"},
                        "right": {"attribute": "Cor", "value": "Branco"}, "immediate": True },
        { "type": "direct", "if": {"attribute": "Cor", "value": "Verde"},
                         "then": {"attribute": "Bebida", "value": "Café"} },
        { "type": "direct", "if": {"attribute": "Cigarro", "value": "Pall Mall"},
                         "then": {"attribute": "Animal", "value": "Pássaros"} },
        { "type": "direct", "if": {"attribute": "Cor", "value": "Amarelo"},
                         "then": {"attribute": "Cigarro", "value": "Dunhill"} },
        { "type": "neighbor", "if": {"attribute": "Cigarro", "value": "Blend"},
                          "neighbor": {"attribute": "Animal", "value": "Gatos"} },
        { "type": "neighbor", "if": {"attribute": "Cigarro", "value": "Blend"},
                          "neighbor": {"attribute": "Bebida", "value": "Água"} },
        { "type": "neighbor", "if": {"attribute": "Animal", "value": "Cavalos"},
                          "neighbor": {"attribute": "Cigarro", "value": "Dunhill"} },
        { "type": "direct", "if": {"attribute": "Cigarro", "value": "BlueMaster"},
                         "then": {"attribute": "Bebida", "value": "Cerveja"} }
    ]
    zebra_fixed = {
        0: {"Nacionalidade": "Norueguês"},
        2: {"Bebida": "Leite"}
    }
    zebra_dimension = 5
    zebra_name = "Quebra-cabeça Zebra Clássico"
    zebra_enunciado = generate_enunciado(zebra_name, zebra_dimension, zebra_domain, zebra_constraints)

    print("=== Enunciado Gerado (Zebra) ===")
    print(zebra_enunciado)

    zebra_solution, zebra_log = solve_puzzle(zebra_domain, zebra_constraints, zebra_fixed, zebra_dimension)
    if zebra_solution is None:
        print("Nenhuma solução encontrada para o puzzle Zebra.")
    else:
        print("=== Passos do Backtracking (Zebra) ===")
        for step in zebra_log:
            print(step)
        print("=== Solução Final (Zebra) ===")
        for i, house in enumerate(zebra_solution):
            print(f"Casa {i+1}: {house}")
        zebra_owner = None
        water_drinker = None
        for house in zebra_solution:
            if house.get("Animal") == "Zebra":
                zebra_owner = house.get("Nacionalidade")
            if house.get("Bebida") == "Água":
                water_drinker = house.get("Nacionalidade")
        print(f"\nQuem tem a Zebra? {zebra_owner}")
        print(f"Quem bebe Água? {water_drinker}")

    # ----- Puzzle dos Carros na Corrida -----
    car_domain = {
        "Piloto": ["Hamilton", "Vettel", "Alonso", "Verstappen", "Norris"],
        "Cor": ["Vermelho", "Azul", "Verde", "Amarelo", "Preto"],
        "Bebida": ["Água", "Chá", "Café", "Refrigerante", "Suco"],
        "Marca": ["Ferrari", "Lamborghini", "Porsche", "Bugatti", "McLaren"],
        "Acessório": ["Relógio", "Óculos", "Chapéu", "Luvas", "Jaqueta"]
    }
    car_constraints = [
        { "type": "position", "position": 0, "attribute": "Piloto", "value": "Hamilton" },
        { "type": "position", "position": 2, "attribute": "Bebida", "value": "Café" },
        { "type": "direct", "if": {"attribute": "Piloto", "value": "Vettel"},
                         "then": {"attribute": "Cor", "value": "Vermelho"} },
        { "type": "direct", "if": {"attribute": "Piloto", "value": "Alonso"},
                         "then": {"attribute": "Bebida", "value": "Chá"} },
        { "type": "direct", "if": {"attribute": "Piloto", "value": "Verstappen"},
                         "then": {"attribute": "Marca", "value": "Bugatti"} },
        { "type": "direct", "if": {"attribute": "Piloto", "value": "Norris"},
                         "then": {"attribute": "Acessório", "value": "Relógio"} },
        { "type": "ordered", "left": {"attribute": "Cor", "value": "Verde"},
                        "right": {"attribute": "Cor", "value": "Preto"}, "immediate": True },
        { "type": "direct", "if": {"attribute": "Cor", "value": "Verde"},
                         "then": {"attribute": "Bebida", "value": "Água"} },
        { "type": "direct", "if": {"attribute": "Marca", "value": "Ferrari"},
                         "then": {"attribute": "Acessório", "value": "Óculos"} },
        { "type": "direct", "if": {"attribute": "Cor", "value": "Amarelo"},
                         "then": {"attribute": "Marca", "value": "Lamborghini"} },
        { "type": "neighbor", "if": {"attribute": "Marca", "value": "Porsche"},
                          "neighbor": {"attribute": "Acessório", "value": "Chapéu"} },
        { "type": "neighbor", "if": {"attribute": "Marca", "value": "Porsche"},
                          "neighbor": {"attribute": "Bebida", "value": "Refrigerante"} },
        { "type": "neighbor", "if": {"attribute": "Acessório", "value": "Luvas"},
                          "neighbor": {"attribute": "Marca", "value": "McLaren"} },
        { "type": "direct", "if": {"attribute": "Marca", "value": "McLaren"},
                         "then": {"attribute": "Bebida", "value": "Suco"} }
    ]
    car_fixed = {
        0: {"Piloto": "Hamilton"},
        2: {"Bebida": "Café"}
    }
    car_dimension = 5
    car_name = "Quebra-cabeça: Carros na Corrida"
    car_enunciado = generate_enunciado(car_name, car_dimension, car_domain, car_constraints)

    print("\n=== Enunciado Gerado (Carros) ===")
    print(car_enunciado)

    car_solution, car_log = solve_puzzle(car_domain, car_constraints, car_fixed, car_dimension)
    if car_solution is None:
        print("Nenhuma solução encontrada para o puzzle dos Carros.")
    else:
        print("=== Passos do Backtracking (Carros) ===")
        for step in car_log:
            print(step)
        print("=== Solução Final (Carros) ===")
        for i, carro in enumerate(car_solution):
            print(f"Carro {i+1}: {carro}")
        hat_pilot = None
        refrigerante_pilot = None
        for carro in car_solution:
            if carro.get("Acessório") == "Chapéu":
                hat_pilot = carro.get("Piloto")
            if carro.get("Bebida") == "Refrigerante":
                refrigerante_pilot = carro.get("Piloto")
        print(f"\nQuem é o piloto que possui o Chapéu? {hat_pilot}")
        print(f"Quem bebe Refrigerante? {refrigerante_pilot}")

if __name__ == '__main__':
    main()
