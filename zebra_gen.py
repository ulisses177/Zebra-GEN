#!/usr/bin/env python3
import itertools
from functools import lru_cache
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from puzzle_examples import Puzzle, ZEBRA_PUZZLE, CARS_PUZZLE
import random
from collections import defaultdict

class PuzzleError(Exception):
    """Classe base para erros relacionados ao puzzle"""
    pass

class InvalidConstraintError(PuzzleError):
    """Erro para restrições inválidas"""
    pass

class NoSolutionError(PuzzleError):
    """Erro quando nenhuma solução é encontrada"""
    pass

# === Parte 1: Motor do Puzzle (Backtracking, Verificação, etc.) ===

# Adicionamos um cache global para estados insatisfatórios
state_cache = {}

def make_state_key(items, pos):
    """
    Cria uma chave imutável para o estado parcial do puzzle.
    Considera as posições 0 até pos-1. Cada item é transformado em uma tupla
    dos pares (atributo, valor) ordenados.
    """
    key = []
    for i in range(pos):
        # Converte o dicionário em tupla ordenada para garantir consistência
        key.append(tuple(sorted(items[i].items())))
    return tuple(key)

def check_constraints_param(items, constraints):
    """
    Verifica se todas as restrições são satisfeitas para uma configuração de items.
    Para soluções parciais, só verifica as restrições que podem ser avaliadas.
    """
    try:
        for constraint in constraints:
            constraint_type = constraint["type"]
            
            if constraint_type == "direct":
                # Para restrições diretas, só verifica se todos os valores necessários estão presentes
                if_attr = constraint["if"]["attribute"]
                if_val = constraint["if"]["value"]
                then_attr = constraint["then"]["attribute"]
                then_val = constraint["then"]["value"]
                
                for item in items:
                    # Pula itens incompletos
                    if None in item.values():
                        continue
                    if item[if_attr] == if_val and item[then_attr] != then_val:
                        return False
                    
            elif constraint_type == "neighbor":
                # Para vizinhos, só verifica se há contradição nas posições já preenchidas
                if_attr = constraint["if"]["attribute"]
                if_val = constraint["if"]["value"]
                neighbor_attr = constraint["neighbor"]["attribute"]
                neighbor_val = constraint["neighbor"]["value"]
                
                # Se encontrarmos o valor 'if', procuramos o valor 'neighbor' nos vizinhos
                for i in range(len(items) - 1):
                    # Pula pares incompletos
                    if None in items[i].values() or None in items[i+1].values():
                        continue
                    if items[i][if_attr] == if_val:
                        if items[i+1][neighbor_attr] != neighbor_val:
                            # Se o próximo item está preenchido e não satisfaz a condição
                            if i > 0 and items[i-1][neighbor_attr] != neighbor_val:
                                return False
                    if items[i+1][if_attr] == if_val:
                        if items[i][neighbor_attr] != neighbor_val:
                            if i < len(items)-2 and items[i+2][neighbor_attr] != neighbor_val:
                                return False
                    
            elif constraint_type == "ordered":
                # Para ordem, só verifica posições já preenchidas
                left_attr = constraint["left"]["attribute"]
                left_val = constraint["left"]["value"]
                right_attr = constraint["right"]["attribute"]
                right_val = constraint["right"]["value"]
                immediate = constraint.get("immediate", False)
                
                left_pos = -1
                right_pos = -1
                for i, item in enumerate(items):
                    if None in item.values():
                        continue
                    if item[left_attr] == left_val:
                        left_pos = i
                    if item[right_attr] == right_val:
                        right_pos = i
                
                # Só verifica se ambas as posições foram encontradas
                if left_pos != -1 and right_pos != -1:
                    if left_pos >= right_pos:
                        return False
                    if immediate and right_pos != left_pos + 1:
                        return False
                    
        return True
        
    except Exception as e:
        raise InvalidConstraintError(f"Erro ao verificar restrições: {str(e)}")

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
    Preenche os itens (de índice 0 a n-1) com atribuições candidatas, usando backtracking.
    Essa versão utiliza caching para evitar reavaliação de estados parciais que já falharam.
    """
    # Gerar a chave do estado para as posições preenchidas até o momento
    key = make_state_key(items, i)
    if key in state_cache:
        log.append(f"[Cache] Estado repetido na posição {i}, pulando")
        return None

    if i == len(items):
        if check_constraints_param(items, constraints):
            return items
        else:
            log.append("Solução completa encontrada mas restrições não satisfeitas")
            state_cache[key] = False
            return None

    candidates = list(generate_candidates_for_item(i, used, fixed_assignments, domain))
    log.append(f"Posição {i}: Testando {len(candidates)} candidatos")
    print(f"[Progresso] Processando posição {i+1}/{len(items)} - {len(candidates)} candidatos gerados")
    
    if not candidates:
        log.append(f"Posição {i}: Nenhum candidato válido encontrado")
        state_cache[key] = False
        return None

    for candidate in candidates:
        items[i] = candidate
        # Atualiza os valores usados para atributos que não estão fixos
        for cat in candidate:
            if not (i in fixed_assignments and cat in fixed_assignments[i]):
                used[cat].add(candidate[cat])
        log.append(f"Posição {i}: Tentando {candidate}")
        if check_constraints_param(items[:i+1], constraints):
            sol = backtrack(i + 1, items, used, log, domain, fixed_assignments, constraints)
            if sol is not None:
                return sol

        log.append(f"Posição {i}: Backtrack necessário para {candidate}")
        for cat in candidate:
            if not (i in fixed_assignments and cat in fixed_assignments[i]):
                used[cat].remove(candidate[cat])
        items[i] = {cat: None for cat in domain.keys()}

    log.append(f"Posição {i}: Nenhuma solução encontrada com os candidatos disponíveis")
    state_cache[key] = False
    return None

def solve_puzzle(puzzle: Puzzle):
    """Resolve o puzzle usando a nova classe Puzzle"""
    return solve_puzzle_internal(
        puzzle.domain,
        puzzle.constraints,
        puzzle.fixed,
        puzzle.dimension
    )

def solve_puzzle_internal(domain, constraints, fixed_assignments, dimension=5):
    """Implementação interna do solucionador"""
    try:
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
        
        if solution is None:
            print("\nLog de execução:")
            for entry in log:
                print(entry)
            raise NoSolutionError("Não foi possível encontrar uma solução para o puzzle")
            
        return solution, log
        
    except PuzzleError:
        raise
    except Exception as e:
        raise PuzzleError(f"Erro inesperado ao resolver o puzzle: {str(e)}")

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
    try:
        # Usa os puzzles pré-definidos
        zebra = ZEBRA_PUZZLE
        cars = CARS_PUZZLE
        
        # Resolve o puzzle Zebra
        print("=== Enunciado Gerado (Zebra) ===")
        print(generate_enunciado(zebra.name, zebra.dimension, 
                               zebra.domain, zebra.constraints))
        
        zebra_solution, zebra_log = solve_puzzle(zebra)
        
        print("\n=== Solução ===")
        for i, item in enumerate(zebra_solution):
            print(f"Casa {i+1}:")
            for cat, val in item.items():
                print(f"  {cat}: {val}")
        
        # Resolve o puzzle dos Carros
        print("\n=== Enunciado Gerado (Carros) ===")
        print(generate_enunciado(cars.name, cars.dimension, 
                               cars.domain, cars.constraints))
        
        cars_solution, cars_log = solve_puzzle(cars)
        
        print("\n=== Solução ===")
        for i, item in enumerate(cars_solution):
            print(f"Posição {i+1}:")
            for cat, val in item.items():
                print(f"  {cat}: {val}")
                
    except PuzzleError as e:
        print(f"Erro ao resolver o puzzle: {str(e)}")
    except Exception as e:
        print(f"Erro inesperado: {str(e)}")

if __name__ == '__main__':
    main()

@dataclass
class DeductiveStep:
    """Representa um passo na cadeia dedutiva"""
    premise: str
    conclusion: str
    explanation: str

class LogicalChain:
    """Mantém a cadeia de deduções lógicas"""
    def __init__(self):
        self.steps: List[DeductiveStep] = []
        self.known_facts: Set[Tuple] = set()  # (pos, attr, value)
        
    def add_step(self, premise: str, conclusion: str, explanation: str):
        self.steps.append(DeductiveStep(premise, conclusion, explanation))

def generate_solution_first(dimension: int, attributes: List[str], 
                          values_per_attribute: Dict[str, List[str]]) -> Tuple[List[Dict], List[Dict], LogicalChain]:
    """
    Gera primeiro uma solução válida, depois deriva as restrições mínimas necessárias.
    
    Returns:
        Tuple[solution, constraints, logical_chain]
    """
    # 1. Gerar uma solução válida (matriz de atributos x items)
    solution = []
    used = {attr: set() for attr in attributes}
    
    for pos in range(dimension):
        item = {}
        for attr in attributes:
            available = [v for v in values_per_attribute[attr] if v not in used[attr]]
            value = random.choice(available)
            item[attr] = value
            used[attr].add(value)
        solution.append(item)
    
    # 2. Construir a cadeia dedutiva e as restrições necessárias
    logical_chain = LogicalChain()
    constraints = []
    
    # 2.1 Começar com uma ou duas fixações estratégicas
    start_pos = random.randint(0, dimension-1)
    start_attr = random.choice(attributes)
    start_value = solution[start_pos][start_attr]
    
    constraints.append({
        "type": "position",
        "position": start_pos,
        "attribute": start_attr,
        "value": start_value
    })
    
    logical_chain.add_step(
        premise=f"Dado inicialmente",
        conclusion=f"Item {start_pos+1} tem {start_attr}={start_value}",
        explanation="Informação fornecida no enunciado"
    )
    logical_chain.known_facts.add((start_pos, start_attr, start_value))
    
    # 2.2 Identificar relações dedutivas necessárias
    while not is_solution_unique(solution, constraints):
        # Encontrar próxima dedução mais forte
        next_deduction = find_strongest_deduction(solution, logical_chain.known_facts)
        if next_deduction is None:
            break
            
        pos, attr, value, deduction_type, related = next_deduction
        constraint = create_constraint(deduction_type, pos, attr, value, related)
        constraints.append(constraint)
        
        logical_chain.add_step(
            premise=f"Considerando {attr}={value} na posição {pos+1}",
            conclusion=format_deduction(deduction_type, related),
            explanation=generate_explanation(deduction_type, related)
        )
        logical_chain.known_facts.add((pos, attr, value))
    
    return solution, constraints, logical_chain

def find_strongest_deduction(solution: List[Dict], known_facts: Set[Tuple]) -> Optional[Tuple]:
    """
    Encontra a próxima dedução mais forte possível dado o que já sabemos.
    Retorna uma tupla (pos, attr, value, deduction_type, related) ou None se não encontrar.
    
    Tipos de dedução:
    - "direct": relação direta entre dois atributos
    - "neighbor": relação de vizinhança
    - "ordered": relação de ordem
    """
    dimension = len(solution)
    all_attributes = set(solution[0].keys())
    
    # 1. Primeiro, procura por relações diretas fortes
    for pos, item in enumerate(solution):
        for attr1 in all_attributes:
            val1 = item[attr1]
            if (pos, attr1, val1) in known_facts:
                # Procura por correlações diretas com outros atributos
                for attr2 in all_attributes:
                    if attr2 != attr1:
                        val2 = item[attr2]
                        # Se encontrarmos um par único na solução
                        pair_count = sum(1 for other_item in solution 
                                       if other_item[attr1] == val1 
                                       and other_item[attr2] == val2)
                        if pair_count == 1:
                            return (pos, attr2, val2, "direct", {
                                "if_attr": attr1,
                                "if_val": val1,
                                "then_attr": attr2,
                                "then_val": val2
                            })
    
    # 2. Procura por relações de vizinhança
    for pos in range(dimension - 1):
        current = solution[pos]
        next_item = solution[pos + 1]
        for attr1 in all_attributes:
            val1 = current[attr1]
            if (pos, attr1, val1) in known_facts:
                for attr2 in all_attributes:
                    val2 = next_item[attr2]
                    # Verifica se essa relação de vizinhança é única
                    neighbor_count = 0
                    for i in range(dimension - 1):
                        if (solution[i][attr1] == val1 and solution[i+1][attr2] == val2) or \
                           (solution[i][attr2] == val2 and solution[i+1][attr1] == val1):
                            neighbor_count += 1
                    if neighbor_count == 1:
                        return (pos + 1, attr2, val2, "neighbor", {
                            "if_attr": attr1,
                            "if_val": val1,
                            "neighbor_attr": attr2,
                            "neighbor_val": val2
                        })
    
    # 3. Procura por relações de ordem
    for pos1 in range(dimension):
        for pos2 in range(pos1 + 1, dimension):
            for attr in all_attributes:
                val1 = solution[pos1][attr]
                val2 = solution[pos2][attr]
                if (pos1, attr, val1) in known_facts:
                    # Verifica se essa ordem é única e necessária
                    order_count = 0
                    for i in range(dimension):
                        for j in range(i + 1, dimension):
                            if solution[i][attr] == val1 and solution[j][attr] == val2:
                                order_count += 1
                    if order_count == 1:
                        return (pos2, attr, val2, "ordered", {
                            "left_attr": attr,
                            "left_val": val1,
                            "right_attr": attr,
                            "right_val": val2,
                            "immediate": pos2 == pos1 + 1
                        })
    
    return None

def is_solution_unique(solution: List[Dict], constraints: List[Dict]) -> bool:
    """
    Verifica se as restrições atuais garantem uma única solução.
    Tenta encontrar uma solução diferente que satisfaça as mesmas restrições.
    """
    dimension = len(solution)
    domain = {attr: set() for attr in solution[0].keys()}
    
    # Constrói o domínio a partir da solução atual
    for item in solution:
        for attr, val in item.items():
            domain[attr].add(val)
    
    # Tenta encontrar uma solução diferente
    def try_alternative(pos: int, items: List[Dict], used: Dict[str, Set[str]]) -> bool:
        if pos == dimension:
            # Verifica se encontramos uma solução diferente
            return items != solution and check_constraints_param(items, constraints)
            
        item = {}
        for attr in domain:
            available = [v for v in domain[attr] if v not in used[attr]]
            for val in available:
                item[attr] = val
                used[attr].add(val)
                items[pos] = item.copy()
                
                if check_constraints_param(items[:pos+1], constraints):
                    if try_alternative(pos + 1, items, used):
                        return True
                        
                used[attr].remove(val)
        
        items[pos] = {attr: None for attr in domain}
        return False
    
    # Inicializa estruturas para a busca
    test_items = [{attr: None for attr in domain} for _ in range(dimension)]
    test_used = {attr: set() for attr in domain}
    
    # Se encontrar uma solução alternativa, retorna False
    return not try_alternative(0, test_items, test_used)

def create_constraint(deduction_type: str, pos: int, attr: str, 
                     value: str, related: Dict) -> Dict:
    """
    Cria uma restrição formal baseada no tipo de dedução.
    """
    # Implementar criação de restrições baseada no tipo de dedução
    pass

def format_deduction(deduction_type: str, related: Dict) -> str:
    """
    Formata a dedução em linguagem natural.
    """
    # Implementar formatação da dedução
    pass

def generate_explanation(deduction_type: str, related: Dict) -> str:
    """
    Gera uma explicação em linguagem natural para a dedução.
    """
    # Implementar geração de explicação
    pass
