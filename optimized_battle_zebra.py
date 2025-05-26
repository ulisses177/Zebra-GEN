import itertools
import random
import json
from collections import defaultdict

def constraint_to_text(constraint):
    ctype = constraint["type"]
    if ctype == "position":
        return f"O item {constraint['position'] + 1} tem {constraint['attribute']} igual a {constraint['value']}."
    elif ctype == "direct":
        return f"Se um item tem {constraint['if']['attribute']} igual a {constraint['if']['value']}, então esse mesmo item tem {constraint['then']['attribute']} igual a {constraint['then']['value']}."
    elif ctype == "ordered":
        return f"O item com {constraint['left']['attribute']} igual a {constraint['left']['value']} vem antes do item com {constraint['right']['attribute']} igual a {constraint['right']['value']}."
    elif ctype == "neighbor":
        return f"Se um item tem {constraint['if']['attribute']} igual a {constraint['if']['value']}, então um dos vizinhos tem {constraint['neighbor']['attribute']} igual a {constraint['neighbor']['value']}."
    elif ctype == "interval":
        return f"O item com {constraint['left']['attribute']} igual a {constraint['left']['value']} está {constraint['distance']} posições antes do item com {constraint['right']['attribute']} igual a {constraint['right']['value']}."
    elif ctype == "count":
        return f"Existem exatamente {constraint['count']} itens com {constraint['attribute']} igual a {constraint['value']}."
    else:
        return "Restrição desconhecida."

def check_constraints(items, constraints):
    for constraint in constraints:
        ctype = constraint["type"]
        if ctype == "position":
            if items[constraint["position"]].get(constraint["attribute"]) != constraint["value"]:
                return False
        elif ctype == "direct":
            for item in items:
                if item.get(constraint["if"]["attribute"]) == constraint["if"]["value"]:
                    if item.get(constraint["then"]["attribute"]) != constraint["then"]["value"]:
                        return False
        elif ctype == "ordered":
            left_indices = [i for i, item in enumerate(items) if item.get(constraint['left']['attribute']) == constraint['left']['value']]
            right_indices = [i for i, item in enumerate(items) if item.get(constraint['right']['attribute']) == constraint['right']['value']]
            if any(l >= r for l in left_indices for r in right_indices):
                return False
        elif ctype == "neighbor":
            for i, item in enumerate(items):
                if item.get(constraint['if']['attribute']) == constraint['if']['value']:
                    neighbors = []
                    if i > 0:
                        neighbors.append(items[i-1])
                    if i < len(items) - 1:
                        neighbors.append(items[i+1])
                    if not any(n.get(constraint['neighbor']['attribute']) == constraint['neighbor']['value'] for n in neighbors):
                        return False
        elif ctype == "interval":
            for i in range(len(items)):
                if items[i].get(constraint['left']['attribute']) == constraint['left']['value']:
                    target_index = i + constraint['distance']
                    if 0 <= target_index < len(items) and items[target_index].get(constraint['right']['attribute']) != constraint['right']['value']:
                        return False
        elif ctype == "count":
            count = sum(1 for item in items if item.get(constraint['attribute']) == constraint['value'])
            if count != constraint['count']:
                return False
    return True

def forward_checking(items, used, domain, index):
    """ Remove valores inválidos do domínio de acordo com restrições """
    for cat in domain.keys():
        if items[index].get(cat) is not None:
            continue
        domain[cat] = [val for val in domain[cat] if val not in used[cat]]

def backtrack(i, items, used, domain, constraints):
    if i == len(items):
        return items if check_constraints(items, constraints) else None
    
    forward_checking(items, used, domain, i)
    candidates = list(itertools.product(*[domain[cat] for cat in domain]))
    
    for candidate in candidates:
        items[i] = dict(zip(domain.keys(), candidate))
        for cat in domain.keys():
            used[cat].add(items[i][cat])
        if check_constraints(items, constraints):
            sol = backtrack(i + 1, items, used, domain, constraints)
            if sol:
                return sol
        for cat in domain.keys():
            used[cat].remove(items[i][cat])
        items[i] = {cat: None for cat in domain.keys()}
    
    return None

def solve_puzzle(domain, constraints, dimension):
    items = [{cat: None for cat in domain.keys()} for _ in range(dimension)]
    used = defaultdict(set)
    return backtrack(0, items, used, domain, constraints)

def generate_puzzle(domain, constraints, dimension):
    return solve_puzzle(domain, constraints, dimension)
