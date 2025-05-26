# Exemplos de puzzles lógicos

from typing import Dict, List, Any
from dataclasses import dataclass
import random
from collections import defaultdict

@dataclass
class Puzzle:
    name: str
    dimension: int
    domain: Dict[str, List[str]]
    constraints: List[Dict[str, Any]]
    fixed: Dict[int, Dict[str, str]]

    def __post_init__(self):
        """Validação dos dados do puzzle"""
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Nome do puzzle inválido")
        
        if not self.domain or not all(isinstance(values, list) for values in self.domain.values()):
            raise ValueError("Domínio inválido")
            
        if not all(len(values) == self.dimension for values in self.domain.values()):
            raise ValueError(f"Todas as categorias devem ter {self.dimension} valores")
        
        for pos, assignments in self.fixed.items():
            if not (0 <= pos < self.dimension):
                raise ValueError(f"Posição fixa {pos} fora do intervalo válido [0, {self.dimension-1}]")
            for cat, val in assignments.items():
                if cat not in self.domain:
                    raise ValueError(f"Categoria {cat} não existe no domínio")
                if val not in self.domain[cat]:
                    raise ValueError(f"Valor {val} não existe no domínio de {cat}")

def generate_puzzle(
    dimension: int,
    attributes: List[str],
    values_per_attribute: Dict[str, List[str]]
) -> Puzzle:
    """
    Gera um puzzle com solução garantida.
    
    Args:
        dimension: Número de posições no puzzle.
        attributes: Lista de atributos (ex: ["Cor", "Animal"]).
        values_per_attribute: Valores possíveis para cada atributo.
    """
    # 1. Gera uma solução válida aleatória
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
    
    # 2. Gerar restrições a partir da solução com mais "poda" do espaço de solução
    constraints = []
    
    # Aumenta o número de restrições diretas
    num_direct = random.randint(dimension * 2, dimension * 3)
    for _ in range(num_direct):
        pos = random.randint(0, dimension - 1)
        attr1, attr2 = random.sample(attributes, 2)
        constraints.append({
            "type": "direct",
            "if": {"attribute": attr1, "value": solution[pos][attr1]},
            "then": {"attribute": attr2, "value": solution[pos][attr2]}
        })
    
    # Aumenta o número de restrições de vizinhança
    num_neighbor = random.randint(dimension, dimension * 2)
    for _ in range(num_neighbor):
        pos = random.randint(0, dimension - 2)
        attr1, attr2 = random.sample(attributes, 2)
        constraints.append({
            "type": "neighbor",
            "if": {"attribute": attr1, "value": solution[pos][attr1]},
            "neighbor": {"attribute": attr2, "value": solution[pos+1][attr2]}
        })
    
    # Aumenta o número de restrições de ordem
    num_ordered = random.randint(dimension, dimension * 2)
    for _ in range(num_ordered):
        pos1, pos2 = sorted(random.sample(range(dimension), 2))
        attr = random.choice(attributes)
        constraints.append({
            "type": "ordered",
            "left": {"attribute": attr, "value": solution[pos1][attr]},
            "right": {"attribute": attr, "value": solution[pos2][attr]},
            "immediate": random.choice([True, False])
        })
    
    # 3. Seleção de posições fixas com dicas iniciais
    num_fixed = random.randint(1, max(2, dimension // 2))
    fixed_positions = random.sample(range(dimension), num_fixed)
    fixed = {}
    
    for pos in fixed_positions:
        attr = random.choice(attributes)
        fixed[pos] = {attr: solution[pos][attr]}
    
    return Puzzle(
        name="Puzzle Gerado Programaticamente",
        dimension=dimension,
        domain=values_per_attribute,
        constraints=constraints,
        fixed=fixed
    )

# Exemplo de uso:
def create_sample_puzzle(dimension: int = 5) -> Puzzle:
    """Cria um puzzle de exemplo com dimensão específica"""
    attributes = ["Cor", "Animal", "Bebida", "Profissão", "Hobby"]
    
    values_per_attribute = {
        "Cor": [f"Cor_{i}" for i in range(dimension)],
        "Animal": [f"Animal_{i}" for i in range(dimension)],
        "Bebida": [f"Bebida_{i}" for i in range(dimension)],
        "Profissão": [f"Profissão_{i}" for i in range(dimension)],
        "Hobby": [f"Hobby_{i}" for i in range(dimension)]
    }
    
    return generate_puzzle(dimension, attributes, values_per_attribute)

# Exemplos de puzzles pré-definidos (podem ser movidos para outro arquivo se necessário)
ZEBRA_PUZZLE = Puzzle(
    name="Quebra-cabeça Zebra Clássico",
    dimension=5,
    domain={
        "Nacionalidade": ["Norueguês", "Dinamarquês", "Britânico", "Alemão", "Sueco"],
        "Cor": ["Amarelo", "Azul", "Vermelho", "Verde", "Branco"],
        "Bebida": ["Água", "Chá", "Leite", "Café", "Cerveja"],
        "Cigarro": ["Dunhill", "Blend", "Pall Mall", "Prince", "BlueMaster"],
        "Animal": ["Gatos", "Cavalos", "Pássaros", "Zebra", "Cachorros"]
    },
    constraints=[
        {"type": "direct", "if": {"attribute": "Nacionalidade", "value": "Britânico"}, 
         "then": {"attribute": "Cor", "value": "Vermelho"}},
        {"type": "direct", "if": {"attribute": "Nacionalidade", "value": "Sueco"}, 
         "then": {"attribute": "Animal", "value": "Cachorros"}},
        {"type": "direct", "if": {"attribute": "Nacionalidade", "value": "Dinamarquês"}, 
         "then": {"attribute": "Bebida", "value": "Chá"}},
        {"type": "neighbor", "if": {"attribute": "Cor", "value": "Verde"}, 
         "neighbor": {"attribute": "Cor", "value": "Branco"}},
        {"type": "direct", "if": {"attribute": "Cor", "value": "Verde"}, 
         "then": {"attribute": "Bebida", "value": "Café"}},
        {"type": "direct", "if": {"attribute": "Cigarro", "value": "Pall Mall"}, 
         "then": {"attribute": "Animal", "value": "Pássaros"}},
        {"type": "direct", "if": {"attribute": "Cor", "value": "Amarelo"}, 
         "then": {"attribute": "Cigarro", "value": "Dunhill"}},
        {"type": "direct", "if": {"attribute": "Cigarro", "value": "BlueMaster"}, 
         "then": {"attribute": "Bebida", "value": "Cerveja"}},
        {"type": "direct", "if": {"attribute": "Nacionalidade", "value": "Alemão"}, 
         "then": {"attribute": "Cigarro", "value": "Prince"}},
        {"type": "neighbor", "if": {"attribute": "Cigarro", "value": "Blend"}, 
         "neighbor": {"attribute": "Animal", "value": "Gatos"}},
        {"type": "neighbor", "if": {"attribute": "Animal", "value": "Cavalos"}, 
         "neighbor": {"attribute": "Cigarro", "value": "Dunhill"}},
        {"type": "neighbor", "if": {"attribute": "Cigarro", "value": "Blend"}, 
         "neighbor": {"attribute": "Bebida", "value": "Água"}}
    ],
    fixed={
        0: {"Nacionalidade": "Norueguês"},
        2: {"Bebida": "Leite"}
    }
)

CARS_PUZZLE = Puzzle(
    name="Quebra-cabeça: Carros na Corrida",
    dimension=5,
    domain={
        "Piloto": ["Hamilton", "Vettel", "Alonso", "Verstappen", "Norris"],
        "Cor": ["Vermelho", "Azul", "Verde", "Amarelo", "Preto"],
        "Bebida": ["Água", "Chá", "Café", "Refrigerante", "Suco"],
        "Marca": ["Ferrari", "Lamborghini", "Porsche", "Bugatti", "McLaren"],
        "Acessório": ["Relógio", "Óculos", "Chapéu", "Luvas", "Jaqueta"]
    },
    constraints=[
        {"type": "direct", "if": {"attribute": "Piloto", "value": "Hamilton"}, 
         "then": {"attribute": "Marca", "value": "McLaren"}},
        {"type": "direct", "if": {"attribute": "Cor", "value": "Vermelho"}, 
         "then": {"attribute": "Bebida", "value": "Café"}},
        {"type": "neighbor", "if": {"attribute": "Acessório", "value": "Óculos"}, 
         "neighbor": {"attribute": "Marca", "value": "Ferrari"}},
        {"type": "direct", "if": {"attribute": "Piloto", "value": "Vettel"}, 
         "then": {"attribute": "Acessório", "value": "Relógio"}},
        {"type": "ordered", "left": {"attribute": "Cor", "value": "Verde"}, 
         "right": {"attribute": "Cor", "value": "Preto"}, "immediate": True}
    ],
    fixed={
        0: {"Piloto": "Hamilton"},
        2: {"Bebida": "Café"}
    }
)

# Criar um puzzle 3x3 personalizado
mini_puzzle = generate_puzzle(
    dimension=3,
    attributes=["Cor", "Animal"],
    values_per_attribute={
        "Cor": ["Vermelho", "Azul", "Verde"],
        "Animal": ["Gato", "Cachorro", "Pássaro"]
    }
)

# Ou usar o gerador de exemplo
puzzle_5x5 = create_sample_puzzle(dimension=5) 