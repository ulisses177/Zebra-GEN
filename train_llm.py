import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import numpy as np
from typing import Dict, List

class PuzzleSolverDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Carregar dataset
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def extract_correct_answer(self, item):
        """
        Extrai a resposta correta do campo solution baseado na feedback_question
        """
        question = item['feedback_question']
        solution = item['solution']
        
        # Extrair o número do item e o atributo da pergunta
        # Exemplo: "Com base na dedução, qual é o {atributo} do item {número}?"
        import re
        match = re.search(r"qual é o ([^ ]+) do item (\d+)", question)
        if not match:
            return None
            
        attribute = match.group(1)
        item_num = int(match.group(2)) - 1  # Converter para índice 0-based
        
        # Pegar o valor correto da solução
        if 0 <= item_num < len(solution):
            return solution[item_num].get(attribute)
        return None
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extrair a resposta correta da solução
        correct_answer = self.extract_correct_answer(item)
        if correct_answer is None:
            raise ValueError(f"Não foi possível extrair a resposta correta para o item {idx}")
        
        # Formatar o prompt com o enunciado
        prompt = f"Resolva o seguinte puzzle:\n{item['enunciado']}\n\n"
        
        # Formatar a resposta esperada com estrutura <think></think>
        thoughts = "\n".join([
            f"Passo {i+1}: {step}" for i, step in enumerate(item['deduction'])
        ])
        
        response = f"<think>\n{thoughts}\n</think>\n{correct_answer}"
        
        # Concatenar prompt e resposta
        full_text = f"{prompt}{response}"
        
        # Tokenizar
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze(),
            "correct_answer": correct_answer  # Adicionar para uso no reward model
        }

class PuzzleRewardModel:
    """Modelo de recompensa que verifica se a resposta está correta"""
    def __init__(self):
        pass
    
    def evaluate_response(self, 
                         generated_response: str, 
                         correct_answer: str, 
                         deduction_steps: List[str]) -> float:
        """
        Avalia a resposta gerada e retorna uma pontuação entre 0 e 1
        
        Args:
            generated_response: Resposta completa gerada pelo modelo
            correct_answer: Resposta correta extraída da solution
            deduction_steps: Lista de passos dedutivos do dataset
        """
        score = 0.0
        
        # Extrair a resposta final (após o </think>)
        try:
            final_answer = generated_response.split("</think>")[-1].strip()
        except:
            return 0.0
            
        # Verificar se tem a estrutura <think></think>
        if "<think>" in generated_response and "</think>" in generated_response:
            score += 0.15  # 15% pelo formato correto
            
        # Extrair os pensamentos
        try:
            thoughts = generated_response.split("<think>")[1].split("</think>")[0].strip()
            if thoughts:
                score += 0.15  # 15% por ter pensamentos
        except:
            pass
            
        # Verificar se a resposta final está correta
        if final_answer.lower() == str(correct_answer).lower():
            score += 0.7  # 70% pelo acerto da resposta
            
        return score

# ... resto do código permanece igual ... 