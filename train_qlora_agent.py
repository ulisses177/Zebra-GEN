#!/usr/bin/env python3
import torch
import json
import re
import time
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import LoraConfig, get_peft_model
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from tqdm import tqdm   # Importa o tqdm para barra de progresso

def prepare_dataset(data_path: str):
    """Prepara o dataset no formato esperado pelo treinamento."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data:
        # Garante que query, deduction e answer são strings
        query = str(item['enunciado'])
        
        # Converte deduction para string se for lista
        if isinstance(item['deduction'], list):
            deduction = " ".join(str(d) for d in item['deduction'])
        else:
            deduction = str(item['deduction'])
            
        answer = str(item['solution'])
        
        # Monta o prompt completo
        prompt = f"Questão: {query}\n\n<think>{deduction}</think>\n\nResposta: {answer}"
        
        formatted_data.append({
            'input_ids': prompt,  # Texto completo para treinamento
            'query': query,       # Apenas a questão (para geração)
            'deduction': deduction,
            'answer': answer
        })
    
    return Dataset.from_list(formatted_data)

def reward_function(responses, answers, **kwargs):
    """
    Função simples de recompensa que pontua:
       - Presença dos marcadores <think>
       - Se há conteúdo entre os marcadores
       - Se a resposta final (após </think>) coincide com a resposta correta
    """
    rewards = []
    for response, correct_answer in zip(responses, answers):
        score = 0.0
        if "<think>" in response and "</think>" in response:
            score += 0.15
            thoughts = response.split("<think>")[1].split("</think>")[0].strip()
            if thoughts:
                score += 0.15
        try:
            final_answer = response.split("</think>")[-1].strip()
            if final_answer.lower() == str(correct_answer).lower():
                score += 0.7
        except Exception as e:
            pass
        rewards.append(score)
    return torch.tensor(rewards)

class RewardModel(torch.nn.Module):
    """
    Modelo de recompensa que avalia as respostas baseando-se no conteúdo dos passos de 
    dedução e na formatação (tokens <think> e resposta final).
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, responses, answers, deductions):
        rewards = []
        for response, correct_answer, deduction in zip(responses, answers, deductions):
            score = 0.0
            if "<think>" in response and "</think>" in response:
                score += 0.15
                thoughts = response.split("<think>")[1].split("</think>")[0].strip()
                # Se parte do conteúdo do passo dedutivo estiver presente nos pensamentos
                if thoughts and any(d.lower() in thoughts.lower() for d in deduction):
                    score += 0.15
            try:
                final_answer = response.split("</think>")[-1].strip()
                if final_answer.lower() == str(correct_answer).lower():
                    score += 0.7
            except Exception as e:
                pass
            rewards.append(score)
        # Retorna o tensor de recompensas
        return torch.tensor(rewards)

def update_lora_parameters(avg_reward: float, ppo_trainer, base_lr: float):
    """
    Agente simples que utiliza a média da recompensa para atualizar dinamicamente o learning rate.
    
    Exemplo: Se a média de recompensa for baixa (< threshold), aumenta o lr
             Caso contrário, diminui o lr (sempre dentro de limites definidos).
    """
    threshold = 0.7
    for param_group in ppo_trainer.optimizer.param_groups:
        current_lr = param_group['lr']
        if avg_reward < threshold:
            new_lr = min(current_lr * 1.1, base_lr * 1.5)  # limitar para não aumentar indefinidamente
        else:
            new_lr = max(current_lr * 0.9, base_lr * 0.5)  # limitar para não reduzir demais
        param_group['lr'] = new_lr
    print(f"Updated learning rate based on average reward {avg_reward:.3f}")

def train_qlora_model(
    dataset_path: str,
    output_dir: str,
    max_seq_length: int = 1024,
    lora_rank: int = 64,
    batch_size: int = 4,
    num_epochs: int = 5
):
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Carregar modelo e tokenizer
    model_name = "meta-llama/Llama-3.2-3B"
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )
    
    # Carregar modelo base
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Se não houver pad_token, adiciona um novo token [PAD]
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("Added [PAD] token as padding token.")
    special_tokens = {"additional_special_tokens": ["<think>", "</think>"]}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Configurar LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Preparar dataset
    dataset = prepare_dataset(dataset_path)
    
    # Modificar o DataLoader para usar um collate_fn personalizado
    def collate_fn(batch):
        # Pega apenas os campos necessários do batch
        input_texts = [item['input_ids'] for item in batch]
        
        # Tokeniza os textos com padding
        inputs = tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt"
        )
        
        # Retorna um dicionário com todos os campos necessários
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'query': [item['query'] for item in batch],
            'deduction': [item['deduction'] for item in batch],
            'answer': [item['answer'] for item in batch]
        }
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn  # Adiciona a função de coleta personalizada
    )
    
    # Otimizador com learning rate adaptativo
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=2
    )
    
    # Modelo de recompensa para avaliação
    reward_model = RewardModel()
    
    # Loop de treinamento
    best_reward = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        # Envolve o dataloader com tqdm para imprimir progresso
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            # Preparar inputs corretamente como dicionário
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            
            # Forward pass com labels igual a input_ids
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            # Atualiza a barra de progresso com a loss atual
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # A cada 5 batches, avaliar e ajustar
            if (batch_idx + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    # Gerar respostas; observe o uso de **inputs para desempacotar o dicionário
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=128,  # Gera 128 tokens adicionais além do prompt
                        num_return_sequences=1,
                        temperature=0.7
                    )
                    responses = tokenizer.batch_decode(generated, skip_special_tokens=True)

                    # Avaliar respostas
                    rewards = reward_model(
                        responses, 
                        batch['answer'], 
                        batch['deduction']
                    )
                    avg_reward = rewards.mean().item()

                    # Ajustar learning rate baseado na recompensa
                    scheduler.step(avg_reward)

                    # Salvar melhor modelo
                    if avg_reward > best_reward:
                        best_reward = avg_reward
                        model.save_pretrained(f"{output_dir}/best_model")

                    # Escreve os resultados do batch
                    tqdm.write(f"Epoch {epoch} | Batch {batch_idx}: Loss: {loss.item():.4f}, Reward: {avg_reward:.4f}")
                model.train()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
    
    # Salvar modelo final
    model.save_pretrained(f"{output_dir}/final_model")
    print("Training completed!")

if __name__ == '__main__':
    train_qlora_model(
        dataset_path="datasets/dataset_checkpoint_120.json",
        output_dir="trained_qlora_model",
        batch_size=4,
        num_epochs=5
    ) 