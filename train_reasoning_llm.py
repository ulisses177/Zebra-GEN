from trl import PPOConfig

# Testar quais parâmetros são aceitos
test_config = PPOConfig(output_dir="./test_output")
print("Parâmetros disponíveis no PPOConfig:")
for param, value in test_config.__dict__.items():
    print(f"{param}: {value}")


import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
import torch
import json
import re




def prepare_dataset(data_path: str):
    """Prepara o dataset no formato correto para treinamento PPO"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data:
        # O enunciado será nosso input para o modelo
        query = item['enunciado']
        
        # A dedução será usada como parte do processo de raciocínio
        deduction = item['deduction']
        
        # A resposta correta vem do campo correct_answer
        answer = item['correct_answer']
        
        formatted_data.append({
            'input_ids': query,  # O texto do enunciado
            'query': query,      # Mantemos uma cópia do texto original
            'deduction': deduction,  # Os passos de dedução para reward
            'answer': answer     # A resposta correta para reward
        })
    
    return Dataset.from_list(formatted_data)

def reward_function(responses, answer, **kwargs):
    """Função de recompensa que avalia as respostas"""
    rewards = []
    
    for response, correct_answer in zip(responses, answer):
        score = 0.0
        
        # Verificar formato <think></think>
        if "<think>" in response and "</think>" in response:
            score += 0.15
            
            # Verificar se tem conteúdo nos pensamentos
            thoughts = response.split("<think>")[1].split("</think>")[0].strip()
            if thoughts:
                score += 0.15
        
        # Extrair resposta final
        try:
            final_answer = response.split("</think>")[-1].strip()
            # 70% do peso para resposta correta
            if final_answer.lower() == str(correct_answer).lower():
                score += 0.7
        except Exception as e:
            pass
            
        rewards.append(score)
    
    return torch.tensor(rewards)

class RewardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, responses, answers, deductions):
        rewards = []
        for response, correct_answer, deduction in zip(responses, answers, deductions):
            score = 0.0
            
            # Verificar formato <think></think>
            if "<think>" in response and "</think>" in response:
                score += 0.15
                
                # Verificar se os pensamentos são similares aos do deduction
                thoughts = response.split("<think>")[1].split("</think>")[0].strip()
                if thoughts and any(d.lower() in thoughts.lower() for d in deduction):
                    score += 0.15
            
            # Extrair resposta final
            try:
                final_answer = response.split("</think>")[-1].strip()
                if final_answer.lower() == str(correct_answer).lower():
                    score += 0.7
            except:
                pass
                
            rewards.append(score)
        
        return torch.tensor(rewards, device=responses.device)

def train_model(
    dataset_path: str,
    output_dir: str,
    max_seq_length: int = 1024,
    lora_rank: int = 64,
    batch_size: int = 4,
    num_steps: int = 1000
):
    # Verificar CUDA
    print("CUDA disponível:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
    
    # Carregar modelo e tokenizer
    model_name = "meta-llama/Llama-3.2-3B"
    print("Carregando modelo base...")
    
    try:
        # Criar modelo base
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("Modelo base carregado com sucesso!")
        
        print("Carregando tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("Tokenizer carregado com sucesso!")
        
        # Adicionar tokens especiais
        print("Adicionando tokens especiais...")
        special_tokens = {"additional_special_tokens": ["<think>", "</think>"]}
        tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
        
        # Configurar LoRA
        print("Configurando LoRA...")
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        print("LoRA configurado com sucesso!")
        
        # Adicionar value head para o PPO
        print("Adicionando value head...")
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("Value head adicionado com sucesso!")

        # Criar modelo de referência
        print("Criando modelo de referência...")
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        for param in ref_model.parameters():
            param.requires_grad = False
        print("Modelo de referência criado com sucesso!")
        
    except Exception as e:
        print(f"Erro ao carregar o modelo: {str(e)}")
        raise e

    # Verificar CUDA e memória
    if torch.cuda.is_available():
        print("\nInformações da GPU:")
        print(f"GPU em uso: {torch.cuda.get_device_name(0)}")
        print(f"Memória alocada: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memória reservada: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Criar modelo de recompensa
    reward_model = RewardModel()
    
    # Configurar PPO
    config = PPOConfig(
        learning_rate=5e-6,
        batch_size=batch_size,
        max_steps=num_steps,
        gradient_accumulation_steps=1,
        max_grad_norm=0.1,
        output_dir=output_dir,
        num_ppo_epochs=4,
        temperature=0.7
    )

    # Inicializar trainer do PPO com os parâmetros corretos
    ppo_trainer = PPOTrainer(
        args=config,  # Mantemos args
        model=model,
        ref_model=ref_model,
        processing_class=tokenizer,  # Substituído 'tokenizer' por 'processing_class'
        train_dataset=prepare_dataset(dataset_path),  # Usando 'train_dataset' em vez de 'dataset'
        reward_model=reward_model
    )

    # Configurar geração
    generation_config = model.generation_config
    generation_config.max_length = max_seq_length
    generation_config.do_sample = True
    generation_config.temperature = 0.7
    model.generation_config = generation_config

    # Treinar
    for epoch in range(num_steps):
        batch = next(iter(ppo_trainer.dataloader))
        queries = batch['query']
        answers = batch['answer']
        deductions = batch['deduction']
        
        # Gerar respostas usando a configuração de geração
        response_tensors = ppo_trainer.generate(
            queries,
            generation_config=generation_config
        )
        
        # Decodificar respostas
        responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
        # Atualizar modelo usando PPO
        stats = ppo_trainer.step(response_tensors, answers, deductions)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: {stats}")

    # Salvar modelo final
    ppo_trainer.save_pretrained(f"{output_dir}/final_model")

if __name__ == "__main__":
    train_model(
        dataset_path="datasets/dataset_checkpoint_120.json",
        output_dir="trained_reasoning_model",
        batch_size=4,
        num_steps=1000
    ) 