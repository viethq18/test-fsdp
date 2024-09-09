import torch 
import os
from transformers import AutoModelForCausalLM

print(torch.cuda.is_available())

model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B-Instruct",
        attn_implementation="flash_attention_2",
    )
rank = int(int(os.environ["RANK"]))

print(model)
print(f"RANK: {rank}")