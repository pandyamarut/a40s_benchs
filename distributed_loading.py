from accelerate import Accelerator
from transformers import AutoModelForCausalLM
import time
import torch

# Initialize Accelerator
accelerator = Accelerator()

# Function to load the model
def load_model():
    model_path = "mistralai/Mistral-7B-v0.1"
    print(f"Model name: {model_path}")
    start = time.time()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,    
        device_map={"": accelerator.process_index},  # or use device_map="auto"
        torch_dtype=torch.bfloat16,
    )
    
    end = time.time()
    print(f"Loading time = {end - start}")
    return model

def main():
    load_model()

if __name__ == "__main__":
    main()

