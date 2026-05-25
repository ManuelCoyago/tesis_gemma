import torch
from transformers import AutoModelForCausalLM

print("Cargando modelo Gemma 3 27B en 16-bits...")
print("VRAM antes de cargar: ", torch.cuda.memory_allocated() / 1024**3, "GB")

# Al intentar cargar el modelo en bfloat16, superará los 40GB de tu A100
try:
    model = AutoModelForCausalLM.from_pretrained(
        "/home/manuel/gemma-3-27b-it", # Ruta local donde lo estamos descargando
        device_map="cuda",             # Forzamos que vaya a la GPU
        torch_dtype=torch.bfloat16     # Precisión de 16 bits
    )
    print("¡Exito! El modelo cargó (Esto no debería pasar con 40GB)")
except RuntimeError as e:
    print("\n--- ERROR DE MEMORIA INTERCEPTADO ---")
    print(e)
    print("-------------------------------------")
    print("\nEste es el error (OOM) significa que se quedo sin vram")