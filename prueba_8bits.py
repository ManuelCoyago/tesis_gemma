import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ruta_modelo = "/home/manuel/gemma-3-27b-it"

print("Configurando cuantización a 8-bits...")
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

print("Cargando el tokenizador...")
tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)

print("Cargando el modelo en la A100...")
model = AutoModelForCausalLM.from_pretrained(
    ruta_modelo,
    device_map="cuda",
    quantization_config=quantization_config
)

print("--- INICIANDO PRUEBA DE RAZONAMIENTO ---")
mensajes = [
    {"role": "user", "content": "Eres un experto en el juego de Tic-Tac-Toe. ¿Cuál es estratégicamente el mejor primer movimiento en un tablero vacío y por qué?"}
]

# 1. Armamos la estructura de texto plano que Gemma entiende
texto_prompt = tokenizer.apply_chat_template(mensajes, tokenize=False, add_generation_prompt=True)

# 2. Convertimos el texto a tensores y los enviamos a la GPU
inputs = tokenizer(texto_prompt, return_tensors="pt").to("cuda")

print("Generando respuesta...\n")
outputs = model.generate(
    **inputs, # <-- El doble asterisco desempaqueta el diccionario correctamente
    max_new_tokens=200,
    temperature=0.7
)

# Cortamos el prompt original para que solo imprima la respuesta nueva
respuesta = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(respuesta)