import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ruta_modelo = "/home/manuel/gemma-3-27b-it"

print("Configurando cuantización a 8-bits...")
# Le decimos a PyTorch que cargue los pesos usando solo 1 byte por parámetro
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

print("Cargando el tokenizador...")
tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)

print("Cargando el modelo en la A100 (esto tomará un par de minutos)...")
model = AutoModelForCausalLM.from_pretrained(
    ruta_modelo,
    device_map="cuda",
    quantization_config=quantization_config
)

# Verificamos cuánta memoria logramos ahorrar
vram_usada = torch.cuda.memory_allocated() / 1024**3
print(f"\n¡Éxito! El modelo cargó perfectamente.")
print(f"VRAM usada: {vram_usada:.2f} GB de 40.0 GB disponibles.\n")

print("--- INICIANDO PRUEBA DE RAZONAMIENTO ---")
# Formato de chat estructurado para modelos Instruct
mensajes = [
    {"role": "user", "content": "Eres un experto en el juego de Tic-Tac-Toe. ¿Cuál es estratégicamente el mejor primer movimiento en un tablero vacío y por qué?"}
]

# Preparamos el prompt con el template de Gemma
inputs = tokenizer.apply_chat_template(
    mensajes, 
    add_generation_prompt=True, 
    return_tensors="pt"
).to("cuda")

print("Generando respuesta...\n")
outputs = model.generate(
    inputs, 
    max_new_tokens=200,
    temperature=0.7 # Un poco de creatividad pero manteniendo lógica
)

# Decodificamos solo la parte nueva generada (ignorando el prompt inicial)
respuesta = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print(respuesta)