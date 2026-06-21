# -*- coding: utf-8 -*-
import os
import time
import pandas as pd
import re
import torch
import gc
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ==========================================
# 1. CONFIGURACIÓN DEL ENTORNO
# ==========================================
print("Iniciando Pipeline Nocturno en Google Cloud (A100 80GB)...")

CARPETA_BASE = "." 
CARPETA_IMAGENES = os.path.join(CARPETA_BASE, "dataset_imagenes_ttt_balanced")
ARCHIVO_MAESTRO = os.path.join(CARPETA_BASE, "dataset_jugadas_ttt_minimax_balanced.csv")

OUT_TEXTO_CRUDO = "resultados_gemma3_27b_ABLACION_TEXTO.csv"
OUT_TEXTO_FINAL = "resultados_gemma3_27b_ABLACION_TEXTO_VALIDADO.csv"
OUT_MULTI_CRUDO = "resultados_gemma3_27b_MULTIMODAL.csv"
OUT_MULTI_FINAL = "resultados_gemma3_27b_MULTIMODAL_VALIDADO.csv"

ITERACIONES_POR_TABLERO = 10 

# ==========================================
# 2. FUNCIONES DE VALIDACIÓN (MINIMAX)
# ==========================================
def verificar_ganador(tablero, jugador):
    lineas = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
    for linea in lineas:
        if tablero[linea[0]] == tablero[linea[1]] == tablero[linea[2]] == jugador: return True
    return False

def extraer_coordenada(texto):
    texto_str = str(texto).upper()
    texto_busqueda = texto_str.split("NEXT MOVE")[-1] if "NEXT MOVE" in texto_str else texto_str
    match = re.search(r'\[\s*(\d)\s*,\s*(\d)\s*\]', texto_busqueda)
    if match:
        f, c = int(match.group(1)), int(match.group(2))
        if 1 <= f <= 3 and 1 <= c <= 3: return f, c
    return None, None

def auditar_jugada(row):
    texto = row['respuesta_bruta_gemma']
    tablero_str = row['tablero']
    jugador = row['rol_asignado']
    jugada_minimax = str(row.get('movimiento_esperado', 'END_GAME'))
    
    if jugada_minimax == "END_GAME":
        se_dio_cuenta = "END_GAME" in str(texto).upper() or "FIN" in str(texto).upper()
        return pd.Series([se_dio_cuenta, None, None, None, se_dio_cuenta])

    f, c = extraer_coordenada(texto)
    if f is None or c is None: return pd.Series([False, False, False, False, False])

    tablero = tablero_str.replace('/', '')
    idx = (f - 1) * 3 + (c - 1)
    legal = (0 <= idx < 9 and tablero[idx] == '-')
    if not legal: return pd.Series([True, False, False, False, False])

    tab_list = list(tablero)
    tab_list[idx] = jugador
    gano = verificar_ganador(tab_list, jugador)
    
    oponente = 'O' if jugador == 'X' else 'X'
    tab_op = list(tablero)
    tab_op[idx] = oponente
    bloqueo = verificar_ganador(tab_op, oponente)

    return pd.Series([True, True, gano, bloqueo, (f"[{f}, {c}]" == jugada_minimax)])

# ==========================================
# 3. CARGA ÚNICA DEL MODELO (Gemma 3 27B)
# ==========================================
print("\n[INIT] Cargando la bestia Gemma 3 27B en GPU A100. Esto tomará un momento...")
MODEL_ID = "google/gemma-3-27b-it"

procesador = AutoProcessor.from_pretrained(MODEL_ID)
procesador.tokenizer.padding_side = "left"
if procesador.tokenizer.pad_token is None:
    procesador.tokenizer.pad_token = procesador.tokenizer.eos_token

modelo = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
print("[OK] ¡Modelo gigante cargado exitosamente en VRAM!")

df_completo = pd.read_csv(ARCHIVO_MAESTRO)
registros = df_completo.to_dict('records')
total_registros = len(registros)

# ==========================================
# FASE 1: ABLACIÓN DE TEXTO PURO
# ==========================================
LOTE_TEXTO = 10
resultados_texto = []
print(f"\n" + "="*50)
print(f"🌟 INICIANDO FASE 1: TEXTO PURO ({ITERACIONES_POR_TABLERO} Iters | Lotes de {LOTE_TEXTO}) 🌟")
print("="*50)

inicio_fase1 = time.time()
for i, fila in enumerate(registros):
    t_inicio = time.time()
    turno_actual, tablero_str = fila['turno_de'], fila['tablero']
    
    filas = tablero_str.split('/')
    tablero_visual = f"""    Col 1 Col 2 Col 3
Row 1: [{filas[0][0]}] [{filas[0][1]}] [{filas[0][2]}]
Row 2: [{filas[1][0]}] [{filas[1][1]}] [{filas[1][2]}]
Row 3: [{filas[2][0]}] [{filas[2][1]}] [{filas[2][2]}]"""

    prompt = f"""You are an expert Tic-Tac-Toe analyst.
The current board state is represented below. Empty spaces are represented by '-'.
{tablero_visual}

You are playing as '{turno_actual}'.
Please respond STRICTLY using the following structure:

1. SPATIAL ANALYSIS:
List the exact coordinates of all pieces currently on the board using the [Row, Column] format (from 1 to 3).
- X positions: [e.g., [1, 2], [3, 3] or 'None']
- O positions: [e.g., [2, 2] or 'None']

2. STRATEGIC REASONING:
Provide a concise plain-text paragraph explaining the current advantage, immediate threats, and your action plan.

3. NEXT MOVE:
If the game is already finished (someone has won or the board is full), output exactly: END_GAME
Otherwise, output your exact next move targeting an EMPTY '-' space using ONLY numeric coordinates: [Row, Column] (e.g., [1, 3] or [3, 2]).

STRICT RULE: Your entire response and reasoning must be written exclusively in ENGLISH. Do NOT use Spanish."""

    try:
        mensajes = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        texto_fmt = procesador.apply_chat_template(mensajes, add_generation_prompt=True)
        
        inputs = procesador(text=[texto_fmt]*LOTE_TEXTO, return_tensors="pt", padding=True).to("cuda")
        outputs = modelo.generate(**inputs, max_new_tokens=300, temperature=0.2, do_sample=True)
        tokens_entrada = inputs["input_ids"].shape[1]
        
        for idx in range(LOTE_TEXTO):
            respuesta = procesador.decode(outputs[idx][tokens_entrada:], skip_special_tokens=True)
            data_res = fila.copy()
            data_res["iteracion_num"] = idx + 1
            data_res["rol_asignado"] = turno_actual
            data_res["respuesta_bruta_gemma"] = respuesta
            resultados_texto.append(data_res)

        print(f"[FASE 1] Tablero [{i+1}/{total_registros}] | Iters: 10 | Tiempo: {round(time.time() - t_inicio, 2)}s")
        if (i + 1) % 20 == 0: pd.DataFrame(resultados_texto).to_csv(OUT_TEXTO_CRUDO, index=False)

    except Exception as e:
        print(f"ERROR Fase 1 - Tablero {fila['tablero']}: {e}")

df_txt = pd.DataFrame(resultados_texto)
df_txt.to_csv(OUT_TEXTO_CRUDO, index=False)
print(f"✅ FASE 1 terminada. Validando...")
df_txt[['formato_valido', 'movimiento_legal', 'gano_partida', 'bloqueo_amenaza', 'match_minimax']] = df_txt.apply(auditar_jugada, axis=1)
df_txt.to_csv(OUT_TEXTO_FINAL, index=False)
print(f"✅ FASE 1 COMPLETADA y auditada en {OUT_TEXTO_FINAL}")

# ==========================================
# LIMPIEZA PROFUNDA DE VRAM
# ==========================================
del resultados_texto
del df_txt
gc.collect()
torch.cuda.empty_cache()
time.sleep(5) 

# ==========================================
# FASE 2: VISIÓN MULTIMODAL
# ==========================================
LOTE_MULTI = 5
resultados_multi = []
print(f"\n" + "="*50)
print(f"🌟 INICIANDO FASE 2: MULTIMODAL ({ITERACIONES_POR_TABLERO} Iters | Lotes de {LOTE_MULTI}) 🌟")
print("="*50)

inicio_fase2 = time.time()
for i, fila in enumerate(registros):
    t_inicio = time.time()
    turno_actual = fila['turno_de']
    
    prompt = f"""You are an expert Tic-Tac-Toe analyst.
The attached image shows the current state of a board. You are playing as '{turno_actual}'.
Please respond STRICTLY using the following structure:

1. SPATIAL ANALYSIS:
List the exact coordinates of all pieces currently on the board using the [Row, Column] format (from 1 to 3).
- X positions: [e.g., [1, 2], [3, 3] or 'None']
- O positions: [e.g., [2, 2] or 'None']

2. STRATEGIC REASONING:
Provide a concise plain-text paragraph explaining the current advantage, immediate threats, and your action plan.

3. NEXT MOVE:
If the game is already finished (someone has won or the board is full), output exactly: END_GAME
Otherwise, output your exact next move using ONLY numeric coordinates: [Row, Column] (e.g., [1, 3] or [3, 2]). Do NOT use words like 'Center' or 'Right'.

STRICT RULE: Your entire response and reasoning must be written exclusively in ENGLISH. Do NOT use Spanish."""

    filename = fila['tablero'].replace('/', '_') + '.png'
    ruta_imagen = os.path.join(CARPETA_IMAGENES, fila['etapa_juego'], filename)

    try:
        imagen = Image.open(ruta_imagen).convert("RGB")
        mensajes = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        texto_fmt = procesador.apply_chat_template(mensajes, add_generation_prompt=True)
        
        iteracion_actual = 1
        for _ in range(2): # Ejecuta 2 pasadas de 5 para llegar a 10
            inputs = procesador(images=[[imagen]]*LOTE_MULTI, text=[texto_fmt]*LOTE_MULTI, return_tensors="pt", padding=True).to("cuda")
            outputs = modelo.generate(**inputs, max_new_tokens=300, temperature=0.2, do_sample=True)
            tokens_entrada = inputs["input_ids"].shape[1]
            
            for idx in range(LOTE_MULTI):
                respuesta = procesador.decode(outputs[idx][tokens_entrada:], skip_special_tokens=True)
                data_res = fila.copy()
                data_res["iteracion_num"] = iteracion_actual
                data_res["rol_asignado"] = turno_actual
                data_res["respuesta_bruta_gemma"] = respuesta
                resultados_multi.append(data_res)
                iteracion_actual += 1

        print(f"[FASE 2] Tablero [{i+1}/{total_registros}] | Iters: 10 (2x5) | Tiempo: {round(time.time() - t_inicio, 2)}s")
        if (i + 1) % 20 == 0: pd.DataFrame(resultados_multi).to_csv(OUT_MULTI_CRUDO, index=False)

    except Exception as e:
        print(f"ERROR Fase 2 - Tablero {fila['tablero']}: {e}")

df_mm = pd.DataFrame(resultados_multi)
df_mm.to_csv(OUT_MULTI_CRUDO, index=False)
print(f"✅ FASE 2 terminada. Validando...")
df_mm[['formato_valido', 'movimiento_legal', 'gano_partida', 'bloqueo_amenaza', 'match_minimax']] = df_mm.apply(auditar_jugada, axis=1)
df_mm.to_csv(OUT_MULTI_FINAL, index=False)

print("\n" + "="*65)
print(f" 🎉 PIPELINE NOCTURNO COMPLETADO CON ÉXITO 🎉")
print("="*65)