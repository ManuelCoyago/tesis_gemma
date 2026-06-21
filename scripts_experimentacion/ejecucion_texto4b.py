# -*- coding: utf-8 -*-
import os
import time
import pandas as pd
import re
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# ==========================================
# 1. CONFIGURACIÓN DEL ENTORNO LOCAL (GCP)
# ==========================================
print("Iniciando entorno en Google Cloud (Debian)...")

CARPETA_BASE = "." 
# Usamos tu dataset original balanceado sin la columna minimax
ARCHIVO_ENTRADA = os.path.join(CARPETA_BASE, "dataset_jugadas_ttt_balanced.csv")
# El oráculo que usaremos solo en la fase de validación
ARCHIVO_ORACULO = os.path.join(CARPETA_BASE, "dataset_jugadas_ttt_minimax_balanced.csv")

# Archivos de salida ESPECÍFICOS PARA 4B
ARCHIVO_INFERENCIA = os.path.join(CARPETA_BASE, "resultados_gemma3_4b_ABLACION_TEXTO.csv")
ARCHIVO_FINAL = os.path.join(CARPETA_BASE, "resultados_gemma3_4b_ABLACION_TEXTO_VALIDADO.csv")

ITERACIONES_POR_TABLERO = 10 
# El 4B es ligero y estamos en texto puro. ¡Lote de 10 iteraciones de un solo golpe!
LOTE_MAXIMO = 10 

# ==========================================
# 2. CARGA DEL MODELO (Gemma 3 4B)
# ==========================================
print("\nCargando Gemma 3 4B en GPU para prueba de ablación (Solo Texto)...")
MODEL_ID = "google/gemma-3-4b-it"

procesador = AutoProcessor.from_pretrained(MODEL_ID)
procesador.tokenizer.padding_side = "left"
if procesador.tokenizer.pad_token is None:
    procesador.tokenizer.pad_token = procesador.tokenizer.eos_token

modelo = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("¡Modelo cargado exitosamente en VRAM!")

# ==========================================
# 3. FASE DE INFERENCIA (EJECUCIÓN MASIVA)
# ==========================================
print(f"\nCargando dataset maestro desde: {ARCHIVO_ENTRADA}")
df_completo = pd.read_csv(ARCHIVO_ENTRADA)
registros = df_completo.to_dict('records')
total_registros = len(registros)
total_inferencias = total_registros * ITERACIONES_POR_TABLERO

resultados = []
print(f"\n¡INICIANDO EXPERIMENTO FINAL MODO TEXTO 4B ({ITERACIONES_POR_TABLERO} iteraciones por tablero en lotes de {LOTE_MAXIMO})!")

inicio_experimento = time.time()
contador_global = 0

for i, fila in enumerate(registros):
    inicio_tablero = time.time()
    turno_actual = fila['turno_de']
    
    tablero_str = fila['tablero']
    
    filas = tablero_str.split('/')
    tablero_visual = f"""
    Col 1 Col 2 Col 3
Row 1: [{filas[0][0]}] [{filas[0][1]}] [{filas[0][2]}]
Row 2: [{filas[1][0]}] [{filas[1][1]}] [{filas[1][2]}]
Row 3: [{filas[2][0]}] [{filas[2][1]}] [{filas[2][2]}]
"""

    prompt_dinamico = f"""You are an expert Tic-Tac-Toe analyst.
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
        mensajes = [{"role": "user", "content": [{"type": "text", "text": prompt_dinamico}]}]
        texto_formateado = procesador.apply_chat_template(mensajes, add_generation_prompt=True)
        
        iteracion_actual = 1
        chunks = [LOTE_MAXIMO] * (ITERACIONES_POR_TABLERO // LOTE_MAXIMO)
        if ITERACIONES_POR_TABLERO % LOTE_MAXIMO != 0:
            chunks.append(ITERACIONES_POR_TABLERO % LOTE_MAXIMO)

        for tamano_lote in chunks:
            lote_textos = [texto_formateado] * tamano_lote

            inputs = procesador(
                text=lote_textos, 
                return_tensors="pt", 
                padding=True
            ).to("cuda")

            outputs = modelo.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.2,
                do_sample=True
            )

            tokens_entrada = inputs["input_ids"].shape[1]
            
            for idx_lote in range(tamano_lote):
                contador_global += 1
                tokens_generados = outputs[idx_lote][tokens_entrada:]
                respuesta_texto = procesador.decode(tokens_generados, skip_special_tokens=True)
                
                data_resultado = fila.copy()
                data_resultado["iteracion_num"] = iteracion_actual
                data_resultado["rol_asignado"] = turno_actual
                data_resultado["respuesta_bruta_gemma"] = respuesta_texto
                
                resultados.append(data_resultado)
                iteracion_actual += 1

        tiempo_tablero = round((time.time() - inicio_tablero), 2)
        print(f"Tablero [{i+1}/{total_registros}] | 4B Texto Puro | {ITERACIONES_POR_TABLERO} Iters en {tiempo_tablero}s | Progreso: {contador_global}/{total_inferencias}")

        if (i + 1) % 20 == 0:
            pd.DataFrame(resultados).to_csv(ARCHIVO_INFERENCIA, index=False)
            print(f"--> [AUTO-SAVE] Respaldo en disco consolidado.")

    except Exception as e:
        print(f"ERROR Crítico en Tablero {fila['tablero']}: {e}")

tiempo_total_m = round((time.time() - inicio_experimento) / 60, 2)
pd.DataFrame(resultados).to_csv(ARCHIVO_INFERENCIA, index=False)
print(f"\n¡INFERENCIA FINAL COMPLETADA en {tiempo_total_m} min! Archivo: {ARCHIVO_INFERENCIA}")

# ==========================================
# 4. FASE DE VALIDACIÓN CORREGIDA (CRUCE CON ORÁCULO)
# ==========================================
print("\nIniciando Validación Automatizada (cruzando con Oráculo Minimax)...")
df_resultados = pd.read_csv(ARCHIVO_INFERENCIA)
df_minimax = pd.read_csv(ARCHIVO_ORACULO)

# Mapeo vital para tener la jugada correcta
diccionario_movimientos = dict(zip(df_minimax['tablero'], df_minimax['movimiento_esperado']))
df_resultados['movimiento_esperado'] = df_resultados['tablero'].map(diccionario_movimientos)

def verificar_ganador(tablero, jugador):
    lineas = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
    for linea in lineas:
        if tablero[linea[0]] == tablero[linea[1]] == tablero[linea[2]] == jugador:
            return True
    return False

def extraer_coordenada(texto):
    texto_str = str(texto).upper()
    if "NEXT MOVE" in texto_str:
        texto_busqueda = texto_str.split("NEXT MOVE")[-1]
    else:
        texto_busqueda = texto_str
        
    match = re.search(r'\[\s*(\d)\s*,\s*(\d)\s*\]', texto_busqueda)
    if match:
        f, c = int(match.group(1)), int(match.group(2))
        if 1 <= f <= 3 and 1 <= c <= 3:
            return f, c
    return None, None

def es_movimiento_legal(tablero_str, f, c):
    tablero = tablero_str.replace('/', '')
    idx = (f - 1) * 3 + (c - 1)
    if 0 <= idx < 9:
        return tablero[idx] == '-'
    return False

def evaluar_consecuencia(tablero_str, f, c, jugador_actual):
    tablero = list(tablero_str.replace('/', ''))
    idx = (f - 1) * 3 + (c - 1)
    oponente = 'O' if jugador_actual == 'X' else 'X'

    tablero_simulado = tablero.copy()
    tablero_simulado[idx] = jugador_actual
    victoria = verificar_ganador(tablero_simulado, jugador_actual)

    tablero_oponente = tablero.copy()
    tablero_oponente[idx] = oponente
    bloqueo = verificar_ganador(tablero_oponente, oponente)

    return victoria, bloqueo

def auditar_jugada(row):
    texto = row['respuesta_bruta_gemma']
    tablero_str = row['tablero']
    jugador = row['rol_asignado']
    jugada_minimax = str(row['movimiento_esperado'])

    if jugada_minimax == "END_GAME":
        se_dio_cuenta = "END_GAME" in str(texto).upper() or "FIN" in str(texto).upper()
        return pd.Series([se_dio_cuenta, None, None, None, se_dio_cuenta])

    f, c = extraer_coordenada(texto)
    formato_valido = (f is not None and c is not None)
    
    if not formato_valido:
        return pd.Series([False, False, False, False, False])

    legal = es_movimiento_legal(tablero_str, f, c)
    if not legal:
         return pd.Series([True, False, False, False, False])

    gano, bloqueo = evaluar_consecuencia(tablero_str, f, c, jugador)
    jugada_gemma_str = f"[{f}, {c}]"
    match_minimax = (jugada_gemma_str == jugada_minimax)

    return pd.Series([True, True, gano, bloqueo, match_minimax])

columnas_metricas = ['formato_valido', 'movimiento_legal', 'gano_partida', 'bloqueo_amenaza', 'match_minimax']
df_resultados[columnas_metricas] = df_resultados.apply(auditar_jugada, axis=1)

total_evaluado = df_resultados['formato_valido'].notna().sum()
total_legales = df_resultados['movimiento_legal'].sum()

print("\n" + "="*65)
print(f" 🚀 REPORTE FINAL TESIS (4B TEXTO PURO - ABLACIÓN) 🚀")
print("="*65)
print(f"Total de evaluaciones: {total_evaluado}")
print(f"✅ Formato Válido: {df_resultados['formato_valido'].sum()} ({df_resultados['formato_valido'].mean()*100:.1f}%)")
print(f"✅ Movimiento Legal: {total_legales} ({df_resultados['movimiento_legal'].mean()*100:.1f}%)")
print(f"🏆 Logró Ganar:      {df_resultados['gano_partida'].sum()}")
print(f"🛡️ Logró Bloquear:   {df_resultados['bloqueo_amenaza'].sum()}")
print(f"🤖 Match Minimax:    {df_resultados['match_minimax'].sum()} ({df_resultados['match_minimax'].mean()*100:.1f}%)")
print("="*65)

print(f"\n🔍 ANÁLISIS DE CONSISTENCIA (10 ITERACIONES POR TABLERO) 🔍")
consistencia = df_resultados.groupby('tablero')['match_minimax'].mean() * 100
print(f"Tableros dominados (100% aciertos): {(consistencia == 100).sum()}")
print(f"Tableros incomprendidos (0% aciertos): {(consistencia == 0).sum()}")
print(f"Tableros inestables (1-99% aciertos): {((consistencia > 0) & (consistencia < 100)).sum()}")
print("="*65)

df_resultados.to_csv(ARCHIVO_FINAL, index=False)
print(f"\n✅ Pipeline Maestro 4B TEXTO finalizado con éxito. Resultados en: {ARCHIVO_FINAL}")