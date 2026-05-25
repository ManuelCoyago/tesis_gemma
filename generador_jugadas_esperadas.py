import pandas as pd
import math

# 1. Función auxiliar para convertir el string 'X-O/---/--X' a una lista de 9 posiciones
def parsear_tablero(tablero_str):
    return list(tablero_str.replace('/', ''))

# 2. Función para evaluar si el estado es terminal
def evaluar_estado_terminal(tablero):
    lineas_ganadoras = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Horizontales
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Verticales
        [0, 4, 8], [2, 4, 6]              # Diagonales
    ]
    for linea in lineas_ganadoras:
        if tablero[linea[0]] == tablero[linea[1]] == tablero[linea[2]] and tablero[linea[0]] != '-':
            return tablero[linea[0]] # Devuelve 'X' o 'O'
    
    if '-' not in tablero:
        return 'Empate'
    return None

# 3. El Algoritmo Minimax recursivo
def minimax(tablero, profundidad, es_maximizador, jugador_ai):
    jugador_humano = 'O' if jugador_ai == 'X' else 'X'
    resultado = evaluar_estado_terminal(tablero)
    
    # Sistema de puntuación
    if resultado == jugador_ai: return 10 - profundidad
    if resultado == jugador_humano: return profundidad - 10
    if resultado == 'Empate': return 0

    if es_maximizador:
        mejor_puntaje = -math.inf
        for i in range(9):
            if tablero[i] == '-':
                tablero[i] = jugador_ai
                puntaje = minimax(tablero, profundidad + 1, False, jugador_ai)
                tablero[i] = '-'
                mejor_puntaje = max(puntaje, mejor_puntaje)
        return mejor_puntaje
    else:
        mejor_puntaje = math.inf
        for i in range(9):
            if tablero[i] == '-':
                tablero[i] = jugador_humano
                puntaje = minimax(tablero, profundidad + 1, True, jugador_ai)
                tablero[i] = '-'
                mejor_puntaje = min(puntaje, mejor_puntaje)
        return mejor_puntaje

# 4. Función principal para calcular el mejor movimiento
def obtener_mejor_movimiento(tablero_str, turno_actual):
    tablero = parsear_tablero(tablero_str)
    
    # Interceptar los estados terminales (Victoria o Empate)
    if evaluar_estado_terminal(tablero) is not None:
        return "END_GAME"
        
    mejor_puntaje = -math.inf
    mejor_movimiento = -1
    
    for i in range(9):
        if tablero[i] == '-':
            tablero[i] = turno_actual
            # Simulamos que el siguiente en jugar es el oponente (minimizar)
            puntaje = minimax(tablero, 0, False, turno_actual)
            tablero[i] = '-'
            
            # En caso de empate técnico, Minimax agarra la primera jugada óptima que encuentre
            if puntaje > mejor_puntaje:
                mejor_puntaje = puntaje
                mejor_movimiento = i
                
    # Convertir el índice plano (0-8) a coordenadas espaciales [Fila, Columna]
    fila = mejor_movimiento // 3
    columna = mejor_movimiento % 3
    return f"[{fila}, {columna}]"

# ==========================================
# PUNTO DE EJECUCIÓN (Llenado del CSV)
# ==========================================
if __name__ == "__main__":
    print("Cargando dataset de estados generados...")
    df = pd.read_csv("dataset_jugadas_ttt.csv")
    
    print("Ejecutando algoritmo Minimax. Calculando 'Ground Truth'...")
    # Aplicar la función a cada fila y crear la nueva columna
    df['movimiento_esperado'] = df.apply(lambda row: obtener_mejor_movimiento(row['tablero'], row['turno_de']), axis=1)
    
    # Guardar el dataset maestro listo para la evaluación
    df.to_csv("dataset_jugadas_ttt_minimax.csv", index=False)
    
    print("\n¡Oráculo matemático completado!")
    print(f"Total de estados etiquetados: {len(df)}")
    print("Revisa el archivo 'dataset_jugadas_ttt_minimax.csv'.")
    