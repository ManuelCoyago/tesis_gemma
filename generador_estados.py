import pandas as pd

# Función para verificar si un jugador ha ganado
def verificar_ganador(tablero, jugador):
    # Filas, columnas y diagonales
    lineas = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Filas
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columnas
        [0, 4, 8], [2, 4, 6]              # Diagonales
    ]
    for linea in lineas:
        if tablero[linea[0]] == tablero[linea[1]] == tablero[linea[2]] == jugador:
            return True
    return False

# Búsqueda en Profundidad (DFS) para generar estados
def generar_estados_dfs(tablero, turno_actual, estados_visitados):
    # Convertir a string para guardarlo y evitar procesar duplicados
    str_tablero = "".join(tablero)
    if str_tablero in estados_visitados:
        return
    
    # Contar fichas y clasificar
    num_fichas = 9 - tablero.count('-')
    
    if num_fichas <= 3:
        etapa = "Inicial"
    elif num_fichas <= 6:
        etapa = "Intermedia"
    else:
        etapa = "Final"
        
    # Formato visual X-O/---/--X
    formato_visual = f"{str_tablero[0:3]}/{str_tablero[3:6]}/{str_tablero[6:9]}"
    
    estados_visitados[str_tablero] = {
        "tablero": formato_visual,
        "num_fichas": num_fichas,
        "etapa_juego": etapa,
        "turno_de": turno_actual
    }
    
    # Condición de parada: Si alguien ya ganó o el tablero está lleno
    oponente = 'O' if turno_actual == 'X' else 'X'
    if verificar_ganador(tablero, oponente) or num_fichas == 9:
        return

    # Generar siguientes movimientos legales
    for i in range(9):
        if tablero[i] == '-':
            nuevo_tablero = list(tablero)
            nuevo_tablero[i] = turno_actual
            generar_estados_dfs(nuevo_tablero, oponente, estados_visitados)

# Inicializar y ejecutar
estados_visitados = {}
tablero_vacio = ['-'] * 9

print("Generando el árbol completo del juego (DFS)...")
generar_estados_dfs(tablero_vacio, 'X', estados_visitados)

# Convertir a DataFrame y guardar
df_estados = pd.DataFrame(list(estados_visitados.values()))

print("\n--- Distribución Matemática Generada ---")
print(df_estados['num_fichas'].value_counts().sort_index())

print("\n--- Resumen por Etapa ---")
print(df_estados['etapa_juego'].value_counts())

df_estados.to_csv("dataset_jugadas_ttt.csv", index=False)
print(f"\nTotal de estados únicos guardados: {len(df_estados)}")