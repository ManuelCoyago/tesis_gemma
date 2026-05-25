import os
import pandas as pd
from PIL import Image, ImageDraw

# ==========================================
# CONFIGURACIÓN ARQUITECTÓNICA (ViT / SigLIP)
# ==========================================
IMG_SIZE = 448       # Resolución óptima
PADDING = 14         # Margen exacto de 1 parche ViT
CELL_SIZE = 140      # Tamaño de celda exacto (448 - 28) / 3
LINE_WIDTH = 14      # Grosor de línea de 1 parche ViT exacto
SYMBOL_MARGIN = 28   # Margen interno para las X y O (2 parches)

# Colores Absolutos (Alta discriminación de canal)
COLOR_BG = (255, 255, 255) # Blanco puro
COLOR_GRID = (0, 0, 0)     # Negro puro
COLOR_X = (0, 0, 255)      # Azul puro (Canal B)
COLOR_O = (255, 0, 0)      # Rojo puro (Canal R)

def draw_board(board_str):
    """
    Toma un string tipo 'X-O/---/--X' y devuelve un objeto Image de PIL.
    """
    # Crear fondo blanco
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=COLOR_BG)
    draw = ImageDraw.Draw(img)

    # 1. DIBUJAR LA CUADRÍCULA DEL TABLERO
    # Líneas Verticales
    draw.line([(PADDING + CELL_SIZE, PADDING), 
               (PADDING + CELL_SIZE, IMG_SIZE - PADDING)], 
              fill=COLOR_GRID, width=LINE_WIDTH)
    draw.line([(PADDING + 2 * CELL_SIZE, PADDING), 
               (PADDING + 2 * CELL_SIZE, IMG_SIZE - PADDING)], 
              fill=COLOR_GRID, width=LINE_WIDTH)
    
    # Líneas Horizontales
    draw.line([(PADDING, PADDING + CELL_SIZE), 
               (IMG_SIZE - PADDING, PADDING + CELL_SIZE)], 
              fill=COLOR_GRID, width=LINE_WIDTH)
    draw.line([(PADDING, PADDING + 2 * CELL_SIZE), 
               (IMG_SIZE - PADDING, PADDING + 2 * CELL_SIZE)], 
              fill=COLOR_GRID, width=LINE_WIDTH)

    # 2. DIBUJAR LAS X y LAS O
    rows = board_str.split('/')
    
    for row_idx, row_chars in enumerate(rows):
        for col_idx, char in enumerate(row_chars):
            if char == '-':
                continue
            
            # Calcular las coordenadas exactas de la casilla actual
            x_min = PADDING + col_idx * CELL_SIZE
            y_min = PADDING + row_idx * CELL_SIZE
            x_max = x_min + CELL_SIZE
            y_max = y_min + CELL_SIZE
            
            if char == 'X':
                # Dibujar X (Dos líneas diagonales)
                draw.line([(x_min + SYMBOL_MARGIN, y_min + SYMBOL_MARGIN), 
                           (x_max - SYMBOL_MARGIN, y_max - SYMBOL_MARGIN)], 
                          fill=COLOR_X, width=LINE_WIDTH)
                draw.line([(x_min + SYMBOL_MARGIN, y_max - SYMBOL_MARGIN), 
                           (x_max - SYMBOL_MARGIN, y_min + SYMBOL_MARGIN)], 
                          fill=COLOR_X, width=LINE_WIDTH)
            elif char == 'O':
                # Dibujar O (Círculo geométrico perfecto)
                draw.ellipse([x_min + SYMBOL_MARGIN, y_min + SYMBOL_MARGIN, 
                              x_max - SYMBOL_MARGIN, y_max - SYMBOL_MARGIN], 
                             outline=COLOR_O, width=LINE_WIDTH)
    return img

def generate_dataset(csv_path, output_dir):
    """
    Lee el CSV, crea la estructura de carpetas y genera todas las imágenes.
    """
    # Crear estructura de carpetas
    etapas = ["Inicial", "Intermedia", "Final"]
    for etapa in etapas:
        # exist_ok=True evita errores si la carpeta ya existe
        os.makedirs(os.path.join(output_dir, etapa), exist_ok=True)
        
    print(f"Cargando dataset desde: {csv_path}...")
    df = pd.read_csv(csv_path)
    total_imgs = len(df)
    
    print(f"Iniciando renderizado de {total_imgs} imágenes. Por favor espera...\n")
    
    for idx, row in df.iterrows():
        tablero = row['tablero']
        etapa = row['etapa_juego']
        
        # Renderizar la imagen
        img = draw_board(tablero)
        
        # Crear un nombre de archivo seguro (ej. 'X-O_---_--X.png')
        filename = tablero.replace('/', '_') + '.png'
        filepath = os.path.join(output_dir, etapa, filename)
        
        # GUARDAR COMO PNG SIN PÉRDIDA
        img.save(filepath, "PNG")
        
        # Imprimir progreso cada 500 imágenes
        if (idx + 1) % 500 == 0:
            print(f"Progreso: {idx + 1} / {total_imgs} imágenes renderizadas...")
            
    print("\n¡Pipeline visual completado exitosamente!")
    print(f"Las {total_imgs} imágenes están listas en la carpeta: '{output_dir}'")

# ==========================================
# PUNTO DE EJECUCIÓN
# ==========================================
if __name__ == "__main__":

    ARCHIVO_CSV = "dataset_jugadas_ttt.csv"
    CARPETA_SALIDA = "dataset_imagenes_ttt"
    
    generate_dataset(ARCHIVO_CSV, CARPETA_SALIDA)