import cv2
import torch
import numpy as np
import os
import csv
from pathlib import Path

# --- CONFIGURAÇÕES E INICIALIZAÇÃO ---
# Detecta se há GPU disponível (mais rápido para processar muitos animais)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

model_type = "MiDaS_small"
midas = torch.hub.load("isl-org/MiDaS", model_type)
midas.to(device).eval()
midas_transforms = torch.hub.load("isl-org/MiDaS", "transforms")
transform = midas_transforms.small_transform if model_type == "MiDaS_small" else midas_transforms.dpt_transform

def estimar_volume(caminho_img):
    img = cv2.imread(str(caminho_img))
    if img is None: 
        return None
    
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    
    # --- SEGMENTAÇÃO MELHORADA ---
    # Normalizamos para o intervalo 0-255
    depth_norm = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Otsu tende a falhar se o fundo for complexo. Adicionamos um Blur para reduzir ruído.
    blurred = cv2.GaussianBlur(depth_norm, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morfologia: Fecha pequenos buracos dentro do animal e remove pontos isolados
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Fecha buracos
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove ruído externo

    # Cálculo da "Massa de Profundidade"
    volume_pixel_depth = np.sum(depth_map[mask == 255])
    
    return float(volume_pixel_depth)

def gerar_dados_calibracao(pasta_fotos, pesos_reais):
    pasta = Path(pasta_fotos)
    if not pasta.exists():
        print(f"Erro: A pasta '{pasta}' não existe.")
        return

    caminho_csv = 'calibragem_animais_resultado.csv'
    formatos_suportados = ('.jpg', '.jpeg', '.png', '.bmp')
    
    with open(caminho_csv, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Arquivo", "Volume_IA", "Peso_Real", "Fator_K", "Status"])
        
        processados = 0
        
        # Itera sobre os arquivos da pasta que estão no dicionário
        for nome_arquivo, peso_real in pesos_reais.items():
            caminho_foto = pasta / nome_arquivo
            
            if not caminho_foto.exists():
                print(f"Aviso: Arquivo {nome_arquivo} não encontrado na pasta.")
                writer.writerow([nome_arquivo, 0, peso_real, 0, "Não encontrado"])
                continue

            print(f"Processando animal: {nome_arquivo}...")
            volume_ia = estimar_volume(caminho_foto)
            
            if volume_ia and volume_ia > 0:
                k_ideal = peso_real / volume_ia
                writer.writerow([nome_arquivo, f"{volume_ia:.4f}", peso_real, f"{k_ideal:.8f}", "Sucesso"])
                processados += 1
            else:
                writer.writerow([nome_arquivo, 0, peso_real, 0, "Erro no processamento"])

    print("-" * 30)
    print(f"Processamento concluído!")
    print(f"Total de animais calibrados: {processados}")
    print(f"Arquivo gerado: {caminho_csv}")

# --- CONFIGURAÇÃO DE ENTRADA ---

# Dicionário mapeando Nome do Arquivo -> Peso Real (kg)
meus_pesos = {
    'boi1.jpg': 550.5,
    'vaca_matriz.png': 480.0,
    'bezerro_02.jpeg': 120.0
}

# Caminho da pasta onde estão as fotos
caminho_fotos = "./3_segmentarImg/file_img"

# Execução
if __name__ == "__main__":
    gerar_dados_calibracao(caminho_fotos, meus_pesos)