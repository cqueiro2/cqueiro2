import cv2
import torch
import numpy as np
import os
import csv
from pathlib import Path

# --- CONFIGURAÇÕES E INICIALIZAÇÃO ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

model_type = "MiDaS_small"
midas = torch.hub.load("isl-org/MiDaS", model_type)
midas.to(device).eval()
midas_transforms = torch.hub.load("isl-org/MiDaS", "transforms")
transform = midas_transforms.small_transform if model_type == "MiDaS_small" else midas_transforms.dpt_transform

def estimar_volume(caminho_img):
    img = cv2.imread(str(caminho_img))
    if img is None: return None
    
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    depth_norm = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Suavização e Máscara
    blurred = cv2.GaussianBlur(depth_norm, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    volume_pixel_depth = np.sum(depth_map[mask == 255])
    return float(volume_pixel_depth)

def buscar_peso_no_dicionario(nome_arquivo, pesos_bovinos):
    """
    Tenta extrair Raça, Sexo e Idade do nome do arquivo.
    Exemplo esperado: 'Nelore_macho_210_dias.jpg'
    """
    try:
        # Remove a extensão e separa por '_'
        partes = nome_arquivo.replace('.jpg', '').replace('.jpeg', '').replace('.png', '').split('_')
        raca = partes[0]      # Nelore
        sexo = partes[1]      # macho
        idade = f"{partes[2]}_{partes[3]}" # 210_dias
        
        return pesos_bovinos[raca][sexo][idade]
    except (KeyError, IndexError):
        return None

def gerar_calibracao_complexa(pasta_fotos, pesos_bovinos):
    pasta = Path(pasta_fotos)
    if not pasta.exists():
        print(f"Erro: Pasta {pasta} não encontrada.")
        return

    caminho_csv = 'calibragem_raca_idade.csv'
    arquivos = [f for f in os.listdir(pasta) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    with open(caminho_csv, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Arquivo", "Raça", "Sexo", "Idade", "Volume_IA", "Peso_Referencia", "K_Fator"])

        processados = 0
        for foto in arquivos:
            peso_ref = buscar_peso_no_dicionario(foto, pesos_bovinos)
            
            if peso_ref:
                print(f"Processando {foto} (Peso Ref: {peso_ref}kg)...")
                volume_ia = estimar_volume(pasta / foto)
                
                if volume_ia and volume_ia > 0:
                    k_fator = peso_ref / volume_ia
                    # Extraindo info para o CSV
                    partes = foto.split('_')
                    writer.writerow([foto, partes[0], partes[1], partes[2], f"{volume_ia:.2f}", peso_ref, f"{k_fator:.8f}"])
                    processados += 1
            else:
                print(f"Pulo: {foto} não segue o padrão 'Raca_Sexo_Idade_dias.jpg'")

    print(f"\nConcluído! {processados} animais processados. Resultado em: {caminho_csv}")

# --- SEU DICIONÁRIO MANTIDO ---
pesos_bovinos = {
    "Nelore": {
        "macho": {"1_dia": 32, "210_dias": 180, "450_dias": 330},
        "femea": {"1_dia": 30, "210_dias": 170, "450_dias": 300}
    },
    "Angus": {
        "macho": {"1_dia": 35, "210_dias": 220, "450_dias": 380},
        "femea": {"1_dia": 33, "210_dias": 210, "450_dias": 350}
    },
    "Hereford": {
        "macho": {"1_dia": 36, "210_dias": 210, "450_dias": 370},
        "femea": {"1_dia": 34, "210_dias": 200, "450_dias": 340}
    },
    "Brahman": {
        "macho": {"1_dia": 34, "210_dias": 200, "450_dias": 360},
        "femea": {"1_dia": 32, "210_dias": 190, "450_dias": 330}
    },
    "Tabapua": {
        "macho": {"1_dia": 33, "210_dias": 195, "450_dias": 355},
        "femea": {"1_dia": 31, "210_dias": 185, "450_dias": 325}
    },
    "Holandes": {
        "macho": {"1_dia": 40, "210_dias": 250, "450_dias": 400},
        "femea": {"1_dia": 38, "210_dias": 240, "450_dias": 370}
    },
    "Girolando": {
        "macho": {"1_dia": 38, "210_dias": 230, "450_dias": 380},
        "femea": {"1_dia": 36, "210_dias": 220, "450_dias": 350}
    }
}

caminho_fotos = "./3_segmentarImg/file_img"

if __name__ == "__main__":
    gerar_calibracao_complexa(caminho_fotos, pesos_bovinos)