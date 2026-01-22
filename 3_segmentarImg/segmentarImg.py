import cv2
import torch
import numpy as np
import os

# --- 1. Configurar o dispositivo e carregar o modelo MiDaS ---
device = torch.device("cpu")
model_type = "MiDaS_small"
midas = torch.hub.load("isl-org/MiDaS", model_type)
midas.to(device).eval()

midas_transforms = torch.hub.load("isl-org/MiDaS", "transforms")
transform = midas_transforms.small_transform if model_type == "MiDaS_small" else midas_transforms.dpt_transform

# --- 2. Configurar as pastas ---
pasta_entrada = './1_DimensioImag/fotos_saida/'  # Pasta onde estão as fotos originais
pasta_saida = './3_segmentarImg/file_img/' # Pasta onde os recortes serão salvos

if not os.path.exists(pasta_saida):
    os.makedirs(pasta_saida)

# --- 3. Processar várias fotos na pasta ---
extensoes_suportadas = ('.jpg', '.jpeg', '.png', '.bmp')

print(f"Iniciando processamento na pasta: {pasta_entrada}")

for nome_arquivo in os.listdir(pasta_entrada):
    if nome_arquivo.lower().endswith(extensoes_suportadas):
        caminho_completo = os.path.join(pasta_entrada, nome_arquivo)
        
        # 4. Ler a imagem
        img = cv2.imread(caminho_completo)
        if img is None:
            print(f"Erro ao carregar: {nome_arquivo}")
            continue

        print(f"Processando: {nome_arquivo}...")

        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = transform(img_rgb).to(device)

        # 5. Executar a Inferência
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # 6. Criar Máscara de Segmentação (Otsu)
        depth_norm = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, mask = cv2.threshold(depth_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 7. Aplicar a máscara para isolar o boi (Fundo Preto)
        segmented_img = cv2.bitwise_and(img, img, mask=mask)

        # 8. Salvar o resultado
        nome_saida = f"seg_{nome_arquivo}"
        cv2.imwrite(os.path.join(pasta_saida, nome_saida), segmented_img)

print("\nConcluído! Todas as imagens foram segmentadas e salvas.")