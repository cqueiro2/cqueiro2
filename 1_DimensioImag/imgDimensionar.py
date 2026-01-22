import cv2 as cv
import os

def redimensionar_letterbox(caminho_origem, caminho_destino, largura_alvo, altura_alvo):

# 1. Criar pasta de destino sse não exister
    if not os.path.exists(caminho_destino):
        os.makedirs(caminho_destino)

# 2. Percorrer os aquivos da pasta
    for nome_arquivo in os.listdir(caminho_origem):
        if nome_arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
           img_path = os.path.join(caminho_origem, nome_arquivo)
           img = cv.imread(img_path)

        if img is None:
           continue

        # 3. Calcular proporções
        h, w = img.shape[:2]
        proporcao = min(largura_alvo / w, altura_alvo / h)
        nova_largura = int(w * proporcao)
        nova_altura = int(h * proporcao)


        # 4. Redimensionar mantendo o aspecto
        img_redimensionada = cv.resize(img, (nova_largura, nova_altura), interpolation=cv.INTER_AREA)

        # 5. Adicionar bordas (Latterboxing)
        delta_w = largura_alvo - nova_largura
        delta_h = altura_alvo - nova_altura
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        cor_borda = [0, 0, 0] #preto
        img_final = cv.copyMakeBorder(img_redimensionada, top, bottom, left, right,
                                      cv.BORDER_CONSTANT, value=cor_borda)
        
         # 6. Salvar resultado
        caminho_final = os.path.join(caminho_destino, nome_arquivo)
        cv.imwrite(caminho_final, img_final)
        print(f"Processada: {nome_arquivo}")

#exemplo de uso
redimensionar_letterbox('fotos_originais','fotos_saida', 600, 640)

