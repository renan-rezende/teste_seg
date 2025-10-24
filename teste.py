from ultralytics import YOLO
import cv2
import time

# Lista de modelos para testar (caminho do modelo / caminho de saida da imagem)
modelos = [
    ("models/GRAY_RESIZE.pt", "images/output/resultado_GRAY.jpg"),
    ("models/RGB_FHD.pt", "images/output/resultado_RGB.jpg"),
    ("models/RGB_FHD_UPISE_DOWN.pt", "images/output/resultado_UPSIE.jpg"),
]

# Caminho da imagem de teste
input_image = "images/input/frame_00010.jpg"

for modelo_path, nome_saida in modelos:
    print(f"\n🔍 Testando modelo: {modelo_path}")
    model = YOLO(modelo_path)
    
    # Medir tempo de referencia
    inicio = time.time()
    results = model.predict(
        source=input_image,
        imgsz=1920,         # mantém resolução nativa (se imagem for 1920px)
        conf=0.7,
        retina_masks=True,  # máscaras com mais qualidade
        verbose=False,
        save=False
    )
    tempo = time.time() - inicio

    r = results[0]  
    num_detec = len(r.boxes)      

    print(f"📦 Detecções encontradas: {num_detec}")
    print(f"Tempo de inferência: {tempo:.3f} s")

    img_seg = r.plot(boxes=False)       # só segmentação (sem bbox)
    cv2.imwrite(nome_saida, img_seg)

    print(f"✅ Imagem salva como: {nome_saida}")

print("\n✅ Testes concluídos!")
