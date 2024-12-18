from ultralytics import YOLO
import cv2

# carregar o modelo
model = YOLO("best.pt")

# Caminho da imagem
image_path = "picture.png"

# Realizar a inferência no modelo
results = model.predict(source=image_path, save=True, save_txt=True)

# Carregar a imagem original para exibir as caixas
image = cv2.imread(image_path)

# Iterar sobre as detecções
for result in results:
    for box in result.boxes.xyxy:  # Coordenadas da caixa delimitadora [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Desenhar a caixa na imagem

# Mostrar a imagem com as caixas
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
