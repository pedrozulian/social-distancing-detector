from detector_core import social_distancing_config as config
from detector_core.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os

# Construir o argumento a analisar e analisar o argumento (para linha de comando)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="", help="endereco (opicional) para input de video")
ap.add_argument("-o", "--output", type=str, default="", help="endereco (opcional) para output do video")
ap.add_argument("-d", "--display", type=int, default=1, help="se deve ou nao ser exibido o video")
args = vars(ap.parse_args())

# Caregar nome das classes COCO em que nosso modelo YOLO foi treinado
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# Deriva os paths para o YOLO configurar modelo e pesos
print('Linha 22')
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])
print('ConfigPath', configPath)
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
print('weightsPath', weightsPath)

# Carrega nosso objeto de detecção treinado YOLO com dataset
print("[INFO] carregando YOLO do disco...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Checando se vamos usar a GPU
if config.USE_GPU:
    print("[INFO] configurando backend e alvos preferiveis a CUDA (GPU)...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Determina somente o *output* nome das camadas que precisamos do YOLO
In = net.getLayerNames()
In = [In[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Inicializa o stream de video e pontua a saida do video
print("[INFO] acessando stream de video...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None


# Loop nos frames do video
while True:
    # Le o proximo frame do arquivo
    (grabbed, frame) = vs.read()

    # Se não houver frame, então finaliza a stream do video
    if not grabbed:
        break
    
    # Redimensona o frame e depois detecta a pessoa
    frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, In, personIdx=LABELS.index("person"))

    # Inicializa o cojunto de index que violam a distancia social
    violate = set()

    # Assegurando que existam ao menos 2 pessoas para calcular a distancia em pares
    if len(results) >= 2:
        # Extrai todos centroids do resultado e computa
        # a distancia Euclideana entre os pares do centroid
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        # Loop sobre o triangulo superior da matriz de distancia
        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
                # Checa se a distancia entre qualquer par de centroid
                # esta abaixo do numero de pixels (ditanciamento) da configuracao
                if D[i, j] < config.MIN_DISTANCE:
                    # Atualiza nossa lista de violacao
                    # com o index dos centroids
                    violate.add(i)
                    violate.add(j)


    # Anotando os frames com triangulos, circulos e texto
    for (i, (prob, bbox, centroid)) in enumerate(results):
        # Extrai a caixa delimitadora e coordenadas do centroid
        # entao inicializa a anotacao com cor
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        # Se o index aparece na lista de violados (violate)
        if i in violate:
            color = (0, 0, 255)
        
        # Retangulo: caixa delimitadora em volta da pessoa
        # Circulo: coordenadas do centroid das pessoas
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)
    
    # Texto com total de pessoas violando distanciamento social
    text = "Violacao do distanciamento social: {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

    # Verifica se o frame de saida deve ser mostrado na tela
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Se apertar q para o script
        if key == ord("q"):
            break
    
    if args["output"] != "" and writer is None:
        # Inicia a leitura do video
        fourcc = cv2.VideoWriter_fourcc("M","J","P","G")
        writer = cv2.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)
    
    if writer is not None:
        writer.write(frame)