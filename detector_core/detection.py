from .social_distancing_config import *
import numpy as np
import cv2

# Parametros
# frame: frame do video ou da webcam
# net: pre-inicializado e pre-treinado modelo que detecta objeto YOLO
# In: nomes das camadas de saida do YOLO CNN
# personIdx: esse index é específico da classe de Pessoa do YOLO
def detect_people(frame, net, In, personIdx=0):
    
    # Pega as dimensões da moldura (que divide a imagem em pixels)
    # e inicializa a lista de resultados
    (H, W) = frame.shape[:2]
    results = []
    
    # Constroi um blob do frame de entrada e avança
    # passa pelo detector de objetos YOLO passando 
    # as caixas de delimitação e probabilidades associadas
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(In)
    
    boxes = []
    centroids = []
    confidences = []
    
    # Loop acima de cada camada de saida
    for output in layerOutputs:
        # Loop para cada detections
        for detection in output:
            # Extrai o ID da classe (probabilidade)
            # do objeto atual detectado
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            # Filtra a detecção para:
            # 1 - assegurar que o objeto detectado foi uma pessoa;
            # 2 - a confianca foi alcancada
            if classID == personIdx and confidence > MIN_CONF:
                # Escalar as cordenadas da caixa de delimitação
                # em relação ao tamanho da imagem, devolvendo 
                # o (x, y) - coordenadas da caixa de delimitação
                # em seguida a largura e altura
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                
                # Usa as coordenadas para derivar
                # o topo e o canto da caixa de delimitação
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))
                
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
    
    # Aplica a supressão non-maxima para 
    # suprimir as caixas de delimitação fracas
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)
    
    # Checando se existe ao menos uma detecção
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Extraindo coordenadas da caixa delimitadora
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            # Atualizando nossa lista de resultados de pessoas para 
            # manter probabilidade de predição, coordenadas e controle
            r = (confidences[i], (x, y, x+w, y+h), centroids[i])
            results.append(r)
    return results

