# Path do diretorio do YOLO
MODEL_PATH = "yolo-coco"

# Configuracao para iniciar a probabilidade minima
# para filtrar deteccoes fracas e limitar ao aplicar
# a supressao nao maxima
MIN_CONF = 0.3
NMS_THRESH = 0.3

# Boolean que indica se NVIDIA CUDA GPU deve ser usada
USE_GPU = False

# Definindo a distancia (em pixels) segura para duas ou mais pessoas
MIN_DISTANCE = 50