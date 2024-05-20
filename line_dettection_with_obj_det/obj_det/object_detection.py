import cv2
import numpy as np
import os

# Получение пути к директории скрипта
script_dir = os.path.dirname(os.path.abspath(__file__))

# Указание полных путей к файлам
weights_path = os.path.join(script_dir, "yolov3.weights")
config_path = os.path.join(script_dir, "yolov3.cfg")
names_path = os.path.join(script_dir, "coco.names")

# Проверка наличия файлов
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"File not found: {weights_path}")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"File not found: {config_path}")
if not os.path.exists(names_path):
    raise FileNotFoundError(f"File not found: {names_path}")

# Загрузка YOLO
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Функция для обнаружения объектов
def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [(boxes[i], class_ids[i]) for i in indexes]
