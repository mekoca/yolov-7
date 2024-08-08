import sys
import os
import cv2
import torch
import numpy as np
import tempfile
from googletrans import Translator
from gtts import gTTS
from playsound import playsound
from threading import Thread
import time

# Add the path where YOLOv7 is located
sys.path.append('/Users/meliskoca/PycharmProjects/yolo/pythonProject3/yolov7')

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import non_max_suppression, scale_coords

def load_model():
    weights_path = '/Users/meliskoca/PycharmProjects/yolo/pythonProject3/yolov7/yolov7.pt'
    try:
        model = attempt_load(weights_path, map_location=torch.device('cpu'))  # Load model to CPU
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def play_audio(text, lang):
    tts = gTTS(text=text, lang=lang)
    tmpfile = tempfile.NamedTemporaryFile(delete=True, suffix='.mp3')
    tts.save(tmpfile.name)
    playsound(tmpfile.name)
    tmpfile.close()

def detect():
    model = load_model()
    if model is None:
        return

    cap = cv2.VideoCapture(0)  # Open default camera
    translator = Translator()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        img = letterbox(frame, new_shape=(640, 640))[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB
        img = np.ascontiguousarray(img)

        with torch.no_grad():
            img = torch.from_numpy(img).to(torch.device('cpu'))
            img = img.float() / 255.0  # Normalize image
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img, augment=False)[0]
            pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy)
                    label = f'{model.names[int(cls)]} {conf:.2f}%'
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    translated_text_en = translator.translate(model.names[int(cls)], dest='en').text
                    Thread(target=play_audio, args=(translated_text_en, 'en')).start()

        cv2.imshow("YOLOv7 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect()
