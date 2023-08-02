import cv2
import numpy as np
from PIL import Image
import os

def train_face_recognition_model(samples_path, cascade_file, output_model_file):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cascade_file)

    def extract_face_samples_and_ids(path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []

        for image_path in image_paths:
            gray_img = Image.open(image_path).convert('L')
            img_arr = np.array(gray_img, 'uint8')

            id = int(os.path.split(image_path)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_arr)

            for (x, y, w, h) in faces:
                face_samples.append(img_arr[y:y+h, x:x+w])
                ids.append(id)

        return face_samples, ids

    print("Training faces. It will take a few seconds. Please wait...")

    faces, ids = extract_face_samples_and_ids(samples_path)
    recognizer.train(faces, np.array(ids))

    recognizer.write(output_model_file)
    print("Model trained. The trained model is saved as:", output_model_file)

samples_path = 'samples'
cascade_file = "haarcascade_frontalface_default.xml"
output_model_file = 'trainer/trainer.yml'

train_face_recognition_model(samples_path, cascade_file, output_model_file)
