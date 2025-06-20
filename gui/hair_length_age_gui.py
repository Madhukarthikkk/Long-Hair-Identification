import tkinter as tk
from tkinter import Label
import random
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk

base_path = "C:/Users/karth/Desktop/jupyter projects/LongHairGenderDetection/model_training"
hair_model_path = os.path.join(base_path, "hair_length_classifier_model.h5")
age_model_path = os.path.join(base_path, "age_model.h5")

hair_model = load_model(hair_model_path)
age_model = load_model(age_model_path, compile=False)

test_image_dir = "C:/Users/karth/Desktop/jupyter projects/LongHairGenderDetection/test_images"
IMG_SIZE = 64

def predict_random_image():
    files = os.listdir(test_image_dir)
    if not files:
        label_result.config(text="No test images found.")
        return

    selected_file = random.choice(files)
    image_path = os.path.join(test_image_dir, selected_file)

    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    hair_prediction = hair_model.predict(img_input)[0]
    predicted_class = np.argmax(hair_prediction)
    confidence = round(np.max(hair_prediction) * 100, 2)
    hair_label = "Long Hair" if predicted_class == 1 else "Short Hair"

    gender_label = "Female" if hair_label == "Long Hair" else "Male"

    age_prediction = age_model.predict(img_input)[0][0]
    age_estimate = int(round(age_prediction))

    pil_img = Image.open(image_path)
    pil_img = pil_img.resize((200, 200))
    photo = ImageTk.PhotoImage(pil_img)
    img_label.config(image=photo)
    img_label.image = photo

    label_result.config(
        text=f"Prediction: {hair_label} | Gender: {gender_label}\nEstimated Age: {age_estimate}\nConfidence: {confidence}%"
    )

root = tk.Tk()
root.title("Hair Length, Gender, and Age Predictor")

btn = tk.Button(root, text="Predict Random Image", command=predict_random_image)
btn.pack(pady=10)

img_label = Label(root)
img_label.pack()

label_result = Label(root, text="", font=("Arial", 14))
label_result.pack(pady=10)

root.mainloop()
