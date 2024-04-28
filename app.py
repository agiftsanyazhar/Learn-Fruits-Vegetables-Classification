import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

model = load_model("Image_Classification.keras")

data_cat = [
    "apple",
    "banana",
    "beetroot",
    "bell pepper",
    "cabbage",
    "capsicum",
    "carrot",
    "cauliflower",
    "chilli pepper",
    "corn",
    "cucumber",
    "eggplant",
    "garlic",
    "ginger",
    "grapes",
    "jalepeno",
    "kiwi",
    "lemon",
    "lettuce",
    "mango",
    "onion",
    "orange",
    "paprika",
    "pear",
    "peas",
    "pineapple",
    "pomegranate",
    "potato",
    "raddish",
    "soy beans",
    "spinach",
    "sweetcorn",
    "sweetpotato",
    "tomato",
    "turnip",
    "watermelon",
]

img_height = 180
img_width = 180

image = "paprika.jpg"

image_load = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_batch = tf.expand_dims(img_arr, 0)

predict = model.predict(img_batch)

score = tf.nn.softmax(predict)
st.image(image)

st.write(
    "vegetable/fruit in image is {} with accuracy of {:.2f}%".format(
        data_cat[np.argmax(score)], np.max(score) * 100
    )
)
