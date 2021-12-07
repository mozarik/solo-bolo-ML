import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image

# Preprocess image being uploaded
IMAGE_SIZE = 224


def prepare_img(img_raw, img_height=IMAGE_SIZE, img_width=IMAGE_SIZE):
    # img = load_img(img_raw, target_size=(img_height, img_width))
    img_array = np.array([img_to_array(img_raw)])
    return preprocess_input(img_array)


def init_model():
    model = load_model("model.h5")
    return model
