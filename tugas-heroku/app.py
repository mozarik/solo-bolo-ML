import preprocess as pp
import streamlit as st
import predict as pred
from PIL import Image

CLASSES = {0: "daisy", 1: "dandelion", 2: "rose", 3: "sunflower", 4: "tulip"}
IMAGE_SIZE = 224

def main():
    # init object
    model = pp.init_model()

    # streamlit code
    st.text('Masukan gambar')
    uploaded_file = st.file_uploader("Upload file", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        img = pp.prepare_img(image)
        prediction = pred.predict(img, model)
        result = CLASSES[prediction[0]]

        button_gen = st.button("Generate Prediction")
        if button_gen:
            st.title("Predicted label for the image is {}".format(result))


if __name__ == "__main__":
    main()
