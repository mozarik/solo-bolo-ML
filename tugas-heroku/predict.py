import preprocess as pp
import numpy as np


def predict(img_raw, model):
    # imgs = pp.prepare_img(img_raw)
    # predictions = model.predict(imgs)
    predictions = np.argmax(model.predict(img_raw), axis=-1)
    return predictions
