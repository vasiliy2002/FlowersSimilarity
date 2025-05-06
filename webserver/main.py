import io
import json
import config
import numpy as np
from tensorflow.keras.utils import img_to_array
from PIL import Image
from fastapi import FastAPI, File
from starlette.requests import Request
from keras.applications import VGG16
from keras import layers, Model
from tensorflow.keras import ops
from sklearn.neighbors import NearestNeighbors
import uvicorn
from numpy import dot
from numpy.linalg import norm

def get_pathes():
    with open(config.IMAGE_PATHES, "r") as f:
        pathes = json.load(f)
    return pathes

def build_knn():
    knn = NearestNeighbors(n_neighbors=config.N_NEIGHBORS)
    data = np.load(config.LIBRARY_PATH)
    knn.fit(data)
    return knn, data

def euclidean_distance(vects):
    """
    Функция для рассчета евклидовой метрики
    """

    x, y = vects
    sum_square = ops.sum(ops.square(x - y), axis=1, keepdims=True)
    return ops.sqrt(ops.maximum(sum_square, keras.backend.epsilon()))


def build_model():
    #------ Сверточная основа ------
    target_shape = (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3)
    base_cnn = VGG16(weights='imagenet',
      include_top=False,
      input_shape=target_shape)

    #------ Классификатор ------
    flatten = layers.Flatten()(base_cnn.output)
    dense1 = layers.Dense(256, activation="relu")(flatten)
    dense1 = layers.BatchNormalization()(dense1)
    output = layers.Dense(64)(dense1)

    embedding = Model(base_cnn.input, output, name="Embedding")

    # Разделение изображений
    inpt = layers.Input((2, *target_shape))
    input_1 = layers.Lambda(lambda x: x[:, 0, :, :, :])(inpt)
    input_2 = layers.Lambda(lambda x: x[:, 1, :, :, :])(inpt)

    # Параллельный запуск с общими весами
    head_1 = embedding(input_1)
    head_2 = embedding(input_2)

    # Вычисление дистанции между векторами, нахождение "схожести"
    merge_layer = layers.Lambda(euclidean_distance, output_shape=(1,))(
        [head_1, head_2]
    )
    normal_layer = layers.BatchNormalization()(merge_layer)
    output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)
    siamese = Model(inputs=[inpt], outputs=output_layer)
    return siamese

def build_encoder(siamese):
    # Формирование кодировщика
    encoder_output = siamese.get_layer("Embedding").output
    encoder_input = siamese.get_layer("Embedding").input
    encoder = Model(inputs=encoder_input, outputs=encoder_output)    
    return encoder

def get_pathes():
    with open(config.IMAGE_PATHES, "r") as f:
        pathes = json.load(f)
    return pathes

def prepare_image(image):
    target = (config.IMAGE_HEIGHT, config.IMAGE_WIDTH)

    if image.mode != "RGB":
        image = image.convert("RGB")


    image = image.resize(target)
    image = img_to_array(image)
    image = image / 255.
    image = np.expand_dims(image, axis=0)

    return image

def cosine_similarity(a, b):
  return dot(a, b)/(norm(a)*norm(b))

app = FastAPI()
model = build_model()
model.load_weights(config.WEIGHTS)
encoder = build_encoder(model)

knn, X = build_knn()
pathes = get_pathes()



@app.get("/")
def index():
    return "Connected"


@app.post("/predict")
def predict(request: Request, img_file: bytes=File(...)):

    if request.method == "POST":
        result = {}
        result['similar_images_pathes'] = list()
        result['distances'] = list()
        result['cosine_sims'] = list()

        image = Image.open(io.BytesIO(img_file))
        image = prepare_image(image)

        image = image.copy(order="C")
        vec = encoder(image)
        vec = vec.numpy()
        dists, neighbors = knn.kneighbors(vec.reshape(1, -1), return_distance=True)
        for j, neighbor in enumerate(neighbors[0][1:]):
            result['similar_images_pathes'].append(pathes[neighbor])
            result['distances'].append(float(dists[0][j+1]))

            sim = cosine_similarity(vec, X[neighbor])
            result['cosine_sims'].append(float(sim))
        return result

    return {'result': 200}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)