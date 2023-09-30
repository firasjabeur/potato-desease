from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



MODEL = tf.keras.models.load_model("../saved_models/2")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return "hello python"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file: UploadFile
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(image_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    #np.argmax([0.00078,  0999, 0.00048]) ==> 1: indice du maximum number
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }




if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)