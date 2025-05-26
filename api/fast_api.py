from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
from PIL import Image
import io
import uvicorn

app = FastAPI()
model = load_model('../models/mu_model_densenet.keras')

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # garder 3 canaux
    image = image.resize((224, 224))  # redimensionner
    img_array = np.array(image).astype('float32')
    img_array = np.expand_dims(img_array, axis=0)  # ajouter dimension batch (1,224,224,3)
    # Normalisation spécifique MobileNetV2
    img_array = preprocess_input(img_array)
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = preprocess_image(image_bytes)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))
    return {"predicted_class": int(predicted_class), "confidence": confidence}

# --- Pour lancer localement ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
