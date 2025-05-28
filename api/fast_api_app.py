from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
from PIL import Image
import cv2
import io
import uvicorn

app = FastAPI()
model = load_model('../models/model_densenet.keras')

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # garder 3 canaux
      # Preprocess for model
    img = np.array(image)
    img_resized = cv2.resize(img, (224, 224))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)
    return img_input

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = preprocess_image(image_bytes)
    #prediction = model.predict(img)
    # Prediction
    
    prediction_proba = model.predict(img)[0][0]
    
    confidence = float(np.max(prediction_proba))
    label = "PNEUMONIA" if prediction_proba > 0.5 else "NORMAL"
    print("label",label, "",type(label))
    return {"predicted_class": label, "confidence": confidence}

# --- Pour lancer localement ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
