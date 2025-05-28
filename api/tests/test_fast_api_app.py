import pytest
import numpy as np
from PIL import Image
import io
import requests

def test_predict_endpoint():
    url = "http://127.0.0.1:8000/predict/"
    file_path = r"C:\Users\User\Desktop\DL_transfer_CNN_pneumonia\data\test\PNEUMONIA\person1_virus_6.jpeg"  # Assurez-vous d’avoir une image JPEG

    with open(file_path, "rb") as f:
        files = {"file": ("sample.jpg", f, "image/jpeg")}
        response = requests.post(url, files=files)

    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert isinstance(data["predicted_class"], str)
    assert isinstance(data["confidence"], float)
