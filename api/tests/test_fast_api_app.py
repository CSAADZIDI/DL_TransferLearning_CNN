from fastapi.testclient import TestClient
from fast_api_app import app

client = TestClient(app)

def test_predict_pneumonia_or_normal():
    image_path = "data/test_image.jpeg"  # remplace par un vrai chemin vers une image test

    with open(image_path, "rb") as image_file:
        response = client.post(
            "/predict/",
            files={"file": ("test_image.jpeg", image_file, "image/jpeg")}
        )

    assert response.status_code == 200

    json_data = response.json()
    assert "predicted_class" in json_data
    assert json_data["predicted_class"] in ["PNEUMONIA", "NORMAL"]

    assert "confidence" in json_data
    assert isinstance(json_data["confidence"], float)
    assert 0.0 <= json_data["confidence"] <= 1.0
